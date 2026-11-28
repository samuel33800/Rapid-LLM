#!/usr/bin/env python3
"""Exhaustive single-hardware sweep for Project 3 (Block Diffusion LLM).

Covers every combination of:
  - Batch size:         1, 8, 16, 24, 32, 48, 64
  - Prompt length:      512, 2048, 8192, 32768
  - Output length:      64, 128, 256, 512
  - Block size (b):     4, 8, 16, 32
  - Refinement rounds:  2, 4, 8

Optimizations (producing mathematically identical results to eval_block_diffusion.py):
  1. Prefill caching: prefill time and prefill peak memory depend only on
     (batch_size, prompt_len) — computed once per pair (28 runs instead of 1,456).
  2. Analytical R-scaling: decode time, model calls, TTFT decode term, and KV fetch
     scale linearly with refinement_rounds R, and decode peak memory is independent
     of R — computed once per (batch_size, prompt_len, decode_len, b) at R=1
     (560 runs instead of 1,456).

Outputs written to `output/all_sweeps/` (by default):
  - 48 CSV files (`sweep_p{prompt}_b{b}_r{R}.csv`), each matching the layout of
    `eval_block_diffusion.py` (all batches x outputs, Normal vs. Block Diffusion).
  - `eval_tables.md`: all 48 Markdown evaluation tables produced by
    `eval_block_diffusion.format_markdown_table`.
  - `compiled_results.txt`: summary report (also printed to stdout) highlighting
    the best block-diffusion speedup, best autoregressive performance, and
    minimal-speedup crossover (closest to 1.0x) for both Total and Decode
    speedups, across all configs and within the GPU memory limit.
"""

import argparse
import contextlib
import csv
import io
import itertools
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

import config
import llm_util
from eval_block_diffusion import (
    DEFAULT_HW_CONFIG,
    DEFAULT_MODEL_CONFIG,
    REPO_ROOT,
    format_markdown_table,
    load_model_config_with_overrides,
)
from inference_timing import TimeCalculationLLMInference
from memory_estimation import MemoryEstimator
from train_timing import LLMExecutionDispatcher


DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "all_sweeps"
DEFAULT_BATCHES = [1, 8, 16, 24, 32, 48, 64]
DEFAULT_PROMPTS = [512, 2048, 8192, 32768]
DEFAULT_OUTPUTS = [64, 128, 256, 512]
DEFAULT_BLOCK_SIZES = [4, 8, 16, 32]
DEFAULT_ROUNDS = [2, 4, 8]

# Per-worker cached configs to avoid re-parsing YAML or pickling HWConfig per task.
_WORKER_HW_CONFIG = None
_WORKER_BASE_YAML = None


def _init_worker(hw_config_path: str, model_config_path: str) -> None:
    global _WORKER_HW_CONFIG, _WORKER_BASE_YAML
    _WORKER_HW_CONFIG = config.parse_config(hw_config_path, config_type="hardware")
    with open(model_config_path, "r", encoding="utf-8") as handle:
        _WORKER_BASE_YAML = yaml.safe_load(handle)


def _run_prefill_worker(task: Tuple[int, int]) -> Tuple[Tuple[int, int], Dict[str, float]]:
    """Compute prefill time and prefill peak memory once for (batch_size, prompt_len)."""
    batch_size, prompt_len = task
    model_config = load_model_config_with_overrides(
        _WORKER_BASE_YAML,
        _WORKER_HW_CONFIG,
        global_batch_size=batch_size,
        prompt_len=prompt_len,
        decode_len=0,
        block_size=1,
        refinement_rounds=1,
    )
    with tempfile.TemporaryDirectory(prefix="rapid_prefill_") as tmp_out:
        tc = TimeCalculationLLMInference(_WORKER_HW_CONFIG, model_config, "LLM", output_dir=tmp_out)
        with contextlib.redirect_stdout(io.StringIO()):
            prefill_time, prefill_energy = tc.calc_time()
        prefill_peak_gib = float(tc.memory_peak_gb)
    return task, {
        "prefill_time_s": float(prefill_time),
        "prefill_energy_j": float(prefill_energy),
        "prefill_peak_gib": prefill_peak_gib,
    }


def _compute_decode_peak_gib(tc: TimeCalculationLLMInference, *, decode_len: int, block_size: int) -> float:
    """Compute final decode peak memory without re-running prefill (mirrors calc_time)."""
    batch_size = tc._effective_transformer_batch()
    prefill_len = tc.seq_len - decode_len
    padded_decode_len = ((decode_len + block_size - 1) // block_size) * block_size
    decode_ctx_len = prefill_len + padded_decode_len

    decode_gemm_shapes = llm_util.process_decode_gemm_shapes(
        tc,
        batch_size=batch_size,
        current_seq_len=decode_ctx_len,
        d_model=tc.hidden_dim,
        num_heads=tc.num_heads,
        kv_heads=tc.kv_heads,
        intermediate_size=tc.moe_intermediate_size if tc.use_moe else tc.intermediate_size,
        vocab_size=tc.vocab_size,
        model_type=tc.model_type,
        query_tokens=block_size,
    )
    (
        decode_pipeline_graph,
        decode_pipeline_root,
        _,
        _,
        decode_transformer_graph,
        decode_transformer_forward_root,
        decode_transformer_backward_root,
        decode_moe_transformer_graph,
        decode_moe_transformer_forward_root,
        decode_moe_transformer_backward_root,
        decode_interconnect_params,
    ), _ = tc.prepare_decode_graphs(
        batch_size=batch_size,
        total_seq_len=decode_ctx_len,
        gemm_shapes=decode_gemm_shapes,
        query_tokens=block_size,
    )
    decode_dispatcher = LLMExecutionDispatcher(
        time_calc=tc,
        pipeline_graph=decode_pipeline_graph,
        pipeline_root=decode_pipeline_root,
        interconnect_params=decode_interconnect_params,
        transformer_graph=decode_transformer_graph,
        transformer_forward_root=decode_transformer_forward_root,
        transformer_backward_root=decode_transformer_backward_root,
        moe_transformer_graph=decode_moe_transformer_graph,
        moe_transformer_forward_root=decode_moe_transformer_forward_root,
        moe_transformer_backward_root=decode_moe_transformer_backward_root,
    )
    mem_estimator = MemoryEstimator(tc)
    decode_memory_root = decode_dispatcher.build_flattened_root_for_memory()
    decode_memory_data = mem_estimator.build_memory_data(
        mode="inference",
        batch_size=batch_size,
        seq_len=block_size,
        gemm_shapes=decode_gemm_shapes,
        kv_cache_tokens=decode_ctx_len,
    )
    tc.pipeline_graph = decode_pipeline_graph
    _, decode_peak_gib = mem_estimator.simulate_peak(
        decode_memory_root,
        decode_memory_data,
        mode="inference",
        filename="memory_graph_decode",
    )
    return float(decode_peak_gib)


def _run_decode_r1_worker(
    task: Tuple[int, int, int, int],
) -> Tuple[Tuple[int, int, int, int], Dict[str, float]]:
    """Compute decode time (at R=1), sample0 time, decode peak memory, and KV byte rate."""
    batch_size, prompt_len, decode_len, block_size = task
    model_config = load_model_config_with_overrides(
        _WORKER_BASE_YAML,
        _WORKER_HW_CONFIG,
        global_batch_size=batch_size,
        prompt_len=prompt_len,
        decode_len=decode_len,
        block_size=block_size,
        refinement_rounds=1,
    )
    with tempfile.TemporaryDirectory(prefix="rapid_decode_") as tmp_out:
        tc = TimeCalculationLLMInference(_WORKER_HW_CONFIG, model_config, "LLM", output_dir=tmp_out)
        with contextlib.redirect_stdout(io.StringIO()):
            decode_time_r1, decode_energy_r1, decode_samples = tc.calc_decode_time()
            decode_peak_gib = _compute_decode_peak_gib(tc, decode_len=decode_len, block_size=block_size)

        head_dim = getattr(tc, "head_dim", None)
        if head_dim is None:
            head_dim = tc.hidden_dim // tc.num_heads
        token_bytes = llm_util.attention_kv_cache_token_bytes(
            getattr(tc, "attention_type", "mha"),
            batch_size=tc._effective_transformer_batch(),
            kv_heads=tc.kv_heads,
            head_dim=head_dim,
            precision_bytes=tc.precision.kv_cache,
            kv_lora_rank=getattr(tc, "kv_lora_rank", None),
            num_heads=getattr(tc, "num_heads", None),
            qk_nope_head_dim=getattr(tc, "qk_nope_head_dim", None),
            qk_rope_head_dim=getattr(tc, "qk_rope_head_dim", None),
            v_head_dim=getattr(tc, "v_head_dim", None),
        )
        if getattr(tc, "disable_kv_cache", False):
            token_bytes = 0.0
        num_layers = int(tc.num_layers)

    sample0_time = float(decode_samples[0].execution_time) if decode_samples else 0.0
    return task, {
        "decode_time_r1": float(decode_time_r1),
        "decode_energy_r1": float(decode_energy_r1),
        "sample0_time": sample0_time,
        "decode_peak_gib": decode_peak_gib,
        "token_bytes": float(token_bytes),
        "num_layers": num_layers,
    }


def _assemble_point(
    *,
    batch_size: int,
    prompt_len: int,
    decode_len: int,
    block_size: int,
    refinement_rounds: int,
    prefill: Dict[str, float],
    decode_r1: Dict[str, float],
) -> Dict[str, object]:
    """Assemble exact metrics for any R by scaling the R=1 decode result."""
    seq_len = prompt_len + decode_len
    num_blocks = ((decode_len + block_size - 1) // block_size) if decode_len > 0 else 0
    decode_model_calls = num_blocks * refinement_rounds
    total_model_calls = decode_model_calls + (1 if prompt_len > 0 else 0)

    prefill_time_s = prefill["prefill_time_s"]
    decode_time_s = decode_r1["decode_time_r1"] * refinement_rounds
    total_time_s = prefill_time_s + decode_time_s
    ttft_s = prefill_time_s + refinement_rounds * decode_r1["sample0_time"]

    prefill_peak_gib = prefill["prefill_peak_gib"]
    decode_peak_gib = decode_r1["decode_peak_gib"]
    max_peak_gib = max(prefill_peak_gib, decode_peak_gib)

    kv_decode_fetch_bytes = (
        decode_r1["token_bytes"]
        * decode_r1["num_layers"]
        * (decode_model_calls * (prompt_len + seq_len) // 2)
    )
    decode_overall_tok_s = ((decode_len / decode_time_s) * batch_size) if decode_time_s > 0.0 else 0.0

    return {
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "decode_len": decode_len,
        "seq_len": seq_len,
        "block_size": block_size,
        "refinement_rounds": refinement_rounds,
        "prefill_time_s": prefill_time_s,
        "decode_time_s": decode_time_s,
        "total_time_s": total_time_s,
        "ttft_s": ttft_s,
        "decode_model_calls": decode_model_calls,
        "total_model_calls": total_model_calls,
        "prefill_peak_gib": prefill_peak_gib,
        "decode_peak_gib": decode_peak_gib,
        "max_peak_gib": max_peak_gib,
        "kv_decode_fetch_gib": kv_decode_fetch_bytes / (1024 ** 3),
        "decode_overall_tok_s": decode_overall_tok_s,
    }


def _gpu_memory_capacity_gib(hw_config) -> Optional[float]:
    tech_cfg = getattr(hw_config, "tech_config", None)
    dram_cfg = getattr(tech_cfg, "DRAM", None) if tech_cfg is not None else None
    size_bytes = getattr(dram_cfg, "size", None)
    if size_bytes is None:
        return None
    return float(size_bytes) / float(1024 ** 3)


def _format_entry(label: str, entry: Dict[str, object], metric: str, hw_mem_gib: Optional[float]) -> str:
    norm = entry["norm"]
    blk = entry["blk"]
    total_sp = entry["total_speedup"]
    decode_sp = entry["decode_speedup"]
    oom_note = ""
    if hw_mem_gib is not None and not entry["fits_memory"]:
        oom_note = f" [OOM > {hw_mem_gib:.2f} GiB]"
    primary_val = total_sp if metric == "total" else decode_sp
    ar_adv = 1.0 / primary_val if primary_val > 0 else float("inf")
    return "\n".join(
        [
            f"  {label}{oom_note}:",
            (
                f"    Params: batch={blk['batch_size']}, prompt={blk['prompt_len']}, "
                f"output={blk['decode_len']}, b={blk['block_size']}, R={blk['refinement_rounds']}"
            ),
            (
                f"    Total time : Normal={norm['total_time_s']:.4f}s vs Block={blk['total_time_s']:.4f}s "
                f"(Total Speedup={total_sp:.4f}x)"
            ),
            (
                f"    Decode time: Normal={norm['decode_time_s']:.4f}s vs Block={blk['decode_time_s']:.4f}s "
                f"(Decode Speedup={decode_sp:.4f}x | AR advantage={ar_adv:.4f}x on {metric})"
            ),
            (
                f"    Model calls: Normal={norm['decode_model_calls']} ({norm['total_model_calls']} total) "
                f"-> Block={blk['decode_model_calls']} ({blk['total_model_calls']} total)"
            ),
            (
                f"    Peak memory: Decode {norm['decode_peak_gib']:.2f} -> {blk['decode_peak_gib']:.2f} GiB | "
                f"Max {norm['max_peak_gib']:.2f} -> {blk['max_peak_gib']:.2f} GiB"
            ),
        ]
    )


def _build_ranking_section(
    title: str,
    entries: List[Dict[str, object]],
    hw_mem_gib: Optional[float],
) -> List[str]:
    lines = ["=" * 78, title, "=" * 78]
    if not entries:
        lines.append("  No configurations matched this filter.")
        lines.append("")
        return lines

    lines.append(f"  Configurations evaluated: {len(entries)}")
    lines.append("")

    for metric_name, key in (
        ("TOTAL END-TO-END SPEEDUP (normal_total / block_total)", "total_speedup"),
        ("DECODE-ONLY SPEEDUP (normal_decode / block_decode)", "decode_speedup"),
    ):
        metric_tag = "total" if key == "total_speedup" else "decode"
        best_bd = max(entries, key=lambda e: e[key])
        best_ar = min(entries, key=lambda e: e[key])
        minimal = min(entries, key=lambda e: abs(e[key] - 1.0))

        lines.append(f"--- {metric_name} ---")
        lines.append(_format_entry("1) Most significant speedup for Block Diffusion", best_bd, metric_tag, hw_mem_gib))
        lines.append(_format_entry("2) Best performance for Autoregression (lowest BD speedup)", best_ar, metric_tag, hw_mem_gib))
        lines.append(_format_entry("3) Minimal speedup / crossover (closest to 1.00x)", minimal, metric_tag, hw_mem_gib))
        lines.append("")

    return lines


def build_compiled_report(
    comparisons: List[Dict[str, object]],
    *,
    hw_name: str,
    model_name: str,
    hw_mem_gib: Optional[float],
    num_csvs: int,
    num_prefill_runs: int,
    num_decode_runs: int,
    elapsed_s: float,
    output_dir: Path,
) -> str:
    in_mem = [c for c in comparisons if c["fits_memory"]]
    oom_count = len(comparisons) - len(in_mem)

    header = [
        "==============================================================================",
        "COMPLETE SINGLE-HARDWARE BLOCK DIFFUSION SWEEP REPORT",
        "==============================================================================",
        f"Hardware config   : {hw_name}" + (f" ({hw_mem_gib:.2f} GiB DRAM)" if hw_mem_gib is not None else ""),
        f"Model config      : {model_name}",
        f"Simulations run   : {num_prefill_runs} unique prefills + {num_decode_runs} unique R=1 decodes (scaled across R)",
        f"Block configs     : {len(comparisons)} total ({len(in_mem)} within memory limit, {oom_count} OOM)",
        f"CSV files written : {num_csvs} in {output_dir}",
        f"Eval tables file  : {output_dir / 'eval_tables.md'}",
        f"Elapsed wall time : {elapsed_s:.2f}s",
        "",
    ]

    sections = []
    sections.extend(_build_ranking_section("RANKINGS — ALL CONFIGURATIONS (INCLUDING OOM)", comparisons, hw_mem_gib))
    if hw_mem_gib is not None:
        sections.extend(
            _build_ranking_section(
                f"RANKINGS — WITHIN GPU MEMORY LIMIT (<= {hw_mem_gib:.2f} GiB)",
                in_mem,
                hw_mem_gib,
            )
        )

    return "\n".join(header + sections)


def _execute_stage(
    tasks: List[tuple],
    worker_fn,
    *,
    workers: int,
    hw_config_path: str,
    model_config_path: str,
    label: str,
    start_time: float,
) -> Dict[tuple, Dict[str, float]]:
    results: Dict[tuple, Dict[str, float]] = {}
    total = len(tasks)
    completed = 0
    progress_step = max(1, total // 10)

    if workers <= 1:
        _init_worker(hw_config_path, model_config_path)
        for task in tasks:
            key, val = worker_fn(task)
            results[key] = val
            completed += 1
            if completed % progress_step == 0 or completed == total:
                elapsed = time.perf_counter() - start_time
                print(f"  [{label}] {completed}/{total} ({100.0 * completed / total:.1f}%) — {elapsed:.1f}s")
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(hw_config_path, model_config_path),
        ) as executor:
            futures = [executor.submit(worker_fn, task) for task in tasks]
            for future in as_completed(futures):
                key, val = future.result()
                results[key] = val
                completed += 1
                if completed % progress_step == 0 or completed == total:
                    elapsed = time.perf_counter() - start_time
                    print(f"  [{label}] {completed}/{total} ({100.0 * completed / total:.1f}%) — {elapsed:.1f}s")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exhaustive single-hardware block-diffusion sweep.")
    parser.add_argument("--hardware_config", type=Path, default=DEFAULT_HW_CONFIG)
    parser.add_argument("--model_config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    parser.add_argument("--prompts", type=int, nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--outputs", type=int, nargs="+", default=DEFAULT_OUTPUTS)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=DEFAULT_BLOCK_SIZES)
    parser.add_argument("--rounds", type=int, nargs="+", default=DEFAULT_ROUNDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1)),
        help="Number of parallel worker processes (default: CPU count).",
    )
    parser.add_argument(
        "--print-tables",
        action="store_true",
        help="Also print all 48 Markdown evaluation tables to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()

    hw_config = config.parse_config(str(args.hardware_config), config_type="hardware")
    hw_mem_gib = _gpu_memory_capacity_gib(hw_config)

    # Stage 1: Unique prefills, sorted largest (batch * prompt) first for better load balancing.
    prefill_tasks = sorted(
        list(itertools.product(args.batches, args.prompts)),
        key=lambda t: t[0] * t[1],
        reverse=True,
    )
    # Stage 2: Unique R=1 decodes (b=1 for AR baseline + all requested block sizes).
    all_block_sizes = sorted(set([1] + list(args.block_sizes)))
    decode_tasks = list(itertools.product(args.batches, args.prompts, args.outputs, all_block_sizes))

    print(
        f"Starting exhaustive sweep with {args.workers} worker(s): "
        f"{len(prefill_tasks)} unique prefills + {len(decode_tasks)} unique R=1 decodes "
        f"-> {len(args.batches) * len(args.prompts) * len(args.outputs) * len(args.block_sizes) * len(args.rounds)} "
        f"block-diffusion configs."
    )

    prefill_cache = _execute_stage(
        prefill_tasks,
        _run_prefill_worker,
        workers=args.workers,
        hw_config_path=str(args.hardware_config),
        model_config_path=str(args.model_config),
        label="prefill",
        start_time=start_time,
    )
    decode_cache = _execute_stage(
        decode_tasks,
        _run_decode_r1_worker,
        workers=args.workers,
        hw_config_path=str(args.hardware_config),
        model_config_path=str(args.model_config),
        label="decode ",
        start_time=start_time,
    )

    # Stage 3: Assemble all points analytically across R, write 48 CSVs + eval_tables.md + report.
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    table_blocks: List[str] = [
        f"# Exhaustive Block Diffusion Evaluation Tables ({args.hardware_config.name} / {args.model_config.name})",
        "",
    ]
    comparisons: List[Dict[str, object]] = []
    num_csvs = 0

    for prompt_len, b, r in itertools.product(args.prompts, args.block_sizes, args.rounds):
        sweep_rows: List[Dict[str, object]] = []
        for out_len in args.outputs:
            for bs in args.batches:
                pf = prefill_cache[(bs, prompt_len)]
                norm_row = _assemble_point(
                    batch_size=bs,
                    prompt_len=prompt_len,
                    decode_len=out_len,
                    block_size=1,
                    refinement_rounds=1,
                    prefill=pf,
                    decode_r1=decode_cache[(bs, prompt_len, out_len, 1)],
                )
                blk_row = _assemble_point(
                    batch_size=bs,
                    prompt_len=prompt_len,
                    decode_len=out_len,
                    block_size=b,
                    refinement_rounds=r,
                    prefill=pf,
                    decode_r1=decode_cache[(bs, prompt_len, out_len, b)],
                )
                sweep_rows.append(norm_row)
                sweep_rows.append(blk_row)

                total_speedup = norm_row["total_time_s"] / blk_row["total_time_s"]
                decode_speedup = norm_row["decode_time_s"] / blk_row["decode_time_s"]
                fits_memory = True
                if hw_mem_gib is not None:
                    fits_memory = (
                        norm_row["max_peak_gib"] <= hw_mem_gib
                        and blk_row["max_peak_gib"] <= hw_mem_gib
                    )
                comparisons.append(
                    {
                        "norm": norm_row,
                        "blk": blk_row,
                        "total_speedup": total_speedup,
                        "decode_speedup": decode_speedup,
                        "fits_memory": fits_memory,
                    }
                )

        csv_path = output_dir / f"sweep_p{prompt_len}_b{b}_r{r}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sweep_rows)
        num_csvs += 1

        table_md = format_markdown_table(sweep_rows, block_size=b, refinement_rounds=r)
        section_header = f"## Prompt={prompt_len}, b={b}, R={r} (`{csv_path.name}`)"
        table_blocks.extend([section_header, "", table_md, ""])
        if args.print_tables:
            print(f"\n{section_header}\n{table_md}")

    tables_path = output_dir / "eval_tables.md"
    tables_path.write_text("\n".join(table_blocks), encoding="utf-8")

    elapsed_s = time.perf_counter() - start_time
    report_text = build_compiled_report(
        comparisons,
        hw_name=args.hardware_config.name,
        model_name=args.model_config.name,
        hw_mem_gib=hw_mem_gib,
        num_csvs=num_csvs,
        num_prefill_runs=len(prefill_tasks),
        num_decode_runs=len(decode_tasks),
        elapsed_s=elapsed_s,
        output_dir=output_dir,
    )

    report_path = output_dir / "compiled_results.txt"
    report_path.write_text(report_text + "\n", encoding="utf-8")
    print("\n" + report_text)


if __name__ == "__main__":
    main()
