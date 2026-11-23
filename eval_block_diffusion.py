#!/usr/bin/env python3
"""Project 3 (Block Diffusion LLM) evaluation sweep script for RAPID-LLM.

Runs Llama-3.1-8B on a100_80GB_1gpu across specified batch sizes and output
lengths, comparing normal autoregressive decode (b=1, R=1) against block
diffusion (default b=8, R=4).

How the global_batch_size / batch_size gotcha is avoided:
    Instead of mutating parsed Python config objects in place (where
    `config.parse_config()` copies `global_batch_size` into `batch_size` and
    applies unit conversions at parse time), this script updates a fresh copy
    of the raw YAML dictionary, writes it to a temporary YAML file, and passes
    it through `config.parse_config()` + `config.validate_configs()`—the exact
    same pipeline used by `run_perf.py`.
"""

import argparse
import contextlib
import copy
import csv
import io
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

import config
from inference_timing import TimeCalculationLLMInference


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_HW_CONFIG = REPO_ROOT / "configs/hardware-config/course_project/a100_80GB_1gpu.yaml"
DEFAULT_MODEL_CONFIG = REPO_ROOT / "configs/model-config/course_project/Llama3.1-8B_inf.yaml"


def load_model_config_with_overrides(
    base_yaml: dict,
    hw_config,
    *,
    global_batch_size: int,
    prompt_len: int,
    decode_len: int,
    block_size: int,
    refinement_rounds: int,
):
    """Build and validate a parsed model config via temporary YAML file."""
    cfg_dict = copy.deepcopy(base_yaml)
    cfg_dict["model_param"]["global_batch_size"] = int(global_batch_size)
    cfg_dict["model_param"]["decode_len"] = int(decode_len)
    cfg_dict["model_param"]["seq_len"] = int(prompt_len) + int(decode_len)
    cfg_dict["project_param"] = {
        "block_size": int(block_size),
        "refinement_rounds": int(refinement_rounds),
    }

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
            yaml.safe_dump(cfg_dict, tmp, sort_keys=False)
            tmp_path = tmp.name
        model_config = config.parse_config(tmp_path, config_type="LLM")
        config.validate_configs(hw_config, model_config)
        return model_config
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _parse_memory_summary(output_dir: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract (prefill_peak_gib, decode_peak_gib, max_peak_gib) from memory_capacity_comparison.txt."""
    summary_path = Path(output_dir) / "memory-summary" / "memory_capacity_comparison.txt"
    if not summary_path.is_file():
        return None, None, None
    text = summary_path.read_text(encoding="utf-8")

    def _extract(pattern: str) -> Optional[float]:
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None

    return (
        _extract(r"Prefill peak memory usage \(per gpu\):\s*([0-9.]+)\s*GiB"),
        _extract(r"Final decode peak memory usage \(per gpu\):\s*([0-9.]+)\s*GiB"),
        _extract(r"Max peak memory usage \(per gpu\):\s*([0-9.]+)\s*GiB"),
    )


def run_single_point(
    hw_config,
    base_model_yaml: dict,
    *,
    global_batch_size: int,
    prompt_len: int,
    decode_len: int,
    block_size: int,
    refinement_rounds: int,
    verbose: bool = False,
) -> Dict[str, object]:
    """Run one inference simulation point and return collected metrics."""
    model_config = load_model_config_with_overrides(
        base_model_yaml,
        hw_config,
        global_batch_size=global_batch_size,
        prompt_len=prompt_len,
        decode_len=decode_len,
        block_size=block_size,
        refinement_rounds=refinement_rounds,
    )

    with tempfile.TemporaryDirectory(prefix="rapid_block_diff_") as tmp_out:
        tc = TimeCalculationLLMInference(hw_config, model_config, "LLM", output_dir=tmp_out)
        if verbose:
            timing = tc.calc_total_inference_time()
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                timing = tc.calc_total_inference_time()
        prefill_peak_gb, decode_peak_gb, max_peak_gb = _parse_memory_summary(tmp_out)

    decode_rates = timing.get("decode_tokens_per_s") or {}
    return {
        "batch_size": global_batch_size,
        "prompt_len": prompt_len,
        "decode_len": decode_len,
        "seq_len": prompt_len + decode_len,
        "block_size": block_size,
        "refinement_rounds": refinement_rounds,
        "prefill_time_s": timing["prefill_time"],
        "decode_time_s": timing["decode_time"],
        "total_time_s": timing["total_inference_time"],
        "ttft_s": timing["time_to_first_token"],
        "decode_model_calls": timing["decode_model_calls"],
        "total_model_calls": timing["total_model_calls"],
        "prefill_peak_gib": prefill_peak_gb if prefill_peak_gb is not None else 0.0,
        "decode_peak_gib": decode_peak_gb if decode_peak_gb is not None else 0.0,
        "max_peak_gib": max_peak_gb if max_peak_gb is not None else float(timing["peak_memory_gb"]),
        "kv_decode_fetch_gib": timing["kv_cache_decode_fetch_bytes"] / (1024 ** 3),
        "decode_overall_tok_s": decode_rates.get("overall", 0.0) * global_batch_size,
    }


def format_markdown_table(rows: List[Dict[str, object]], *, block_size: int, refinement_rounds: int) -> str:
    """Format side-by-side Normal vs. Block Diffusion comparison table in Markdown."""
    by_key = {
        (r["decode_len"], r["batch_size"], r["block_size"], r["refinement_rounds"]): r
        for r in rows
    }
    outputs = sorted({r["decode_len"] for r in rows})
    batches = sorted({r["batch_size"] for r in rows})

    headers = [
        "Output",
        "Batch",
        "Normal Total (s)",
        f"Block (b={block_size},R={refinement_rounds}) Total (s)",
        "Total Speedup",
        "Normal Decode (s)",
        "Block Decode (s)",
        "Decode Speedup",
        "Calls (Normal -> Block)",
        "Decode Peak GiB (Normal -> Block)",
        "Max Peak GiB (Normal -> Block)",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---:"] * len(headers)) + " |",
    ]

    for out_len in outputs:
        for bs in batches:
            norm = by_key[(out_len, bs, 1, 1)]
            blk = by_key[(out_len, bs, block_size, refinement_rounds)]
            total_speedup = norm["total_time_s"] / blk["total_time_s"]
            decode_speedup = norm["decode_time_s"] / blk["decode_time_s"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(out_len),
                        str(bs),
                        f"{norm['total_time_s']:.4f}",
                        f"{blk['total_time_s']:.4f}",
                        f"{total_speedup:.2f}x",
                        f"{norm['decode_time_s']:.4f}",
                        f"{blk['decode_time_s']:.4f}",
                        f"{decode_speedup:.2f}x",
                        f"{norm['decode_model_calls']} ({norm['total_model_calls']}) -> {blk['decode_model_calls']} ({blk['total_model_calls']})",
                        f"{norm['decode_peak_gib']:.2f} -> {blk['decode_peak_gib']:.2f}",
                        f"{norm['max_peak_gib']:.2f} -> {blk['max_peak_gib']:.2f}",
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Project 3 block-diffusion evaluation sweep.")
    parser.add_argument("--hardware_config", type=Path, default=DEFAULT_HW_CONFIG)
    parser.add_argument("--model_config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--prompt", type=int, default=512)
    parser.add_argument("--outputs", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--csv", type=Path, default=None, help="Optional path to write full CSV results.")
    parser.add_argument("--verbose", action="store_true", help="Show per-run RAPID-LLM stdout.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hw_config = config.parse_config(str(args.hardware_config), config_type="hardware")
    with open(args.model_config, "r", encoding="utf-8") as handle:
        base_model_yaml = yaml.safe_load(handle)

    rows: List[Dict[str, object]] = []
    modes = [(1, 1), (args.block_size, args.rounds)]
    for out_len in args.outputs:
        for bs in args.batches:
            for b, r in modes:
                rows.append(
                    run_single_point(
                        hw_config,
                        base_model_yaml,
                        global_batch_size=bs,
                        prompt_len=args.prompt,
                        decode_len=out_len,
                        block_size=b,
                        refinement_rounds=r,
                        verbose=args.verbose,
                    )
                )

    print(
        f"Hardware: {args.hardware_config.name} | Model: {args.model_config.name} | "
        f"Prompt: {args.prompt} | Block Diffusion: b={args.block_size}, R={args.rounds}\n"
    )
    print(format_markdown_table(rows, block_size=args.block_size, refinement_rounds=args.rounds))
    print("\nNote: Calls shown as `decode_calls (total_calls_including_prefill)`.")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV results to {args.csv}")


if __name__ == "__main__":
    main()

