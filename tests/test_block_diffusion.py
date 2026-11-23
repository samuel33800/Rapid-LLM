"""Unit tests for Project 3 (Block Diffusion LLM) modeling in RAPID-LLM."""

import copy
import math
import os
import tempfile
from pathlib import Path

import pytest
import yaml

import config
import llm_util
from inference_timing import TimeCalculationLLMInference


REPO_ROOT = Path(__file__).resolve().parents[1]
HW_CONFIG_PATH = REPO_ROOT / "configs/hardware-config/course_project/a100_80GB_1gpu.yaml"
LLAMA_CONFIG_PATH = REPO_ROOT / "configs/model-config/course_project/Llama3.1-8B_inf.yaml"


def _load_hw_and_llama(
    *,
    global_batch_size: int = 1,
    prompt_len: int = 512,
    decode_len: int = 128,
    project_param=None,
):
    """Parse A100 80GB + Llama-3.1-8B configs through the full YAML pipeline."""
    hw_config = config.parse_config(str(HW_CONFIG_PATH), config_type="hardware")
    with open(LLAMA_CONFIG_PATH, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["model_param"]["global_batch_size"] = int(global_batch_size)
    raw["model_param"]["decode_len"] = int(decode_len)
    raw["model_param"]["seq_len"] = int(prompt_len) + int(decode_len)
    if project_param is None:
        raw.pop("project_param", None)
    else:
        raw["project_param"] = copy.deepcopy(project_param)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
            yaml.safe_dump(raw, tmp, sort_keys=False)
            tmp_path = tmp.name
        model_config = config.parse_config(tmp_path, config_type="LLM")
        config.validate_configs(hw_config, model_config)
        return hw_config, model_config
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_b1_r1_matches_omitted_project_param(tmp_path):
    hw_a, m_a = _load_hw_and_llama(global_batch_size=4, prompt_len=512, decode_len=128, project_param=None)
    hw_b, m_b = _load_hw_and_llama(
        global_batch_size=4,
        prompt_len=512,
        decode_len=128,
        project_param={"block_size": 1, "refinement_rounds": 1},
    )

    res_a = TimeCalculationLLMInference(hw_a, m_a, "LLM", output_dir=str(tmp_path / "a")).calc_total_inference_time()
    res_b = TimeCalculationLLMInference(hw_b, m_b, "LLM", output_dir=str(tmp_path / "b")).calc_total_inference_time()

    assert res_a["prefill_time"] == res_b["prefill_time"]
    assert res_a["decode_time"] == res_b["decode_time"]
    assert res_a["total_inference_time"] == res_b["total_inference_time"]
    assert res_a["time_to_first_token"] == res_b["time_to_first_token"]
    assert res_a["kv_cache_decode_fetch_bytes"] == res_b["kv_cache_decode_fetch_bytes"]
    assert res_a["decode_model_calls"] == res_b["decode_model_calls"] == 128
    assert res_a["total_model_calls"] == res_b["total_model_calls"] == 129
    assert res_a["peak_memory_gb"] == res_b["peak_memory_gb"]
    assert res_a["decode_tokens_per_s"] == res_b["decode_tokens_per_s"]


def test_autoregressive_decoder_gemm_scales_with_query_tokens(tmp_path):
    hw, m = _load_hw_and_llama(global_batch_size=2, prompt_len=512, decode_len=128, project_param=None)
    tc = TimeCalculationLLMInference(hw, m, "LLM", output_dir=str(tmp_path))

    g1 = llm_util.autoregressive_decoder_gemm(
        tc, batch_size=2, current_seq_len=520, d_model=4096, num_heads=32, kv_heads=8,
        intermediate_size=14336, vocab_size=128256, model_type="llama", query_tokens=1,
    )
    g8 = llm_util.autoregressive_decoder_gemm(
        tc, batch_size=2, current_seq_len=520, d_model=4096, num_heads=32, kv_heads=8,
        intermediate_size=14336, vocab_size=128256, model_type="llama", query_tokens=8,
    )

    assert g8["qkv_proj"] == (2, 8, 4096, g1["qkv_proj"][3])
    assert g8["attention_score"] == (2 * 8, 8 * 4, 128, 520)
    assert g8["attention_output"] == (2 * 8, 8 * 4, 520, 128)
    assert g8["output_proj"] == (2, 8, 4096, 4096)
    assert g8["ffn1"] == (2, 8, 4096, 2 * 14336)
    assert g8["ffn2"] == (2, 8, 14336, 4096)
    assert g8["linear"] == (2, 8, 4096, 128256)
    assert g8["router"] == (2, 8, 4096, 1)

    with pytest.raises(ValueError):
        llm_util.autoregressive_decoder_gemm(
            tc, batch_size=2, current_seq_len=520, d_model=4096, num_heads=32, kv_heads=8,
            intermediate_size=14336, vocab_size=128256, model_type="llama", query_tokens=0,
        )


@pytest.mark.parametrize(
    "bad_param",
    [
        {"block_size": 0, "refinement_rounds": 4},
        {"block_size": 8, "refinement_rounds": 0},
        {"block_size": -2, "refinement_rounds": 4},
    ],
)
def test_invalid_block_diffusion_params_raise(bad_param, tmp_path):
    hw, m = _load_hw_and_llama(project_param=bad_param)
    tc = TimeCalculationLLMInference(hw, m, "LLM", output_dir=str(tmp_path))
    with pytest.raises(ValueError):
        tc.calc_total_inference_time()


def test_block_diffusion_calls_kv_fetch_ttft_and_rates(tmp_path):
    hw_norm, m_norm = _load_hw_and_llama(
        global_batch_size=1, prompt_len=512, decode_len=128,
        project_param={"block_size": 1, "refinement_rounds": 1},
    )
    hw_blk, m_blk = _load_hw_and_llama(
        global_batch_size=1, prompt_len=512, decode_len=128,
        project_param={"block_size": 8, "refinement_rounds": 4},
    )

    tc_norm = TimeCalculationLLMInference(hw_norm, m_norm, "LLM", output_dir=str(tmp_path / "norm"))
    tc_blk = TimeCalculationLLMInference(hw_blk, m_blk, "LLM", output_dir=str(tmp_path / "blk"))

    res_norm = tc_norm.calc_total_inference_time()
    res_blk = tc_blk.calc_total_inference_time()

    # Model call counts: 128/8 * 4 = 64 decode calls (+1 prefill)
    assert res_norm["decode_model_calls"] == 128
    assert res_norm["total_model_calls"] == 129
    assert res_blk["decode_model_calls"] == 64
    assert res_blk["total_model_calls"] == 65

    # KV store unchanged; KV fetch scales by R/b = 4/8 = 0.5x
    assert res_blk["kv_cache_decode_store_bytes"] == res_norm["kv_cache_decode_store_bytes"]
    assert res_blk["kv_cache_decode_fetch_bytes"] == res_norm["kv_cache_decode_fetch_bytes"] // 2

    # TTFT includes all R=4 refinement calls for the first block
    _, _, blk_samples = tc_blk.calc_decode_time()
    assert [s.step_id for s in blk_samples] == [0, 15]
    assert [s.current_seq_len for s in blk_samples] == [520, 640]
    expected_ttft = res_blk["prefill_time"] + 4 * blk_samples[0].execution_time
    assert math.isclose(res_blk["time_to_first_token"], expected_ttft, rel_tol=1e-12)

    # Rates account for b/R = 2 tokens/call and interpolate midpoint properly
    rates = res_blk["decode_tokens_per_s"]
    assert rates["start"] >= rates["midpoint"] >= rates["end"]
    assert math.isclose(rates["overall"], 128.0 / res_blk["decode_time"], rel_tol=1e-9)
    assert rates["end"] <= rates["overall"] <= rates["start"]


def test_block_diffusion_partial_single_block(tmp_path):
    hw, m = _load_hw_and_llama(
        global_batch_size=1, prompt_len=512, decode_len=4,
        project_param={"block_size": 8, "refinement_rounds": 4},
    )
    tc = TimeCalculationLLMInference(hw, m, "LLM", output_dir=str(tmp_path))
    decode_time, _, samples = tc.calc_decode_time()

    assert [s.step_id for s in samples] == [0]
    assert [s.current_seq_len for s in samples] == [516]
    assert math.isclose(decode_time, 4 * samples[0].execution_time, rel_tol=1e-12)


def test_block_diffusion_decode_peak_memory_scales_with_block_size(tmp_path):
    hw1, m1 = _load_hw_and_llama(
        global_batch_size=64, prompt_len=512, decode_len=128,
        project_param={"block_size": 1, "refinement_rounds": 1},
    )
    hw8, m8 = _load_hw_and_llama(
        global_batch_size=64, prompt_len=512, decode_len=128,
        project_param={"block_size": 8, "refinement_rounds": 4},
    )

    tc1 = TimeCalculationLLMInference(hw1, m1, "LLM", output_dir=str(tmp_path / "tc1"))
    tc1.calc_time()
    tc8 = TimeCalculationLLMInference(hw8, m8, "LLM", output_dir=str(tmp_path / "tc8"))
    tc8.calc_time()

    mem1 = llm_util.estimate_inference_memory(hw1, copy.deepcopy(m1), mode="LLM", output_dir=str(tmp_path / "m1"))
    mem8 = llm_util.estimate_inference_memory(hw8, copy.deepcopy(m8), mode="LLM", output_dir=str(tmp_path / "m8"))

    assert mem8["decode_peak_gb"] > mem1["decode_peak_gb"]
    assert math.isclose(tc1.memory_peak_gb, mem1["max_peak_gb"], rel_tol=1e-9)
    assert math.isclose(tc8.memory_peak_gb, mem8["max_peak_gb"], rel_tol=1e-9)


def test_block_diffusion_batch_crossover(tmp_path):
    """At batch=1 (memory-bound), b=8,R=4 is faster; at batch=64 (compute-bound), slower."""
    hw1_n, m1_n = _load_hw_and_llama(global_batch_size=1, prompt_len=512, decode_len=128, project_param={"block_size": 1, "refinement_rounds": 1})
    hw1_b, m1_b = _load_hw_and_llama(global_batch_size=1, prompt_len=512, decode_len=128, project_param={"block_size": 8, "refinement_rounds": 4})
    dt1_n, _, _ = TimeCalculationLLMInference(hw1_n, m1_n, "LLM", output_dir=str(tmp_path / "1n")).calc_decode_time()
    dt1_b, _, _ = TimeCalculationLLMInference(hw1_b, m1_b, "LLM", output_dir=str(tmp_path / "1b")).calc_decode_time()
    assert dt1_b < dt1_n

    hw64_n, m64_n = _load_hw_and_llama(global_batch_size=64, prompt_len=512, decode_len=128, project_param={"block_size": 1, "refinement_rounds": 1})
    hw64_b, m64_b = _load_hw_and_llama(global_batch_size=64, prompt_len=512, decode_len=128, project_param={"block_size": 8, "refinement_rounds": 4})
    dt64_n, _, _ = TimeCalculationLLMInference(hw64_n, m64_n, "LLM", output_dir=str(tmp_path / "64n")).calc_decode_time()
    dt64_b, _, _ = TimeCalculationLLMInference(hw64_b, m64_b, "LLM", output_dir=str(tmp_path / "64b")).calc_decode_time()
    assert dt64_b > dt64_n


def test_prefill_only_with_block_diffusion(tmp_path):
    hw, m = _load_hw_and_llama(
        global_batch_size=1, prompt_len=512, decode_len=0,
        project_param={"block_size": 8, "refinement_rounds": 4},
    )
    res = TimeCalculationLLMInference(hw, m, "LLM", output_dir=str(tmp_path)).calc_total_inference_time()
    assert res["decode_time"] == 0.0
    assert res["decode_model_calls"] == 0
    assert res["total_model_calls"] == 1
    assert res["time_to_first_token"] == res["prefill_time"]

