# SPDX-License-Identifier: Apache-2.0
"""HPU_KIMI_PREFILL_TIMING helper: env parsing, rank gating, rate limit and the log line. Runs without habana/vllm."""
import importlib.util
import logging
import sys
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).resolve().parents[2] / "vllm_gaudi" / "v1" / "worker" / "kimi_prefill_timing.py"


@pytest.fixture
def timing(monkeypatch):
    name = "kimi_prefill_timing_under_test"
    spec = importlib.util.spec_from_file_location(name, PATH)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("value, period", [(None, 0), ("", 0), ("0", 0), ("off", 0), ("1", 1), ("true", 1), ("10", 10),
                                           ("-3", 0), ("x", 0)])
def test_parse_timing_env(timing, value, period):
    assert timing.parse_timing_env(value) == period


@pytest.mark.parametrize("driver, all_ranks, enabled", [(True, None, True), (False, None, False), (False, "1", True)])
def test_only_the_driver_times_unless_all_ranks(timing, monkeypatch, driver, all_ranks, enabled):
    monkeypatch.setenv(timing.ENV_TIMING, "1")
    if all_ranks is None:
        monkeypatch.delenv(timing.ENV_ALL_RANKS, raising=False)
    else:
        monkeypatch.setenv(timing.ENV_ALL_RANKS, all_ranks)
    assert timing.PrefillTimer.from_env(is_driver_worker=driver, get_rank=lambda: 3,
                                        sync=lambda: None).enabled is enabled


def test_off_by_default(timing, monkeypatch):
    monkeypatch.delenv(timing.ENV_TIMING, raising=False)

    def no_tp_group():
        raise AssertionError("the TP rank must not be looked up when timing is off")

    assert not timing.PrefillTimer.from_env(is_driver_worker=True, get_rank=no_tp_group, sync=lambda: None).enabled


def test_count_video_chunks(timing):
    assert timing.count_video_chunks({"video_num_chunks": torch.tensor([5, 3])}) == 8
    assert timing.count_video_chunks({"video_num_chunks": [torch.tensor(5), torch.tensor(2)]}) == 7
    assert timing.count_video_chunks({"pixel_values": torch.zeros(1)}) == 0


def test_now_syncs_the_device_before_reading_the_clock(timing):
    calls = []
    timer = timing.PrefillTimer(1, 0, sync=lambda: calls.append("sync"), clock=lambda: calls.append("clock") or 1.0)
    assert timer.now() == 1.0
    assert calls == ["sync", "clock"]


def test_log_line_rate_limit_and_means(timing, caplog):
    timer = timing.PrefillTimer(2, 0, sync=lambda: None)
    with caplog.at_level(logging.INFO, logger="vllm.gaudi.prefill_timing"):
        timer.begin_step()
        timer.add_encoder(0.8, 1, 5)
        assert timer.finish_prefill(num_reqs=1, tokens=9234, padded_tokens=10240, mm_inputs_s=0.85,
                                    lm_forward_s=3.0) is None
        timer.begin_step()
        line = timer.finish_prefill(num_reqs=2, tokens=100, padded_tokens=128, mm_inputs_s=0.05, lm_forward_s=1.0)
    assert line == ("Prefill timing: rank=0 prefill=2 reqs=2 tokens=100 padded_tokens=128 mm_items=0 video_chunks=0 "
                    "encoder_ms=0.0 mm_inputs_ms=50.0 lm_forward_ms=1000.0 total_ms=1050.0 | mean of 2: "
                    "encoder_ms=400.0 mm_inputs_ms=450.0 lm_forward_ms=2000.0 total_ms=2450.0")
    assert caplog.text.count("Prefill timing: rank=") == 1
