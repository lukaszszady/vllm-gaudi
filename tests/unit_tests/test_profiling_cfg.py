# SPDX-License-Identifier: Apache-2.0
"""Token budget of the VLLM_PROFILE_PROMPT / VLLM_PROFILE_DECODE dummy batch (pure python module, loaded by path)."""

import importlib.util
import os
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "vllm_gaudi" / "extension" / "profiling_cfg.py"
_spec = importlib.util.spec_from_file_location("profiling_cfg_under_test", _PATH)
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)


def test_prompt_only_schedules_exactly_bs_times_query_tokens():
    assert cfg.profiling_max_batched_tokens((1, 8192, 0), None, 4096) == 8192
    assert cfg.profiling_max_batched_tokens((2, 1024, 3), None, 8192) == 2048


def test_mixed_profile_reserves_one_token_per_decode_request():
    # build #853: PROMPT=1,8192,0 + DECODE=1,53 schedules 8192 + 1 tokens; 8192 overflowed the position buffer
    assert cfg.profiling_max_batched_tokens((1, 8192, 0), (1, 1, 53), 8192) == 8193
    assert cfg.profiling_max_batched_tokens((1, 512, 0), (16, 1, 64), 8192) == 512 + 16


def test_decode_only_never_shrinks_the_budget_below_the_batch():
    assert cfg.profiling_max_batched_tokens(None, (1, 1, 53), 8192) == 8192
    assert cfg.profiling_max_batched_tokens(None, (16384, 1, 64), 8192) == 16384


def test_bucketing_budget_is_a_multiple_of_the_mamba_chunk_in_profile_mode():
    # build #859: 8192 + 1 = 8193 is not divisible by the default KDA/GDN chunk size 128 -> bucketing init raised
    budget = cfg.profiling_max_batched_tokens((1, 8192, 0), (1, 1, 53), 8192)
    assert budget % 128 != 0
    assert cfg.bucketing_max_batched_tokens(budget, 128, profiling=True) == 8320
    assert cfg.bucketing_max_batched_tokens(8192, 128, profiling=True) == 8192
    assert cfg.bucketing_max_batched_tokens(budget, 256, profiling=True) == 8192 + 256


def test_bucketing_budget_is_untouched_outside_profile_mode_or_without_mamba_layers():
    assert cfg.bucketing_max_batched_tokens(8193, 128, profiling=False) == 8193  # real misconfig still fails loudly
    assert cfg.bucketing_max_batched_tokens(8193, 0, profiling=True) == 8193


class _OneArgProf:
    """Like the Habana profiler wrapper: export_chrome_trace accepts exactly one positional argument."""

    def export_chrome_trace(self, path):
        with open(path, "w") as f:
            f.write('{"traceEvents": []}')


def test_trace_handler_writes_gzip_with_single_arg_export(tmp_path):
    import gzip
    cfg.make_trace_handler(str(tmp_path / "out"), 8)(_OneArgProf())
    files = list((tmp_path / "out").iterdir())
    assert len(files) == 1 and files[0].name.startswith("8_") and files[0].name.endswith(".pt.trace.json.gz")
    assert gzip.open(files[0]).read() == b'{"traceEvents": []}'


def test_trace_handler_plain_json_without_gzip(tmp_path):
    cfg.make_trace_handler(str(tmp_path), 0, use_gzip=False)(_OneArgProf())
    assert [f.name.endswith(".pt.trace.json") for f in tmp_path.iterdir()] == [True]


def test_trace_handler_swallows_export_errors(tmp_path):

    class Broken:

        def export_chrome_trace(self, path):
            raise TypeError("takes 2 positional arguments but 3 were given")

    cfg.make_trace_handler(str(tmp_path), 1)(Broken())
    assert list(tmp_path.iterdir()) == []


def test_trace_names_include_rank_and_pid_and_are_distinct(tmp_path):
    import time
    handler = cfg.make_trace_handler(str(tmp_path), 3, use_gzip=False)
    handler(_OneArgProf())
    time.sleep(0.01)
    handler(_OneArgProf())
    names = sorted(f.name for f in tmp_path.iterdir())
    assert len(set(names)) == 2
    assert all(n.startswith(f"3_{__import__('os').getpid()}_") for n in names)


def test_trace_handler_logger_is_under_vllm_tree():
    assert cfg._logger.name.startswith("vllm.")


def _load_setup_profiler(captured):
    """Compile only setup_profiler from extension/profiler.py (the module imports Habana / vLLM at load time)."""
    import ast
    import types
    path = _PATH.with_name("profiler.py")
    tree = ast.parse(path.read_text())
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "setup_profiler")
    fake_profiler = types.SimpleNamespace(
        schedule=lambda **kw: "schedule",
        ProfilerActivity=types.SimpleNamespace(CPU="cpu", HPU="hpu"),
        profile=lambda **kw: captured.update(kw) or "profiler",
    )
    ns = {"torch": types.SimpleNamespace(profiler=fake_profiler), "make_trace_handler": cfg.make_trace_handler}
    exec(compile(ast.Module([func], []), str(path), "exec"), ns)
    return ns["setup_profiler"]


def test_setup_profiler_passes_rank_to_the_trace_handler(tmp_path):
    captured = {}
    setup = _load_setup_profiler(captured)
    assert setup(warmup=0, active=1, output_dir=str(tmp_path), rank=5) == "profiler"
    captured["on_trace_ready"](_OneArgProf())
    (trace, ) = tmp_path.iterdir()
    assert trace.name.startswith("5_")


class _TorchProfLikeHabana:
    """Mimics torch.profiler.profile over the Habana wrapper (build #867).

    The outer ``export_chrome_trace`` forwards ``(path, metadata)`` to ``self.profiler``, whose Habana wrapper only
    accepts ``(path)`` -> TypeError. Only a direct call on the inner profiler works.
    """

    class _Inner:

        def export_chrome_trace(self, path):
            with open(path, "w") as f:
                f.write('{"traceEvents": [1]}')

    def __init__(self):
        self.profiler = self._Inner()
        self._trace_metadata = {}

    def export_chrome_trace(self, path, use_python_export=False):
        self.profiler.export_chrome_trace(path, self._trace_metadata)


def test_trace_handler_calls_inner_profiler_when_outer_forwards_extra_arg(tmp_path):
    import gzip
    cfg.make_trace_handler(str(tmp_path), 0)(_TorchProfLikeHabana())
    files = list(tmp_path.iterdir())
    assert len(files) == 1 and files[0].name.endswith(".pt.trace.json.gz")
    assert gzip.open(files[0]).read() == b'{"traceEvents": [1]}'


def test_trace_handler_falls_back_to_kineto_results_save(tmp_path):

    class Kineto:

        def save(self, path):
            with open(path, "w") as f:
                f.write("{}")

    class Inner:
        kineto_results = Kineto()

        def export_chrome_trace(self, path, metadata=None):
            raise TypeError("boom")

    class Prof:
        profiler = Inner()

        def export_chrome_trace(self, path):
            raise TypeError("boom")

    cfg.make_trace_handler(str(tmp_path), 0, use_gzip=False)(Prof())
    assert [f.name.endswith(".pt.trace.json") for f in tmp_path.iterdir()] == [True]


def test_partial_file_of_a_failed_attempt_never_counts_as_a_trace(tmp_path):

    class Inner:

        def export_chrome_trace(self, path, metadata=None):
            with open(path, "w") as f:
                f.write("{partial")
            raise OSError("disk full")

    class Prof:
        profiler = Inner()

        def export_chrome_trace(self, path):
            pass  # "succeeds" without writing anything

    path = str(tmp_path / "t.json")
    try:
        cfg._export_trace(Prof(), path)
    except RuntimeError as exc:
        assert "no file written" in str(exc) and "disk full" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
    assert not os.path.exists(path)


def test_all_attempts_failing_leaves_no_file_and_handler_writes_nothing(tmp_path):

    class Inner:

        def export_chrome_trace(self, path, metadata=None):
            with open(path, "w") as f:
                f.write("{partial")
            raise OSError("boom")

    class Prof:
        profiler = Inner()

        def export_chrome_trace(self, path):
            raise OSError("boom")

    cfg.make_trace_handler(str(tmp_path), 0, use_gzip=False)(Prof())
    assert list(tmp_path.iterdir()) == []


def test_exit_delay_env_parsing():
    assert cfg.profile_exit_delay_s({}) == 0.0
    assert cfg.profile_exit_delay_s({"VLLM_PROFILE_EXIT_DELAY_S": "120"}) == 120.0
    assert cfg.profile_exit_delay_s({"VLLM_PROFILE_EXIT_DELAY_S": "1.5"}) == 1.5
    assert cfg.profile_exit_delay_s({"VLLM_PROFILE_EXIT_DELAY_S": "-5"}) == 0.0
    assert cfg.profile_exit_delay_s({"VLLM_PROFILE_EXIT_DELAY_S": "abc"}) == 0.0
    assert cfg.profile_exit_delay_s({"VLLM_PROFILE_EXIT_DELAY_S": ""}) == 0.0


def test_hold_before_exit_logs_trace_names_and_sleeps(monkeypatch, caplog):
    monkeypatch.setattr(cfg, "WRITTEN_TRACES", ["/d/0_1_2.pt.trace.json.gz"])
    slept = []
    with caplog.at_level("WARNING", logger="vllm.gaudi.profiling"):
        cfg.hold_before_exit(30, sleep=slept.append)
    assert slept == [30]
    assert "0_1_2.pt.trace.json.gz" in caplog.text and "holding 30 s before exit" in caplog.text


def test_hold_before_exit_is_a_noop_without_delay(caplog):
    slept = []
    cfg.hold_before_exit(0, sleep=slept.append)
    assert slept == [] and caplog.text == ""


def test_trace_handler_records_written_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WRITTEN_TRACES", [])
    cfg.make_trace_handler(str(tmp_path), 0, use_gzip=False)(_OneArgProf())
    assert len(cfg.WRITTEN_TRACES) == 1 and cfg.WRITTEN_TRACES[0].endswith(".pt.trace.json")


def test_profile_blocks_mode_env():
    assert cfg.profile_blocks_scattered({}) is False
    assert cfg.profile_blocks_scattered({"VLLM_PROFILE_BLOCKS": "repeat"}) is False
    assert cfg.profile_blocks_scattered({"VLLM_PROFILE_BLOCKS": " Scatter "}) is True
    with pytest.raises(ValueError):
        cfg.profile_blocks_scattered({"VLLM_PROFILE_BLOCKS": "random"})


def test_pad_owner_block_in_scheduler_units():
    # Kimi-K3: 128-token kernel blocks, 384-token scheduler blocks; the pad kernel block 3648 lies in block 1216
    assert cfg.pad_owner_block(3648, 128, 384) == 1216
    assert cfg.pad_owner_block(3648, 128, 128) == 3648


def test_scattered_blocks_are_distinct_and_skip_null_and_excluded_blocks():
    # VLLM_PROFILE_DECODE=24,3648 with 384-token scheduler blocks: 24 requests x 51 blocks
    alloc = cfg.ScatteredBlockAllocator(3648, exclude={1216})
    ids = [b for _ in range(24) for b in alloc.take(0, 51)]
    assert len(set(ids)) == len(ids) == 24 * 51
    assert 0 not in ids and 1216 not in ids and max(ids) < 3648
    assert not alloc.wrapped
    first = ids[:51]
    assert first != sorted(first)  # scattered, not one contiguous run
    assert max(first) - min(first) > 51


def test_scattered_blocks_are_deterministic_and_independent_per_group():
    a, b = cfg.ScatteredBlockAllocator(100), cfg.ScatteredBlockAllocator(100)
    assert a.take(0, 10) == b.take(0, 10)
    assert a.take(1, 10) != a.take(0, 10)
    assert b.take(1, 1) == cfg.ScatteredBlockAllocator(100).take(1, 1)


def test_scattered_blocks_wrap_when_the_pool_is_exhausted():
    alloc = cfg.ScatteredBlockAllocator(5, exclude={2})
    assert alloc.pool_size == 3
    ids = alloc.take(0, 3)
    assert sorted(ids) == [1, 3, 4] and not alloc.wrapped
    assert alloc.take(0, 2) == ids[:2] and alloc.wrapped
    with pytest.raises(ValueError):
        cfg.ScatteredBlockAllocator(2, exclude={1})


def test_profile_sampling_unset_keeps_greedy():
    assert cfg.profile_sampling_kwargs(None) is None
    assert cfg.profile_sampling_kwargs("  ") is None


def test_profile_sampling_default_uses_model_defaults():
    # Kimi-K3's generation_config.json has no sampling keys -> SamplingParams defaults (temperature 1.0)
    assert cfg.profile_sampling_kwargs("default", {}) == {}
    defaults = dict(temperature=0.6, top_p=0.95, max_tokens=10)
    assert cfg.profile_sampling_kwargs("default", defaults) == dict(temperature=0.6, top_p=0.95)


def test_profile_sampling_explicit_items_override_defaults():
    assert cfg.profile_sampling_kwargs("temperature=1.0, top_p=0.95") == dict(temperature=1.0, top_p=0.95)
    kwargs = cfg.profile_sampling_kwargs("default,top_k=50", dict(temperature=0.7, top_k=20))
    assert kwargs == dict(temperature=0.7, top_k=50) and isinstance(kwargs["top_k"], int)


def test_profile_sampling_rejects_unknown_items():
    for spec in ("greedy", "seed=1", "top_p", "top_k=abc"):
        with pytest.raises(ValueError):
            cfg.profile_sampling_kwargs(spec)
