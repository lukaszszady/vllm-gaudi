# SPDX-License-Identifier: Apache-2.0
import os
import sys
import threading
import time
import types

import pytest
import torch

from vllm_gaudi.v1.worker import hpu_weight_page_cache as wpc


def _fake_iterator(calls):

    def iterator(files, use_tqdm_on_load, strategy=None, *args, **kwargs):
        calls.append((list(files), strategy, kwargs))
        for path in files:
            yield os.path.basename(path), torch.zeros(1)

    return iterator


@pytest.fixture(autouse=True)
def stub_vllm_modules(monkeypatch):
    """The wrapper only needs tqdm and vLLM's weight_utils / default_loader to exist; stub them when not installed."""
    if "tqdm.auto" not in sys.modules:
        try:
            import tqdm.auto  # noqa: F401
        except ImportError:
            tqdm_pkg, tqdm_auto = types.ModuleType("tqdm"), types.ModuleType("tqdm.auto")
            tqdm_auto.tqdm = lambda it, **kwargs: it
            tqdm_pkg.auto = tqdm_auto
            monkeypatch.setitem(sys.modules, "tqdm", tqdm_pkg)
            monkeypatch.setitem(sys.modules, "tqdm.auto", tqdm_auto)
    try:
        from vllm.model_executor.model_loader import default_loader  # noqa: F401
    except ImportError:
        default_loader = types.ModuleType("vllm.model_executor.model_loader.default_loader")
        default_loader.safetensors_weights_iterator = _fake_iterator([])
        weight_utils = types.ModuleType("vllm.model_executor.model_loader.weight_utils")
        loader_pkg = types.ModuleType("vllm.model_executor.model_loader")
        loader_pkg.default_loader, loader_pkg.weight_utils = default_loader, weight_utils
        for name, module in (("vllm", types.ModuleType("vllm")), ("vllm.model_executor",
                                                                  types.ModuleType("vllm.model_executor")),
                             ("vllm.model_executor.model_loader",
                              loader_pkg), ("vllm.model_executor.model_loader.default_loader", default_loader),
                             ("vllm.model_executor.model_loader.weight_utils", weight_utils)):
            monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def shards(tmp_path):
    paths = []
    for i in (10, 2, 1):
        path = tmp_path / f"model-{i:05d}-of-00010.safetensors"
        path.write_bytes(b"x")
        paths.append(str(path))
    return paths


@pytest.fixture
def fadvised(monkeypatch):
    dropped = []

    def fake_fadvise(fd, offset, length, advice):
        assert (offset, length, advice) == (0, 0, os.POSIX_FADV_DONTNEED)
        dropped.append(os.readlink(f"/proc/self/fd/{fd}"))

    monkeypatch.setattr(os, "posix_fadvise", fake_fadvise)
    return dropped


@pytest.mark.parametrize("lag", ["0", "1", "5"])
def test_every_shard_is_dropped_once_in_load_order(monkeypatch, shards, fadvised, lag):
    monkeypatch.setenv("VLLM_HPU_DROP_WEIGHT_PAGE_CACHE_LAG", lag)
    calls = []
    wrapped = wpc.wrap_safetensors_iterator(_fake_iterator(calls))

    names = [name for name, _ in wrapped(shards, True, None, None, safetensors_prefetch_num_threads=2)]

    ordered = sorted(shards, key=lambda p: int(os.path.basename(p).split("-")[1]))
    assert names == [os.path.basename(p) for p in ordered]
    assert [c[0] for c in calls] == [[p] for p in ordered]
    assert all(c[2] == {"safetensors_prefetch_num_threads": 2} for c in calls)
    assert fadvised == [os.path.realpath(p) for p in ordered]


def test_shard_is_dropped_only_after_lag_more_shards(monkeypatch, shards, fadvised):
    monkeypatch.setenv("VLLM_HPU_DROP_WEIGHT_PAGE_CACHE_LAG", "1")
    wrapped = wpc.wrap_safetensors_iterator(_fake_iterator([]))
    it = wrapped(shards, False)
    next(it)  # first shard yielded, nothing dropped yet
    assert fadvised == []
    next(it)  # second shard started; the first one is dropped only when the second is consumed
    assert fadvised == []
    next(it)
    deadline = time.monotonic() + 2  # the drop runs in the background thread
    while not fadvised and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(fadvised) == 1
    it.close()  # stopping early still drops the remaining shards
    assert len(fadvised) == 3


@pytest.mark.parametrize("strategy", ["torchao", "prefetch"])
def test_cross_shard_strategies_keep_upstream_iterator(shards, fadvised, strategy):
    calls = []
    wrapped = wpc.wrap_safetensors_iterator(_fake_iterator(calls))
    list(wrapped(shards, False, strategy))
    assert len(calls) == 1 and calls[0][0] == shards and calls[0][1] == strategy
    assert fadvised == []


@pytest.mark.parametrize("strategy,expected", [(None, "lazy"), ("lazy", "lazy"), ("eager", "eager")])
def test_default_strategy_is_passed_as_lazy_per_shard(shards, fadvised, strategy, expected):
    calls = []
    list(wpc.wrap_safetensors_iterator(_fake_iterator(calls))(shards, False, strategy))
    assert [c[1] for c in calls] == [expected] * len(shards)


def test_context_manager_patches_and_restores_default_loader(monkeypatch):
    from vllm.model_executor.model_loader import default_loader
    original = default_loader.safetensors_weights_iterator
    with wpc.drop_weight_page_cache_during_load():
        assert default_loader.safetensors_weights_iterator is not original
        assert default_loader.safetensors_weights_iterator.__wrapped__ is original
    assert default_loader.safetensors_weights_iterator is original

    monkeypatch.setenv("VLLM_HPU_DROP_WEIGHT_PAGE_CACHE", "0")
    with wpc.drop_weight_page_cache_during_load():
        assert default_loader.safetensors_weights_iterator is original


@pytest.mark.parametrize("value,mode", [(None, "auto"), ("", "auto"), ("auto", "auto"), ("bogus", "auto"), ("1", "on"),
                                        ("true", "on"), ("0", "off"), ("off", "off")])
def test_drop_mode(monkeypatch, value, mode):
    if value is None:
        monkeypatch.delenv("VLLM_HPU_DROP_WEIGHT_PAGE_CACHE", raising=False)
    else:
        monkeypatch.setenv("VLLM_HPU_DROP_WEIGHT_PAGE_CACHE", value)
    assert wpc.drop_mode() == mode


def test_auto_mode_keeps_upstream_iterator_for_small_checkpoints(monkeypatch, shards, fadvised):
    monkeypatch.setattr(wpc, "_mem_total_bytes", lambda: 1000)  # shards are 3 bytes: far below 25%
    calls, seen = [], []
    list(wpc.wrap_safetensors_iterator(_fake_iterator(calls), seen, force=False)(shards, False, None))
    assert len(calls) == 1 and calls[0][0] == shards and calls[0][1] is None  # untouched, no forced "lazy"
    assert fadvised == [] and seen == []


def test_auto_mode_is_active_for_large_checkpoints_and_forced_mode_always(monkeypatch, shards, fadvised):
    monkeypatch.setenv("VLLM_HPU_DROP_WEIGHT_PAGE_CACHE_LAG", "0")
    monkeypatch.setattr(wpc, "_mem_total_bytes", lambda: 8)  # 3 bytes > 25% of 8
    calls, seen = [], []
    list(wpc.wrap_safetensors_iterator(_fake_iterator(calls), seen, force=False)(shards, False, None))
    assert len(calls) == 3 and {c[1] for c in calls} == {"lazy"} and len(fadvised) == 3 and len(seen) == 3

    monkeypatch.setattr(wpc, "_mem_total_bytes", lambda: 10**12)
    calls.clear()
    list(wpc.wrap_safetensors_iterator(_fake_iterator(calls), [], force=True)(shards, False, None))
    assert len(calls) == 3


def test_all_seen_shards_are_dropped_again_after_get_model(monkeypatch, shards, fadvised):
    from vllm.model_executor.model_loader import default_loader
    monkeypatch.setenv("VLLM_HPU_DROP_WEIGHT_PAGE_CACHE", "1")
    monkeypatch.setenv("VLLM_HPU_DROP_WEIGHT_PAGE_CACHE_LAG", "0")
    monkeypatch.setattr(default_loader, "safetensors_weights_iterator", _fake_iterator([]))
    with wpc.drop_weight_page_cache_during_load():
        list(default_loader.safetensors_weights_iterator(shards, False))
        in_loader = len(fadvised)
        assert in_loader == 3
    assert len(fadvised) == 2 * 3  # the final pass covers every shard once more
    assert sorted(fadvised[in_loader:]) == sorted(os.path.realpath(p) for p in shards)


def test_final_drop_also_runs_when_get_model_raises(monkeypatch, shards, fadvised):
    from vllm.model_executor.model_loader import default_loader
    monkeypatch.setenv("VLLM_HPU_DROP_WEIGHT_PAGE_CACHE", "1")
    monkeypatch.setenv("VLLM_HPU_DROP_WEIGHT_PAGE_CACHE_LAG", "5")
    monkeypatch.setattr(default_loader, "safetensors_weights_iterator", _fake_iterator([]))
    with pytest.raises(RuntimeError), wpc.drop_weight_page_cache_during_load():
        list(default_loader.safetensors_weights_iterator(shards, False))
        raise RuntimeError("process_weights_after_loading failed")
    assert len(fadvised) == 6


def test_failing_final_drop_does_not_replace_the_get_model_exception(monkeypatch, shards, fadvised):
    from vllm.model_executor.model_loader import default_loader
    monkeypatch.setenv("VLLM_HPU_DROP_WEIGHT_PAGE_CACHE", "1")
    original = _fake_iterator([])
    monkeypatch.setattr(default_loader, "safetensors_weights_iterator", original)
    with pytest.raises(RuntimeError, match="get_model failed"), wpc.drop_weight_page_cache_during_load():
        list(default_loader.safetensors_weights_iterator(shards, False))

        def broken(*args, **kwargs):
            raise ValueError("fadvise broke")

        monkeypatch.setattr(wpc, "host_memory_line", broken)
        raise RuntimeError("get_model failed")
    assert default_loader.safetensors_weights_iterator is original


def test_unknown_memtotal_keeps_dropping_and_says_so(monkeypatch, shards, fadvised, caplog):
    monkeypatch.setenv("VLLM_HPU_DROP_WEIGHT_PAGE_CACHE_LAG", "0")
    monkeypatch.setattr(wpc, "_mem_total_bytes", lambda: 0)
    messages = []
    monkeypatch.setattr(wpc.logger, "info", lambda msg, *args: messages.append(msg % args))
    calls = []
    list(wpc.wrap_safetensors_iterator(_fake_iterator(calls), [], force=False)(shards, False, None))
    assert len(calls) == 3 and len(fadvised) == 3
    assert any("MemTotal unknown" in m for m in messages)


def test_host_memory_line_reports_threshold_metric():
    line = wpc.host_memory_line()
    for key in ("MemTotal=", "MemFree=", "MemAvailable=", "Active(file)=", "VmRSS=", "used_pct(MemTotal-MemFree)="):
        assert key in line


def _slow_fadvise(monkeypatch, delay, dropped, threads=None):

    def fake_fadvise(fd, offset, length, advice):
        time.sleep(delay)
        dropped.append(os.readlink(f"/proc/self/fd/{fd}"))
        if threads is not None:
            threads.add(threading.current_thread().name)

    monkeypatch.setattr(os, "posix_fadvise", fake_fadvise)


def test_drops_run_in_background_and_do_not_delay_iteration(monkeypatch, shards):
    monkeypatch.setenv("VLLM_HPU_DROP_WEIGHT_PAGE_CACHE_LAG", "0")
    dropped, threads = [], set()
    _slow_fadvise(monkeypatch, 0.5, dropped, threads)
    wrapped = wpc.wrap_safetensors_iterator(_fake_iterator([]))

    it = wrapped(shards, True, None)
    start = time.monotonic()
    names = [next(it)[0] for _ in range(2)]  # shards 0 and 1; shard 0 was queued before shard 1 was yielded
    elapsed = time.monotonic() - start
    assert elapsed < 0.4  # a synchronous 0.5 s drop would have been paid in between
    names += [name for name, _ in it]  # exhausting the iterator waits for the remaining drops

    assert names == [
        "model-00001-of-00010.safetensors", "model-00002-of-00010.safetensors", "model-00010-of-00010.safetensors"
    ]
    assert len(dropped) == 3
    assert threads == {"hpu-weight-page-cache-drop"}


@pytest.mark.parametrize("how", ["early_stop", "consumer_raises"])
def test_all_started_shards_are_dropped_when_consumer_stops(monkeypatch, shards, how):
    monkeypatch.setenv("VLLM_HPU_DROP_WEIGHT_PAGE_CACHE_LAG", "4")
    dropped = []
    _slow_fadvise(monkeypatch, 0.05, dropped)
    wrapped = wpc.wrap_safetensors_iterator(_fake_iterator([]))

    it = wrapped(shards, True, None)
    next(it)
    next(it)
    if how == "early_stop":
        it.close()
    else:
        with pytest.raises(RuntimeError):
            it.throw(RuntimeError("consumer failed"))

    assert len(dropped) == 2  # opened shards only; finally waited for the background thread


def test_all_shards_dropped_when_iterator_completes(monkeypatch, shards):
    monkeypatch.setenv("VLLM_HPU_DROP_WEIGHT_PAGE_CACHE_LAG", "1")
    dropped = []
    _slow_fadvise(monkeypatch, 0.05, dropped)
    list(wpc.wrap_safetensors_iterator(_fake_iterator([]))(shards, True, None))
    assert sorted(dropped) == sorted(shards)


def test_summary_reports_count_and_background_time(monkeypatch, shards):
    messages = []
    monkeypatch.setattr(
        wpc, "logger",
        types.SimpleNamespace(info=lambda m, *a: messages.append(m % a),
                              warning=lambda m, *a: messages.append(m % a),
                              debug=lambda *a: None))
    monkeypatch.setenv("VLLM_HPU_DROP_WEIGHT_PAGE_CACHE_LAG", "0")
    _slow_fadvise(monkeypatch, 0.05, [])
    list(wpc.wrap_safetensors_iterator(_fake_iterator([]))(shards, True, None))
    summary = [m for m in messages if "loader finished" in m]
    assert len(summary) == 1
    assert "called for 3 shards (0 failed)" in summary[0] and "background fadvise time total" in summary[0]


def test_exception_in_drop_thread_does_not_crash_loading(monkeypatch, shards):
    monkeypatch.setenv("VLLM_HPU_DROP_WEIGHT_PAGE_CACHE_LAG", "0")
    monkeypatch.setattr(wpc, "drop_file_page_cache", lambda path: 1 / 0)
    names = [n for n, _ in wpc.wrap_safetensors_iterator(_fake_iterator([]))(shards, True, None)]
    assert len(names) == 3


def test_wait_is_bounded(monkeypatch, shards):
    monkeypatch.setenv("VLLM_HPU_DROP_WEIGHT_PAGE_CACHE_LAG", "0")
    monkeypatch.setenv("VLLM_HPU_DROP_WEIGHT_PAGE_CACHE_WAIT", "0.1")
    _slow_fadvise(monkeypatch, 0.6, [])
    start = time.monotonic()
    list(wpc.wrap_safetensors_iterator(_fake_iterator([]))(shards, True, None))
    assert time.monotonic() - start < 1.0
    time.sleep(2)  # let the daemon thread finish before other tests
