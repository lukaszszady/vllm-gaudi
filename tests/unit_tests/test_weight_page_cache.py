# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch

from vllm_gaudi.v1.worker import hpu_weight_page_cache as wpc


def _fake_iterator(calls):

    def iterator(files, use_tqdm_on_load, strategy=None, *args, **kwargs):
        calls.append((list(files), strategy, kwargs))
        for path in files:
            yield os.path.basename(path), torch.zeros(1)

    return iterator


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
    assert len(fadvised) == 1
    it.close()  # stopping early still drops the remaining shards
    assert len(fadvised) == 3


def test_torchao_strategy_keeps_upstream_iterator(shards, fadvised):
    calls = []
    wrapped = wpc.wrap_safetensors_iterator(_fake_iterator(calls))
    list(wrapped(shards, False, "torchao"))
    assert len(calls) == 1 and calls[0][0] == shards
    assert fadvised == []


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


def test_host_memory_line_reports_threshold_metric():
    line = wpc.host_memory_line()
    for key in ("MemTotal=", "MemFree=", "MemAvailable=", "Active(file)=", "VmRSS=", "used_pct(MemTotal-MemFree)="):
        assert key in line
