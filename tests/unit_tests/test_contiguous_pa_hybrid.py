# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests: contiguous PA for MLA decode in hybrid (KDA/GDN/mamba) models.

Under non-contiguous PA every MLA decode step gathers the referenced latent KV blocks with index_select
(VLLMKVCache.fetch_from_cache) and the attention BMM reads the copy again. Contiguous PA replaces the gather
with cache.narrow(0, 0, n) and masks unused rows through block_groups / block_usage. These tests check that
flat_pa_mla gives the same result for both layouts, that the defragmenter (unsafe for hybrid state caches)
stays off for hybrids, and that the decode buckets cover every highest block id up to the pool size.

The ops / utils / defragmentation modules are loaded from their files with habana_frameworks and vllm
stubbed, so the tests run on a plain CPU torch install.
"""

import importlib.util
import itertools
import math
import pathlib
import sys
import types
from unittest.mock import MagicMock

import pytest
import torch

from vllm_gaudi.extension.bucketing.common import (HPUBucketingManager, cap_contiguous_pa_decode_buckets,
                                                   find_equal_or_closest_greater_config)
from vllm_gaudi.extension.runtime import clear_config

_EXT_DIR = pathlib.Path(__file__).resolve().parents[2] / "vllm_gaudi" / "extension"
_OPS_STUBS = (
    "habana_frameworks",
    "habana_frameworks.torch",
    "habana_frameworks.torch.core",
    "habana_frameworks.torch.utils",
    "habana_frameworks.torch.utils.experimental",
    "vllm",
    "vllm.model_executor",
    "vllm.model_executor.layers",
    "vllm.model_executor.layers.quantization",
    "vllm.model_executor.layers.quantization.utils",
    "vllm.model_executor.layers.quantization.base_config",
    "vllm.model_executor.layers.fused_moe",
    "vllm.model_executor.layers.fused_moe.activation",
    "vllm_gaudi.extension.runtime",
    "vllm_gaudi.extension.utils",
    "vllm_gaudi.extension.scales",
)

# Kimi-K3 decode buckets from its bucketing file and its pool: 1374 scheduler blocks x 3 kernel blocks.
_KIMI_DECODE_BUCKETS = [(1, 1, 152), (2, 1, 304), (4, 1, 456), (4, 1, 608), (8, 1, 760), (8, 1, 1000), (8, 1, 1216),
                        (16, 1, 1520), (16, 1, 1976), (16, 1, 2432), (32, 1, 3040), (32, 1, 3952), (32, 1, 4864),
                        (45, 1, 5472), (45, 1, 6080), (45, 1, 6840)]
_KIMI_KERNEL_BLOCKS = 1374 * 3


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ops(monkeypatch):
    for name in _OPS_STUBS:
        monkeypatch.setitem(sys.modules, name, MagicMock())
    sys.modules["vllm_gaudi.extension.runtime"].get_config = lambda: types.SimpleNamespace(
        fp32_softmax=False, fused_block_softmax=False, fused_block_softmax_adjustment=False)
    return _load("contiguous_pa_ops_under_test", _EXT_DIR / "ops.py")


@pytest.fixture
def ext_utils(monkeypatch):
    for name in ("habana_frameworks", "habana_frameworks.torch"):
        monkeypatch.setitem(sys.modules, name, MagicMock())
    return _load("contiguous_pa_utils_under_test", _EXT_DIR / "utils.py")


def _make_kv_cache(ext_utils, monkeypatch, use_contiguous_pa):
    monkeypatch.setattr(ext_utils, "get_config", lambda: types.SimpleNamespace(use_contiguous_pa=use_contiguous_pa))
    return ext_utils.VLLMKVCache()


def _paged_attn_buffers(block_tables, seq_lens, block_size, pad_block_id, block_bucket_size, contiguous):
    """Mirror of HPUModelRunner.get_habana_paged_attn_buffers + _set_block_mapping (fake-HPU branch)."""
    batch_size = len(block_tables)
    last_usage = [(seq_len - 1) % block_size + 1 for seq_len in seq_lens]
    block_list = list(itertools.chain(*block_tables))
    block_groups = list(itertools.chain(*([i] * len(bt) for i, bt in enumerate(block_tables))))
    block_usage = list(
        itertools.chain(*([block_size] * (len(bt) - 1) + [lbu] for bt, lbu in zip(block_tables, last_usage))))
    if contiguous:
        indices = [None] * block_bucket_size
        for i, bid in enumerate(block_list):
            indices[bid] = i

        def pad(values, pad_value):
            return [pad_value if i is None else values[i] for i in indices]
    else:

        def pad(values, pad_value):
            return values + [pad_value] * (block_bucket_size - len(values))

    block_list = torch.tensor(pad(block_list, pad_block_id))
    block_groups = torch.tensor(pad(block_groups, -1))
    block_usage = torch.tensor(pad(block_usage, 1))

    mask = torch.arange(block_size).unsqueeze(0) >= block_usage.unsqueeze(-1)
    attn_bias = torch.zeros(mask.shape).masked_fill_(mask, -math.inf)
    block_mapping = torch.nn.functional.one_hot(torch.relu(block_groups), num_classes=batch_size).float()
    block_mapping.masked_fill_(block_groups.lt(0).unsqueeze(-1), 0)
    block_groups = block_groups.masked_fill(block_groups.lt(0), batch_size)
    return block_list, block_groups, block_mapping, attn_bias


def _mla_decode(ops, kv_cache_module, query, key_cache, buffers, block_size, kv_lora_rank):
    block_list, block_groups, block_mapping, attn_bias = buffers
    return ops.flat_pa_mla(query=query,
                           key_cache=key_cache,
                           value_cache=None,
                           block_list=block_list,
                           block_mapping=block_mapping,
                           block_bias=attn_bias,
                           block_groups=block_groups,
                           block_size=block_size,
                           scale=0.125,
                           matmul_qk_op=torch.matmul,
                           matmul_av_op=torch.matmul,
                           batch2block_matmul_op=torch.matmul,
                           block2batch_matmul_op=torch.matmul,
                           keys_fetch_func=kv_cache_module.fetch_from_cache,
                           values_fetch_func=None,
                           kv_lora_rank=kv_lora_rank)


def test_mla_decode_contiguous_slice_matches_gather(ops, ext_utils, monkeypatch):
    torch.manual_seed(0)
    block_size, num_blocks, heads, head_dim, kv_lora_rank = 4, 12, 3, 10, 8
    # Scattered block ids incl. the highest real block; kernel block 0 is unused, as in the runner.
    block_tables = [[5, 1, 9], [11], [3, 7]]
    seq_lens = [10, 2, 8]
    # One extra row: the pad block lives past the real pool (kernel units).
    key_cache = torch.randn((num_blocks + 1) * block_size, 1, head_dim)
    query = torch.randn(len(block_tables), heads, head_dim)

    gather = _make_kv_cache(ext_utils, monkeypatch, use_contiguous_pa=False)
    ref = _mla_decode(ops, gather, query, key_cache,
                      _paged_attn_buffers(block_tables, seq_lens, block_size, num_blocks, 8, contiguous=False),
                      block_size, kv_lora_rank)

    narrow = _make_kv_cache(ext_utils, monkeypatch, use_contiguous_pa=True)
    out = _mla_decode(ops, narrow, query, key_cache,
                      _paged_attn_buffers(block_tables, seq_lens, block_size, num_blocks, num_blocks, contiguous=True),
                      block_size, kv_lora_rank)

    assert out.shape == (len(block_tables), heads, kv_lora_rank)
    torch.testing.assert_close(out, ref)


def test_mla_decode_contiguous_slice_bf16_within_tolerance(ops, ext_utils, monkeypatch):
    torch.manual_seed(1)
    block_size, num_blocks, heads, head_dim, kv_lora_rank = 8, 16, 4, 12, 8
    block_tables = [[15, 2], [4, 6, 8], [10]]
    seq_lens = [9, 20, 1]
    key_cache = torch.randn((num_blocks + 1) * block_size, 1, head_dim).bfloat16()
    query = torch.randn(len(block_tables), heads, head_dim).bfloat16()

    def run(use_contiguous_pa, bucket):
        buffers = _paged_attn_buffers(block_tables, seq_lens, block_size, num_blocks, bucket, use_contiguous_pa)
        buffers = tuple(t.bfloat16() if t.is_floating_point() else t for t in buffers)
        kv_cache = _make_kv_cache(ext_utils, monkeypatch, use_contiguous_pa)
        return _mla_decode(ops, kv_cache, query, key_cache, buffers, block_size, kv_lora_rank)

    torch.testing.assert_close(run(True, num_blocks), run(False, 8), atol=2e-2, rtol=2e-2)


def test_contiguous_fetch_is_a_view_without_gather(ext_utils, monkeypatch):
    cache = torch.randn(6, 4, 1, 3)
    kv_cache = _make_kv_cache(ext_utils, monkeypatch, use_contiguous_pa=True)
    out = kv_cache.fetch_from_cache(cache, torch.arange(5))
    assert out.data_ptr() == cache.data_ptr()
    assert out.shape[0] == 5


@pytest.fixture
def defrag_module(monkeypatch, ext_utils):
    monkeypatch.setitem(sys.modules, "vllm_gaudi.extension.utils", ext_utils)
    module = _load("contiguous_pa_defrag_under_test", _EXT_DIR / "defragmentation.py")
    config = types.SimpleNamespace(defrag=True,
                                   VLLM_DEFRAG_THRESHOLD=None,
                                   VLLM_DEFRAG_WITH_GRAPHS=False,
                                   bridge_mode='eager',
                                   prefix_caching=False)
    monkeypatch.setattr(module, "get_config", lambda: config)
    return module


@pytest.mark.parametrize("hybrid", [True, False])
def test_defragmenter_disabled_for_hybrid_models(defrag_module, hybrid):
    kv_caches = ((torch.empty(0, device='meta'), torch.empty(0, device='meta')), )
    defrag = defrag_module.OnlineDefragmenter(kv_caches, block_size=128, hybrid=hybrid)
    assert defrag.enabled is not hybrid
    defrag.update_state({'req': [7, 9]}, [])
    assert defrag.resolve(7) == 7


def test_cap_contiguous_pa_decode_buckets_kimi():
    buckets = cap_contiguous_pa_decode_buckets(_KIMI_DECODE_BUCKETS, _KIMI_KERNEL_BLOCKS)
    assert max(ctx for _, _, ctx in buckets) == _KIMI_KERNEL_BLOCKS
    assert [b for b in buckets if b[0] == 45] == [(45, 1, _KIMI_KERNEL_BLOCKS)]
    assert [b for b in buckets if b[0] == 32] == [(32, 1, 3040), (32, 1, 3952), (32, 1, _KIMI_KERNEL_BLOCKS)]
    # Smaller batches keep their buckets for short contexts.
    assert {(1, 1, 152), (1, 1, _KIMI_KERNEL_BLOCKS), (16, 1, 2432)} <= set(buckets)

    for bs in sorted({b[0] for b in _KIMI_DECODE_BUCKETS}):
        for max_id in (1, 151, 152, 1375, 3000, _KIMI_KERNEL_BLOCKS - 1, _KIMI_KERNEL_BLOCKS):
            found = find_equal_or_closest_greater_config(buckets, (bs, 1, max_id))
            # Lookups stay within the batch size bucket and never exceed the pool.
            assert found is not None and found[0] == bs
            assert max_id <= found[2] <= _KIMI_KERNEL_BLOCKS


def _file_manager(monkeypatch, tmp_path, use_contiguous_pa, full_block_range):
    bucket_file = tmp_path / "buckets.txt"
    bucket_file.write_text("\n".join(str(b) for b in _KIMI_DECODE_BUCKETS) + "\n")
    monkeypatch.setenv("VLLM_BUCKETING_FROM_FILE", str(bucket_file))
    monkeypatch.setenv("VLLM_CONTIGUOUS_PA", "true" if use_contiguous_pa else "false")
    clear_config()
    manager = HPUBucketingManager()
    manager.initialize(max_num_seqs=45,
                       max_num_prefill_seqs=1,
                       block_size=128,
                       max_num_batched_tokens=4096,
                       max_model_len=262144)
    manager.num_hpu_blocks = _KIMI_KERNEL_BLOCKS
    manager.decode_full_block_range = full_block_range
    manager.generate_decode_buckets()
    return manager


@pytest.mark.parametrize(("use_contiguous_pa", "full_block_range", "capped"), [(True, True, True), (True, False, False),
                                                                               (False, True, False)])
def test_generate_decode_buckets_full_block_range(monkeypatch, tmp_path, use_contiguous_pa, full_block_range, capped):
    monkeypatch.setattr(HPUBucketingManager, "_active_instance", None)
    try:
        manager = _file_manager(monkeypatch, tmp_path, use_contiguous_pa, full_block_range)
        expected = (cap_contiguous_pa_decode_buckets(_KIMI_DECODE_BUCKETS, _KIMI_KERNEL_BLOCKS)
                    if capped else _KIMI_DECODE_BUCKETS)
        assert manager.decode_buckets == expected
    finally:
        clear_config()
