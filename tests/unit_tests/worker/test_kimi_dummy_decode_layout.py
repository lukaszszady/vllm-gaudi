# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel block ids of the contiguous-PA dummy decode batch, Kimi-K3 branch of _prepare_dummy_scenario."""

import pytest

from tests.unit_tests.worker.test_hpu_mm_embeddings import HPUModelRunner, _make_mock_runner


def _layout(is_kimi_k3, num_sched_blocks, pool, decode_num_blocks, bs=2):
    runner = _make_mock_runner({}, {}, is_kimi_k3=is_kimi_k3)
    runner.is_pooling_model = False
    runner.speculative_config = None
    runner.use_contiguous_pa = True
    runner.interleaved_sliding_window = False
    runner.attn_block_size = 128
    runner.kv_cache_config = type("Cfg", (), {"num_blocks": num_sched_blocks})()
    runner._num_attn_blocks = pool
    runner.max_model_len = 1 << 20
    added = []
    runner._add_dummy_request.side_effect = lambda *a, **kw: added.append(kw)
    HPUModelRunner._prepare_dummy_scenario(runner, None, (bs, 1, decode_num_blocks))
    return added


def test_shared_layout_caps_block_id_at_scheduler_pool():
    added = _layout(False, num_sched_blocks=100, pool=400, decode_num_blocks=350)
    assert {kw["block_id"] for kw in added} == {99}
    assert {kw["total_tokens"] for kw in added} == {128}


@pytest.mark.parametrize("decode_num_blocks,block_id,tokens", [(350, 87, 2 * 128 - 1), (400, 99, 4 * 128 - 1),
                                                               (4, 0, 4 * 128 - 1)])
def test_kimi_layout_places_the_highest_kernel_block_at_bucket_minus_one(decode_num_blocks, block_id, tokens):
    # 100 scheduler blocks, 4 kernel blocks each: kernel id decode_num_blocks - 1 = block_id * 4 + remainder
    added = _layout(True, num_sched_blocks=100, pool=400, decode_num_blocks=decode_num_blocks)
    assert {kw["block_id"] for kw in added} == {block_id}
    assert {kw["total_tokens"] for kw in added} == {tokens}


def test_kimi_layout_caps_at_the_kernel_pool():
    added = _layout(True, num_sched_blocks=100, pool=400, decode_num_blocks=10_000)
    assert {kw["block_id"] for kw in added} == {99}


def test_kimi_layout_without_virtual_split_matches_shared_layout():
    added = _layout(True, num_sched_blocks=100, pool=100, decode_num_blocks=60)
    assert {kw["block_id"] for kw in added} == {59}
    assert {kw["total_tokens"] for kw in added} == {128}
