# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests: pad block / pad slot ids and contiguous-PA dummy decode placement in kernel-block units.

Under hybrid virtual block splitting (Kimi-K3: scheduler block 384 tokens, kernel block 128, ratio 3)
the attention cache is allocated with get_kv_cache_shape(num_kernel_blocks + 1): kernel rows
0..num_kernel_blocks - 1 belong to scheduler blocks (s -> s * 3 .. s * 3 + 2), the last row is the pad row.
"""

import itertools

import pytest

from vllm_gaudi.v1.worker.kv_block_layout import (attn_pad_ids, contiguous_dummy_decode_layout, pad_slot_cycle)

_SCHED_BLOCK, _KERNEL_BLOCK = 384, 128
_RATIO = _SCHED_BLOCK // _KERNEL_BLOCK
_NUM_BLOCKS = 1374  # scheduler blocks of the Kimi-K3 pool
_NUM_KERNEL_BLOCKS = _NUM_BLOCKS * _RATIO


def _kernel_blocks_of(sched_block):
    return range(sched_block * _RATIO, (sched_block + 1) * _RATIO)


def test_pad_ids_point_at_trailing_pad_row_under_virtual_split():
    num_attn_blocks, pad_block_id, pad_slot_id = attn_pad_ids(_NUM_BLOCKS, _NUM_KERNEL_BLOCKS, _KERNEL_BLOCK)
    cache_rows = _NUM_KERNEL_BLOCKS + 1  # get_kv_cache_shape(num_kernel_blocks + 1, ...)
    assert num_attn_blocks == _NUM_KERNEL_BLOCKS
    assert pad_block_id == _NUM_KERNEL_BLOCKS == cache_rows - 1
    assert pad_slot_id // _KERNEL_BLOCK == pad_block_id and pad_slot_id % _KERNEL_BLOCK == 0
    live = {k for s in range(_NUM_BLOCKS) for k in _kernel_blocks_of(s)}
    assert pad_block_id not in live
    # The old scheduler-unit pad id is a live kernel block (first one of scheduler block 458).
    assert _NUM_BLOCKS in _kernel_blocks_of(_NUM_BLOCKS // _RATIO)


def test_pad_ids_unchanged_without_virtual_split():
    assert attn_pad_ids(1000, None, 128) == (1000, 1000, 1000 * 128)
    # Equal scheduler and kernel block size: kernel count == scheduler count.
    assert attn_pad_ids(1000, 1000, 128) == (1000, 1000, 1000 * 128)


@pytest.mark.parametrize("num_pad_rows", [1, 127, 128, 129, 300, 1000])
def test_pad_slots_stay_in_pad_row(num_pad_rows):
    _, pad_block_id, pad_slot_id = attn_pad_ids(_NUM_BLOCKS, _NUM_KERNEL_BLOCKS, _KERNEL_BLOCK)
    cache_slots = (_NUM_KERNEL_BLOCKS + 1) * _KERNEL_BLOCK
    # Runner decode: base = _PAD_BLOCK_ID * attn_block_size; spec decode: base = _PAD_SLOT_ID.
    for base in (pad_block_id * _KERNEL_BLOCK, pad_slot_id):
        slots = list(itertools.islice(pad_slot_cycle(base, _KERNEL_BLOCK), num_pad_rows))
        assert all(s // _KERNEL_BLOCK == pad_block_id for s in slots)
        assert max(slots) < cache_slots
        assert len(set(slots)) == min(num_pad_rows, _KERNEL_BLOCK)


def test_pad_slots_with_scheduler_block_length_would_leave_the_cache():
    # Regression guard for hpu_eagle: cycling over block_size (384) instead of the pad row length (128)
    # walks past the end of the kernel-block cache.
    _, _, pad_slot_id = attn_pad_ids(_NUM_BLOCKS, _NUM_KERNEL_BLOCKS, _KERNEL_BLOCK)
    cache_slots = (_NUM_KERNEL_BLOCKS + 1) * _KERNEL_BLOCK
    slots = list(itertools.islice(pad_slot_cycle(pad_slot_id, _SCHED_BLOCK), 200))
    assert max(slots) >= cache_slots


@pytest.mark.parametrize("bucket", [1, 2, 3, 4, 152, 6840, _NUM_KERNEL_BLOCKS, _NUM_KERNEL_BLOCKS + 500])
def test_contiguous_dummy_decode_ratio3_new_token_in_last_kernel_block(bucket):
    block_id, seq_len = contiguous_dummy_decode_layout(bucket, _NUM_KERNEL_BLOCKS, _NUM_BLOCKS, _KERNEL_BLOCK,
                                                       _KERNEL_BLOCK, False)
    target = min(bucket, _NUM_KERNEL_BLOCKS) - 1
    assert block_id == target // _RATIO < _NUM_BLOCKS
    # The new decode token sits at position seq_len (positions 0..seq_len): its kernel block is
    # block_id * ratio + seq_len // kernel_block, which must be the highest id, bucket - 1.
    assert block_id * _RATIO + seq_len // _KERNEL_BLOCK == target
    assert seq_len < _SCHED_BLOCK


@pytest.mark.parametrize("bucket", [1, 7, 500, 999, 1000, 4000])
@pytest.mark.parametrize("sliding", [False, True])
def test_contiguous_dummy_decode_ratio1_as_before(bucket, sliding):
    num_blocks, block_size, min_tokens = 1000, 128, 5 * 128
    block_id, seq_len = contiguous_dummy_decode_layout(bucket, num_blocks, num_blocks, block_size, min_tokens,
                                                       sliding)
    # Previous code: block_id = min(bucket - 1, num_blocks - 1), min_tokens_per_seq unchanged.
    assert (block_id, seq_len) == (min(bucket - 1, num_blocks - 1), min_tokens)


def test_contiguous_dummy_decode_before_kv_cache_init():
    # _num_attn_blocks is 0 until initialize_kv_cache: no ZeroDivisionError, falls back to ratio 1.
    assert contiguous_dummy_decode_layout(152, 0, 1374, 128, 128, False) == (151, 128)
    assert contiguous_dummy_decode_layout(5000, 0, 1374, 128, 128, False) == (1373, 128)
