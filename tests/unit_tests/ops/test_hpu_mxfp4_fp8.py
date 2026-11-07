# SPDX-License-Identifier: Apache-2.0
"""MXFP4 -> per-row FP8 conversion of Kimi-K3 routed experts (HPU_KIMI_MOE_FP8_LAYERS). CPU only."""
import pytest
import torch

from vllm_gaudi.ops.hpu_mxfp4_fp8 import (EXACT_EXPONENT_SPAN, layer_index, layer_selected, mxfp4_experts_to_fp8,
                                          mxfp4_to_fp8_rows, parse_layer_spec)
from vllm_gaudi.ops.hpu_situ_moe import dequant_mxfp4_reference


def _random_mxfp4(rows, cols, exp_lo, exp_hi, seed=0, lead=()):
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (*lead, rows, cols // 2), generator=g, dtype=torch.int32).to(torch.uint8)
    scales = torch.randint(exp_lo, exp_hi + 1, (*lead, rows, cols // 32), generator=g, dtype=torch.int32)
    return packed, scales.to(torch.uint8)


def _restore(fp8, row_scale):
    return fp8.to(torch.float32) * row_scale.unsqueeze(-1)


@pytest.mark.parametrize("exp_lo,exp_hi", [(127, 127), (120, 127), (127 - EXACT_EXPONENT_SPAN, 127), (1, 15),
                                           (240, 254)])
def test_exact_within_the_exponent_span(exp_lo, exp_hi):
    packed, scales = _random_mxfp4(16, 256, exp_lo, exp_hi)

    fp8, row_scale, inexact = mxfp4_to_fp8_rows(packed, scales)

    assert fp8.dtype == torch.float8_e4m3fn and fp8.shape == (16, 256)
    assert row_scale.dtype == torch.float32 and row_scale.shape == (16, )
    assert inexact == 0
    assert torch.equal(_restore(fp8, row_scale), dequant_mxfp4_reference(packed, scales, torch.float32))


def test_fp8_values_stay_finite_and_below_the_e4m3_max():
    packed = torch.full((4, 64), 0x77, dtype=torch.uint8)  # 6.0 in both nibbles
    scales = torch.full((4, 4), 127, dtype=torch.uint8)

    fp8, row_scale, inexact = mxfp4_to_fp8_rows(packed, scales)

    values = fp8.to(torch.float32)
    assert inexact == 0 and torch.isfinite(values).all() and values.abs().max() <= 448
    assert torch.equal(_restore(fp8, row_scale), torch.full((4, 128), 6.0))


def test_blocks_beyond_the_span_are_counted_and_rounded():
    packed = torch.full((1, 32), 0x11, dtype=torch.uint8)  # 0.5 everywhere
    scales = torch.tensor([[127, 127 - EXACT_EXPONENT_SPAN - 4]], dtype=torch.uint8)

    fp8, row_scale, inexact = mxfp4_to_fp8_rows(packed, scales)

    restored = _restore(fp8, row_scale)
    reference = dequant_mxfp4_reference(packed, scales, torch.float32)
    assert inexact == 32
    assert torch.equal(restored[0, :32], reference[0, :32])
    assert (restored[0, 32:] - reference[0, 32:]).abs().max() <= reference[0, 32:].abs().max()


def test_zero_rows_and_signed_zero():
    packed = torch.tensor([[0x00, 0x88]], dtype=torch.uint8).repeat(2, 8)
    scales = torch.tensor([[0], [255 - 1]], dtype=torch.uint8)

    fp8, row_scale, inexact = mxfp4_to_fp8_rows(packed, scales)

    assert inexact == 0 and torch.equal(_restore(fp8, row_scale), torch.zeros(2, 32))


def test_experts_match_the_row_conversion():
    packed, scales = _random_mxfp4(8, 64, 118, 127, seed=3, lead=(3, ))

    fp8, row_scale, inexact = mxfp4_experts_to_fp8(packed, scales)

    assert fp8.shape == (3, 8, 64) and row_scale.shape == (3, 8) and inexact == 0
    for e in range(3):
        w, s, _ = mxfp4_to_fp8_rows(packed[e], scales[e])
        assert torch.equal(fp8[e].view(torch.uint8), w.view(torch.uint8)) and torch.equal(row_scale[e], s)
    assert torch.equal(_restore(fp8, row_scale), dequant_mxfp4_reference(packed, scales, torch.float32))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mxfp4_to_fp8_rows(torch.zeros(4, 16, dtype=torch.uint8), torch.zeros(4, 2, dtype=torch.uint8))


@pytest.mark.parametrize("spec,expected", [("", set()), ("0", set()), (" off ", set()), (None, set()), ("all", None),
                                           ("3", {3}), ("1-3,7", {1, 2, 3, 7}), ("5-5, 9,", {5, 9})])
def test_parse_layer_spec(spec, expected):
    assert parse_layer_spec(spec) == expected


@pytest.mark.parametrize("spec", ["x", "3-", "-3", "4-2", "1;2"])
def test_parse_layer_spec_rejects(spec):
    with pytest.raises(ValueError):
        parse_layer_spec(spec)


def test_layer_index_and_selection():
    name = "language_model.model.layers.12.mlp.experts"
    assert layer_index(name) == 12
    assert layer_index("model.sublayers.3.mlp") is None
    assert layer_index(None) is None
    assert layer_selected("10-12", name) and not layer_selected("1-11", name)
    assert layer_selected("all", None) and not layer_selected("1", None) and not layer_selected("", name)
