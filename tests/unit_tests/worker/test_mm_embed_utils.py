# SPDX-License-Identifier: Apache-2.0
"""Row selection by an is_embed mask must equal boolean-mask indexing, without ever indexing with a bool tensor.

Pure torch: runs on CPU without vllm or habana.
"""
import importlib.util
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).resolve().parents[3] / "vllm_gaudi" / "v1" / "worker" / "mm_embed_utils.py"
HIDDEN = 8
MASKS = {
    "gaps": [True, False, False, True, True, False, True, False],
    "all_true": [True] * 8,
    "all_false": [False] * 8,
    "single": [False, False, False, True, False, False, False, False],
}


@pytest.fixture(scope="module")
def utils():
    spec = importlib.util.spec_from_file_location("mm_embed_utils_under_test", PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mask(name):
    return torch.tensor(MASKS[name], dtype=torch.bool)


@pytest.mark.parametrize("name", MASKS)
def test_select_equals_boolean_mask_indexing(utils, name):
    rows = torch.randn(8, HIDDEN)
    mask = _mask(name)
    got = utils.select_embed_rows(rows, mask)
    assert got.shape == rows[mask].shape
    assert torch.equal(got, rows[mask])


@pytest.mark.parametrize("name", MASKS)
def test_scatter_equals_boolean_mask_assignment(utils, name):
    mask = _mask(name)
    output = torch.randn(int(mask.sum()), HIDDEN)
    expected = output.new_full((mask.shape[0], HIDDEN), torch.nan)
    expected[mask] = output
    got = utils.scatter_embed_rows(output, mask, fill_value=torch.nan)
    assert torch.equal(got.isnan(), expected.isnan())
    assert torch.equal(got.nan_to_num(), expected.nan_to_num())


def test_scatter_then_select_round_trips(utils):
    mask = _mask("gaps")
    output = torch.randn(int(mask.sum()), HIDDEN)
    scattered = utils.scatter_embed_rows(output, mask, fill_value=torch.nan)
    # the runner slices the cached item and selects with the matching slice of the mask
    assert torch.equal(utils.select_embed_rows(scattered, mask), output)
    assert torch.equal(utils.select_embed_rows(scattered[2:7], mask[2:7]), scattered[2:7][mask[2:7]])


def test_preserves_dtype_and_never_indexes_with_a_bool_tensor(utils, monkeypatch):
    rows = torch.randn(8, HIDDEN).to(torch.bfloat16)

    # a bool tensor reaching Tensor.__getitem__/__setitem__ is exactly what the Habana graph compiler rejects
    orig_get, orig_set = torch.Tensor.__getitem__, torch.Tensor.__setitem__

    def guard(key):
        if isinstance(key, torch.Tensor) and key.dtype == torch.bool:
            raise AssertionError("boolean-mask indexing")

    monkeypatch.setattr(torch.Tensor, "__getitem__", lambda self, key: (guard(key), orig_get(self, key))[1])
    monkeypatch.setattr(torch.Tensor, "__setitem__", lambda self, key, v: (guard(key), orig_set(self, key, v))[1])
    mask = _mask("gaps")
    sel = utils.select_embed_rows(rows, mask)
    assert sel.dtype == torch.bfloat16
    out = utils.scatter_embed_rows(sel, mask, fill_value=torch.nan)
    assert out.dtype == torch.bfloat16 and out.shape == rows.shape
