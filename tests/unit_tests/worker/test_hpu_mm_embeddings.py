# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for _gather_mm_embeddings with 2D padded inputs.

Tests the multimodal embedding position computation introduced in PR #1126,
which fixes prefill batching for 2D padded [bs, padded_seq] token layouts.
Also covers _execute_mm_encoder with prefix-cached (stripped) mm items.
"""

import sys
from unittest.mock import MagicMock

import pytest
import torch

# Stub habana_frameworks so the test can run on CPU (no HPU required).
# In CI on HPU machines this is harmless — the real module is already loaded.
if "habana_frameworks" not in sys.modules:
    _hf = MagicMock()
    for submod in [
            "habana_frameworks",
            "habana_frameworks.torch",
            "habana_frameworks.torch.core",
            "habana_frameworks.torch.hpu",
            "habana_frameworks.torch.internal",
            "habana_frameworks.torch.internal.bridge_config",
            "habana_frameworks.torch.utils",
            "habana_frameworks.torch.utils.experimental",
            "habana_frameworks.torch.utils.internal",
    ]:
        sys.modules.setdefault(submod, _hf)

    # habana_frameworks normally patches torch to add torch.hpu
    if not hasattr(torch, "hpu"):
        torch.hpu = MagicMock()

from vllm.multimodal.inputs import (MultiModalBatchedField, MultiModalFeatureSpec, MultiModalFieldElem,
                                    MultiModalKwargsItem, PlaceholderRange)
from vllm.multimodal.utils import strip_covered_mm_data
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_gaudi.v1.worker import hpu_model_runner
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner

HIDDEN_SIZE = 16


def _make_encoder_output(length: int) -> torch.Tensor:
    """Create a fake encoder output tensor of shape (length, HIDDEN_SIZE)."""
    return torch.randn(length, HIDDEN_SIZE)


def _make_mm_feature(identifier: str, offset: int, length: int) -> MultiModalFeatureSpec:
    """Create a MultiModalFeatureSpec with a PlaceholderRange (no is_embed mask)."""
    return MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier=identifier,
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


def _make_request_state(
    req_id: str,
    prompt_len: int,
    mm_features: list,
    num_computed_tokens: int = 0,
) -> CachedRequestState:
    """Create a CachedRequestState with the given mm features."""
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        mm_features=mm_features,
        sampling_params=SamplingParams(),
        generator=None,
        block_ids=([0], ),
        num_computed_tokens=num_computed_tokens,
        output_token_ids=[],
    )


def _make_scheduler_output(req_tokens: dict[str, int]) -> SchedulerOutput:
    """Create a minimal SchedulerOutput with num_scheduled_tokens."""
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=req_tokens,
        total_num_scheduled_tokens=sum(req_tokens.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


class _FakeCpuGpuBuffer:
    """Minimal stand-in for CpuGpuBuffer used by is_mm_embed."""

    def __init__(self, size: int):
        self._cpu = torch.zeros(size, dtype=torch.bool)

    @property
    def cpu(self) -> torch.Tensor:
        return self._cpu

    def copy_to_gpu(self, size: int) -> torch.Tensor:
        return self._cpu[:size].clone()


def _make_mock_runner(
    requests: dict,
    encoder_cache: dict,
    max_tokens: int = 1024,
) -> MagicMock:
    """Create a mock HPUModelRunner with the attributes needed by _gather_mm_embeddings."""
    runner = MagicMock(spec=HPUModelRunner)
    runner.requests = requests
    runner.encoder_cache = encoder_cache
    runner.uses_mrope = False
    runner.is_kimi_k3 = False
    runner.device = "cpu"
    runner.is_mm_embed = _FakeCpuGpuBuffer(max_tokens)
    return runner


class TestGatherMmEmbeddings1DContiguous:
    """Test _gather_mm_embeddings with 1D contiguous (non-padded) layout."""

    def test_two_requests_contiguous(self):
        """Two requests batched in 1D layout: positions via req_start_idx."""
        # req_0: 10 tokens, mm at offset=2 length=4
        # req_1: 8 tokens, mm at offset=1 length=3
        encoder_out_0 = _make_encoder_output(4)
        encoder_out_1 = _make_encoder_output(3)

        requests = {
            "req_0": _make_request_state("req_0", 10, [_make_mm_feature("hash_0", offset=2, length=4)]),
            "req_1": _make_request_state("req_1", 8, [_make_mm_feature("hash_1", offset=1, length=3)]),
        }
        encoder_cache = {"hash_0": encoder_out_0, "hash_1": encoder_out_1}

        runner = _make_mock_runner(requests, encoder_cache)
        sched_out = _make_scheduler_output({"req_0": 10, "req_1": 8})

        mm_embeds, is_mm_embed = HPUModelRunner._gather_mm_embeddings(
            runner,
            sched_out,
            req_ids=["req_0", "req_1"],
            total_num_scheduled_tokens=18,
            padded_seq_len=None,
        )

        is_mm = is_mm_embed[:18]
        # req_0: positions [2:6] (offset=2, length=4)
        assert is_mm[2:6].all(), f"req_0 mm positions wrong: {is_mm}"
        # req_1: positions [10+1:10+4] = [11:14]
        assert is_mm[11:14].all(), f"req_1 mm positions wrong: {is_mm}"
        # non-mm positions must be False
        assert not is_mm[0:2].any()
        assert not is_mm[6:11].any()
        assert not is_mm[14:18].any()
        assert len(mm_embeds) == 2


class TestGatherMmEmbeddings2DPadded:
    """Test _gather_mm_embeddings with 2D padded layout (PR #1126 fix)."""

    def test_two_requests_padded(self):
        """Two requests in 2D padded layout: positions via batch_row * padded_seq_len."""
        padded_seq_len = 16
        total_tokens = 2 * padded_seq_len  # 32

        encoder_out_0 = _make_encoder_output(4)
        encoder_out_1 = _make_encoder_output(3)

        requests = {
            "req_0": _make_request_state("req_0", 10, [_make_mm_feature("hash_0", offset=2, length=4)]),
            "req_1": _make_request_state("req_1", 8, [_make_mm_feature("hash_1", offset=1, length=3)]),
        }
        encoder_cache = {"hash_0": encoder_out_0, "hash_1": encoder_out_1}

        runner = _make_mock_runner(requests, encoder_cache)
        sched_out = _make_scheduler_output({"req_0": 10, "req_1": 8})

        mm_embeds, is_mm_embed = HPUModelRunner._gather_mm_embeddings(
            runner,
            sched_out,
            req_ids=["req_0", "req_1"],
            total_num_scheduled_tokens=total_tokens,
            padded_seq_len=padded_seq_len,
        )

        is_mm = is_mm_embed[:total_tokens]
        # req_0 (batch_row=0): [0*16+2 : 0*16+2+4] = [2:6]
        assert is_mm[2:6].all(), f"req_0 mm positions wrong: {is_mm}"
        # req_1 (batch_row=1): [1*16+1 : 1*16+1+3] = [17:20]
        assert is_mm[17:20].all(), f"req_1 mm positions wrong: {is_mm}"
        # padding region and non-mm positions must be False
        assert not is_mm[0:2].any()
        assert not is_mm[6:17].any()
        assert not is_mm[20:32].any()
        assert len(mm_embeds) == 2

    def test_padded_partial_prefill(self):
        """2D padded batch with partial prefill (num_computed_tokens > 0)."""
        padded_seq_len = 16

        # req_0: 10 scheduled tokens, num_computed_tokens=1
        # mm at offset=2, length=4
        #   start_idx = max(1-2, 0) = 0
        #   end_idx   = min(1-2+10, 4) = 4
        #   req_start_pos = 0*16 + 2 - 1 = 1
        #   marks [1+0 : 1+4] = [1:5]
        encoder_out = _make_encoder_output(4)

        requests = {
            "req_0":
            _make_request_state("req_0", 11, [_make_mm_feature("hash_0", offset=2, length=4)], num_computed_tokens=1),
        }
        encoder_cache = {"hash_0": encoder_out}

        runner = _make_mock_runner(requests, encoder_cache)
        sched_out = _make_scheduler_output({"req_0": 10})

        mm_embeds, is_mm_embed = HPUModelRunner._gather_mm_embeddings(
            runner,
            sched_out,
            req_ids=["req_0"],
            total_num_scheduled_tokens=padded_seq_len,
            padded_seq_len=padded_seq_len,
        )

        is_mm = is_mm_embed[:padded_seq_len]
        # marks [1:5]
        assert is_mm[1:5].all(), f"partial prefill mm positions wrong: {is_mm}"
        assert not is_mm[0:1].any()
        assert not is_mm[5:16].any()
        assert len(mm_embeds) == 1

    def test_padded_positions_differ_from_contiguous(self):
        """Verify that 2D padded and 1D contiguous produce different position maps."""
        encoder_out_0 = _make_encoder_output(4)
        encoder_out_1 = _make_encoder_output(3)

        mm_feat_0 = _make_mm_feature("hash_0", offset=2, length=4)
        mm_feat_1 = _make_mm_feature("hash_1", offset=1, length=3)

        requests = {
            "req_0": _make_request_state("req_0", 10, [mm_feat_0]),
            "req_1": _make_request_state("req_1", 8, [mm_feat_1]),
        }
        encoder_cache = {"hash_0": encoder_out_0, "hash_1": encoder_out_1}

        padded_seq_len = 16
        total_padded = 2 * padded_seq_len
        sched_out = _make_scheduler_output({"req_0": 10, "req_1": 8})

        # Run with padded_seq_len (2D)
        runner_padded = _make_mock_runner(requests, encoder_cache)
        _, is_mm_padded = HPUModelRunner._gather_mm_embeddings(
            runner_padded,
            sched_out,
            req_ids=["req_0", "req_1"],
            total_num_scheduled_tokens=total_padded,
            padded_seq_len=padded_seq_len,
        )

        # Run without padded_seq_len (1D contiguous)
        runner_contig = _make_mock_runner(requests, encoder_cache)
        _, is_mm_contig = HPUModelRunner._gather_mm_embeddings(
            runner_contig,
            sched_out,
            req_ids=["req_0", "req_1"],
            total_num_scheduled_tokens=18,
            padded_seq_len=None,
        )

        is_padded = is_mm_padded[:total_padded]
        is_contig = is_mm_contig[:18]

        # req_1 mm in padded: starts at position 17 (1*16+1)
        # req_1 mm in contiguous: starts at position 11 (10+1)
        assert is_padded[17].item() is True
        assert is_contig[11].item() is True
        # padded version should NOT have mm at contiguous position 11
        # (that falls in padding region of req_0)
        assert is_padded[11].item() is False


def _make_pixel_mm_feature(identifier: str, offset: int, length: int) -> MultiModalFeatureSpec:
    """An image feature carrying a pixel payload plus keep_on_cpu M-RoPE metadata."""
    data = MultiModalKwargsItem({
        "pixel_values":
        MultiModalFieldElem(data=torch.empty(4), field=MultiModalBatchedField()),
        "image_grid_thw":
        MultiModalFieldElem(data=torch.ones(1, 3, dtype=torch.long), field=MultiModalBatchedField(keep_on_cpu=True)),
    })
    return MultiModalFeatureSpec(data=data,
                                 modality="image",
                                 identifier=identifier,
                                 mm_position=PlaceholderRange(offset=offset, length=length))


class TestExecuteMmEncoderPrefixCached:
    """_execute_mm_encoder must not encode items the scheduler stripped."""

    def test_covered_item_is_not_encoded(self):
        # A repeated image fully inside a prefix-cache hit whose encoder output was
        # already evicted: the scheduler strips its pixels (M-RoPE keeps grid_thw).
        num_computed = 256
        features = strip_covered_mm_data(
            [_make_pixel_mm_feature("hash_cached", 0, 100),
             _make_pixel_mm_feature("hash_new", 300, 50)],
            num_computed,
            uses_mrope=True,
        )
        assert list(features[0].data.keys()) == ["image_grid_thw"]

        def embed_multimodal(**kwargs):
            # Like the real model: no pixel payload -> no embeddings.
            if "pixel_values" not in kwargs:
                return []
            return [torch.zeros(50, HIDDEN_SIZE) for _ in range(kwargs["pixel_values"].shape[0])]

        requests = {"req_0": _make_request_state("req_0", 400, features, num_computed_tokens=num_computed)}
        runner = _make_mock_runner(requests, encoder_cache={})
        runner.pin_memory = False
        runner.model = MagicMock()
        runner.model.embed_multimodal.side_effect = embed_multimodal

        HPUModelRunner._execute_mm_encoder(runner, _make_scheduler_output({"req_0": 144}), ["req_0"])

        assert runner.model.embed_multimodal.call_count == 1
        assert set(runner.encoder_cache) == {"hash_new"}


# Video placeholders carry an is_embed mask with gaps (separator/timestamp tokens); images have None.
_VIDEO_LENGTH = 12
_VIDEO_IS_EMBED = torch.tensor([1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0], dtype=torch.bool)


@pytest.fixture
def no_bool_mask_on_tensors(monkeypatch):
    """Make any Tensor get/set item with a boolean tensor key raise: HPU cannot build that data-dependent shape."""
    get_item, set_item = torch.Tensor.__getitem__, torch.Tensor.__setitem__

    def _check(key):
        keys = key if isinstance(key, tuple) else (key, )
        for k in keys:
            if isinstance(k, torch.Tensor) and k.dtype == torch.bool:
                raise AssertionError("boolean-mask indexing of a tensor")

    def guarded_get(self, key):
        _check(key)
        return get_item(self, key)

    def guarded_set(self, key, value):
        _check(key)
        return set_item(self, key, value)

    monkeypatch.setattr(torch.Tensor, "__getitem__", guarded_get)
    monkeypatch.setattr(torch.Tensor, "__setitem__", guarded_set)


def _make_video_feature(identifier: str, offset: int, with_pixels: bool = False) -> MultiModalFeatureSpec:
    position = PlaceholderRange(offset=offset, length=_VIDEO_LENGTH, is_embed=_VIDEO_IS_EMBED.clone())
    if with_pixels:
        feature = _make_pixel_mm_feature(identifier, offset, _VIDEO_LENGTH)
        return MultiModalFeatureSpec(data=feature.data, modality="video", identifier=identifier, mm_position=position)
    return MultiModalFeatureSpec(data=None, modality="video", identifier=identifier, mm_position=position)


def _scattered_encoder_output(rows: torch.Tensor) -> torch.Tensor:
    """What the encoder cache holds for an item with is_embed: embedding rows at the True positions, NaN elsewhere."""
    cached = torch.full((_VIDEO_LENGTH, HIDDEN_SIZE), torch.nan)
    cached[torch.nonzero(_VIDEO_IS_EMBED, as_tuple=True)[0]] = rows
    return cached


class TestMmEmbedMaskCallSites:
    """Both runner call sites with a non-None is_embed must avoid boolean-mask indexing of device tensors."""

    @pytest.mark.parametrize("num_computed", [0, 5])
    def test_gather_selects_the_embedding_rows(self, no_bool_mask_on_tensors, monkeypatch, num_computed):
        offset = 3
        rows = _make_encoder_output(int(_VIDEO_IS_EMBED.sum()))
        encoder_cache = {"vid": _scattered_encoder_output(rows)}
        requests = {
            "req_0":
            _make_request_state("req_0", 40, [_make_video_feature("vid", offset)], num_computed_tokens=num_computed)
        }
        runner = _make_mock_runner(requests, encoder_cache)
        spy = MagicMock(wraps=hpu_model_runner.select_embed_rows)
        monkeypatch.setattr(hpu_model_runner, "select_embed_rows", spy)

        scheduled = 40 - num_computed
        mm_embeds, is_mm_embed = HPUModelRunner._gather_mm_embeddings(runner,
                                                                      _make_scheduler_output({"req_0": scheduled}),
                                                                      req_ids=["req_0"],
                                                                      total_num_scheduled_tokens=scheduled)

        # The part of the placeholder not yet computed: [start_idx, _VIDEO_LENGTH)
        start_idx = max(num_computed - offset, 0)
        mask = _VIDEO_IS_EMBED[start_idx:]
        expected = encoder_cache["vid"][start_idx:][torch.nonzero(mask, as_tuple=True)[0]]
        assert spy.call_count == 1
        assert len(mm_embeds) == 1 and torch.equal(mm_embeds[0], expected)
        assert not torch.isnan(mm_embeds[0]).any()
        expected_mask = torch.zeros(scheduled, dtype=torch.bool)
        first = offset + start_idx - num_computed
        expected_mask[first:first + len(mask)] = mask
        assert torch.equal(is_mm_embed[:scheduled], expected_mask)

    def test_gather_without_is_embed_does_not_use_the_helper(self, monkeypatch):
        encoder_cache = {"img": _make_encoder_output(4)}
        requests = {"req_0": _make_request_state("req_0", 10, [_make_mm_feature("img", offset=2, length=4)])}
        runner = _make_mock_runner(requests, encoder_cache)
        spy = MagicMock(wraps=hpu_model_runner.select_embed_rows)
        monkeypatch.setattr(hpu_model_runner, "select_embed_rows", spy)

        mm_embeds, _ = HPUModelRunner._gather_mm_embeddings(runner,
                                                            _make_scheduler_output({"req_0": 10}),
                                                            req_ids=["req_0"],
                                                            total_num_scheduled_tokens=10)

        assert spy.call_count == 0
        assert torch.equal(mm_embeds[0], encoder_cache["img"])

    @pytest.mark.parametrize("num_computed", [0, 5])
    def test_encoder_cache_layout_is_what_gather_selects_from(self, monkeypatch, num_computed):
        # _execute_mm_encoder keeps the original placeholder scatter (the HPU build ran it with a device mask; its
        # source rows are not required to match the is_embed count there). Only _gather_mm_embeddings uses host
        # indices, so it must read exactly the layout the encoder stores: one row per placeholder position.
        rows = _make_encoder_output(int(_VIDEO_IS_EMBED.sum()))
        requests = {
            "req_0":
            _make_request_state("req_0",
                                40, [_make_video_feature("vid", 3, with_pixels=True)],
                                num_computed_tokens=num_computed)
        }
        runner = _make_mock_runner(requests, encoder_cache={})
        runner.pin_memory = False
        runner.model = MagicMock()
        runner.model.embed_multimodal.return_value = [rows]
        # The encoder path must not go through the host-index helpers.
        select_spy = MagicMock(wraps=hpu_model_runner.select_embed_rows)
        monkeypatch.setattr(hpu_model_runner, "select_embed_rows", select_spy)
        assert not hasattr(hpu_model_runner, "scatter_embed_rows")
        index_copy = MagicMock(side_effect=AssertionError("index_copy_ in the encoder path"))
        monkeypatch.setattr(torch.Tensor, "index_copy_", index_copy)

        HPUModelRunner._execute_mm_encoder(runner, _make_scheduler_output({"req_0": 40}), ["req_0"])

        cached = runner.encoder_cache["vid"]
        assert select_spy.call_count == 0
        assert cached.shape == (_VIDEO_LENGTH, HIDDEN_SIZE)
        embed_idx = torch.nonzero(_VIDEO_IS_EMBED, as_tuple=True)[0]
        gap_idx = torch.nonzero(~_VIDEO_IS_EMBED, as_tuple=True)[0]
        assert torch.equal(cached.index_select(0, embed_idx), rows)
        assert torch.isnan(cached.index_select(0, gap_idx)).all()

        scheduled = 40 - num_computed
        mm_embeds, _ = HPUModelRunner._gather_mm_embeddings(runner,
                                                            _make_scheduler_output({"req_0": scheduled}),
                                                            req_ids=["req_0"],
                                                            total_num_scheduled_tokens=scheduled)
        start_idx = max(num_computed - 3, 0)
        expected_rows = int(_VIDEO_IS_EMBED[start_idx:].sum())
        assert select_spy.call_count == 1
        assert mm_embeds[0].shape == (expected_rows, HIDDEN_SIZE)
        assert not torch.isnan(mm_embeds[0]).any()
