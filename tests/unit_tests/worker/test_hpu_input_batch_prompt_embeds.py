# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm_gaudi.v1.worker.hpu_input_batch import CachedRequestState, InputBatch

VOCAB_SIZE = 128
HIDDEN = 8


def _make_batch(max_num_reqs: int = 4) -> InputBatch:
    return InputBatch(
        max_num_reqs=max_num_reqs,
        max_model_len=64,
        max_num_batched_tokens=64,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=VOCAB_SIZE,
        block_sizes=[1],
        kernel_block_sizes=[1],
    )


def _make_req(req_id: str, prompt_token_ids=None, prompt_embeds=None, prompt_is_token_ids=None, output=None):
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=prompt_token_ids,
        prompt_embeds=prompt_embeds,
        prompt_is_token_ids=prompt_is_token_ids,
        sampling_params=SamplingParams(temperature=0.0),
        mm_features=[],
        block_ids=([], ),
        generator=None,
        num_computed_tokens=0,
        output_token_ids=list(output or []),
    )


def test_cached_request_state_num_prompt_tokens_from_embeds():
    state = _make_req("a", prompt_embeds=torch.randn(5, HIDDEN), output=[7, 8])
    assert state.num_prompt_tokens == 5
    assert state.num_tokens == 7
    assert state.get_token_id(5) == 7
    with pytest.raises(ValueError):
        state.get_token_id(2)


def test_add_request_embeds_only():
    batch = _make_batch()
    embeds = torch.randn(5, HIDDEN)
    batch.token_ids_cpu[0, :10] = 99
    req = _make_req("a", prompt_embeds=embeds, output=[7])
    req.num_computed_tokens = 5
    idx = batch.add_request(req)
    assert batch.num_prompt_tokens[idx] == 5
    assert batch.num_tokens[idx] == 6
    assert (batch.token_ids_cpu[idx, :5] == 0).all()
    assert batch.token_ids_cpu[idx, 5] == 7
    stored, mask = batch.req_prompt_embeds[idx]
    assert torch.equal(stored, embeds)
    assert mask is None


def test_add_request_mixed_mode_zeroes_embed_positions():
    batch = _make_batch()
    is_token_ids = [True, False, False, True]
    idx = batch.add_request(
        _make_req("a",
                  prompt_token_ids=[11, 9999, 9999, 12],
                  prompt_embeds=torch.randn(4, HIDDEN),
                  prompt_is_token_ids=is_token_ids))
    assert batch.token_ids_cpu[idx, :4].tolist() == [11, 0, 0, 12]
    assert batch.req_prompt_embeds[idx][1].tolist() == is_token_ids


def test_token_request_has_no_embeds():
    batch = _make_batch()
    idx = batch.add_request(_make_req("a", prompt_token_ids=[1, 2, 3]))
    assert batch.num_prompt_tokens[idx] == 3
    assert idx not in batch.req_prompt_embeds


def test_embeds_follow_request_through_swap_remove_condense():
    batch = _make_batch()
    e0, e2 = torch.randn(3, HIDDEN), torch.randn(4, HIDDEN)
    batch.add_request(_make_req("a", prompt_embeds=e0))
    batch.add_request(_make_req("b", prompt_token_ids=[1, 2]))
    batch.add_request(_make_req("c", prompt_embeds=e2))

    batch.swap_states(0, 1)
    assert 0 not in batch.req_prompt_embeds
    assert torch.equal(batch.req_prompt_embeds[1][0], e0)

    batch.swap_states(1, 2)
    assert torch.equal(batch.req_prompt_embeds[1][0], e2)
    assert torch.equal(batch.req_prompt_embeds[2][0], e0)

    removed = batch.remove_request("b")
    assert removed == 0
    batch.condense([removed])
    assert batch.req_id_to_index == {"a": 0, "c": 1}
    assert torch.equal(batch.req_prompt_embeds[0][0], e0)
    assert torch.equal(batch.req_prompt_embeds[1][0], e2)
    assert 2 not in batch.req_prompt_embeds

    assert batch.remove_request("c") == 1
    assert 1 not in batch.req_prompt_embeds
