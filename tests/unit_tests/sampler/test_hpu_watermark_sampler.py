# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the watermark-aware sampler against the upstream Gumbel watermarker."""

from dataclasses import dataclass

import pytest
import torch
from vllm.config.watermarking import WatermarkConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.watermarking import create_watermarker

from vllm_gaudi.v1._vendored.sample.logits_processor import LogitsProcessors
from vllm_gaudi.v1._vendored.sample.metadata import SamplingMetadata
from vllm_gaudi.v1._vendored.sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample
from vllm_gaudi.v1._vendored.sample.sampler import Sampler
from vllm_gaudi.v1.sample.hpu_watermark_sampler import HPUWatermarkSampler

VOCAB_SIZE = 64
KEY = 42
CONTEXT_WIDTH = 4


@dataclass
class _Request:
    sampling_params: SamplingParams
    prompt_token_ids: list[int]
    output_token_ids: list[int]


def _watermark_sampler(**kwargs) -> HPUWatermarkSampler:
    config = WatermarkConfig(key=KEY, context_width=CONTEXT_WIDTH, **kwargs)
    return HPUWatermarkSampler(create_watermarker(config),
                               deduplicate_contexts=config.deduplicate_contexts,
                               deduplicate_contexts_max_history=config.deduplicate_contexts_max_history)


def _metadata(params: list[SamplingParams], generators: dict[int, torch.Generator] | None = None) -> SamplingMetadata:
    temperature = torch.tensor([p.temperature for p in params], dtype=torch.float32)
    all_greedy = all(p.temperature == 0 for p in params)
    all_random = all(p.temperature != 0 for p in params)
    top_k = torch.tensor([p.top_k if p.top_k > 0 else VOCAB_SIZE for p in params])
    top_p = torch.tensor([p.top_p for p in params], dtype=torch.float32)
    return SamplingMetadata(
        temperature=None if all_greedy else temperature,
        all_greedy=all_greedy,
        all_random=all_random,
        top_p=top_p if any(p.top_p < 1 for p in params) else None,
        top_k=top_k if any(0 < p.top_k < VOCAB_SIZE for p in params) else None,
        generators=generators or {},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(len(params)),
        presence_penalties=torch.zeros(len(params)),
        repetition_penalties=torch.ones(len(params)),
        output_token_ids=[[] for _ in params],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def _logits(rows: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(rows, VOCAB_SIZE, generator=generator)


def _processed(logits: torch.Tensor, params: list[SamplingParams]) -> torch.Tensor:
    temperature = torch.tensor([p.temperature if p.temperature else 1.0 for p in params])
    metadata = _metadata(params)
    return apply_top_k_top_p(logits / temperature.unsqueeze(1), metadata.top_k, metadata.top_p)


def _watermarked(**kwargs) -> SamplingParams:
    return SamplingParams(temperature=0.8, top_p=0.95, watermarking=True, **kwargs)


def test_make_inputs_contexts_and_activity():
    sampler = _watermark_sampler()
    requests = [
        _Request(_watermarked(), [1, 2, 3], [10, 11]),
        _Request(_watermarked(), [1, 2, 3], [10, 11, 12, 13, 14, 15]),
        _Request(SamplingParams(temperature=0.8, watermarking=False), [1], [5, 6, 7, 8]),
        _Request(SamplingParams(temperature=0, watermarking=True), [1], [5, 6, 7, 8]),
        _Request(_watermarked(), [1], []),
    ]
    inputs = sampler.make_inputs(requests, "cpu")
    assert inputs is not None
    assert inputs.contexts.tolist() == [
        [-1, -1, 10, 11],
        [12, 13, 14, 15],
        [5, 6, 7, 8],
        [5, 6, 7, 8],
        [-1, -1, -1, -1],
    ]
    assert inputs.active.tolist() == [True, True, False, False, True]


def test_make_inputs_without_watermarked_rows_is_none():
    sampler = _watermark_sampler()
    requests = [
        _Request(SamplingParams(temperature=0.8, watermarking=False), [1], [5]),
        _Request(SamplingParams(temperature=0, watermarking=True), [1], [5]),
    ]
    assert sampler.make_inputs(requests, "cpu") is None


@pytest.mark.parametrize("scope, expected", [("none", True), ("single_turn", False), ("all", False)])
def test_make_inputs_repeated_context_is_not_watermarked(scope, expected):
    sampler = _watermark_sampler(deduplicate_contexts=scope)
    history = [7, 8, 9, 3, 7, 8, 9, 3]
    inputs = sampler.make_inputs([_Request(_watermarked(), [1], history)], "cpu")
    assert inputs.contexts.tolist() == [[7, 8, 9, 3]]
    assert inputs.active.tolist() == [expected]


def test_make_inputs_all_scope_searches_prompt_and_skips_partial_context():
    sampler = _watermark_sampler(deduplicate_contexts="all")
    in_prompt = _Request(_watermarked(), [4, 5, 6, 7, 1], [4, 5, 6, 7])
    partial = _Request(_watermarked(), [1, 2], [9])
    fresh = _Request(_watermarked(), [1, 2], [9, 10, 11, 12])
    inputs = sampler.make_inputs([in_prompt, partial, fresh], "cpu")
    assert inputs.active.tolist() == [False, False, True]
    single_turn = _watermark_sampler(deduplicate_contexts="single_turn").make_inputs([in_prompt], "cpu")
    assert single_turn.active.tolist() == [True]


def test_watermarked_rows_match_upstream_watermarker():
    sampler = _watermark_sampler()
    outputs = [[10, 11, 12, 13, 14], [20, 21], [], [3, 4, 5, 6, 7, 8], [9]] * 3
    params = [_watermarked(top_k=8) if i % 2 else _watermarked() for i in range(len(outputs))]
    requests = [_Request(p, [1, 2], o) for p, o in zip(params, outputs)]
    logits = _logits(len(params))
    metadata = _metadata(params)
    metadata.watermark = sampler.make_inputs(requests, "cpu")

    sampled, _ = sampler.sample(logits.clone(), metadata)

    contexts = torch.tensor([[11, 12, 13, 14], [-1, -1, 20, 21], [-1, -1, -1, -1], [5, 6, 7, 8], [-1, -1, -1, 9]] * 3)
    expected = create_watermarker(WatermarkConfig(key=KEY, context_width=CONTEXT_WIDTH)).sample(
        _processed(logits, params), contexts).token_ids
    assert sampled.tolist() == expected.tolist()
    assert sampled.tolist() != _processed(logits, params).argmax(dim=-1).tolist()


def test_mixed_batch_keeps_other_rows_ordinary():
    sampler = _watermark_sampler()
    greedy = SamplingParams(temperature=0, watermarking=True)
    plain = SamplingParams(temperature=0.7, watermarking=False)
    marked = _watermarked()
    repeated = _watermarked()
    params = [marked, greedy, plain, repeated]
    requests = [
        _Request(marked, [1], [30, 31, 32, 33, 34]),
        _Request(greedy, [1], [30, 31, 32, 33, 34]),
        _Request(plain, [1], [30, 31, 32, 33, 34]),
        _Request(repeated, [1], [7, 8, 9, 3, 7, 8, 9, 3]),
    ]
    logits = _logits(len(params))
    generators = {2: torch.Generator().manual_seed(5), 3: torch.Generator().manual_seed(6)}
    metadata = _metadata(params, generators)
    metadata.watermark = sampler.make_inputs(requests, "cpu")

    sampled, _ = sampler.sample(logits.clone(), metadata)

    processed = _processed(logits, params)
    watermarked = create_watermarker(WatermarkConfig(key=KEY, context_width=CONTEXT_WIDTH)).sample(
        processed, torch.tensor([[31, 32, 33, 34]] * 4)).token_ids
    probs = processed.softmax(dim=-1)
    ordinary = random_sample(probs.clone(), {2: torch.Generator().manual_seed(5), 3: torch.Generator().manual_seed(6)})
    assert sampled[0] == watermarked[0]
    assert sampled[1] == logits[1].argmax()
    assert sampled[2] == ordinary[2]
    assert sampled[3] == ordinary[3]


def test_batch_without_watermarked_rows_is_identical_to_sampler():
    sampler = _watermark_sampler()
    params = [SamplingParams(temperature=0.7, top_p=0.9), SamplingParams(temperature=0)]
    logits = _logits(len(params))

    def run(sampler_cls_instance):
        generators = {0: torch.Generator().manual_seed(3)}
        return sampler_cls_instance.sample(logits.clone(), _metadata(params, generators))

    sampled, logprobs = run(sampler)
    reference, reference_logprobs = run(Sampler())
    assert sampled.tolist() == reference.tolist()
    assert logprobs is None and reference_logprobs is None
