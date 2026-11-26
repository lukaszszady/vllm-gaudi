# SPDX-License-Identifier: Apache-2.0
"""Sampler for HPU that applies the engine's text watermark to the requests that ask for it."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import torch
from vllm.config.model import LogprobsMode
from vllm.logger import init_logger
from vllm.sampling_params import SamplingType
from vllm.v1.watermarking.watermarker import Watermarker

from vllm_gaudi.v1._vendored.sample.metadata import SamplingMetadata, WatermarkInputs
from vllm_gaudi.v1._vendored.sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample
from vllm_gaudi.v1._vendored.sample.sampler import _SAMPLING_EPS, Sampler

if TYPE_CHECKING:
    from vllm_gaudi.v1.worker.hpu_input_batch import CachedRequestState

logger = init_logger(__name__)


class _RandomSampler:
    """Ordinary random sampling, used by the watermarker for the rows it skips."""

    def __init__(self, generators: dict[int, torch.Generator], use_fp64_gumbel: bool) -> None:
        self.generators = generators
        self.use_fp64_gumbel = use_fp64_gumbel

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return random_sample(probs, self.generators, self.use_fp64_gumbel)


def _is_repeated_context(
    history: Sequence[int],
    context: Sequence[int],
    max_history: int | None,
    skip_partial_context: bool,
    num_output_tokens: int,
) -> bool:
    width = len(context)
    if skip_partial_context and num_output_tokens < width:
        return True
    history_start = max(0, len(history) - max(0, max_history)) if max_history is not None else 0
    padded = [-1] * width + list(history)
    last = context[-1]
    context = list(context)
    for position in range(history_start, len(history)):
        if padded[position + width - 1] == last and padded[position:position + width] == context:
            return True
    return False


class HPUWatermarkSampler(Sampler):
    """A Sampler whose random rows of watermarked requests are chosen by the watermarker.

    The watermark inputs of a batch are built on the host by ``make_inputs`` and ride on
    ``SamplingMetadata.watermark``; a batch without them takes the unchanged Sampler path.
    """

    def __init__(
        self,
        watermarker: Watermarker,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        use_fp64_gumbel: bool = False,
        deduplicate_contexts: Literal["none", "single_turn", "all"] = "single_turn",
        deduplicate_contexts_max_history: int | None = 8192,
    ) -> None:
        super().__init__(logprobs_mode, use_fp64_gumbel)
        self.watermarker = watermarker
        self.deduplicate_contexts = deduplicate_contexts
        self.deduplicate_contexts_max_history = deduplicate_contexts_max_history

    def make_inputs(self, requests: Sequence["CachedRequestState"],
                    device: torch.device | str) -> WatermarkInputs | None:
        """Watermark inputs for one sampler row per entry of ``requests``; None if no row is watermarked."""
        width = self.watermarker.context_width
        contexts: list[list[int]] = []
        active: list[bool] = []
        any_watermarked = False
        for request in requests:
            params = request.sampling_params
            watermarked = params is not None and bool(params.watermarking)
            if watermarked and params.sampling_type == SamplingType.GREEDY:
                logger.warning_once("Watermarking is enabled, but greedy decoding (temperature=0) cannot be "
                                    "watermarked. This request will use ordinary greedy sampling.")
                watermarked = False
            any_watermarked |= watermarked
            outputs = request.output_token_ids
            context = [-1] * max(0, width - len(outputs)) + list(outputs[-width:])
            contexts.append(context)
            repeated = False
            if watermarked and self.deduplicate_contexts != "none":
                include_prompt = self.deduplicate_contexts == "all"
                history = list(request.prompt_token_ids) + list(outputs) if include_prompt else list(outputs)
                repeated = _is_repeated_context(history, context, self.deduplicate_contexts_max_history, include_prompt,
                                                len(outputs))
            active.append(watermarked and not repeated)
        if not any_watermarked:
            return None
        return WatermarkInputs(
            contexts=torch.tensor(contexts, dtype=torch.int64).to(device, non_blocking=True),
            active=torch.tensor(active, dtype=torch.bool).to(device, non_blocking=True),
        )

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        watermark = sampling_metadata.watermark
        if watermark is None:
            return super().sample(logits, sampling_metadata, logprobs_mode_override)

        assert sampling_metadata.temperature is not None
        greedy_sampled = None if sampling_metadata.all_random else self.greedy_sample(logits)

        logits = self.apply_temperature(logits, sampling_metadata.temperature, sampling_metadata.all_random)
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)
        logits = apply_top_k_top_p(logits, sampling_metadata.top_k, sampling_metadata.top_p)

        processed_logprobs = None
        topk_topp_mode = self.topk_topp_sampler.logprobs_mode
        if topk_topp_mode == "processed_logits":
            processed_logprobs = logits
        elif topk_topp_mode == "processed_logprobs":
            processed_logprobs = logits.log_softmax(dim=-1, dtype=torch.float32)

        sampled = self.watermarker.sample(
            logits,
            watermark.contexts,
            _RandomSampler(sampling_metadata.generators, self.use_fp64_gumbel),
            skip_mask=~watermark.active,
        ).token_ids
        if greedy_sampled is not None:
            sampled = torch.where(sampling_metadata.temperature < _SAMPLING_EPS, greedy_sampled, sampled)
        return sampled, processed_logprobs
