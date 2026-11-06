# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wall-time split of a prefill step: multimodal encoder vs language-model forward (``HPU_KIMI_PREFILL_TIMING``).

Off by default. With ``HPU_KIMI_PREFILL_TIMING=N`` (N >= 1) the HPU runner synchronizes the device around the
multimodal encoder call and around the LM forward of every serving prefill and logs every N-th one::

    Prefill timing: rank=0 prefill=12 reqs=1 tokens=9234 padded_tokens=10240 mm_items=1 video_chunks=5 \
encoder_ms=812.3 mm_inputs_ms=851.0 lm_forward_ms=3021.7 total_ms=3872.7 | mean of 12: encoder_ms=810.9 \
mm_inputs_ms=849.6 lm_forward_ms=3018.2 total_ms=3867.8

``encoder_ms`` is the ``embed_multimodal`` calls of the step (0 when nothing was encoded), ``mm_inputs_ms`` the whole
``_get_model_mm_inputs`` (encoder, embedding gather, ``embed_input_ids``) and ``lm_forward_ms`` the
``_execute_model_generic`` call of the prefill. Warmup steps are not timed. Only the driver worker logs, unless
``HPU_KIMI_PREFILL_TIMING_ALL_RANKS=1`` (set ``RAY_DEDUP_LOGS=0`` too, Ray folds lines that differ only in numbers).

The synchronizations remove the host/device overlap of the step, so a timed run is slower than an untimed one: use
it for the split, not for throughput.

No habana or vllm imports here, so the helper can be tested on a CPU host; the runner passes the device sync.
"""

import logging
import os
import time
from collections.abc import Callable, Mapping
from typing import Any

ENV_TIMING = "HPU_KIMI_PREFILL_TIMING"
ENV_ALL_RANKS = "HPU_KIMI_PREFILL_TIMING_ALL_RANKS"

# Keyword of the per-video chunk count in the Kimi-K3 video kwargs (kimi_k3_video.VIDEO_NUM_CHUNKS).
_VIDEO_NUM_CHUNKS = "video_num_chunks"

_FIELDS = ("encoder_ms", "mm_inputs_ms", "lm_forward_ms", "total_ms")

# Must stay in the "vllm.*" tree: other loggers are silent in vllm-gaudi.
logger = logging.getLogger("vllm.gaudi.prefill_timing")


def parse_timing_env(value: str | None) -> int:
    """Logging period from ``HPU_KIMI_PREFILL_TIMING``: 0 = off, N = every N-th prefill."""
    if value is None or value.strip().lower() in ("", "0", "false", "off", "no"):
        return 0
    if value.strip().lower() in ("true", "on", "yes"):
        return 1
    try:
        period = int(value)
    except ValueError:
        logger.warning("%s=%r is not an integer; prefill timing stays off", ENV_TIMING, value)
        return 0
    return max(period, 0)


def count_video_chunks(mm_kwargs_group: Mapping[str, Any]) -> int:
    """Number of Kimi-K3 video chunks in one batched multimodal kwargs group (0 for other inputs)."""
    num_chunks = mm_kwargs_group.get(_VIDEO_NUM_CHUNKS)
    if num_chunks is None:
        return 0
    if isinstance(num_chunks, (list, tuple)):
        return sum(int(n.sum()) if hasattr(n, "sum") else int(n) for n in num_chunks)
    return int(num_chunks.sum())


class PrefillTimer:
    """Per-runner state of ``HPU_KIMI_PREFILL_TIMING``; every method is a no-op when timing is off."""

    def __init__(self,
                 period: int,
                 rank: int,
                 sync: Callable[[], None],
                 clock: Callable[[], float] = time.perf_counter):
        self.period = period
        self.rank = rank
        self._sync = sync
        self._clock = clock
        self.num_prefills = 0
        self._sums = dict.fromkeys(_FIELDS, 0.0)
        self._encoder_s = 0.0
        self._mm_items = 0
        self._video_chunks = 0

    @classmethod
    def from_env(cls, *, is_driver_worker: bool, get_rank: Callable[[], int], sync: Callable[[],
                                                                                             None]) -> "PrefillTimer":
        """``get_rank`` is called only when timing is on (no TP group lookup on the default path)."""
        period = parse_timing_env(os.environ.get(ENV_TIMING))
        if period and not is_driver_worker and parse_timing_env(os.environ.get(ENV_ALL_RANKS)) == 0:
            period = 0
        rank = get_rank() if period else -1
        if period:
            logger.info("Prefill timing on (%s=%d, rank %d): device syncs around the encoder and the LM forward",
                        ENV_TIMING, period, rank)
        return cls(period, rank, sync)

    @property
    def enabled(self) -> bool:
        return self.period > 0

    def now(self) -> float:
        """Device-synchronized wall clock."""
        self._sync()
        return self._clock()

    def begin_step(self) -> None:
        self._encoder_s = 0.0
        self._mm_items = 0
        self._video_chunks = 0

    def add_encoder(self, seconds: float, mm_items: int, video_chunks: int) -> None:
        self._encoder_s += seconds
        self._mm_items += mm_items
        self._video_chunks += video_chunks

    def finish_prefill(self, *, num_reqs: int, tokens: int, padded_tokens: int, mm_inputs_s: float,
                       lm_forward_s: float) -> str | None:
        """Accumulate one timed prefill; log (and return) the line on every ``period``-th one."""
        self.num_prefills += 1
        values = {
            "encoder_ms": self._encoder_s * 1e3,
            "mm_inputs_ms": mm_inputs_s * 1e3,
            "lm_forward_ms": lm_forward_s * 1e3,
            "total_ms": (mm_inputs_s + lm_forward_s) * 1e3,
        }
        for key, value in values.items():
            self._sums[key] += value
        if self.num_prefills % self.period:
            return None
        current = " ".join(f"{key}={values[key]:.1f}" for key in _FIELDS)
        means = " ".join(f"{key}={self._sums[key] / self.num_prefills:.1f}" for key in _FIELDS)
        line = (f"Prefill timing: rank={self.rank} prefill={self.num_prefills} reqs={num_reqs} tokens={tokens} "
                f"padded_tokens={padded_tokens} mm_items={self._mm_items} video_chunks={self._video_chunks} "
                f"{current} | mean of {self.num_prefills}: {means}")
        logger.info(line)
        return line
