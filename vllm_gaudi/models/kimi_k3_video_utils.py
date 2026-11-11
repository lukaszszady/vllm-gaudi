# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure helpers of the Kimi-K3 video input path (no vLLM or torch imports).

Kept apart from ``kimi_k3_video.py`` so that frame sampling, chunking, timestamps and prompt construction can be
unit-tested on a CPU host without the Kimi-K3 classes of vLLM.

Kimi-K3 publishes no video prompt format. The helpers follow the Kimi-K2.5/K2.6 one, because K3 reuses the K2.5 vision
tower, its image processor (``KimiK25FusedVisionProcessor``) and the ``<|media_*|>`` special tokens: the video is
sampled at ``min(sample_fps, source fps)``, cut into chunks of ``temporal_merge_kernel_size`` frames (the tower
mean-pools each chunk over time into one picture worth of tokens) and every chunk gets its own timestamped block::

    00:00:02.000<|media_begin|>video<|media_content|><|media_pad|>*N<|media_end|>

Only the ``<|media_pad|>`` positions carry vision embeddings.
"""

from collections.abc import Sequence

# Modality placeholder the chat parser inserts for a ``video_url`` content part. Like the image one it is ordinary text
# (not a special token); the processor replaces it with the chunk blocks below.
K3_VIDEO_PLACEHOLDER = "<|kimi_video_placeholder|>"

# Kimi-K3 ``media_proc_cfg`` fall-backs, used only when the checkpoint config lacks the key.
DEFAULT_TEMPORAL_CHUNK_SIZE = 4  # temporal_merge_kernel_size
DEFAULT_MAX_SAMPLE_FPS = 8.0  # sample_fps

# Request-level video decoding defaults of the K3 chat renderer (--media-io-kwargs and per-request media_io_kwargs take
# precedence). The loader keeps min(num_frames, source frames, floor(duration * fps)) frames, so these two cap the
# decoded video at 8 frames per second and at 256 frames; the final sampling and chunking is done by
# ``plan_video_chunks``.
K3_VIDEO_LOADER_FPS = DEFAULT_MAX_SAMPLE_FPS
K3_VIDEO_LOADER_MAX_FRAMES = 256

# Shape of the dummy video used for profiling and MM warmup when ``--limit-mm-per-prompt`` gives no override: 20 frames
# of 864x480 at 2 fps are 5 chunks of 558 vision tokens.
DUMMY_VIDEO_WIDTH = 864
DUMMY_VIDEO_HEIGHT = 480
DUMMY_VIDEO_NUM_FRAMES = 20
DUMMY_VIDEO_FPS = 2.0

_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600
_MS_PER_SECOND = 1000


def sample_frame_indices(num_loaded: int, source_total: int, source_fps: float, max_sample_fps: float) -> list[int]:
    """Indices (into the ``num_loaded`` decoded frames) kept at ``min(max_sample_fps, source_fps)``.

    Same arithmetic as the K2.6 ``navit_resize_video``: ``max(round(total * sample_fps / fps), 1)`` frames, evenly
    spread (linspace, rounded) over the decoded frames and never more than were decoded.
    """
    if num_loaded <= 0:
        raise ValueError("The video has no frames")
    if source_fps <= 0:
        raise ValueError(f"Invalid video fps: {source_fps}")
    sample_fps = min(max_sample_fps, source_fps)
    total = max(source_total, num_loaded)
    sampled = min(max(round(total * sample_fps / source_fps), 1), num_loaded)
    if sampled == 1:
        return [0]
    step = (num_loaded - 1) / (sampled - 1)
    return [round(i * step) for i in range(sampled)]


def split_into_chunks(indices: Sequence[int], chunk_size: int) -> list[list[int]]:
    """Cut the sampled frame indices into chunks of ``chunk_size``; the last chunk may be shorter."""
    if chunk_size < 1:
        raise ValueError(f"Invalid chunk size: {chunk_size}")
    return [list(indices[i:i + chunk_size]) for i in range(0, len(indices), chunk_size)]


def format_timestamp(seconds: float) -> str:
    """``hh:mm:ss.fff`` (the K3 ``timestamp_mode``)."""
    total_ms = max(round(seconds * _MS_PER_SECOND), 0)
    hours, rest = divmod(total_ms, _SECONDS_PER_HOUR * _MS_PER_SECOND)
    minutes, rest = divmod(rest, _SECONDS_PER_MINUTE * _MS_PER_SECOND)
    secs, millis = divmod(rest, _MS_PER_SECOND)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def plan_video_chunks(
    num_loaded: int,
    source_total: int,
    source_fps: float,
    source_indices: Sequence[int] | None,
    max_sample_fps: float = DEFAULT_MAX_SAMPLE_FPS,
    chunk_size: int = DEFAULT_TEMPORAL_CHUNK_SIZE,
) -> tuple[list[list[int]], list[float]]:
    """Chunks (lists of decoded-frame indices) of a video and the start time in seconds of every chunk.

    ``source_indices`` are the positions in the source video of the decoded frames (the loader metadata
    ``frames_indices``); the start time of a chunk is the source position of its first frame divided by the source fps.
    Without them the decoded frames are taken as evenly spread over the source duration.
    """
    kept = sample_frame_indices(num_loaded, source_total, source_fps, max_sample_fps)
    chunks = split_into_chunks(kept, chunk_size)
    starts = []
    for chunk in chunks:
        first = chunk[0]
        if source_indices is not None and len(source_indices) == num_loaded:
            position = source_indices[first]
        else:
            position = first * max(source_total, num_loaded) / num_loaded
        starts.append(position / source_fps)
    return chunks, starts


def build_video_prompt(
    chunk_starts: Sequence[float],
    tokens_per_chunk: Sequence[int],
    media_token: str,
) -> str:
    """Prompt text replacing one video placeholder: one timestamped media block per chunk, joined without separator."""
    if len(chunk_starts) != len(tokens_per_chunk):
        raise ValueError("chunk_starts and tokens_per_chunk must have the same length")
    return "".join(f"{format_timestamp(start)}<|media_begin|>video<|media_content|>"
                   f"{media_token * tokens}<|media_end|>" for start, tokens in zip(chunk_starts, tokens_per_chunk))
