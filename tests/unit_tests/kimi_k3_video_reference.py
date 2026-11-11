# SPDX-License-Identifier: Apache-2.0
"""Reference vision-token counts of the Kimi-K3 video path, for the unit tests only."""
import math
from collections.abc import Mapping


def navit_chunk_tokens(
    width: int,
    height: int,
    media_proc_cfg: Mapping[str, object],
) -> int:
    """Vision tokens of ONE video chunk, whatever its length (the tower pools the frames of a chunk over time).

    Re-implements the size arithmetic of ``navit_resize_video`` for a chunk (``nframes`` = chunk length, avg fps 1,
    unlimited sample fps) so the count can be checked without vLLM. The patch limit per frame is
    ``min(in_patch_limit_each_frame, in_patch_limit_video / sampled_frames)``; the second term never binds for 4-frame
    chunks with the K3 config, so it is not modelled. The processor itself counts with the checkpoint image processor;
    this is the reference for the unit tests.
    """
    patch_size = int(media_proc_cfg["patch_size"])  # type: ignore[call-overload]
    merge = int(media_proc_cfg["merge_kernel_size"])  # type: ignore[call-overload]
    per_frame_limit = media_proc_cfg.get("in_patch_limit_each_frame") or media_proc_cfg["in_patch_limit"]
    side_limit = int(media_proc_cfg["patch_limit_on_one_side"])  # type: ignore[call-overload]

    s1 = math.sqrt(per_frame_limit /
                   (max(1.0, width // patch_size) * max(1.0, height // patch_size)))  # type: ignore[operator]
    s2 = side_limit * patch_size / width
    s3 = side_limit * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w = min(max(1, int(width * scale)), side_limit * patch_size)
    new_h = min(max(1, int(height * scale)), side_limit * patch_size)

    factor = merge * patch_size
    pad_w = (factor - new_w % factor) % factor
    pad_h = (factor - new_h % factor) % factor
    return ((new_h + pad_h) // factor) * ((new_w + pad_w) // factor)


def video_vision_tokens(
    utils,
    width: int,
    height: int,
    num_frames: int,
    source_fps: float,
    media_proc_cfg: Mapping[str, object],
) -> int:
    """Total ``<|media_pad|>`` count of a ``num_frames`` video (what the scheduler reserves for the item)."""
    chunk_size = int(media_proc_cfg.get("temporal_merge_kernel_size",
                                        utils.DEFAULT_TEMPORAL_CHUNK_SIZE))  # type: ignore[call-overload]
    max_fps = float(media_proc_cfg.get("sample_fps", utils.DEFAULT_MAX_SAMPLE_FPS))  # type: ignore[arg-type]
    chunks, _ = utils.plan_video_chunks(num_frames, num_frames, source_fps, None, max_fps, chunk_size)
    return len(chunks) * navit_chunk_tokens(width, height, media_proc_cfg)
