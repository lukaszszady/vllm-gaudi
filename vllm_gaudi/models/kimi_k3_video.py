# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Video input for Kimi-K3 on HPU.

The vLLM Kimi-K3 model is image-only. This module adds the ``video`` modality to it, for Kimi-K3 only: nothing is
registered on Kimi-K2.5/K2.6 or any other class and the shared helpers of ``vllm.models.kimi_k3`` are not modified.

How a ``video_url`` request flows through the patched model:

1. The chat renderer decodes the video with the vLLM video loader (frames + metadata) and inserts
   ``<|kimi_video_placeholder|>`` into the prompt (``get_placeholder_str``). ``_K3_MEDIA_IO_DEFAULTS`` gets the loader
   defaults (fps / frame cap) of ``kimi_k3_video_utils``.
2. ``KimiK3VideoProcessor`` samples the frames at ``min(sample_fps, source fps)``, cuts them into chunks of
   ``temporal_merge_kernel_size`` (4) frames and runs the checkpoint image processor on ``video_chunk`` media, which
   gives one ``[t, h, w]`` grid per chunk. The placeholder is replaced by one timestamped media block per chunk.
3. ``embed_multimodal`` runs the shared vision tower on the chunks (``tpool_patch_merger`` mean-pools each chunk over
   time) and concatenates the chunk embeddings of a video into ONE tensor per video, the contract of the HPU runner.

Video tensors use their own keyword names (``video_pixel_values`` / ``video_grid_thws`` / ``video_num_chunks``) so the
model can tell a video group from an image group. The vision tower is not eager on HPU: in torch.compile mode regional
compilation compiles every ``nn.ModuleList`` of the model, including ``vision_tower.encoder.blocks``, and in lazy mode
the tower runs in an HPU graph. A video chunk (T=4, 36x62 patches for 864x480) is a new input shape, so it creates new
graphs. The multimodal warmup precompiles them from the video shape of ``--limit-mm-per-prompt`` (see
``docs/features/kimi_k3_video.md``); a video shape that was not warmed up compiles on the first request.

The K3 classes cannot be imported while the plugin loads, so everything that subclasses them is built by
``install_kimi_k3_video`` when the K3 patches are applied.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from vllm_gaudi.models.kimi_k3_video_utils import (
    DEFAULT_MAX_SAMPLE_FPS,
    DEFAULT_TEMPORAL_CHUNK_SIZE,
    DUMMY_VIDEO_FPS,
    DUMMY_VIDEO_HEIGHT,
    DUMMY_VIDEO_NUM_FRAMES,
    DUMMY_VIDEO_WIDTH,
    K3_VIDEO_LOADER_FPS,
    K3_VIDEO_LOADER_MAX_FRAMES,
    K3_VIDEO_PLACEHOLDER,
    build_video_prompt,
    plan_video_chunks,
)

# Names of the video keyword arguments of the model (see the module docstring).
VIDEO_PIXEL_VALUES = "video_pixel_values"
VIDEO_GRID_THWS = "video_grid_thws"
VIDEO_NUM_CHUNKS = "video_num_chunks"

# Marker set on the patched model class; makes the installation idempotent.
_INSTALLED_FLAG = "_hpu_kimi_k3_video"

# Must stay in the "vllm.*" tree: other loggers are silent in vllm-gaudi.
logger = logging.getLogger("vllm.gaudi.kimi_k3_video")


def _split_video_item(item: Any) -> tuple[Any, Mapping[str, Any]]:
    """``(frames, metadata)`` of a parsed video item; the K3 data parser always asks the loader for metadata."""
    if isinstance(item, tuple) and len(item) == 2:
        return item[0], item[1] or {}
    return item, {}


def video_chunk_plan(frames: Any, metadata: Mapping[str, Any], media_proc_cfg: Mapping[str, Any]):
    """Chunks and chunk start times of a decoded video (``frames``: ``[N, H, W, 3]``) from its loader metadata."""
    num_loaded = len(frames)
    source_fps = metadata.get("fps")
    if not source_fps:
        logger.warning("Video loader gave no fps; assuming %s fps for the frame sampling and chunk timestamps",
                       DUMMY_VIDEO_FPS)
        source_fps = DUMMY_VIDEO_FPS
    source_fps = float(source_fps)
    return plan_video_chunks(
        num_loaded,
        int(metadata.get("total_num_frames") or num_loaded),
        source_fps,
        metadata.get("frames_indices"),
        float(media_proc_cfg.get("sample_fps", DEFAULT_MAX_SAMPLE_FPS)),
        int(media_proc_cfg.get("temporal_merge_kernel_size", DEFAULT_TEMPORAL_CHUNK_SIZE)),
    )


def _make_classes():
    """Build the K3 video processing classes (needs the K3 classes of vLLM)."""
    import numpy as np
    import torch
    from PIL import Image
    from transformers import BatchFeature
    from vllm.models.kimi_k3.common.mm_preprocess import (
        KimiK3DummyInputsBuilder,
        KimiK3MultiModalProcessor,
        KimiK3ProcessingInfo,
    )
    from vllm.multimodal.inputs import MultiModalFieldConfig
    from vllm.multimodal.parse import MultiModalDataParser, VideoProcessorItems
    from vllm.multimodal.processing import PromptReplacement, PromptUpdateDetails, cached_encode

    class HpuKimiK3VideoProcessingInfo(KimiK3ProcessingInfo):

        def get_supported_mm_limits(self) -> Mapping[str, int | None]:
            return {"image": None, "video": None}

        def get_data_parser(self) -> MultiModalDataParser:
            # The chunk plan needs the source fps / frame positions of the loader metadata.
            return MultiModalDataParser(
                expected_hidden_size=self._get_expected_hidden_size(),
                allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
                video_needs_metadata=True,
            )

    class HpuKimiK3VideoDummyInputsBuilder(KimiK3DummyInputsBuilder):

        def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
            return (super().get_dummy_text(mm_counts) + K3_VIDEO_PLACEHOLDER * mm_counts.get("video", 0))

        def get_dummy_mm_data(self, seq_len, mm_counts, mm_options=None):
            data = dict(super().get_dummy_mm_data(seq_len, mm_counts, mm_options))
            num_videos = mm_counts.get("video", 0)
            if not num_videos:
                return data
            overrides = mm_options.get("video") if mm_options else None
            width = getattr(overrides, "width", None) or DUMMY_VIDEO_WIDTH
            height = getattr(overrides, "height", None) or DUMMY_VIDEO_HEIGHT
            num_frames = getattr(overrides, "num_frames", None) or DUMMY_VIDEO_NUM_FRAMES
            video = np.full((num_frames, height, width, 3), 255, dtype=np.uint8)
            metadata = {
                "total_num_frames": num_frames,
                "fps": DUMMY_VIDEO_FPS,
                "duration": num_frames / DUMMY_VIDEO_FPS,
                "video_backend": "dummy",
                "frames_indices": list(range(num_frames)),
                "do_sample_frames": False,
            }
            data["video"] = [(video, dict(metadata)) for _ in range(num_videos)]
            return data

    class KimiK3VideoProcessor(KimiK3MultiModalProcessor):

        def _media_proc_cfg(self) -> Mapping[str, Any]:
            return self.info.image_processor.media_proc_cfg

        def _call_hf_processor(self, hf_data, hf_kwargs):
            hf_data = dict(hf_data)
            videos = hf_data.pop("videos", None)
            outputs = {}
            if hf_data:  # images (and text): the unchanged image-only path
                outputs.update(super()._call_hf_processor(hf_data, hf_kwargs))
            if videos:
                outputs.update(self._process_videos(videos))
            return BatchFeature(data=outputs)

        def _process_videos(self, videos: Sequence[Any]) -> dict[str, torch.Tensor]:
            cfg = self._media_proc_cfg()
            medias: list[dict[str, Any]] = []
            num_chunks: list[int] = []
            for item in videos:
                frames, metadata = _split_video_item(item)
                chunks, _ = video_chunk_plan(frames, metadata, cfg)
                num_chunks.append(len(chunks))
                for chunk in chunks:
                    # The checkpoint image processor takes PIL frames (it rejects ndarrays).
                    medias.append({
                        "type": "video_chunk",
                        "video_chunk": [Image.fromarray(np.asarray(frames[i])) for i in chunk],
                    })
            processed = self.info.image_processor.preprocess(medias,
                                                             return_tensors="pt",
                                                             do_rescale=True,
                                                             do_normalize=True)
            # The checkpoint image processor returns float32 patches. Images get the model dtype from
            # InputProcessingContext.call_hf_processor (floating tensors are cast to model_config.dtype), which this
            # direct call bypasses; without the cast the bf16 vision tower fails on a float32 x bf16 matmul.
            pixel_values = processed["pixel_values"]
            if pixel_values.is_floating_point():
                pixel_values = pixel_values.to(dtype=self.info.ctx.model_config.dtype)
            return {
                VIDEO_PIXEL_VALUES: pixel_values,
                VIDEO_GRID_THWS: processed["grid_thws"],
                VIDEO_NUM_CHUNKS: torch.tensor(num_chunks, dtype=torch.long),
            }

        def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
            fields = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
            if VIDEO_GRID_THWS not in hf_inputs:
                return fields
            chunks_per_video = hf_inputs[VIDEO_NUM_CHUNKS]
            patches_per_chunk = hf_inputs[VIDEO_GRID_THWS].prod(-1)
            patches_per_video = torch.stack(
                [s.sum() for s in torch.split(patches_per_chunk, chunks_per_video.tolist())])
            fields[VIDEO_PIXEL_VALUES] = MultiModalFieldConfig.flat_from_sizes("video", patches_per_video)
            fields[VIDEO_GRID_THWS] = MultiModalFieldConfig.flat_from_sizes("video", chunks_per_video, keep_on_cpu=True)
            fields[VIDEO_NUM_CHUNKS] = MultiModalFieldConfig.batched("video", keep_on_cpu=True)
            return fields

        def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
            updates = list(super()._get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs))
            if "video" not in mm_items:
                return updates

            tokenizer = self.info.get_tokenizer()
            media_token = self.info.media_token
            media_token_id = self.info.media_token_id
            cfg = self._media_proc_cfg()
            videos = mm_items.get_items("video", VideoProcessorItems)

            def get_replacement(item_idx: int) -> PromptUpdateDetails:
                frames, metadata = _split_video_item(videos.get(item_idx))
                frames = np.asarray(frames)
                chunks, starts = video_chunk_plan(frames, metadata, cfg)
                # Tokens of a chunk do not depend on its pixels, only on the frame size: count them with blank frames
                # of the real size through the same calculator the image path uses.
                blank = Image.new("RGB", (frames.shape[2], frames.shape[1]))
                tokens = [
                    self.info.media_tokens_calculator({
                        "type": "video_chunk",
                        "video_chunk": [blank] * len(chunk)
                    }) for chunk in chunks
                ]
                return PromptUpdateDetails.select_token_id(
                    cached_encode(tokenizer, build_video_prompt(starts, tokens, media_token), add_special_tokens=False),
                    media_token_id,
                )

            updates.append(
                PromptReplacement(
                    modality="video",
                    target=cached_encode(tokenizer, K3_VIDEO_PLACEHOLDER, add_special_tokens=False),
                    replacement=get_replacement,
                ))
            return updates

    return (HpuKimiK3VideoProcessingInfo, HpuKimiK3VideoDummyInputsBuilder, KimiK3VideoProcessor)


def _patch_model_class(model_cls, amd_model) -> None:
    """Video-aware ``get_placeholder_str``, ``_mark_tower_model`` and ``embed_multimodal`` on the K3 model class."""
    import torch

    orig_placeholder = model_cls.get_placeholder_str.__func__
    orig_mark_tower = model_cls._mark_tower_model
    orig_embed = model_cls.embed_multimodal

    def get_placeholder_str(cls, modality: str, i: int):
        if modality == "video":
            return K3_VIDEO_PLACEHOLDER
        return orig_placeholder(cls, modality, i)

    def _mark_tower_model(self, vllm_config, modalities, *args, **kwargs):
        # The tower serves both modalities; marked as image-only it would be skipped (left as a stub layer) when
        # only video is enabled by --limit-mm-per-prompt.
        if modalities == "image" or modalities == {"image"}:
            modalities = {"image", "video"}
        return orig_mark_tower(self, vllm_config, modalities, *args, **kwargs)

    def embed_multimodal(self, **kwargs: object):
        pixel_values = kwargs.get(VIDEO_PIXEL_VALUES)
        if pixel_values is None:
            return orig_embed(self, **kwargs)
        if kwargs.get("pixel_values") is not None:
            raise ValueError("Kimi-K3: image and video items must be encoded in separate groups")
        grid_thws = kwargs[VIDEO_GRID_THWS]
        num_chunks = kwargs[VIDEO_NUM_CHUNKS]
        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values, dim=0)
        if isinstance(grid_thws, list):
            grid_thws = torch.cat(list(grid_thws), dim=0)
        if isinstance(num_chunks, list):
            num_chunks = torch.stack(list(num_chunks))
        grid_thws = grid_thws.reshape(-1, grid_thws.shape[-1])
        # One tensor per chunk row from the tower; the runner needs one per video.
        per_chunk = amd_model.vision_tower_forward(
            self.vision_tower,
            pixel_values,
            grid_thws,
            mm_projector=self.mm_projector,
            use_data_parallel=self.use_data_parallel,
        )
        outputs = []
        start = 0
        for count in num_chunks.reshape(-1).tolist():
            outputs.append(torch.cat(list(per_chunk[start:start + count]), dim=0))
            start += count
        return outputs

    model_cls.get_placeholder_str = classmethod(get_placeholder_str)
    model_cls._mark_tower_model = _mark_tower_model
    model_cls.embed_multimodal = embed_multimodal


def install_kimi_k3_video() -> None:
    """Add the ``video`` modality to ``vllm.models.kimi_k3.amd.model.KimiK3ForConditionalGeneration`` (K3 only).

    Raises on any problem; the caller (``patches._patch_kimi_k3_video``) records the failure and logs a WARNING.
    """
    import vllm.models.kimi_k3.amd.model as amd_model
    import vllm.renderers.kimi_k3 as k3_renderer
    from vllm.multimodal import MULTIMODAL_REGISTRY

    model_cls = amd_model.KimiK3ForConditionalGeneration
    if getattr(model_cls, _INSTALLED_FLAG, False):
        return
    info_cls, dummy_cls, processor_cls = _make_classes()

    _patch_model_class(model_cls, amd_model)
    MULTIMODAL_REGISTRY.register_processor(processor_cls, info=info_cls, dummy_inputs=dummy_cls)(model_cls)
    # In place: the renderer reads the dict at request time. Server-level and request-level media_io_kwargs still win.
    k3_renderer._K3_MEDIA_IO_DEFAULTS.setdefault("video", {
        "num_frames": K3_VIDEO_LOADER_MAX_FRAMES,
        "fps": K3_VIDEO_LOADER_FPS,
    })
    setattr(model_cls, _INSTALLED_FLAG, True)
    logger.info("Kimi-K3 video input enabled (loader fps=%s, max frames=%s)", K3_VIDEO_LOADER_FPS,
                K3_VIDEO_LOADER_MAX_FRAMES)
