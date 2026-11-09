# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dummy processor inputs for the multimodal warmup of the HPU runner.

Kept free of HPU imports so that it can be unit tested on a CPU host.
"""


def build_dummy_processor_inputs(processor, seq_len, mm_counts, mm_options):
    """Build ProcessorInputs for dummy items of the given modalities and options.

    Newer vLLM builds them in ``BaseMultiModalProcessor.get_dummy_inputs``; older versions had
    ``get_dummy_processor_inputs`` on the dummy inputs builder, and the builders of newer models
    (e.g. Kimi-K3) do not have it at all.
    """
    get_dummy_inputs = getattr(processor, "get_dummy_inputs", None)
    if get_dummy_inputs is None:
        get_dummy_inputs = processor.dummy_inputs.get_dummy_processor_inputs
    return get_dummy_inputs(seq_len=seq_len, mm_counts=mm_counts, mm_options=mm_options)


def build_raw_image_processor_inputs(processor, modality, count, width, height):
    """Build ProcessorInputs from raw WxH images, bypassing the upstream per-dimension clamp in
    ``DummyInputsBuilder._get_dummy_images``.

    Mirrors ``BaseMultiModalProcessor.get_dummy_inputs`` (dummy_text + parse_mm_data + tokenization) but
    supplies our own solid image at the requested raw WxH. The upstream builder clamps width and height
    *independently* against the model's max-feature size, distorting the aspect ratio so the warmed grid
    stops matching real traffic. Feeding the raw image straight through processor.apply lets the model's own
    resize (smart_resize / navit_resize / ...) pick exactly the same grid it would for a real image of that
    resolution -- including scaling an oversized image down -- so warmup and serving grids are identical by
    construction.
    """
    from PIL import Image
    from vllm.multimodal.processing.inputs import ProcessorInputs

    images = [Image.new("RGB", (width, height), color=255)] * count

    # vision_chunk (Kimi-K2.5/K2.6) does not accept bare PIL images: its parser expects VisionChunkImage dicts
    # ({"type": "image", "image": PIL}). 'image' takes the raw PIL directly. Wrap accordingly so the raw WxH
    # still flows through the model's own resize (navit_resize) and the warmed grid matches real serving traffic.
    mm_items = ([{"type": "image", "image": img} for img in images] if modality == 'vision_chunk' else images)

    dummy_text = processor.dummy_inputs.get_dummy_text({modality: count})
    mm_data_items = processor.info.parse_mm_data({modality: mm_items}, validate=False)

    # Upstream vllm#53093 removed the text components (str prompt + tokenization_kwargs) from ProcessorInputs;
    # the prompt is now a pre-tokenized list[int] and tokenization happens in the caller.
    tokenizer = processor.info.ctx.tokenizer
    if tokenizer is None:
        dummy_prompt: list[int] = []
    else:
        dummy_prompt = tokenizer.encode(
            dummy_text,
            **processor.info.default_tok_params.get_encode_kwargs(),
        )

    return ProcessorInputs(prompt=dummy_prompt, mm_data_items=mm_data_items)
