# SPDX-License-Identifier: Apache-2.0
"""Row selection by a PlaceholderRange.is_embed mask without boolean-mask indexing on the device.

Indexing a device tensor with a boolean mask (``t[mask]``, ``t[mask] = x``) has a data-dependent output shape, which
the Habana graph compiler cannot build (``synNodeCreateWithId failed for node: slice``). The mask lives on the host, so
the integer positions are computed there and the device only sees a static-shape ``index_select`` / ``index_copy_``.
"""
import torch


def embed_positions(is_embed: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """Positions of the True entries of a host mask, as an int64 index tensor on ``device``."""
    return torch.nonzero(is_embed.cpu(), as_tuple=True)[0].to(device)


def select_embed_rows(rows: torch.Tensor, is_embed: torch.Tensor) -> torch.Tensor:
    """Same result as ``rows[is_embed]`` for a 1D bool mask over dim 0, without a device boolean mask."""
    return rows.index_select(0, embed_positions(is_embed, rows.device))


def scatter_embed_rows(output: torch.Tensor, is_embed: torch.Tensor, fill_value: float) -> torch.Tensor:
    """Same result as ``t = full(...); t[is_embed] = output`` with ``t`` of shape [len(is_embed), hidden]."""
    placeholders = output.new_full((is_embed.shape[0], output.shape[-1]), fill_value=fill_value)
    return placeholders.index_copy_(0, embed_positions(is_embed, output.device), output)


def collect_step_encoder_items(req_ids, requests, scheduled_encoder_inputs, num_scheduled_tokens, encoder_cache):
    """Items to encode before prefilling ``req_ids``: ``[(mm_hash, mm_feature)]``, one entry per missing hash.

    The scheduler dedups identical ``mm_hash`` values within a step and schedules the encoder for only one of the
    requests. A prefill of another request of the step, that runs first, must still encode the item it gathers, so
    the payload is taken from any request of the step (``scheduled_encoder_inputs``). An item is needed when its span
    overlaps the tokens scheduled for the request (the condition of ``_gather_mm_embeddings``); spans fully covered
    by computed tokens, e.g. consumed by an earlier prefill chunk, are never encoded again.
    """
    payloads = {}
    for req_id, input_ids in scheduled_encoder_inputs.items():
        req_state = requests.get(req_id)
        if req_state is None:
            continue
        for input_id in input_ids:
            feature = req_state.mm_features[input_id]
            if feature.data is not None:
                payloads.setdefault(feature.identifier, feature)

    items = {}
    for req_id in req_ids:
        req_state = requests[req_id]
        computed = req_state.num_computed_tokens
        end = computed + num_scheduled_tokens[req_id]
        for feature in req_state.mm_features:
            mm_hash = feature.identifier
            if mm_hash in encoder_cache or mm_hash in items:
                continue
            pos_info = feature.mm_position
            if pos_info.offset >= end or pos_info.offset + pos_info.length <= computed:
                continue
            source = feature if feature.data is not None else payloads.get(mm_hash)
            if source is not None:
                items[mm_hash] = source
    return list(items.items())
