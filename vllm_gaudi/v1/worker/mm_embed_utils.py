# SPDX-License-Identifier: Apache-2.0
"""Helpers for the mm encoder step of the HPU model runner."""


def collect_step_encoder_items(req_ids, requests, scheduled_encoder_inputs, num_scheduled_tokens, encoder_cache):
    """Items to encode before prefilling ``req_ids``: ``[(mm_hash, mm_feature)]``, one entry per missing hash.

    The scheduler dedups identical ``mm_hash`` values within a step and schedules the encoder for only one of the
    requests. A prefill of another request of the step, that runs first, must still encode the item it gathers, so
    the payload is taken from any request of the step (``scheduled_encoder_inputs``). An item is needed when its span
    overlaps the tokens scheduled for the request (the condition of ``_gather_mm_embeddings``); spans fully covered
    by computed tokens, e.g. consumed by an earlier prefill chunk, are never encoded again.
    """
    payloads: dict = {}
    for req_id, input_ids in scheduled_encoder_inputs.items():
        req_state = requests.get(req_id)
        if req_state is None:
            continue
        for input_id in input_ids:
            feature = req_state.mm_features[input_id]
            if feature.data is not None:
                payloads.setdefault(feature.identifier, feature)

    items: dict = {}
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
