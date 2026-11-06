# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 encoder item selection when the scheduler dedups an identical video across requests of one step.

Pure python: runs on CPU without vllm or habana.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[3] / "vllm_gaudi" / "v1" / "worker" / "mm_embed_utils.py"
VIDEO_LEN = 100


@pytest.fixture(scope="module")
def collect():
    spec = importlib.util.spec_from_file_location("mm_embed_utils_under_test", PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.collect_step_encoder_items


def _request(mm_hash, computed=0, offset=10, data="pixels"):
    feature = SimpleNamespace(identifier=mm_hash,
                              data=data,
                              modality="video",
                              mm_position=SimpleNamespace(offset=offset, length=VIDEO_LEN))
    return SimpleNamespace(num_computed_tokens=computed, mm_features=[feature])


def _run_step(collect, prefill_order, requests, scheduled, num_tokens, cache):
    """Mimics _execute_mm_encoder + _gather_mm_embeddings per bs=1 prefill; returns (#encodes, gathered hashes)."""
    encodes = 0
    gathered = []
    for req_id in prefill_order:
        items = collect([req_id], requests, scheduled, num_tokens, cache)
        for mm_hash, _ in items:
            cache[mm_hash] = f"embeds-{mm_hash}"
            encodes += 1
        for feature in requests[req_id].mm_features:
            assert feature.identifier in cache, f"Encoder cache miss for {feature.identifier}"
            gathered.append((req_id, feature.identifier))
    return encodes, gathered


@pytest.mark.parametrize("prefill_order", [["a", "b"], ["b", "a"]])
def test_same_hash_encoded_once_whatever_the_prefill_order(collect, prefill_order):
    requests = {"a": _request("h1"), "b": _request("h1")}
    scheduled = {"a": [0]}
    num_tokens = {"a": 200, "b": 200}
    cache = {}
    encodes, gathered = _run_step(collect, prefill_order, requests, scheduled, num_tokens, cache)
    assert encodes == 1
    assert sorted(gathered) == [("a", "h1"), ("b", "h1")]


def test_single_call_dedups_by_hash(collect):
    requests = {"a": _request("h1"), "b": _request("h1")}
    items = collect(["a", "b"], requests, {"a": [0]}, {"a": 200, "b": 200}, {})
    assert [h for h, _ in items] == ["h1"]


def test_payload_taken_from_scheduled_request_when_own_data_is_missing(collect):
    requests = {"a": _request("h1", data="pixels-a"), "b": _request("h1", data=None)}
    items = collect(["b"], requests, {"a": [0]}, {"a": 200, "b": 200}, {})
    assert [(h, f.data) for h, f in items] == [("h1", "pixels-a")]


def test_cached_hash_is_not_encoded(collect):
    requests = {"a": _request("h1")}
    assert collect(["a"], requests, {"a": [0]}, {"a": 200}, {"h1": "embeds"}) == []


def test_span_consumed_by_earlier_chunk_is_not_encoded_again(collect):
    # The video (offset 10..110) is fully computed; the next chunk only prefills text after it.
    requests = {"a": _request("h1", computed=150)}
    assert collect(["a"], requests, {}, {"a": 50}, {}) == []


def test_span_after_scheduled_tokens_is_not_encoded_yet(collect):
    requests = {"a": _request("h1", computed=0, offset=500)}
    assert collect(["a"], requests, {}, {"a": 200}, {}) == []


def test_no_payload_anywhere_is_skipped(collect):
    requests = {"a": _request("h1", data=None)}
    assert collect(["a"], requests, {}, {"a": 200}, {}) == []


def test_distinct_hashes_are_all_encoded(collect):
    requests = {"a": _request("h1"), "b": _request("h2")}
    items = collect(["a", "b"], requests, {"a": [0], "b": [0]}, {"a": 200, "b": 200}, {})
    assert [h for h, _ in items] == ["h1", "h2"]
