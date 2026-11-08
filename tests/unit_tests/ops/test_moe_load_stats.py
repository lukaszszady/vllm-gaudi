# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the diagnostic MoE EP load statistics (vllm_gaudi/extension/moe_load_stats.py)."""

import json

import numpy as np
import torch
import torch._dynamo as dynamo
from torch._dynamo.testing import CompileCounter

from vllm_gaudi.extension import moe_load_stats as mls

EP, LOCAL = 4, 3
GLOBAL = EP * LOCAL


def _reference(routing, chunk_size):
    counts = np.bincount(routing.reshape(-1), minlength=GLOBAL).reshape(EP, LOCAL).astype(float)
    experts = (counts > 0).sum(1)
    chunks = np.ceil(counts / chunk_size).sum(1) if chunk_size else experts
    return np.concatenate(
        [counts.sum(1), experts, chunks,
         counts.max(1), [chunks.max(), chunks.max() - chunks.mean(), 1]])


def _bucket(tokens):
    return max(i for i, lo in enumerate(mls.TOKEN_BUCKETS) if tokens >= lo)


def test_accumulate_matches_numpy_per_bucket():
    rng = np.random.default_rng(0)
    buf = mls.new_buffer(EP, "cpu")
    expert_buf = mls.new_expert_buffer(GLOBAL, "cpu")
    expected = np.zeros((3, len(mls.TOKEN_BUCKETS), mls.num_columns(EP)))
    expected_experts = np.zeros((4, len(mls.TOKEN_BUCKETS), GLOBAL))
    for tokens, chunk_size in ((3, 2), (5, 2), (40, 4), (300, 0)):
        routing = rng.integers(0, GLOBAL, size=(tokens, 2))
        mls.accumulate(buf, torch.tensor(routing, dtype=torch.int32), GLOBAL, LOCAL, chunk_size, expert_buf)
        v, b = _reference(routing, chunk_size), _bucket(tokens)
        expected[0, b] += v
        expected[1, b] += v * v
        expected[2, b] = np.maximum(expected[2, b], v)
        counts = np.bincount(routing.reshape(-1), minlength=GLOBAL).astype(float)
        active = (counts > 0).astype(float)
        chunks = np.ceil(counts / chunk_size) if chunk_size else active
        expected_experts[:, b] += np.stack([counts, active, chunks, counts * counts])
    np.testing.assert_allclose(buf.numpy(), expected, rtol=1e-6)
    np.testing.assert_allclose(expert_buf.numpy(), expected_experts, rtol=1e-6)
    assert buf[0, _bucket(3), -1] == 2 and buf[0, _bucket(40), -1] == 1


def test_out_of_range_ids_are_ignored():
    buf = mls.new_buffer(EP, "cpu")
    mls.accumulate(buf, torch.tensor([[0, -1], [GLOBAL, 1]], dtype=torch.int32), GLOBAL, LOCAL, 1)
    rows = buf[0, 0, :EP]
    assert rows.tolist() == [2.0, 0.0, 0.0, 0.0]


class _Moe(torch.nn.Module):
    _load_stats = None
    _expert_stats = None
    global_num_experts = GLOBAL
    num_experts = LOCAL

    def forward(self, x, routing):
        if self._load_stats is not None:
            mls.accumulate(self._load_stats, routing, GLOBAL, LOCAL, 2, self._expert_stats)
        return x * 2


class _Layer(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.moe_op = _Moe()
        # the op is reachable under a second name, as in FusedMoE with routed_experts
        self.routed = torch.nn.Module()
        self.routed.moe_op = self.moe_op

    def forward(self, x, routing):
        return self.moe_op(x + 1, routing)


def test_compiled_layers_share_one_graph_and_keep_the_mutation():
    dynamo.reset()
    layers = [_Layer() for _ in range(3)]
    for layer in layers:
        layer.moe_op._load_stats = mls.new_buffer(EP, "cpu")
        layer.moe_op._expert_stats = mls.new_expert_buffer(GLOBAL, "cpu")
    counter = CompileCounter()
    compiled = [torch.compile(layer, backend=counter, fullgraph=True) for layer in layers]
    routing = torch.tensor([[0, 4], [5, 11]], dtype=torch.int32)
    for _ in range(2):
        for layer in compiled:
            layer(torch.randn(2, 8), routing)
    assert counter.frame_count == 1
    for layer in layers:
        assert layer.moe_op._load_stats[0, 0, -1] == 2
        assert layer.moe_op._load_stats[0, 0, :EP].tolist() == [2.0, 4.0, 0.0, 2.0]
        assert layer.moe_op._expert_stats[0, 0, [0, 4, 5, 11]].tolist() == [2.0, 2.0, 2.0, 2.0]
    dynamo.reset()


def test_collector_steps_reset_and_report(tmp_path, caplog):
    model = torch.nn.ModuleList([_Layer(), _Layer()])
    path = tmp_path / "stats.log"
    expert_path = tmp_path / "experts.log"
    stats = mls.MoELoadStats(interval=2, path=str(path), rank=0, expert_path=str(expert_path))
    assert stats.attach(model, "cpu", op_types=(_Moe, )) == 2
    assert [name for name, _ in stats.ops] == ["0", "1"]
    assert (stats.ep_size, stats.local_num_experts) == (EP, LOCAL)
    routing = torch.tensor([[0, 1], [3, 9]], dtype=torch.int32)

    model[0](torch.ones(2, 4), routing)
    stats.reset()
    assert stats.total.sum() == 0 and model[0].moe_op._load_stats.sum() == 0
    assert stats.expert_total.sum() == 0 and model[0].moe_op._expert_stats.sum() == 0

    with caplog.at_level("INFO", logger="vllm"):
        for _ in range(4):
            for layer in model:
                layer(torch.ones(2, 4), routing)
            stats.on_step()
    assert stats.copies == 2
    assert all(op._load_stats.sum() == 0 and op._expert_stats.sum() == 0 for _, op in stats.ops)
    assert stats.total[0, :, 0, -1].tolist() == [4.0, 4.0]
    messages = [r.getMessage() for r in caplog.records if "MoE load stats copy" in r.getMessage()]
    assert messages[0].startswith("MoE load stats copy one (steps 2)")
    assert messages[1].startswith("MoE load stats copy two (steps 4)")
    # chunk size 2: rank 0 gets rows 2 on experts 0 and 1 (2 chunks), ranks 1 and 3 one chunk, rank 2 none
    assert "chunks/rank mean 1.0 (ranks 0.0..2.0) rank-max 2.0 max-minus-mean 1.00" in messages[1]
    text = path.read_text()
    assert "model steps 4, copies 2, layers 2" in text
    assert "## tokens 1-16: forwards per layer 4" in text
    assert "0 n 4 spread 1.00/0.00/1 rank-max 2.0 chunks [2.0 1.0 0.0 1.0] max_rows 1" in text
    header, *records = [json.loads(line) for line in expert_path.read_text().splitlines()]
    assert (header["steps"], header["copies"], header["layers"]) == (4, 2, ["0", "1"])
    assert header["features"] == ["rows", "active", "chunks", "rows_sq"]
    assert [(r["layer"], r["bucket"], r["forwards"]) for r in records] == [(0, "1-16", 4), (1, "1-16", 4)]
    expected_rows = [4 if e in (0, 1, 3, 9) else 0 for e in range(GLOBAL)]
    assert records[0]["rows"] == expected_rows
    assert records[0]["active"] == expected_rows
    assert records[0]["chunks"] == expected_rows
    assert records[0]["rows_sq"] == expected_rows


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_HPU_MOE_LOAD_STATS", raising=False)
    assert mls.MoELoadStats.from_env() is None
    monkeypatch.setenv("VLLM_HPU_MOE_LOAD_STATS", "1")
    monkeypatch.setenv("VLLM_HPU_MOE_LOAD_STATS_INTERVAL", "7")
    stats = mls.MoELoadStats.from_env()
    assert stats.interval == 7 and stats.path is None and stats.expert_path is None
    monkeypatch.setenv("VLLM_HPU_MOE_LOAD_STATS_FILE", "/root/results/moe_load_stats.log")
    assert mls.MoELoadStats.from_env().expert_path == "/root/results/moe_expert_stats.log"
    assert mls._in_words(120) == "one-two-zero"
