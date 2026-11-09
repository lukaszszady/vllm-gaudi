# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the static expert permutation (VLLM_HPU_MOE_EXPERT_PERM_FILE).

EP ranks are simulated by one MXFP4 MoE op per rank whose HPU kernel is replaced by an unfused CPU MoE over the
local experts; the rank outputs are summed as the all-reduce does.
"""

import importlib.util
import json
import pathlib
import sys
import types
from unittest.mock import MagicMock

import pytest
import torch
import torch._dynamo as dynamo
import torch.nn.functional as F
from torch._dynamo.testing import CompileCounter

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_STUBBED_MODULES = (
    "habana_frameworks",
    "habana_frameworks.torch",
    "habana_frameworks.torch.core",
    "habana_frameworks.torch.utils",
    "habana_frameworks.torch.utils.experimental",
    "vllm",
    "vllm.model_executor",
    "vllm.model_executor.layers",
    "vllm.model_executor.layers.quantization",
    "vllm.model_executor.layers.quantization.utils",
    "vllm.model_executor.layers.quantization.base_config",
    "vllm.model_executor.layers.fused_moe",
    "vllm.model_executor.layers.fused_moe.activation",
    "vllm_gaudi.extension.runtime",
    "vllm_gaudi.extension.utils",
    "vllm_gaudi.extension.scales",
)

NUM_EXPERTS, EP_SIZE, HIDDEN, INTER, TOP_K = 8, 4, 6, 4, 3
LOCAL = NUM_EXPERTS // EP_SIZE


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


perm_mod = _load("moe_expert_perm_under_test", _ROOT / "vllm_gaudi" / "extension" / "moe_expert_perm.py")


def cpu_moe_fwd(hidden_states, expert_routing_table, router_weights, w12_list, w3_list, d_scale_w12, d_scale_w3,
                block_size, activation, experts_min, experts_max, chunk_size, total_experts):
    w12, w3 = torch.stack(w12_list), torch.stack(w3_list)
    out = torch.zeros_like(hidden_states)
    for k in range(expert_routing_table.shape[1]):
        local = expert_routing_table[:, k].long() - experts_min
        valid = (local >= 0) & (local <= experts_max - experts_min)
        idx = local.clamp(0, w12.shape[0] - 1)
        gate, up = torch.einsum("th,toh->to", hidden_states, w12[idx]).chunk(2, dim=-1)
        y = torch.einsum("ti,thi->th", F.silu(gate) * up, w3[idx])
        out = out + torch.where(valid[:, None], router_weights[:, k:k + 1] * y, torch.zeros_like(y))
    return out


@pytest.fixture
def ops(monkeypatch):
    for name in _STUBBED_MODULES:
        monkeypatch.setitem(sys.modules, name, MagicMock())
    sys.modules["vllm_gaudi.extension.runtime"].get_config = lambda: types.SimpleNamespace(moe_chunk=[],
                                                                                           moe_token_boundary=[])
    sys.modules["vllm.model_executor.layers.fused_moe.activation"].MoEActivation = type("MoEActivation", (), {})
    module = _load("mxfp4_ops_perm_under_test", _ROOT / "vllm_gaudi" / "extension" / "ops.py")
    monkeypatch.setattr(module, "_mxfp4_fused_fwd", cpu_moe_fwd)
    dynamo.reset()
    dynamo.utils.counters.clear()
    yield module
    dynamo.reset()


class _RoutedExperts(torch.nn.Module):
    """The attributes of vLLM's RoutedExperts that the permutation uses, with its linear expert map."""

    def __init__(self, ep_rank, layer_name="model.layers.3.block_sparse_moe.experts"):
        super().__init__()
        self.layer_name = layer_name
        self.moe_config = types.SimpleNamespace(ep_rank=ep_rank)
        self.local_num_experts = LOCAL
        self.global_num_experts = NUM_EXPERTS
        self.expert_map_manager = types.SimpleNamespace(routing_tables=None)
        expert_map = torch.full((NUM_EXPERTS, ), -1, dtype=torch.int32)
        expert_map[ep_rank * LOCAL:(ep_rank + 1) * LOCAL] = torch.arange(LOCAL, dtype=torch.int32)
        self.register_buffer("_expert_map", expert_map, persistent=False)

    def _map_global_expert_id_to_local_expert_id(self, expert_id):
        return int(self._expert_map[expert_id])

    def forward(self, x, topk_ids, topk_weights):
        # as _hpu_mxfp4_routed_apply (patches.py)
        expert_perm = getattr(self, "_hpu_expert_perm", None)
        if expert_perm is not None:
            topk_ids = perm_mod.remap(expert_perm, topk_ids)
        return self.moe_op(x, topk_ids, topk_weights, permuted_weights=True)


def _checkpoint(seed=0):
    g = torch.Generator().manual_seed(seed)
    w13 = torch.randn(NUM_EXPERTS, 2 * INTER, HIDDEN, generator=g, dtype=torch.float64)
    w2 = torch.randn(NUM_EXPERTS, HIDDEN, INTER, generator=g, dtype=torch.float64)
    return w13, w2


def _build_rank(ops, ep_rank, w13, w2, perm=None, layer_name="model.layers.3.block_sparse_moe.experts"):
    layer = _RoutedExperts(ep_rank, layer_name)
    if perm is not None:
        perm_mod.apply_to_layer(layer, perm)
    local_w13 = torch.zeros(LOCAL, 2 * INTER, HIDDEN, dtype=torch.float64)
    local_w2 = torch.zeros(LOCAL, HIDDEN, INTER, dtype=torch.float64)
    # RoutedExperts.weight_loader: global expert id -> local slot, -1 skips the expert
    for expert_id in range(NUM_EXPERTS):
        slot = layer._map_global_expert_id_to_local_expert_id(expert_id)
        if slot >= 0:
            local_w13[slot] = w13[expert_id]
            local_w2[slot] = w2[expert_id]
    lo = ep_rank * LOCAL
    layer.moe_op = ops.VllmMixtureOfExpertsOpMXFP4(NUM_EXPERTS, LOCAL, lo, lo + LOCAL - 1, in_graph_call=True)
    for slot in range(LOCAL):
        layer.moe_op.w13_list[slot].set_weight(local_w13[slot])
        layer.moe_op.w13_list[slot].set_scale(torch.ones(1))
        layer.moe_op.w2_list[slot].set_weight(local_w2[slot])
        layer.moe_op.w2_list[slot].set_scale(torch.ones(1))
    layer.moe_op._cache_weight_lists()
    # eager calls: no nested hpu_backend compile on CPU
    layer.moe_op._compiled_forward = cpu_moe_fwd
    return layer


def _ep_forward(ranks, x, topk_ids, topk_weights):
    return sum(rank(x, topk_ids, topk_weights) for rank in ranks)


def _routing(tokens=5, seed=1):
    g = torch.Generator().manual_seed(seed)
    topk_ids = torch.stack([torch.randperm(NUM_EXPERTS, generator=g)[:TOP_K] for _ in range(tokens)]).int()
    topk_weights = torch.rand(tokens, TOP_K, generator=g, dtype=torch.float64)
    x = torch.randn(tokens, HIDDEN, generator=g, dtype=torch.float64)
    return x, topk_ids, topk_weights


def _dense_reference(x, topk_ids, topk_weights, w13, w2):
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for k in range(TOP_K):
            e = int(topk_ids[t, k])
            gate, up = (w13[e] @ x[t]).chunk(2)
            out[t] += topk_weights[t, k] * (w2[e] @ (F.silu(gate) * up))
    return out


def test_identity_permutation_is_bit_exact(ops):
    w13, w2 = _checkpoint()
    x, topk_ids, topk_weights = _routing()
    baseline = _ep_forward([_build_rank(ops, r, w13, w2) for r in range(EP_SIZE)], x, topk_ids, topk_weights)
    identity = list(range(NUM_EXPERTS))
    permuted = _ep_forward([_build_rank(ops, r, w13, w2, identity) for r in range(EP_SIZE)], x, topk_ids, topk_weights)
    assert torch.equal(permuted, baseline)
    torch.testing.assert_close(baseline, _dense_reference(x, topk_ids, topk_weights, w13, w2))


def test_permutation_places_experts_and_matches_baseline(ops):
    w13, w2 = _checkpoint()
    x, topk_ids, topk_weights = _routing(tokens=7)
    perm = [5, 0, 7, 2, 6, 1, 3, 4]
    ranks = [_build_rank(ops, r, w13, w2, perm) for r in range(EP_SIZE)]
    for r, layer in enumerate(ranks):
        held = sorted(e for e in range(NUM_EXPERTS) if r * LOCAL <= perm[e] < (r + 1) * LOCAL)
        assert [int(s) >= 0 for s in layer._expert_map].count(True) == LOCAL
        assert sorted(torch.nonzero(layer._expert_map >= 0).flatten().tolist()) == held
        for e in held:
            assert torch.equal(layer.moe_op.w13_list[perm[e] - r * LOCAL].weight, w13[e])
    baseline = _ep_forward([_build_rank(ops, r, w13, w2) for r in range(EP_SIZE)], x, topk_ids, topk_weights)
    torch.testing.assert_close(_ep_forward(ranks, x, topk_ids, topk_weights), baseline, rtol=1e-12, atol=1e-12)


def test_remap_compiles_once_for_layers_with_different_permutations(ops):
    w13, w2 = _checkpoint()
    perms = [[5, 0, 7, 2, 6, 1, 3, 4], [1, 2, 3, 4, 5, 6, 7, 0], list(range(NUM_EXPERTS))]
    layers = [_build_rank(ops, 1, w13, w2, p, f"model.layers.{i + 1}.mlp.experts") for i, p in enumerate(perms)]
    counter = CompileCounter()
    compiled = [torch.compile(layer, backend=counter, fullgraph=True) for layer in layers]
    for tokens in (5, 5):
        x, topk_ids, topk_weights = _routing(tokens=tokens)
        for layer, fn in zip(layers, compiled):
            torch.testing.assert_close(fn(x, topk_ids, topk_weights), layer(x, topk_ids, topk_weights))
    assert not dynamo.utils.counters["graph_break"]
    assert counter.frame_count == 1


def test_remap_reads_each_layers_table_as_a_graph_input(ops):
    w13, w2 = _checkpoint()
    perms = [[5, 0, 7, 2, 6, 1, 3, 4], [1, 2, 3, 4, 5, 6, 7, 0]]
    layers = [_build_rank(ops, 1, w13, w2, p, f"model.layers.{i + 1}.mlp.experts") for i, p in enumerate(perms)]
    graph_inputs = []

    def backend(gm, example_inputs):
        graph_inputs.extend(example_inputs)
        return gm

    compiled = [torch.compile(layer, backend=backend, fullgraph=True) for layer in layers]
    x, topk_ids, topk_weights = _routing()
    for layer, fn in zip(layers, compiled):
        torch.testing.assert_close(fn(x, topk_ids, topk_weights), layer(x, topk_ids, topk_weights))
    assert any(t is layers[0]._hpu_expert_perm for t in graph_inputs)
    # an in-place update of the table must reach the compiled graph (not folded into a constant)
    layers[1]._hpu_expert_perm.copy_(torch.tensor(perms[0]))
    torch.testing.assert_close(compiled[1](x, topk_ids, topk_weights), layers[1](x, topk_ids, topk_weights))
    assert not dynamo.utils.counters["graph_break"]


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_remap(ids_dtype):
    perm = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
    ids = torch.tensor([[0, 3], [1, 2]], dtype=ids_dtype)
    out = perm_mod.remap(perm, ids)
    assert out.dtype == ids_dtype
    assert out.tolist() == [[2, 1], [0, 3]]


def _integer_gathers(gm):
    """(op, index dtype) of the integer-indexed gathers (index_select, gather, tensor-indexed getitem) in a graph."""
    gathers = []
    for node in gm.graph.nodes:
        if node.op not in ("call_function", "call_method"):
            continue
        name = getattr(node.target, "__name__", node.target)
        if name in ("index_select", "gather"):
            index = node.kwargs.get("index", node.args[2] if len(node.args) > 2 else None)
        elif name == "getitem":
            index = node.args[1]
        else:
            continue
        value = index.meta.get("example_value") if isinstance(index, torch.fx.Node) else None
        if isinstance(value, torch.Tensor) and not value.is_floating_point() and value.dtype != torch.bool:
            gathers.append((name, value.dtype))
    return gathers


def test_random_permutation_file_matches_baseline_in_compile(ops, monkeypatch, tmp_path):
    # index_select of the remap table returns wrong expert ids in HPU graphs: int64 tensor indexing only
    for name, value in (("NUM_EXPERTS", 32), ("LOCAL", 8), ("TOP_K", 4)):
        monkeypatch.setitem(globals(), name, value)
    g = torch.Generator().manual_seed(3)
    perm = torch.randperm(NUM_EXPERTS, generator=g).tolist()
    path = _file(tmp_path, {"num_experts": NUM_EXPERTS, "layers": {"3": perm}})
    perms = perm_mod.load(path, NUM_EXPERTS, frozenset({3}))
    w13, w2 = _checkpoint()
    baseline_ranks = [_build_rank(ops, r, w13, w2) for r in range(EP_SIZE)]
    ranks = [_build_rank(ops, r, w13, w2, perms[3]) for r in range(EP_SIZE)]
    gathers = []

    def backend(gm, example_inputs):
        gathers.extend(_integer_gathers(gm))
        return gm

    compiled = [torch.compile(rank, backend=backend, fullgraph=True) for rank in ranks]
    x, topk_ids, topk_weights = _routing(tokens=16, seed=4)
    baseline = _ep_forward(baseline_ranks, x, topk_ids, topk_weights)
    torch.testing.assert_close(baseline, _dense_reference(x, topk_ids, topk_weights, w13, w2))
    torch.testing.assert_close(_ep_forward(compiled, x, topk_ids, topk_weights), baseline, rtol=1e-12, atol=1e-12)
    assert gathers and set(gathers) == {("getitem", torch.int64)}


def test_apply_to_layer_keeps_shared_buffers():
    shared_mask = torch.ones(NUM_EXPERTS + 1, dtype=torch.int32)
    layer = _RoutedExperts(0)
    layer.register_buffer("expert_mask", shared_mask, persistent=False)
    perm_mod.apply_to_layer(layer, [7, 6, 5, 4, 3, 2, 1, 0])
    assert shared_mask.tolist() == [1] * (NUM_EXPERTS + 1)
    assert layer.expert_mask.tolist() == [0] * 6 + [1, 1, 1]
    assert layer._map_global_expert_id_to_local_expert_id(7) == 0
    assert layer._map_global_expert_id_to_local_expert_id(0) == -1
    assert layer._hpu_expert_perm.dtype == torch.int64


def test_apply_to_layer_rejects_round_robin():
    layer = _RoutedExperts(0)
    layer.expert_map_manager.routing_tables = (None, None, None)
    with pytest.raises(ValueError, match="linear"):
        perm_mod.apply_to_layer(layer, list(range(NUM_EXPERTS)))


def _file(tmp_path, data):
    path = tmp_path / "perm.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_load_validates_the_file(tmp_path):
    good = {"num_experts": 4, "layers": {"1": [3, 2, 1, 0], "2": [0, 1, 2, 3]}}
    assert perm_mod.load(_file(tmp_path, good), 4, frozenset({1, 2})) == {1: [3, 2, 1, 0], 2: [0, 1, 2, 3]}
    bad = [
        ({
            "num_experts": 5,
            "layers": {
                "1": [0, 1, 2, 3]
            }
        }, "num_experts"),
        ({
            "layers": {
                "1": [0, 1, 2]
            }
        }, "expected a list of 4"),
        ({
            "layers": {
                "1": [0, 1, 1, 3]
            }
        }, "not a permutation"),
        ({
            "layers": {
                "1": [0, 1, 2, 3.0]
            }
        }, "not a permutation"),
        ({
            "layers": {
                "x": [0, 1, 2, 3]
            }
        }, "decoder layer index"),
        ({
            "layers": [[0, 1, 2, 3]]
        }, "'layers' mapping"),
    ]
    for i, (data, match) in enumerate(bad):
        with pytest.raises(ValueError, match=match):
            perm_mod.parse(data, 4, source=f"case {i}")
    with pytest.raises(ValueError, match=r"missing \[3\], unexpected \[\]"):
        perm_mod.parse(good, 4, {1, 2, 3})


def test_layer_indices():
    assert perm_mod.decoder_layer_index("model.language_model.layers.17.block_sparse_moe.experts") == 17
    # RoutedExperts.layer_name of KimiK3ForConditionalGeneration and of the KimiK3MTP drafter
    for name, index in [("language_model.model.layers.1.block_sparse_moe.experts", 1),
                        ("language_model.model.layers.92.block_sparse_moe.experts", 92),
                        ("model.layers.92.block_sparse_moe.experts", 92),
                        ("model.layers.93.block_sparse_moe.experts", 93),
                        ("model.layers.93.mtp_block.block_sparse_moe.experts", 93)]:
        assert perm_mod.decoder_layer_index(name) == index
    with pytest.raises(ValueError):
        perm_mod.decoder_layer_index("lm_head")
    kimi_k3 = types.SimpleNamespace(num_hidden_layers=93, first_k_dense_replace=1, moe_layer_freq=1)
    assert perm_mod.moe_layer_indices(kimi_k3) == set(range(1, 93))
    mtp = types.SimpleNamespace(method="mtp")
    assert not perm_mod.is_drafter_layer(92, kimi_k3, mtp)
    assert perm_mod.is_drafter_layer(93, kimi_k3, mtp)
    cfg = types.SimpleNamespace(num_hidden_layers=5, first_k_dense_replace=1, moe_layer_freq=1)
    assert perm_mod.moe_layer_indices(cfg) == {1, 2, 3, 4}
    assert perm_mod.moe_layer_indices(types.SimpleNamespace()) is None
    assert perm_mod.local_expert_map([3, 0, 2, 1], 1, 2) == [1, -1, 0, -1]


def test_offline_lpt_tool_writes_a_valid_balanced_permutation(tmp_path, capsys):
    tool = _load("moe_expert_perm_tool_under_test", _ROOT / "tools" / "moe_expert_perm.py")
    g = torch.Generator().manual_seed(0)
    layers = ["model.layers.1.block_sparse_moe.experts", "model.layers.2.block_sparse_moe.experts"]
    lines = [json.dumps({"ep_size": EP_SIZE, "local_num_experts": LOCAL, "layers": layers})]
    for li in range(len(layers)):
        # hot experts packed on the first ranks
        chunks = (torch.linspace(40, 1, NUM_EXPERTS) + torch.rand(NUM_EXPERTS, generator=g)).round().int().tolist()
        for bucket in ("1-16", "17-64"):
            lines.append(
                json.dumps({
                    "layer": li,
                    "bucket": bucket,
                    "forwards": 10,
                    "rows": chunks,
                    "active": chunks,
                    "chunks": chunks,
                    "rows_sq": chunks
                }))
    stats = tmp_path / "moe_expert_stats.log"
    stats.write_text("\n".join(lines) + "\n")
    out = tmp_path / "perm.json"
    tool.main(["--stats", str(stats), "--fit-buckets", "1-16", "--eval", "17-64", "--out", str(out)])

    perms = perm_mod.load(str(out), NUM_EXPERTS, frozenset({1, 2}))
    weights, _ = tool.load_weights(str(stats), ["17-64"], "chunks")
    identity = {i: list(range(NUM_EXPERTS)) for i in perms}
    before = tool.imbalance(weights, identity, EP_SIZE, LOCAL)
    after = tool.imbalance(weights, perms, EP_SIZE, LOCAL)
    assert after[0] < before[0] and after[1] < before[1]
    assert "wrote" in capsys.readouterr().out

    # stats collected with the permutation active are indexed by physical slot
    physical = tmp_path / "physical.log"
    rows = [json.loads(line) for line in lines]
    for rec in rows[1:]:
        perm = perms[tool.decoder_layer_index(layers[rec["layer"]])]
        for f in ("rows", "active", "chunks", "rows_sq"):
            phys = [0] * NUM_EXPERTS
            for e, v in enumerate(rec[f]):
                phys[perm[e]] = v
            rec[f] = phys
    physical.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    logical, _ = tool.load_weights(str(physical), ["17-64"], "chunks", applied=perms)
    assert logical == weights


def test_remap_table_follows_the_expert_map_and_weight_device():
    layer = _RoutedExperts(0)
    layer._expert_map = layer._expert_map.to("meta")
    perm_mod.apply_to_layer(layer, list(range(NUM_EXPERTS)))
    assert layer._hpu_expert_perm.device.type == "meta"
    assert layer._hpu_expert_perm.dtype == torch.int64

    layer = _RoutedExperts(0)
    perm_mod.apply_to_layer(layer, list(range(NUM_EXPERTS)))
    layer.w13_weight = torch.nn.Parameter(torch.empty(LOCAL, 1, device="meta"), requires_grad=False)
    perm_mod.place_on_weight_device(layer)
    assert layer._hpu_expert_perm.device.type == "meta"
    assert layer._hpu_expert_perm.dtype == torch.int64
    assert "_hpu_expert_perm" in dict(layer.named_buffers())

    plain = _RoutedExperts(0)
    perm_mod.place_on_weight_device(plain)
    assert getattr(plain, "_hpu_expert_perm", None) is None


def _vllm_config(monkeypatch, speculative_config=None):
    hf_config = types.SimpleNamespace(num_hidden_layers=3, first_k_dense_replace=1, moe_layer_freq=1)
    config = types.SimpleNamespace(parallel_config=types.SimpleNamespace(enable_ep_weight_filter=False),
                                   model_config=types.SimpleNamespace(hf_text_config=hf_config),
                                   speculative_config=speculative_config)
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", types.SimpleNamespace(get_current_vllm_config=lambda: config))


def test_mtp_drafter_layers_keep_the_default_placement(monkeypatch, tmp_path):
    _vllm_config(monkeypatch, types.SimpleNamespace(method="mtp"))
    perm = [7, 6, 5, 4, 3, 2, 1, 0]
    path = _file(tmp_path, {"num_experts": NUM_EXPERTS, "layers": {"1": perm, "2": perm}})
    target = _RoutedExperts(0, "model.layers.2.block_sparse_moe.experts")
    drafter = _RoutedExperts(0, "model.layers.3.mtp_block.block_sparse_moe.experts")
    default_map = drafter._expert_map.clone()
    perm_mod.apply_file(target, path)
    perm_mod.apply_file(drafter, path)
    assert target._hpu_expert_perm.tolist() == perm
    assert getattr(drafter, "_hpu_expert_perm", None) is None
    assert torch.equal(drafter._expert_map, default_map)
    assert perm_mod.is_drafter_layer(2, types.SimpleNamespace(num_hidden_layers=3), None) is False
    assert perm_mod.is_drafter_layer(3, types.SimpleNamespace(num_hidden_layers=3),
                                     types.SimpleNamespace(method="deepseek_mtp"))


def test_separate_draft_model_is_rejected(monkeypatch, tmp_path):
    _vllm_config(monkeypatch, types.SimpleNamespace(method="eagle3"))
    path = _file(tmp_path, {"num_experts": NUM_EXPERTS, "layers": {"1": list(range(NUM_EXPERTS))}})
    with pytest.raises(ValueError, match="speculative method 'eagle3'"):
        perm_mod.apply_file(_RoutedExperts(0, "model.layers.1.block_sparse_moe.experts"), path)
