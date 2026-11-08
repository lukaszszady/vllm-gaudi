# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static per-layer placement of the routed experts across EP ranks (VLLM_HPU_MOE_EXPERT_PERM_FILE).

The file is JSON: {"num_experts": N, "layers": {"<decoder layer index>": [perm[0], ..., perm[N - 1]], ...}}.
Logical expert e of a layer is loaded into physical slot perm[e], so EP rank r holds the experts whose slots fall in
[r * local, (r + 1) * local), and the routing table is remapped to physical slots before the MoE op.
"""

import functools
import json
import logging
import regex as re
from typing import Optional

import torch

# vLLM installs handlers on the "vllm" logger tree only
logger = logging.getLogger("vllm.gaudi.moe_expert_perm")
_logged = False

_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def validate_permutation(perm, num_experts: int, where: str) -> list:
    if not isinstance(perm, list) or len(perm) != num_experts:
        raise ValueError(f"{where}: expected a list of {num_experts} expert slots")
    if not all(isinstance(v, int) and not isinstance(v, bool)
               for v in perm) or sorted(perm) != list(range(num_experts)):
        raise ValueError(f"{where}: not a permutation of 0..{num_experts - 1}")
    return perm


def parse(data: dict, num_experts: int, expected_layers: Optional[set] = None, source: str = "") -> dict:
    """Validated {decoder layer index: permutation} of a loaded permutation file."""
    if not isinstance(data, dict) or not isinstance(data.get("layers"), dict):
        raise ValueError(f"{source}: expected an object with a 'layers' mapping")
    if data.get("num_experts", num_experts) != num_experts:
        raise ValueError(f"{source}: num_experts {data['num_experts']} does not match the model ({num_experts})")
    perms = {}
    for key, perm in data["layers"].items():
        if not str(key).isdigit():
            raise ValueError(f"{source}: layer key {key!r} is not a decoder layer index")
        perms[int(key)] = validate_permutation(perm, num_experts, f"{source} layer {key}")
    if expected_layers is not None and set(perms) != set(expected_layers):
        missing = sorted(set(expected_layers) - set(perms))
        extra = sorted(set(perms) - set(expected_layers))
        raise ValueError(f"{source}: {len(perms)} layers for {len(expected_layers)} MoE layers of the model "
                         f"(missing {missing}, unexpected {extra})")
    return perms


@functools.lru_cache(maxsize=4)
def load(path: str, num_experts: int, expected_layers: Optional[frozenset] = None) -> dict:
    with open(path) as f:
        data = json.load(f)
    return parse(data, num_experts, expected_layers, source=path)


def decoder_layer_index(layer_name: str) -> int:
    m = _LAYER_INDEX_RE.search(layer_name or "")
    if m is None:
        raise ValueError(f"no decoder layer index in MoE layer name {layer_name!r}")
    return int(m.group(1))


def moe_layer_indices(hf_config) -> Optional[frozenset]:
    """Decoder layers with routed experts (first_k_dense_replace / moe_layer_freq), None if not derivable."""
    num_layers = getattr(hf_config, "num_hidden_layers", None)
    if num_layers is None:
        return None
    first_dense = getattr(hf_config, "first_k_dense_replace", 0) or 0
    freq = getattr(hf_config, "moe_layer_freq", 1) or 1
    return frozenset(i for i in range(num_layers) if i >= first_dense and i % freq == 0)


def local_expert_map(perm: list, ep_rank: int, local_num_experts: int) -> list:
    """Global expert id -> local slot on ep_rank, -1 when the expert is placed on another rank."""
    lo = ep_rank * local_num_experts
    return [p - lo if lo <= p < lo + local_num_experts else -1 for p in perm]


def remap(perm: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
    """Routing table of global expert ids -> physical slots."""
    # int64 indexing instead of an int32 index_select
    return perm[topk_ids.long()].to(topk_ids.dtype)


def apply_to_layer(layer, perm: list) -> None:
    """Make the weight loader of a routed-experts layer place expert e in slot perm[e] and store the remap table."""
    if getattr(getattr(layer, "expert_map_manager", None), "routing_tables", None) is not None:
        raise ValueError(f"{layer.layer_name}: expert permutation needs the linear expert placement")
    ep_rank = layer.moe_config.ep_rank
    local = layer.local_num_experts
    slots = local_expert_map(perm, ep_rank, local)
    if sum(s >= 0 for s in slots) != local:
        raise ValueError(f"{layer.layer_name}: permutation gives EP rank {ep_rank} a wrong number of experts")
    old_map = getattr(layer, "_expert_map", None)
    map_device = old_map.device if old_map is not None else None
    expert_map = torch.tensor(slots, dtype=torch.int32, device=map_device)
    # new tensors: the ExpertMapManager and its buffers may be shared between layers
    layer.register_buffer("_expert_map", expert_map, persistent=False)
    mask = getattr(layer, "expert_mask", None)
    if mask is not None:
        mask = mask.clone()
        mask[:len(slots)] = (expert_map >= 0).to(mask.dtype)
        layer.register_buffer("expert_mask", mask, persistent=False)
    to_local = layer._map_global_expert_id_to_local_expert_id
    layer._map_global_expert_id_to_local_expert_id = (lambda expert_id: slots[expert_id]
                                                      if expert_id < len(slots) else to_local(expert_id))
    # moved to the weight device in place_on_weight_device
    layer.register_buffer("_hpu_expert_perm",
                          torch.tensor(perm, dtype=torch.int64, device=map_device),
                          persistent=False)


def place_on_weight_device(layer) -> None:
    """Move the remap table to the device of the loaded expert weights (no host->device copy per step)."""
    perm = getattr(layer, "_hpu_expert_perm", None)
    if perm is None:
        return
    device = next((p.device for p in layer.parameters(recurse=False)), None)
    if device is not None and perm.device != device:
        layer.register_buffer("_hpu_expert_perm", perm.to(device), persistent=False)


def is_drafter_layer(layer_index: int, hf_config, speculative_config) -> bool:
    """MoE layer of an MTP drafter (numbered after the target's decoder layers), left without a permutation.

    Other speculative methods load a separate draft model whose layer indices can collide with the target's.
    """
    if speculative_config is None:
        return False
    method = str(getattr(speculative_config, "method", None) or "")
    if method != "mtp" and not method.endswith("_mtp"):
        raise ValueError(f"VLLM_HPU_MOE_EXPERT_PERM_FILE does not work with speculative method {method!r}")
    num_layers = getattr(hf_config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError("VLLM_HPU_MOE_EXPERT_PERM_FILE with MTP needs num_hidden_layers in the model config")
    return layer_index >= num_layers


def apply_file(layer, path: str) -> None:
    """apply_to_layer with the permutation of the layer's decoder layer from a permutation file."""
    from vllm.config import get_current_vllm_config
    vllm_config = get_current_vllm_config()
    if vllm_config.parallel_config.enable_ep_weight_filter:
        raise ValueError("VLLM_HPU_MOE_EXPERT_PERM_FILE does not work with enable_ep_weight_filter")
    model_config = vllm_config.model_config
    hf_config = getattr(model_config, "hf_text_config", None) or model_config.hf_config
    index = decoder_layer_index(layer.layer_name)
    if is_drafter_layer(index, hf_config, getattr(vllm_config, "speculative_config", None)):
        return
    perms = load(path, layer.global_num_experts, moe_layer_indices(hf_config))
    if index not in perms:
        raise ValueError(f"{path}: no permutation for decoder layer {index} ({layer.layer_name})")
    apply_to_layer(layer, perms[index])
    global _logged
    if not _logged:
        _logged = True
        logger.info("MoE expert permutation from %s: %d layers, EP rank %d holds %d of %d experts", path, len(perms),
                    layer.moe_config.ep_rank, layer.local_num_experts, layer.global_num_experts)
