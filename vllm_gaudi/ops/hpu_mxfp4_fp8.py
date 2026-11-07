# SPDX-License-Identifier: Apache-2.0
"""MXFP4 (E2M1 + E8M0 per 32) -> FP8 E4M3 with one power-of-two scale per output row (pure PyTorch).

Every E2M1 value is exact in E4M3. Each block is shifted by its exponent distance to the row maximum, offset by
``FP8_ROW_HEADROOM`` so that 6 * 2**FP8_ROW_HEADROOM stays below the E4M3 maximum (448). Blocks up to
``EXACT_EXPONENT_SPAN`` exponents below the row maximum convert exactly; smaller ones round to E4M3 subnormals.
"""

from __future__ import annotations

import re

import torch

from vllm_gaudi.ops.hpu_situ_moe import MXFP4_BLOCK, _luts

FP8_ROW_HEADROOM = 6
# smallest E2M1 step (0.5) times 2**(HEADROOM - span) must stay a multiple of the E4M3 subnormal step 2**-9
EXACT_EXPONENT_SPAN = FP8_ROW_HEADROOM + 8


def mxfp4_to_fp8_rows(packed: torch.Tensor, scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """packed uint8 [rows, cols // 2] + E8M0 uint8 [rows, cols // 32] -> (fp8 [rows, cols], fp32 scale [rows], inexact).

    ``inexact`` counts elements whose FP8 value times the row scale differs from the MXFP4 value.
    """
    rows, half = packed.shape
    groups = scales.shape[-1]
    if half * 2 != groups * MXFP4_BLOCK or scales.shape[0] != rows:
        raise ValueError(f"packed {tuple(packed.shape)} and scales {tuple(scales.shape)} do not match")
    byte_lut, _ = _luts(torch.float32, packed.device)
    values = torch.nn.functional.embedding(packed.to(torch.int32), byte_lut).reshape(rows, groups, MXFP4_BLOCK)
    exps = scales.to(torch.int32)
    row_exp = exps.amax(dim=-1, keepdim=True)
    # clamped so far-out blocks stay nonzero in fp32 and are counted as inexact
    shift = (exps - row_exp + FP8_ROW_HEADROOM).clamp(min=-140).to(torch.float32)
    values.mul_(torch.exp2(shift).unsqueeze(-1))
    fp8 = values.reshape(rows, groups * MXFP4_BLOCK).to(torch.float8_e4m3fn)
    inexact = int((fp8.to(torch.float32) != values.reshape(rows, -1)).sum())
    row_scale = torch.exp2((row_exp.squeeze(-1) - 127 - FP8_ROW_HEADROOM).to(torch.float32))
    return fp8, row_scale, inexact


def mxfp4_experts_to_fp8(packed: torch.Tensor, scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Per-expert ``mxfp4_to_fp8_rows`` over [E, rows, cols // 2]; one expert at a time to bound temporaries."""
    experts, rows, half = packed.shape
    out = torch.empty((experts, rows, half * 2), dtype=torch.float8_e4m3fn, device=packed.device)
    out_scale = torch.empty((experts, rows), dtype=torch.float32, device=packed.device)
    inexact = 0
    for e in range(experts):
        w, s, n = mxfp4_to_fp8_rows(packed[e], scales[e])
        out[e].copy_(w)
        out_scale[e].copy_(s)
        inexact += n
    return out, out_scale, inexact


_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def parse_layer_spec(spec: str | None) -> set[int] | None:
    """HPU_KIMI_MOE_FP8_LAYERS: "" (off), "all", or comma-separated indices / inclusive ranges ("1-8,40").

    Returns an empty set when off and None for all layers.
    """
    spec = (spec or "").strip().lower()
    if spec in ("", "0", "off", "none"):
        return set()
    if spec == "all":
        return None
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        if not lo.isdigit() or (sep and not hi.isdigit()):
            raise ValueError(f"HPU_KIMI_MOE_FP8_LAYERS: bad entry {part!r} in {spec!r}")
        start, end = int(lo), int(hi) if sep else int(lo)
        if end < start:
            raise ValueError(f"HPU_KIMI_MOE_FP8_LAYERS: empty range {part!r}")
        layers.update(range(start, end + 1))
    return layers


def layer_index(layer_name: str | None) -> int | None:
    """Decoder layer index from a module prefix such as "language_model.model.layers.12.mlp.experts"."""
    match = _LAYER_INDEX.search(layer_name or "")
    return int(match.group(1)) if match else None


def layer_selected(spec: str | None, layer_name: str | None) -> bool:
    layers = parse_layer_spec(spec)
    if layers is None:
        return True
    index = layer_index(layer_name)
    return index is not None and index in layers
