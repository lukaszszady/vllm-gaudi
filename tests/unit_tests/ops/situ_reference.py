# SPDX-License-Identifier: Apache-2.0
"""Reference MXFP4 dequantization used by test_hpu_situ_moe.py to check ``dequant_mxfp4``."""

import torch

# E2M1 magnitudes; the high bit of a nibble is the sign. Low nibble = even element, high nibble = odd element.
_FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
MXFP4_BLOCK = 32


def dequant_mxfp4_reference(packed: torch.Tensor,
                            scales: torch.Tensor,
                            dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """fp32 stack + ldexp version; the LUT version must equal it."""
    *lead, rows, half = packed.shape
    groups = scales.shape[-1]
    lut = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=packed.device)
    values = torch.stack((lut[(packed & 0x0F).long()], lut[(packed >> 4).long()]), dim=-1)
    values = values.reshape(*lead, rows, groups, MXFP4_BLOCK)
    values = torch.ldexp(values, (scales.to(torch.int32) - 127).unsqueeze(-1))
    return values.reshape(*lead, rows, groups * MXFP4_BLOCK).to(dtype)
