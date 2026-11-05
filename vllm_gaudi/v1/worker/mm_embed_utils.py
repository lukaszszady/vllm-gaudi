# SPDX-License-Identifier: Apache-2.0
"""Row selection by a PlaceholderRange.is_embed mask without boolean-mask indexing on the device.

Indexing a device tensor with a boolean mask (``t[mask]``) has a data-dependent output shape, which
the Habana graph compiler cannot build (``synNodeCreateWithId failed for node: slice``). The mask lives on the host, so
the integer positions are computed there and the device only sees a static-shape ``index_select``.
"""
import torch


def embed_positions(is_embed: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """Positions of the True entries of a host mask, as an int64 index tensor on ``device``."""
    return torch.nonzero(is_embed.cpu(), as_tuple=True)[0].to(device)


def select_embed_rows(rows: torch.Tensor, is_embed: torch.Tensor) -> torch.Tensor:
    """Same result as ``rows[is_embed]`` for a 1D bool mask over dim 0, without a device boolean mask."""
    return rows.index_select(0, embed_positions(is_embed, rows.device))
