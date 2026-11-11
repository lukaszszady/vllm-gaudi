# SPDX-License-Identifier: Apache-2.0
"""Per-chunk reference of the chunk-parallel KDA prefill, used by test_hpu_kda_pytorch.py to check
``kda_chunked_steps_vec``."""

import torch


def _unit_lower_inverse(a: torch.Tensor) -> torch.Tensor:
    """(I + A)^-1 for a strictly lower triangular A [..., C, C] by recursive block doubling.

    The 1x1 diagonal blocks of I + A are 1; neighbouring blocks are merged level by level with the exact block
    formula ``[[T, 0], [-B (C T), B]]`` (T, B the inverses of the two diagonal blocks, C the coupling block), all
    pairs of a level in one batched matmul: log2(C) levels instead of C sequential substitution steps, no in-place
    writes. It is algebraically the same as forward substitution and as accurate (checked against a float64
    inverse, including nearly parallel keys with beta ~ 1), whereas a Neumann / repeated-squaring expansion loses
    all precision there. A size that is not a power of two is padded with identity rows.
    """
    size = a.shape[-1]
    padded = 1
    while padded < size:
        padded *= 2
    if padded != size:
        a = torch.nn.functional.pad(a, (0, padded - size, 0, padded - size))
    lead = a.shape[:-2]
    inv = a.new_ones(*lead, padded, 1, 1)  # inverses of the 1 x 1 diagonal blocks
    block, count = 1, padded
    while count > 1:
        blocks = a.reshape(*lead, count, block, count, block)
        # coupling between block 2p+1 (rows) and block 2p (columns), for every pair p
        coupling = torch.diagonal(blocks[..., 1::2, :, 0::2, :], dim1=-4, dim2=-2)  # [..., block, block, count/2]
        coupling = coupling.permute(*range(len(lead)), -1, -3, -2)  # [..., count/2, block, block]
        top, bottom = inv[..., 0::2, :, :], inv[..., 1::2, :, :]
        lower_left = -torch.matmul(bottom, torch.matmul(coupling, top))
        inv = torch.cat((torch.cat((top, torch.zeros_like(top)), dim=-1), torch.cat((lower_left, bottom), dim=-1)),
                        dim=-2)  # [..., count/2, 2 * block, 2 * block]
        block, count = 2 * block, count // 2
    return inv.reshape(*lead, padded, padded)[..., :size, :size]


def kda_chunked_steps(state: torch.Tensor,
                      q: torch.Tensor,
                      k: torch.Tensor,
                      v: torch.Tensor,
                      g: torch.Tensor,
                      beta: torch.Tensor,
                      chunk_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunk-parallel (WY / UT form) delta rule, identical to ``kda_recurrent_steps`` (reference for the batched
    ``kda_chunked_steps_vec`` that serving runs).

    q,k [B,L,H,K] (normalized, q scaled); v [B,L,H,V]; g [B,L,H,K] LOG forget gate (<= 0); beta [B,L,H];
    state [B,H,V,K]. Chunks are processed one after another (state carried), tokens of a chunk in parallel.
    With G_i the cumulative log gate inside the chunk, u_i the pseudo values and S_0 the state before the chunk::

        (I + A) U = beta * (V - (K * exp(G)) S_0^T)       A_ij = beta_i sum_k k_i k_j exp(G_i - G_j)   (j < i)
        O = (Q * exp(G)) S_0^T + Aqk U                    Aqk_ij = sum_k q_i k_j exp(G_i - G_j)        (j <= i)
        S = S_0 * exp(G_C) + U^T (K * exp(G_C - G))

    Only exp(G_i - G_j) with j <= i (always <= 1) is evaluated, so nothing overflows however negative the gate
    gets; the pairwise [C, C, K] tensor exists for one chunk at a time. Tokens past the real length must have
    g = 0 and beta = 0 (exact no-ops); the sequence is zero-padded to a multiple of ``chunk_size`` the same way.
    """
    if chunk_size <= 0 or (chunk_size > 16 and chunk_size % 16 != 0):
        # keep the chunk (and with it every matmul of the inverse) tile friendly: multiples of 16 above 16
        raise ValueError(f"chunk_size must be positive and, above 16, a multiple of 16, got {chunk_size}")
    bsz, seq_len, heads, kdim = q.shape
    vdim = v.shape[-1]
    chunk = chunk_size
    pad = -seq_len % chunk
    if pad:
        zeros = lambda x: x.new_zeros(bsz, pad, *x.shape[2:])  # noqa: E731
        q, k, v = (torch.cat([x, zeros(x)], dim=1) for x in (q, k, v))
        g = torch.cat([g, zeros(g)], dim=1)
        beta = torch.cat([beta, zeros(beta)], dim=1)
    num_chunks = (seq_len + pad) // chunk

    def to_chunks(x):  # [B, N*C, H, D] -> [B, H, N, C, D]
        return x.reshape(bsz, num_chunks, chunk, heads, -1).permute(0, 3, 1, 2, 4)

    qc, kc, vc, gc = to_chunks(q), to_chunks(k), to_chunks(v), to_chunks(g)
    bc = beta.reshape(bsz, num_chunks, chunk, heads).permute(0, 3, 1, 2).unsqueeze(-1)  # [B,H,N,C,1]
    cum = torch.cumsum(gc, dim=3)  # G, [B,H,N,C,K]
    last = cum[:, :, :, -1:, :]  # G_C
    q_gamma = qc * torch.exp(cum)
    k_gamma = kc * torch.exp(cum)
    k_tail = kc * torch.exp(last - cum)
    gamma_last = torch.exp(last.squeeze(3))  # [B,H,N,K]

    strict = torch.tril(torch.ones(chunk, chunk, dtype=q.dtype, device=q.device), -1)
    lower = torch.tril(torch.ones(chunk, chunk, dtype=q.dtype, device=q.device))

    outs = []
    for n in range(num_chunks):
        g_n, k_n, q_n = cum[:, :, n], kc[:, :, n], qc[:, :, n]  # [B,H,C,K]
        # exp(G_i - G_j) for j <= i; the clamp keeps the masked-out upper triangle finite (0 * inf would be NaN)
        decay = torch.exp(torch.clamp(g_n.unsqueeze(3) - g_n.unsqueeze(2), max=0.0))  # [B,H,C,C,K]
        kk = k_n.unsqueeze(3) * k_n.unsqueeze(2)
        qk = q_n.unsqueeze(3) * k_n.unsqueeze(2)
        a_mat = (kk * decay).sum(-1) * strict * bc[:, :, n]  # beta_i on the row
        aqk = (qk * decay).sum(-1) * lower
        inv = _unit_lower_inverse(a_mat)
        s_t = state.transpose(-1, -2)  # [B,H,K,V]
        rhs = bc[:, :, n] * (vc[:, :, n] - torch.matmul(k_gamma[:, :, n], s_t))
        pseudo = torch.matmul(inv, rhs)  # U, [B,H,C,V]
        out_n = torch.matmul(q_gamma[:, :, n], s_t) + torch.matmul(aqk, pseudo)
        state = state * gamma_last[:, :, n].unsqueeze(-2) + torch.matmul(pseudo.transpose(-1, -2), k_tail[:, :, n])
        outs.append(out_n)
    out = torch.stack(outs, dim=2)  # [B,H,N,C,V]
    out = out.permute(0, 2, 3, 1, 4).reshape(bsz, num_chunks * chunk, heads, vdim)
    return out[:, :seq_len], state
