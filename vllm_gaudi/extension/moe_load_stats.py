# SPDX-License-Identifier: Apache-2.0
"""Diagnostic per-layer MoE expert-parallel load statistics (VLLM_HPU_MOE_LOAD_STATS=1).

Every rank sees the full routing table of the batch, so each MoE forward counts, for every EP rank, the
token rows routed to its local experts, its active experts, its expert chunks (sum of ceil(rows / chunk_size)
per active expert) and its largest expert. The counts are accumulated on device per token-count bucket
(sum, sum of squares, max) and copied to the host every VLLM_HPU_MOE_LOAD_STATS_INTERVAL model steps. Rank 0
logs a short summary per copy and rewrites VLLM_HPU_MOE_LOAD_STATS_FILE with the cumulative per-layer table and
VLLM_HPU_MOE_EXPERT_STATS_FILE (default moe_expert_stats.log next to the former) with per-expert totals as JSON lines.
All ranks accumulate and copy, so the extra device work and syncs are the same on every rank.
"""

import atexit
import json
import logging
import math
import os
from typing import Optional

import torch

logger = logging.getLogger("vllm.gaudi.moe_load_stats")

# lower bounds of the token-count buckets (rows of the routing table, padding included)
TOKEN_BUCKETS = (1, 17, 65, 257, 1025, 2049, 4097, 8193)
FEATURES = ("rows", "experts", "chunks", "max_rows")
# per-forward columns after the per-rank features
_CHUNKS_MAX, _CHUNKS_SPREAD, _COUNT = 0, 1, 2
# per global expert, summed over forwards
EXPERT_FEATURES = ("rows", "active", "chunks", "rows_sq")


def enabled() -> bool:
    return os.environ.get("VLLM_HPU_MOE_LOAD_STATS", "0").strip().lower() in ("1", "true")


def num_columns(ep_size: int) -> int:
    return len(FEATURES) * ep_size + 3


def new_buffer(ep_size: int, device) -> torch.Tensor:
    """[sum | sum of squares | max, token bucket, column] accumulator of one MoE layer."""
    return torch.zeros(3, len(TOKEN_BUCKETS), num_columns(ep_size), dtype=torch.float32, device=device)


def new_expert_buffer(global_num_experts: int, device) -> torch.Tensor:
    """[expert feature, token bucket, global expert] accumulator of one MoE layer."""
    return torch.zeros(len(EXPERT_FEATURES), len(TOKEN_BUCKETS), global_num_experts, dtype=torch.float32, device=device)


def bucket_label(i: int) -> str:
    if i + 1 < len(TOKEN_BUCKETS):
        return f"{TOKEN_BUCKETS[i]}-{TOKEN_BUCKETS[i + 1] - 1}"
    return f">={TOKEN_BUCKETS[i]}"


def accumulate(buf: torch.Tensor,
               routing_table: torch.Tensor,
               global_num_experts: int,
               local_num_experts: int,
               chunk_size: int,
               expert_buf: Optional[torch.Tensor] = None) -> None:
    """Add the EP load of one forward (routing_table [tokens, top_k] of global expert ids) to buf in place."""
    ids = routing_table.reshape(-1).long()
    valid = (ids >= 0) & (ids < global_num_experts)
    counts = torch.zeros(global_num_experts, dtype=torch.float32, device=ids.device)
    counts = counts.scatter_add(0, torch.where(valid, ids, torch.zeros_like(ids)), valid.float())
    active = (counts > 0).float()
    expert_chunks = torch.ceil(counts / chunk_size) if chunk_size > 0 else active
    per_rank = counts.view(-1, local_num_experts)
    experts = active.view(-1, local_num_experts).sum(1)
    chunks = expert_chunks.view(-1, local_num_experts).sum(1)
    chunks_max = chunks.amax()
    tokens = torch.ones_like(routing_table[:, 0], dtype=torch.float32).sum()
    lower = torch.tensor(TOKEN_BUCKETS, dtype=torch.float32, device=ids.device)
    upper = torch.tensor(TOKEN_BUCKETS[1:] + (math.inf, ), dtype=torch.float32, device=ids.device)
    in_bucket = ((tokens >= lower) & (tokens < upper)).float().unsqueeze(1)
    v = torch.cat([
        per_rank.sum(1), experts, chunks,
        per_rank.amax(1),
        chunks_max.view(1), (chunks_max - chunks.mean()).view(1),
        torch.ones(1, dtype=torch.float32, device=ids.device)
    ])
    v = in_bucket * v
    buf.copy_(torch.stack([buf[0] + v, buf[1] + v * v, torch.maximum(buf[2], v)]))
    if expert_buf is not None:
        ev = torch.stack([counts, active, expert_chunks, counts * counts])
        expert_buf.copy_(expert_buf + in_bucket.view(1, -1, 1) * ev.unsqueeze(1))


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _in_words(n: int) -> str:
    # Ray log dedup ignores digits: spell the copy index so consecutive summaries are not collapsed
    names = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
    return "-".join(names[int(d)] for d in str(n))


class MoELoadStats:
    """Owns the device accumulators of all MXFP4 MoE ops of a model and their host-side totals."""

    def __init__(self,
                 interval: int,
                 path: Optional[str],
                 rank: Optional[int] = None,
                 expert_path: Optional[str] = None):
        self.interval = max(1, interval)
        self.path = path
        self.expert_path = expert_path
        self.rank = _rank() if rank is None else rank
        self.ops: list = []
        self.steps = 0
        self.copies = 0
        self.ep_size = 0
        self.local_num_experts = 0
        self.total = None
        self.expert_total = None

    @classmethod
    def from_env(cls) -> Optional["MoELoadStats"]:
        if not enabled():
            return None
        interval = int(os.environ.get("VLLM_HPU_MOE_LOAD_STATS_INTERVAL", "100"))
        path = os.environ.get("VLLM_HPU_MOE_LOAD_STATS_FILE") or None
        expert_path = os.environ.get("VLLM_HPU_MOE_EXPERT_STATS_FILE") or None
        if expert_path is None and path is not None:
            expert_path = os.path.join(os.path.dirname(path), "moe_expert_stats.log")
        return cls(interval, path, expert_path=expert_path)

    def attach(self, model: torch.nn.Module, device, op_types=None) -> int:
        if op_types is None:
            from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOpMXFP4
            op_types = (VllmMixtureOfExpertsOpMXFP4, )
        seen = {id(op) for _, op in self.ops}
        for name, module in model.named_modules():
            op = getattr(module, "moe_op", None)
            # the same op can be reachable from several modules
            if not isinstance(op, op_types) or op.global_num_experts % op.num_experts or id(op) in seen:
                continue
            ep_size = op.global_num_experts // op.num_experts
            if self.ops and (ep_size, op.num_experts) != (self.ep_size, self.local_num_experts):
                continue
            self.ep_size, self.local_num_experts = ep_size, op.num_experts
            op._load_stats = new_buffer(ep_size, device)
            op._expert_stats = new_expert_buffer(op.global_num_experts, device)
            seen.add(id(op))
            self.ops.append((name, op))
        if self.ops:
            self.total = torch.zeros(3,
                                     len(self.ops),
                                     len(TOKEN_BUCKETS),
                                     num_columns(self.ep_size),
                                     dtype=torch.float64)
            self.expert_total = torch.zeros(len(self.ops),
                                            len(EXPERT_FEATURES),
                                            len(TOKEN_BUCKETS),
                                            self.ep_size * self.local_num_experts,
                                            dtype=torch.float64)
            if self.rank == 0:
                atexit.register(self.flush)
        logger.info("MoE load stats on %d layers, %d EP ranks x %d local experts, copy every %d steps, file %s",
                    len(self.ops), self.ep_size, self.local_num_experts, self.interval, self.path)
        return len(self.ops)

    def reset(self) -> None:
        if not self.ops:
            return
        torch._foreach_zero_(self._buffers())
        self.total.zero_()
        self.expert_total.zero_()
        self.steps = 0

    def on_step(self) -> None:
        if not self.ops:
            return
        self.steps += 1
        if self.steps % self.interval == 0:
            self.collect()
            self.emit()

    def _buffers(self) -> list:
        return [op._load_stats for _, op in self.ops] + [op._expert_stats for _, op in self.ops]

    def collect(self) -> None:
        n = len(self.ops)
        bufs = self._buffers()
        # one device-to-host copy for all accumulators
        flat = torch.cat([b.reshape(-1) for b in bufs]).to("cpu", torch.float64)
        torch._foreach_zero_(bufs)
        layer_size = bufs[0].numel()
        host = flat[:n * layer_size].view(n, *bufs[0].shape).transpose(0, 1)
        self.expert_total += flat[n * layer_size:].view(n, *bufs[n].shape)
        self.total[0] += host[0]
        self.total[1] += host[1]
        torch.maximum(self.total[2], host[2], out=self.total[2])

    def flush(self) -> None:
        try:
            self.collect()
            self.emit(final=True)
        except Exception as e:  # interpreter shutdown: the device may be gone
            logger.warning("MoE load stats final copy failed: %s", e)

    def emit(self, final: bool = False) -> None:
        if self.rank != 0:
            return
        self.copies += 1
        logger.info("MoE load stats copy %s%s (steps %d): %s", _in_words(self.copies), " final" if final else "",
                    self.steps, self.summary_line())
        if self.path:
            with open(self.path, "w") as f:
                f.write("\n".join(self.report_lines()) + "\n")
        if self.expert_path:
            tmp = self.expert_path + ".tmp"
            with open(tmp, "w") as f:
                f.write("\n".join(self.expert_lines()) + "\n")
            os.replace(tmp, self.expert_path)

    def expert_lines(self) -> list:
        """JSON lines: a header, then one record per (layer, token bucket) with forwards."""
        lines = [
            json.dumps({
                "steps": self.steps,
                "copies": self.copies,
                "ep_size": self.ep_size,
                "local_num_experts": self.local_num_experts,
                "buckets": [bucket_label(b) for b in range(len(TOKEN_BUCKETS))],
                "features": list(EXPERT_FEATURES),
                "layers": [name for name, _ in self.ops],
            })
        ]
        counts = self.total[0, :, :, self._extra(_COUNT)]
        for li in range(len(self.ops)):
            for b in range(len(TOKEN_BUCKETS)):
                if counts[li, b] == 0:
                    continue
                rec = {"layer": li, "bucket": bucket_label(b), "forwards": int(counts[li, b])}
                for fi, f in enumerate(EXPERT_FEATURES):
                    rec[f] = [int(v) for v in self.expert_total[li, fi, b].round().tolist()]
                lines.append(json.dumps(rec, separators=(",", ":")))
        return lines

    # host-side views of the totals
    def _col(self, feature: str, rank: int) -> int:
        return FEATURES.index(feature) * self.ep_size + rank

    def _extra(self, i: int) -> int:
        return len(FEATURES) * self.ep_size + i

    def summary_line(self) -> str:
        s, _, m = self.total
        parts = []
        for b in range(len(TOKEN_BUCKETS)):
            n = s[:, b, self._extra(_COUNT)]
            if n.sum() == 0:
                continue
            nn = n.clamp(min=1)
            chunks = s[:, b, FEATURES.index("chunks") * self.ep_size:(FEATURES.index("chunks") + 1) * self.ep_size]
            mean_rank = (chunks / nn[:, None]).mean(0)
            spread = s[:, b, self._extra(_CHUNKS_SPREAD)] / nn
            parts.append(f"tokens {bucket_label(b)} forwards/layer {int(n.max())} chunks/rank mean "
                         f"{mean_rank.mean():.1f} (ranks {mean_rank.min():.1f}..{mean_rank.max():.1f}) "
                         f"rank-max {(s[:, b, self._extra(_CHUNKS_MAX)] / nn).mean():.1f} "
                         f"max-minus-mean {spread.mean():.2f} (worst layer {spread.max():.2f}, "
                         f"single forward {m[:, b, self._extra(_CHUNKS_SPREAD)].max():.1f})")
        return "; ".join(parts) or "no MoE forwards"

    def report_lines(self) -> list:
        s, q, m = self.total
        ep = self.ep_size
        lines = [
            f"# MoE EP load stats: model steps {self.steps}, copies {self.copies}, layers {len(self.ops)}, "
            f"EP ranks {ep}, local experts {self.local_num_experts}",
            "# rows = token rows routed to the rank's experts, experts = active local experts, chunks = sum of "
            "ceil(rows / chunk_size) over active experts, max_rows = rows of the rank's largest expert;",
            "# rank-max / max-minus-mean = per forward max of chunks over EP ranks and its excess over the mean; "
            "mean/std/max over forwards",
        ]
        for b in range(len(TOKEN_BUCKETS)):
            n = s[:, b, self._extra(_COUNT)]
            if n.sum() == 0:
                continue
            lines.append("")
            lines.append(f"## tokens {bucket_label(b)}: forwards per layer {int(n.max())}")
            nn = n.clamp(min=1)
            tot = n.sum()

            def pooled(col):
                mean = s[:, b, col].sum() / tot
                std = (q[:, b, col].sum() / tot - mean * mean).clamp(min=0).sqrt()
                return mean, std, m[:, b, col].max()

            lines.append("### per EP rank, pooled over layers and forwards: mean / std / max")
            lines.append("rank " + " ".join(f"{f:>22}" for f in FEATURES))
            for r in range(ep):
                cells = []
                for f in FEATURES:
                    mean, std, mx = pooled(self._col(f, r))
                    cells.append(f"{mean:8.1f}/{std:6.1f}/{mx:6.0f}")
                lines.append(f"{r:4d} " + " ".join(f"{c:>22}" for c in cells))
            for i, label in ((_CHUNKS_MAX, "rank-max chunks"), (_CHUNKS_SPREAD, "max-minus-mean chunks")):
                mean, std, mx = pooled(self._extra(i))
                lines.append(f"{label}: {mean:.2f} / {std:.2f} / {mx:.1f}")
            lines.append("### per layer: forwards, max-minus-mean chunks mean/std/max, rank-max chunks mean, "
                         "chunks mean per EP rank, max_rows max")
            ch0 = FEATURES.index("chunks") * ep
            for li, (name, _) in enumerate(self.ops):
                if n[li] == 0:
                    continue
                col = self._extra(_CHUNKS_SPREAD)
                mean = s[li, b, col] / nn[li]
                std = (q[li, b, col] / nn[li] - mean * mean).clamp(min=0).sqrt()
                per_rank = " ".join(f"{v:.1f}" for v in (s[li, b, ch0:ch0 + ep] / nn[li]).tolist())
                mr0 = FEATURES.index("max_rows") * ep
                lines.append(f"{name} n {int(n[li])} spread {mean:.2f}/{std:.2f}/{m[li, b, col]:.0f} "
                             f"rank-max {s[li, b, self._extra(_CHUNKS_MAX)] / nn[li]:.1f} "
                             f"chunks [{per_rank}] max_rows {m[li, b, mr0:mr0 + ep].max():.0f}")
        return lines
