# SPDX-License-Identifier: Apache-2.0
"""Sizing of the scheduler token budget for the profile-one-synthetic-batch mode (no vLLM / Habana imports)."""

from __future__ import annotations

import contextlib
import gzip
import logging
import os
import random
import shutil
import time
from collections.abc import Callable, Iterable, Mapping

# under the 'vllm' logger tree: vLLM attaches its handler there, other names would drop the INFO lines
_logger = logging.getLogger('vllm.gaudi.profiling')


def profiling_max_batched_tokens(prompt_cfg: tuple | None, decode_cfg: tuple | None, current: int) -> int:
    """Token budget (``max_num_batched_tokens``) that fits the dummy batch of VLLM_PROFILE_PROMPT / _DECODE.

    ``prompt_cfg`` is ``(bs, query, ctx_blocks)``, ``decode_cfg`` ``(bs, 1, blocks)``. The prompt requests
    schedule ``bs * query`` tokens and every decode request one more token, so a mixed step (both variables set)
    needs ``bs * query + decode_bs``; sizing it as ``bs * query`` alone made the 8193rd token overflow the
    position buffer in ``_prepare_inputs``. A decode-only profile never shrinks the budget below its batch size.
    """
    tokens = current
    if prompt_cfg:
        tokens = prompt_cfg[0] * prompt_cfg[1]
    if decode_cfg:
        tokens = tokens + decode_cfg[0] if prompt_cfg else max(tokens, decode_cfg[0])
    return tokens


def profiling_max_reqs(prompt_cfg: tuple | None, decode_cfg: tuple | None, current: int) -> int:
    """Request slots (e.g. the compact GDN state pool) the dummy batch needs: a mixed step holds both batches."""
    reqs = (prompt_cfg[0] if prompt_cfg else 0) + (decode_cfg[0] if decode_cfg else 0)
    return max(current, reqs)


def bucketing_max_batched_tokens(tokens: int, mamba_chunk_size: int, profiling: bool) -> int:
    """Token budget handed to the bucketing manager.

    HPUBucketingManager.initialize raises unless ``max_num_batched_tokens`` is divisible by the mamba-like chunk
    size (128 for KDA/GDN models without an explicit one). A mixed profile budget (``bs * query + decode_bs``,
    e.g. 8192 + 1) is not, which killed the engine right after start (build #859). Only the profile mode rounds
    up; a misconfigured normal serve still fails loudly.
    """
    if not profiling or mamba_chunk_size <= 0:
        return tokens
    return -(-tokens // mamba_chunk_size) * mamba_chunk_size


def profile_blocks_scattered(environ: Mapping[str, str] | None = None) -> bool:
    """VLLM_PROFILE_BLOCKS: 'repeat' (default) gives every dummy block id 0, so the paged-attention KV gather
    re-reads one cached block; 'scatter' gives each dummy request its own blocks spread over the KV pool."""
    environ = os.environ if environ is None else environ
    mode = (environ.get('VLLM_PROFILE_BLOCKS') or 'repeat').strip().lower()
    if mode not in ('repeat', 'scatter'):
        raise ValueError(f"VLLM_PROFILE_BLOCKS must be 'repeat' or 'scatter', got {mode!r}")
    return mode == 'scatter'


def pad_owner_block(pad_kernel_block: int, kernel_block_size: int, block_size: int) -> int:
    """Scheduler block (``block_size`` tokens) containing the decode pad kernel block, which padded batch slots
    write to; dummy requests must not own it."""
    return pad_kernel_block * kernel_block_size // block_size


class ScatteredBlockAllocator:
    """Distinct scheduler block ids for the dummy requests of the profile mode.

    Every KV cache group draws from its own fixed random permutation of ``[1, num_blocks)`` (block 0 is vLLM's
    null block) minus ``exclude``, so a request's blocks are scattered over the pool like after a long serve.
    Ids repeat only when a group's pool is exhausted; ``wrapped`` tells whether that happened.
    """

    def __init__(self, num_blocks: int, exclude: Iterable[int] = (), seed: int = 0):
        excluded = set(exclude)
        self._base = [b for b in range(1, num_blocks) if b not in excluded]
        if not self._base:
            raise ValueError(f"no block left for dummy requests (num_blocks={num_blocks}, exclude={sorted(excluded)})")
        self._seed = seed
        self._pools: dict[int, list[int]] = {}
        self._used: dict[int, int] = {}
        self.wrapped = False

    @property
    def pool_size(self) -> int:
        return len(self._base)

    def take(self, group: int, n: int) -> list[int]:
        pool = self._pools.get(group)
        if pool is None:
            pool = self._pools[group] = list(self._base)
            random.Random(self._seed + group).shuffle(pool)
        used = self._used.get(group, 0)
        self._used[group] = used + n
        self.wrapped = self.wrapped or used + n > len(pool)
        return [pool[(used + i) % len(pool)] for i in range(n)]


# SamplingParams fields VLLM_PROFILE_SAMPLING may set (a generation config may also carry e.g. max_tokens).
_SAMPLING_FIELDS = {
    'temperature': float,
    'top_p': float,
    'top_k': int,
    'min_p': float,
    'repetition_penalty': float,
    'presence_penalty': float,
    'frequency_penalty': float,
}


def profile_sampling_kwargs(spec: str | None, model_defaults: Mapping | None = None) -> dict | None:
    """SamplingParams kwargs for the dummy requests from VLLM_PROFILE_SAMPLING (None = unset: greedy as before).

    'default' takes the model's generation-config defaults (``model_defaults``) on top of the SamplingParams
    defaults (temperature 1.0, no top-k/top-p), i.e. what a request without sampling fields gets; ``vllm bench
    serve`` no longer sends temperature 0. 'key=value' items set fields explicitly and combine with 'default',
    e.g. 'default,top_p=0.95' or 'temperature=0.7,top_k=50'.
    """
    if spec is None or not spec.strip():
        return None
    kwargs: dict = {}
    for item in (s.strip() for s in spec.split(',')):
        if not item:
            continue
        if item == 'default':
            defaults = model_defaults or {}
            kwargs.update({k: _SAMPLING_FIELDS[k](v) for k, v in defaults.items() if k in _SAMPLING_FIELDS})
            continue
        key, sep, value = item.partition('=')
        key = key.strip()
        if not sep or key not in _SAMPLING_FIELDS:
            raise ValueError(f"VLLM_PROFILE_SAMPLING item {item!r}: expected 'default' or one of "
                             f"{sorted(_SAMPLING_FIELDS)} as key=value")
        kwargs[key] = _SAMPLING_FIELDS[key](value.strip())
    return kwargs


# Paths of the traces written by make_trace_handler in this process (logged by hold_before_exit).
WRITTEN_TRACES: list = []


def profile_exit_delay_s(environ: Mapping[str, str] | None = None) -> float:
    """VLLM_PROFILE_EXIT_DELAY_S: seconds every rank holds after the traces are written, before the run ends with
    'Finished profiling' (0 = no hold). Gives time to copy the traces off the pod; invalid/negative values mean 0."""
    environ = os.environ if environ is None else environ
    try:
        return max(0.0, float(environ.get('VLLM_PROFILE_EXIT_DELAY_S', '0') or 0))
    except ValueError:
        return 0.0


def hold_before_exit(delay_s: float, sleep: Callable[[float], None] = time.sleep) -> None:
    """Log the written trace paths at WARNING and sleep ``delay_s`` s (no-op for 0).

    Called on every rank after the post-profiling barrier with the same delay, so ranks stay aligned.
    """
    if delay_s <= 0:
        return
    _logger.warning("Profiling traces ready: %s; holding %g s before exit", ", ".join(WRITTEN_TRACES)
                    or "none written on this rank", delay_s)
    sleep(delay_s)


def _remove_if_exists(path: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.remove(path)


def _export_trace(prof, path: str) -> None:
    """Write the chrome trace of ``prof`` to ``path`` with a fallback chain.

    ``torch.profiler.profile.export_chrome_trace`` always calls ``self.profiler.export_chrome_trace(path,
    self._trace_metadata)`` (torch/profiler/profiler.py), but the Habana ``habana_autograd_profile_wrapper``
    (habana_frameworks/torch/activity_profiler/activity_profiler.py) only takes ``(self, path)``, hence
    ``TypeError: ... takes 2 positional arguments but 3 were given`` for any outer call. So call the inner
    profiler directly with the path only; fall back to the outer method and to the raw kineto results.
    """
    errors = []
    inner = getattr(prof, "profiler", None)
    attempts = []
    if inner is not None and hasattr(inner, "export_chrome_trace"):
        attempts.append(("profiler.export_chrome_trace", lambda: inner.export_chrome_trace(path)))
    attempts.append(("export_chrome_trace", lambda: prof.export_chrome_trace(path)))
    kineto = getattr(inner, "kineto_results", None)
    if kineto is not None and hasattr(kineto, "save"):
        attempts.append(("kineto_results.save", lambda: kineto.save(path)))
    for name, attempt in attempts:
        _remove_if_exists(path)  # a partial file of a failed attempt must never count as a trace
        try:
            attempt()
            if os.path.exists(path):
                return
            errors.append(f"{name}: no file written")
        except Exception as exc:
            errors.append(f"{name}: {exc!r}")
    _remove_if_exists(path)
    raise RuntimeError("; ".join(errors))


# Below 70 MB so every part stays a Jenkins artifact (< 75 MB).
COPY_PART_BYTES = 64 * 1024 * 1024
_DIGIT_WORDS = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')


def rank_tag(rank: int) -> str:
    """Rank spelled in words ('one-five' for 15): Ray log dedup ignores digits, so numbers alone collapse lines."""
    return '-'.join(_DIGIT_WORDS[int(d)] for d in str(rank))


def profile_copy_dir(environ: Mapping[str, str] | None = None) -> str | None:
    """VLLM_PROFILE_COPY_DIR: directory receiving a copy of every written trace as ``*.log`` parts (unset = off)."""
    environ = os.environ if environ is None else environ
    return environ.get('VLLM_PROFILE_COPY_DIR') or None


def copy_trace_parts(path: str, copy_dir: str, rank: int, part_bytes: int = COPY_PART_BYTES) -> list[str]:
    """Copy ``path`` to ``<copy_dir>/trace_rank<R>.<suffix>.log``, split into ``.partNN.log`` pieces if larger
    than ``part_bytes``. Each piece is written as ``*.tmp`` and renamed when complete. Returns the pieces."""
    os.makedirs(copy_dir, exist_ok=True)
    base = os.path.join(copy_dir, f"trace_rank{rank}." + os.path.basename(path).split('.', 1)[1])
    size = os.path.getsize(path)
    count = max(1, -(-size // part_bytes))
    names = [base + ".log"] if count == 1 else [f"{base}.part{i:02d}.log" for i in range(count)]
    with open(path, "rb") as src:
        for name in names:
            with open(name + ".tmp", "wb") as dst:
                left = part_bytes
                while left > 0:
                    chunk = src.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    dst.write(chunk)
                    left -= len(chunk)
            os.replace(name + ".tmp", name)
    return names


def _mem_available_mb() -> int:
    with contextlib.suppress(OSError, ValueError, IndexError), open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) // 1024
    return -1


def make_trace_handler(output_dir: str, rank: int, use_gzip: bool = True, copy_dir: str | None = None) -> Callable:
    """``on_trace_ready`` callback writing ``<output_dir>/<rank>_<pid>_<ms>.pt.trace.json[.gz]``.

    ``torch.profiler.tensorboard_trace_handler`` goes through ``prof.export_chrome_trace``, which passes extra
    arguments the Habana profiler wrapper does not accept (see ``_export_trace``). This handler writes the plain
    trace through ``_export_trace`` and gzips in Python; with ``copy_dir`` (default VLLM_PROFILE_COPY_DIR) it also
    copies the result via ``copy_trace_parts``. Failures are logged, never raised.
    """
    copy_dir = profile_copy_dir() if copy_dir is None else copy_dir
    tag = rank_tag(rank)

    def handler(prof) -> None:
        path = os.path.join(output_dir, f"{rank}_{os.getpid()}_{int(time.time() * 1000)}.pt.trace.json")
        try:
            _logger.warning("Exporting profiler trace on rank %d (%s), host MemAvailable %d MB", rank, tag,
                            _mem_available_mb())
            os.makedirs(output_dir, exist_ok=True)
            _export_trace(prof, path)
            if use_gzip:
                with open(path, "rb") as src, gzip.open(path + ".gz", "wb", compresslevel=1) as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                os.remove(path)
                path += ".gz"
            WRITTEN_TRACES.append(path)
            _logger.warning("Profiler trace written on rank %d (%s): %s (%d bytes)", rank, tag, path,
                            os.path.getsize(path))
        except Exception as exc:
            _logger.warning("Cannot write the profiler trace on rank %d (%s, %s): %r", rank, tag, path, exc)
            return
        if copy_dir:
            try:
                parts = copy_trace_parts(path, copy_dir, rank)
                _logger.warning("Profiler trace copied on rank %d (%s): %s", rank, tag, ", ".join(parts))
            except Exception as exc:
                _logger.warning("Cannot copy the profiler trace on rank %d (%s) to %s: %r", rank, tag, copy_dir, exc)

    return handler


def forward_traces_to_rank0(group,
                            copy_dir: str | None = None,
                            host: str | None = None,
                            chunk_bytes: int = COPY_PART_BYTES,
                            dist=None) -> list[str]:
    """Send the last trace of every rank on another host than rank 0 to rank 0 over ``group`` (gloo), where
    it is written with ``copy_trace_parts`` into rank 0's ``copy_dir`` (default VLLM_PROFILE_COPY_DIR).

    Collective: every rank of ``group`` must call it, also ranks without a trace. Metadata is gathered first, so
    all ranks derive the same transfer plan; files go in ``chunk_bytes`` pieces. Failures are logged, never
    raised. Returns the parts written on rank 0.
    """
    import socket

    import torch
    dist = torch.distributed if dist is None else dist
    copy_dir = profile_copy_dir() if copy_dir is None else copy_dir
    host = socket.gethostname() if host is None else host
    rank = dist.get_rank()
    tag = rank_tag(rank)
    path = WRITTEN_TRACES[-1] if WRITTEN_TRACES else None
    size = 0
    with contextlib.suppress(OSError):
        size = os.path.getsize(path) if path else 0
    meta: list = [None] * dist.get_world_size(group)
    try:
        dist.all_gather_object(meta, (rank, host, size, bool(copy_dir)), group=group)
    except Exception as exc:
        _logger.warning("Cannot gather profiler trace metadata on rank %d (%s): %r", rank, tag, exc)
        return []
    root = next((m for m in meta if m[0] == 0), None)
    if root is None or not root[3]:
        return []
    senders = [(r, s) for r, h, s, _ in meta if r != 0 and h != root[1] and s > 0]
    written: list[str] = []
    for src, total in senders:
        if rank not in (0, src):
            continue
        sizes = [min(chunk_bytes, total - off) for off in range(0, total, chunk_bytes)]
        try:
            if rank == src:
                _send_trace(dist, group, path, sizes)
                _logger.warning("Profiler trace sent to rank 0 from rank %d (%s): %d bytes", rank, tag, total)
            else:
                parts = _recv_trace(dist, group, src, sizes, copy_dir)
                written += parts
                _logger.warning("Profiler trace of rank %d (%s) copied on rank 0: %s", src, rank_tag(src),
                                ", ".join(parts))
        except Exception as exc:
            _logger.warning("Cannot forward the profiler trace of rank %d (%s) to rank 0: %r", src, rank_tag(src), exc)
    return written


def _send_trace(dist, group, path: str, sizes: list[int]) -> None:
    import torch
    src = None
    with contextlib.suppress(OSError):
        src = open(path, "rb")  # noqa: SIM115
    try:
        for n in sizes:
            buf = bytearray(n)
            if src is not None:
                src.readinto(buf)  # a short read leaves zeros; the chunk count must match the receiver's
            dist.send(torch.frombuffer(buf, dtype=torch.uint8), dst=0, group=group)
    finally:
        if src is not None:
            src.close()


def _recv_trace(dist, group, src: int, sizes: list[int], copy_dir: str) -> list[str]:
    import torch
    recv_dir = os.path.join(copy_dir, "recv")
    tmp = os.path.join(recv_dir, f"{src}_recv.pt.trace.json.gz")
    dst = None
    error = None
    try:
        os.makedirs(recv_dir, exist_ok=True)
        dst = open(tmp, "wb")  # noqa: SIM115
    except OSError as exc:
        error = exc
    try:
        for n in sizes:
            buf = torch.empty(n, dtype=torch.uint8)
            dist.recv(buf, src=src, group=group)
            if dst is not None:
                dst.write(buf.numpy().tobytes())
    finally:
        if dst is not None:
            dst.close()
    try:
        if error is not None:
            raise error
        return copy_trace_parts(tmp, copy_dir, src)
    finally:
        _remove_if_exists(tmp)
        with contextlib.suppress(OSError):
            os.rmdir(recv_dir)
