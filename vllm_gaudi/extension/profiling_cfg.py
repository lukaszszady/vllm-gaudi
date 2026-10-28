# SPDX-License-Identifier: Apache-2.0
"""Sizing of the scheduler token budget for the profile-one-synthetic-batch mode (no vLLM / Habana imports)."""

from __future__ import annotations

import gzip
import logging
import os
import random
import shutil
import time
from typing import Callable, Iterable, Mapping, Optional

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


def profile_blocks_scattered(environ: Optional[Mapping[str, str]] = None) -> bool:
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


def profile_sampling_kwargs(spec: Optional[str], model_defaults: Optional[Mapping] = None) -> Optional[dict]:
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


def profile_exit_delay_s(environ: Optional[Mapping[str, str]] = None) -> float:
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
    _logger.warning("Profiling traces ready: %s; holding %g s before exit",
                    ", ".join(WRITTEN_TRACES) or "none written on this rank", delay_s)
    sleep(delay_s)


def _remove_if_exists(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


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


def make_trace_handler(output_dir: str, rank: int, use_gzip: bool = True) -> Callable:
    """``on_trace_ready`` callback writing ``<output_dir>/<rank>_<pid>_<ms>.pt.trace.json[.gz]``.

    ``torch.profiler.tensorboard_trace_handler`` goes through ``prof.export_chrome_trace``, which passes extra
    arguments the Habana profiler wrapper does not accept (see ``_export_trace``). This handler writes the plain
    trace through ``_export_trace`` and gzips in Python. Failures are logged, never raised.
    """

    def handler(prof) -> None:
        path = os.path.join(output_dir, f"{rank}_{os.getpid()}_{int(time.time() * 1000)}.pt.trace.json")
        try:
            os.makedirs(output_dir, exist_ok=True)
            _export_trace(prof, path)
            if use_gzip:
                with open(path, "rb") as src, gzip.open(path + ".gz", "wb") as dst:
                    shutil.copyfileobj(src, dst)
                os.remove(path)
                path += ".gz"
            WRITTEN_TRACES.append(path)
            _logger.info("Profiler trace written on rank %d: %s (%d bytes)", rank, path, os.path.getsize(path))
        except Exception as exc:
            _logger.warning("Cannot write the profiler trace on rank %d (%s): %r", rank, path, exc)

    return handler
