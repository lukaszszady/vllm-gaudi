# SPDX-License-Identifier: Apache-2.0
"""Release the host page cache of checkpoint shards while the model weights are loaded.

The default loader reads every *.safetensors shard through mmap, so the page cache keeps all pages that were read
(for a 1.5 TB checkpoint ~700 GB per node with EP). That memory is reclaimable, but MemFree stays low until the
kernel needs it, and the Habana PyTorch bridge treats MemTotal - MemFree above
PT_HPU_HOST_MEMORY_THRESHOLD_PERCENT (90 by default) as host memory pressure and evicts its recipe cache, so every
graph is recompiled afterwards. The weights live on the HPU after loading and the shards are not read again, so
their cached pages are dropped with posix_fadvise(POSIX_FADV_DONTNEED) as soon as the loader is done with them.

Environment:
  VLLM_HPU_DROP_WEIGHT_PAGE_CACHE      1 (default) drops the cache of consumed shards, 0 keeps the upstream loader
  VLLM_HPU_DROP_WEIGHT_PAGE_CACHE_LAG  shards kept cached behind the current one (default 4). The ranks of a node
                                       read the same shards at about the same time; the lag lets slower ranks still
                                       hit the cache instead of re-reading a shard that a faster rank dropped.
"""
import contextlib
import functools
import os
from collections.abc import Callable, Generator, Iterator

import torch
from tqdm.auto import tqdm

from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()

_ENV_DROP = "VLLM_HPU_DROP_WEIGHT_PAGE_CACHE"
_ENV_LAG = "VLLM_HPU_DROP_WEIGHT_PAGE_CACHE_LAG"
_DEFAULT_LAG = 4
_LOG_EVERY_SHARDS = 16
# Strategies whose iteration state does not cross shard boundaries, so the iterator can be called once per shard.
# "torchao" carries leftover tensors from one shard to the next and keeps the upstream iterator.
_PER_SHARD_STRATEGIES = (None, "lazy", "eager", "prefetch")
_MEMINFO_KEYS = ("MemTotal", "MemFree", "MemAvailable", "Cached", "Shmem", "Active(file)", "Inactive(file)")


def drop_enabled() -> bool:
    return os.getenv(_ENV_DROP, "1").strip().lower() not in ("0", "false", "no", "off")


def _lag() -> int:
    value = os.getenv(_ENV_LAG, "")
    try:
        return max(0, int(value)) if value else _DEFAULT_LAG
    except ValueError:
        logger.warning("%s=%r is not an integer, using %d", _ENV_LAG, value, _DEFAULT_LAG)
        return _DEFAULT_LAG


def host_memory_line() -> str:
    """Host memory figures (MiB) that matter for the bridge threshold, plus the RSS of this process."""
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in _MEMINFO_KEYS:
                    values[key] = int(rest.split()[0]) // 1024
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    values["VmRSS"] = int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError) as exc:
        return f"meminfo not available: {exc}"
    total, free = values.get("MemTotal", 0), values.get("MemFree", 0)
    used_pct = 100.0 * (total - free) / total if total else 0.0
    return " ".join(f"{k}={v}" for k, v in values.items()) + f" used_pct(MemTotal-MemFree)={used_pct:.1f}"


def drop_file_page_cache(path: str) -> bool:
    """Ask the kernel to forget the cached, unmapped pages of a file. Pages still mapped elsewhere stay."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as exc:
        logger.debug("page cache of %s not dropped: %s", path, exc)
        return False
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return True
    except OSError as exc:
        logger.debug("page cache of %s not dropped: %s", path, exc)
        return False
    finally:
        os.close(fd)


def wrap_safetensors_iterator(original: Callable[..., Iterator[tuple[str, torch.Tensor]]]):
    """Return a drop-in replacement of safetensors_weights_iterator that drops the cache of consumed shards."""
    from vllm.model_executor.model_loader import weight_utils

    sort_key = getattr(weight_utils, "_natural_sort_key", None)
    use_tqdm = getattr(weight_utils, "enable_tqdm", lambda flag: flag)
    bar_format = getattr(weight_utils, "_BAR_FORMAT", None)

    @functools.wraps(original)
    def iterator(hf_weights_files: list[str],
                 use_tqdm_on_load: bool,
                 safetensors_load_strategy: str | None = None,
                 *args,
                 **kwargs) -> Generator[tuple[str, torch.Tensor], None, None]:
        if safetensors_load_strategy not in _PER_SHARD_STRATEGIES:
            yield from original(hf_weights_files, use_tqdm_on_load, safetensors_load_strategy, *args, **kwargs)
            return
        files = sorted(hf_weights_files, key=sort_key) if sort_key else sorted(hf_weights_files)
        lag = _lag()
        logger.info("Weight page cache: dropping shards after use (lag %d of %d shards); before: %s", lag, len(files),
                    host_memory_line())
        dropped = failed = 0
        started = 0  # shards opened so far
        next_drop = 0  # first shard whose cache is not dropped yet
        try:
            for i, st_file in enumerate(
                    tqdm(files,
                         desc="Loading safetensors checkpoint shards",
                         disable=not use_tqdm(use_tqdm_on_load),
                         bar_format=bar_format)):
                started = i + 1
                # One shard per call: the shard's safe_open handle (and its mmap) is closed when the call ends.
                yield from original([st_file], False, safetensors_load_strategy, *args, **kwargs)
                while next_drop <= i - lag:
                    ok = drop_file_page_cache(files[next_drop])
                    dropped, failed, next_drop = dropped + ok, failed + (not ok), next_drop + 1
                if started % _LOG_EVERY_SHARDS == 0:
                    logger.info("Weight page cache: %d/%d shards loaded: %s", started, len(files), host_memory_line())
        finally:
            # Also runs when loading stops early, so no opened shard stays cached because of this process.
            for st_file in files[next_drop:started]:
                ok = drop_file_page_cache(st_file)
                dropped, failed = dropped + ok, failed + (not ok)
            logger.info("Weight page cache: dropped %d shards (%d failed); after: %s", dropped, failed,
                        host_memory_line())

    return iterator


@contextlib.contextmanager
def drop_weight_page_cache_during_load():
    """Use the page-cache-dropping iterator in vLLM's default loader for the duration of the block."""
    if not drop_enabled():
        yield
        return
    try:
        from vllm.model_executor.model_loader import default_loader
        original = default_loader.safetensors_weights_iterator
    except (ImportError, AttributeError) as exc:
        logger.warning("Weight page cache drop disabled, default loader not patchable: %s", exc)
        yield
        return
    default_loader.safetensors_weights_iterator = wrap_safetensors_iterator(original)
    try:
        yield
    finally:
        default_loader.safetensors_weights_iterator = original
