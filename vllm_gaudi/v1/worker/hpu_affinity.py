###############################################################################
# Copyright (C) 2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
"""Optional CPU/NUMA affinity of an HPU worker process.

Ray workers start without any CPU affinity, so the host Python of a rank may
run on a different NUMA node than its card. ``VLLM_HPU_BIND_NUMA`` pins the
process (and every thread created afterwards, as they inherit the mask) to:

* ``1``    - all CPUs of the NUMA node the card is attached to,
* ``core`` - a disjoint slice of the physical cores of that node (hyperthread
  siblings included); the node's cores are split evenly among the cards
  attached to it, so ranks do not compete for the same cores,
* ``0``    - (default) no binding; the affinity is only logged.

The card is identified without opening the device: ``HABANA_VISIBLE_MODULES``
(module id, set per worker by Ray / the platform) is mapped to a PCI address
through the habanalabs sysfs attributes; ``HL_VISIBLE_PCI_ADDRESS`` is the
fallback. Nothing here raises: on any failure the process stays unbound.
"""

import contextlib
import glob
import os

from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()

_SYSFS = "/sys"
_HABANA_VENDOR = "0x1da3"
_MODES = ("0", "1", "core")


def bind_mode() -> str:
    value = os.environ.get("VLLM_HPU_BIND_NUMA", "0").strip().lower()
    if value in ("", "0", "false", "off", "no"):
        return "0"
    if value in ("1", "true", "on", "yes"):
        return "1"
    if value == "core":
        return "core"
    logger.warning("HPU affinity: unknown VLLM_HPU_BIND_NUMA=%r (expected 0, 1 or core), not binding", value)
    return "0"


def parse_cpu_list(text: str) -> set[int]:
    """Parses a sysfs cpulist such as ``0-3,8,10-11``."""
    cpus: set[int] = set()
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def format_cpu_list(cpus) -> str:
    """Compact form of a CPU set, e.g. ``0-27,112-139``."""
    ordered = sorted(cpus)
    if not ordered:
        return "-"
    ranges = []
    start = prev = ordered[0]
    for cpu in ordered[1:]:
        if cpu != prev + 1:
            ranges.append((start, prev))
            start = cpu
        prev = cpu
    ranges.append((start, prev))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _habana_cards(sysfs: str) -> dict[str, int | None]:
    """Maps PCI address -> module id for all Habana cards of the host."""
    cards: dict[str, int | None] = {}
    for dev in sorted(glob.glob(os.path.join(sysfs, "bus/pci/devices/*"))):
        if _read(os.path.join(dev, "vendor")) != _HABANA_VENDOR:
            continue
        pci = os.path.basename(dev)
        cards[pci] = None
        for attr in glob.glob(os.path.join(dev, "accel/accel*/module_id")) + [os.path.join(dev, "module_id")]:
            text = _read(attr)
            if text is not None:
                with contextlib.suppress(ValueError):
                    cards[pci] = int(text, 0)
                break
    return cards


def find_card_pci(local_rank: int, sysfs: str = _SYSFS, environ=None) -> str | None:
    """PCI address of the card used by this process, without opening it."""
    env = os.environ if environ is None else environ
    cards = _habana_cards(sysfs)
    modules = [m.strip() for m in env.get("HABANA_VISIBLE_MODULES", "").split(",") if m.strip()]
    if modules:
        # Ray gives each worker its own card; with a list use local_rank.
        idx = local_rank if 0 <= local_rank < len(modules) and len(modules) > 1 else 0
        try:
            module_id = int(modules[idx])
        except ValueError:
            module_id = None
        for pci, mid in cards.items():
            if mid is not None and mid == module_id:
                return pci
    pcis = [p.strip() for p in env.get("HL_VISIBLE_PCI_ADDRESS", "").split(",") if p.strip()]
    if pcis:
        if len(pcis) == 1:
            return pcis[0]
        return pcis[local_rank] if 0 <= local_rank < len(pcis) else None
    return None


def _numa_node(pci: str, sysfs: str) -> int | None:
    text = _read(os.path.join(sysfs, "bus/pci/devices", pci, "numa_node"))
    try:
        node = int(text) if text is not None else -1
    except ValueError:
        return None
    return node if node >= 0 else None


def _node_cpus(pci: str, node: int | None, sysfs: str) -> set[int]:
    text = _read(os.path.join(sysfs, "bus/pci/devices", pci, "local_cpulist"))
    if not text and node is not None:
        text = _read(os.path.join(sysfs, "devices/system/node", f"node{node}", "cpulist"))
    return parse_cpu_list(text) if text else set()


def _physical_cores(cpus: set[int], sysfs: str) -> list[list[int]]:
    """Groups ``cpus`` into physical cores (thread siblings), ordered by first cpu."""
    seen: set[int] = set()
    cores: list[list[int]] = []
    for cpu in sorted(cpus):
        if cpu in seen:
            continue
        text = _read(os.path.join(sysfs, f"devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"))
        siblings = (parse_cpu_list(text) if text else {cpu}) & cpus
        siblings.add(cpu)
        seen |= siblings
        cores.append(sorted(siblings))
    return cores


def _core_slice(pci: str, node: int | None, node_cpus: set[int], sysfs: str) -> set[int]:
    """The share of the node's physical cores that belongs to this card."""
    peers = [p for p in _habana_cards(sysfs) if _numa_node(p, sysfs) == node]
    if pci not in peers:
        peers.append(pci)
    peers.sort()
    cores = _physical_cores(node_cpus, sysfs)
    n, k = len(peers), peers.index(pci)
    if len(cores) < n:
        raise RuntimeError(f"only {len(cores)} physical cores for {n} cards on NUMA node {node}")
    lo, hi = k * len(cores) // n, (k + 1) * len(cores) // n
    return {cpu for core in cores[lo:hi] for cpu in core}


def apply_affinity(rank: int, local_rank: int, sysfs: str = _SYSFS) -> None:
    """Logs (and, if enabled, sets) the CPU affinity of this process. Never raises."""
    mode = "0"
    pci: str | None = None
    node: int | None = None
    before: set[int] = set()
    try:
        mode = bind_mode()
        before = set(os.sched_getaffinity(0))
        target: set[int] | None = None
        pci = find_card_pci(local_rank, sysfs)
        if pci is None:
            if mode != "0":
                logger.warning(
                    "HPU affinity: card of local_rank=%s not identified (HABANA_VISIBLE_MODULES=%r, "
                    "HL_VISIBLE_PCI_ADDRESS=%r), not binding", local_rank, os.environ.get("HABANA_VISIBLE_MODULES"),
                    os.environ.get("HL_VISIBLE_PCI_ADDRESS"))
                mode = "0"
        else:
            node = _numa_node(pci, sysfs)
            if mode != "0":
                node_cpus = _node_cpus(pci, node, sysfs)
                if mode == "core":
                    node_cpus = _core_slice(pci, node, node_cpus, sysfs)
                target = node_cpus & before
                if not target:
                    logger.warning("HPU affinity: no allowed CPU on the card's NUMA node (%s), not binding", node)
                    target = None
                    mode = "0"
        if target:
            os.sched_setaffinity(0, target)
    except Exception as exc:
        logger.warning("HPU affinity: binding failed (%s: %s), continuing unbound", type(exc).__name__, exc)
        mode = "0"
    try:
        now = set(os.sched_getaffinity(0))
    except Exception:
        now = before
    logger.info("HPU affinity: rank=%s local_rank=%s card_pci=%s numa_node=%s mode=%s cpus=%s (before: %d cpus)", rank,
                local_rank, pci, node, mode, format_cpu_list(now), len(before))
