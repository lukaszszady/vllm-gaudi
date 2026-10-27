# SPDX-License-Identifier: Apache-2.0
import logging
import os

import pytest

from vllm_gaudi.v1.worker import hpu_affinity as aff

# Fake host: 2 NUMA nodes, 8 physical cores each with hyperthreads
# (node0: cpus 0-7 + 16-23, node1: cpus 8-15 + 24-31), 2 Habana cards per node.
CARDS = {  # pci -> (numa node, module id)
    "0000:19:00.0": (0, 3),
    "0000:1a:00.0": (0, 5),
    "0000:b3:00.0": (1, 0),
    "0000:b4:00.0": (1, 7),
}


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def sysfs(tmp_path):
    for cpu in range(32):
        core = cpu % 16
        _write(tmp_path / f"devices/system/cpu/cpu{cpu}/topology/thread_siblings_list", f"{core},{core + 16}\n")
    for pci, (node, module) in CARDS.items():
        dev = tmp_path / "bus/pci/devices" / pci
        _write(dev / "vendor", "0x1da3\n")
        _write(dev / "numa_node", f"{node}\n")
        _write(dev / "local_cpulist", "0-7,16-23\n" if node == 0 else "8-15,24-31\n")
        _write(dev / "accel/accel0/module_id", f"{module}\n")
    # a non-Habana device must be ignored
    _write(tmp_path / "bus/pci/devices/0000:00:01.0/vendor", "0x8086\n")
    return str(tmp_path)


@pytest.fixture
def cpus(monkeypatch):
    state = {"mask": set(range(32)), "set_calls": []}
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(state["mask"]), raising=False)

    def setaff(pid, mask):
        state["set_calls"].append(set(mask))
        state["mask"] = set(mask)

    monkeypatch.setattr(os, "sched_setaffinity", setaff, raising=False)
    return state


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("VLLM_HPU_BIND_NUMA", "HABANA_VISIBLE_MODULES", "HL_VISIBLE_PCI_ADDRESS"):
        monkeypatch.delenv(name, raising=False)


def _info(caplog):
    return [r.getMessage() for r in caplog.records if "HPU affinity" in r.getMessage()]


def test_cpu_list_helpers():
    assert aff.parse_cpu_list("0-2,8,10-11\n") == {0, 1, 2, 8, 10, 11}
    assert aff.format_cpu_list({0, 1, 2, 8, 10, 11}) == "0-2,8,10-11"
    assert aff.format_cpu_list(set()) == "-"


def test_default_logs_but_does_not_bind(sysfs, cpus, monkeypatch, caplog):
    monkeypatch.setenv("HABANA_VISIBLE_MODULES", "5")
    with caplog.at_level(logging.INFO, logger="vllm"):
        aff.apply_affinity(rank=1, local_rank=1, sysfs=sysfs)
    assert cpus["set_calls"] == []
    (line, ) = _info(caplog)
    assert "rank=1 local_rank=1 card_pci=0000:1a:00.0 numa_node=0 mode=0 cpus=0-31 (before: 32 cpus)" in line


def test_mode_1_binds_to_card_numa_node(sysfs, cpus, monkeypatch, caplog):
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "1")
    monkeypatch.setenv("HABANA_VISIBLE_MODULES", "7")
    with caplog.at_level(logging.INFO, logger="vllm"):
        aff.apply_affinity(rank=3, local_rank=3, sysfs=sysfs)
    assert cpus["set_calls"] == [set(range(8, 16)) | set(range(24, 32))]
    assert "numa_node=1 mode=1 cpus=8-15,24-31 (before: 32 cpus)" in _info(caplog)[0]


def test_mode_1_intersects_with_allowed_cpus(sysfs, cpus, monkeypatch):
    cpus["mask"] = {0, 1, 2, 3, 8, 9}
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "1")
    monkeypatch.setenv("HABANA_VISIBLE_MODULES", "3")
    aff.apply_affinity(0, 0, sysfs=sysfs)
    assert cpus["set_calls"] == [{0, 1, 2, 3}]


@pytest.mark.parametrize("module,expected", [
    ("3", {0, 1, 2, 3, 16, 17, 18, 19}),
    ("5", {4, 5, 6, 7, 20, 21, 22, 23}),
    ("0", {8, 9, 10, 11, 24, 25, 26, 27}),
    ("7", {12, 13, 14, 15, 28, 29, 30, 31}),
])
def test_core_mode_gives_disjoint_physical_core_slices(sysfs, cpus, monkeypatch, module, expected):
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "core")
    monkeypatch.setenv("HABANA_VISIBLE_MODULES", module)
    aff.apply_affinity(0, 0, sysfs=sysfs)
    assert cpus["set_calls"] == [expected]


def test_card_found_by_pci_address_fallback(sysfs, cpus, monkeypatch):
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "1")
    monkeypatch.setenv("HL_VISIBLE_PCI_ADDRESS", "0000:19:00.0,0000:b3:00.0")
    aff.apply_affinity(1, 1, sysfs=sysfs)
    assert cpus["set_calls"] == [set(range(8, 16)) | set(range(24, 32))]


def test_unidentified_card_warns_and_stays_unbound(sysfs, cpus, monkeypatch, caplog):
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "1")
    with caplog.at_level(logging.INFO, logger="vllm"):
        aff.apply_affinity(0, 0, sysfs=sysfs)
    assert cpus["set_calls"] == []
    assert any(r.levelno == logging.WARNING and "not identified" in r.getMessage() for r in caplog.records)
    assert "card_pci=None" in _info(caplog)[-1] and "mode=0" in _info(caplog)[-1]


def test_setaffinity_failure_never_raises(sysfs, cpus, monkeypatch, caplog):
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "1")
    monkeypatch.setenv("HABANA_VISIBLE_MODULES", "3")

    def boom(pid, mask):
        raise PermissionError("denied")

    monkeypatch.setattr(os, "sched_setaffinity", boom, raising=False)
    with caplog.at_level(logging.INFO, logger="vllm"):
        aff.apply_affinity(0, 0, sysfs=sysfs)
    assert any(r.levelno == logging.WARNING and "binding failed" in r.getMessage() for r in caplog.records)
    assert "mode=0" in _info(caplog)[-1]


def test_missing_sysfs_never_raises(tmp_path, cpus, monkeypatch):
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "core")
    monkeypatch.setenv("HABANA_VISIBLE_MODULES", "3")
    aff.apply_affinity(0, 0, sysfs=str(tmp_path))
    assert cpus["set_calls"] == []


def test_unknown_mode_is_treated_as_off(sysfs, cpus, monkeypatch):
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "bogus")
    monkeypatch.setenv("HABANA_VISIBLE_MODULES", "3")
    aff.apply_affinity(0, 0, sysfs=sysfs)
    assert cpus["set_calls"] == []
