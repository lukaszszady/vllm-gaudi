# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from vllm_gaudi.v1.worker import hpu_affinity as aff

# Fake host: 2 NUMA nodes, 8 physical cores each with hyperthreads
# (node0: cpus 0-7 + 16-23, node1: cpus 8-15 + 24-31), 2 Habana cards per node.
CARDS = {  # pci -> (numa node, module id, accel index)
    "0000:19:00.0": (0, 3, 0),
    "0000:1a:00.0": (0, 5, 1),
    "0000:b3:00.0": (1, 0, 2),
    "0000:b4:00.0": (1, 7, 3),
}
NODE0 = set(range(8)) | set(range(16, 24))
NODE1 = set(range(8, 16)) | set(range(24, 32))


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _add_card(root, pci, node, module, index, cpulist):
    dev = root / "bus/pci/devices" / pci
    _write(dev / "vendor", "0x1da3\n")
    _write(dev / "numa_node", f"{node}\n")
    _write(dev / "local_cpulist", cpulist)
    _write(dev / "accel/accel0/module_id", f"{module}\n")
    link = root / "class/accel" / f"accel{index}" / "device"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(dev)


def _make_sysfs(root, cards=CARDS):
    for cpu in range(32):
        core = cpu % 16
        _write(root / f"devices/system/cpu/cpu{cpu}/topology/thread_siblings_list", f"{core},{core + 16}\n")
    for pci, (node, module, index) in cards.items():
        _add_card(root, pci, node, module, index, "0-7,16-23\n" if node == 0 else "8-15,24-31\n")
    _write(root / "bus/pci/devices/0000:00:01.0/vendor", "0x8086\n")  # not Habana


@pytest.fixture
def sysfs(tmp_path):
    _make_sysfs(tmp_path / "sys")
    return str(tmp_path / "sys")


@pytest.fixture
def proc(tmp_path):
    """Fake /proc with 3 threads and no open Habana node."""
    root = tmp_path / "proc"
    for tid in ("100", "101", "102"):
        (root / "self/task" / tid).mkdir(parents=True)
    (root / "self/fd").mkdir(parents=True)
    return root


def open_device(proc, index, node="accel/accel"):
    link = proc / "self/fd" / str(10 + len(list((proc / "self/fd").iterdir())))
    link.symlink_to(f"/dev/{node}{index}")


def close_devices(proc):
    for link in (proc / "self/fd").iterdir():
        link.unlink()


@pytest.fixture
def cpus(monkeypatch):
    state = {"mask": set(range(32)), "calls": [], "fail_tids": set()}
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(state["mask"]), raising=False)

    def setaff(pid, mask):
        if pid in state["fail_tids"]:
            raise PermissionError("denied")
        state["calls"].append((pid, set(mask)))
        state["mask"] = set(mask)

    monkeypatch.setattr(os, "sched_setaffinity", setaff, raising=False)
    state["threads"] = []
    monkeypatch.setattr(aff.torch, "set_num_threads", lambda n: state["threads"].append(n))
    monkeypatch.setattr(aff.torch, "get_num_threads", lambda: state["threads"][-1] if state["threads"] else 99)
    return state


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("VLLM_HPU_BIND_NUMA", "HABANA_VISIBLE_MODULES", "HL_VISIBLE_PCI_ADDRESS", "OMP_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def empty_dev(tmp_path, monkeypatch):
    """The container's /dev must not leak into the tests: an empty one makes the sysfs cards the visible ones."""
    path = tmp_path / "empty_dev"
    path.mkdir()
    monkeypatch.setattr(aff, "_DEV", str(path))
    return path


@pytest.fixture
def logs(monkeypatch):
    """vllm's logger does not propagate, so caplog cannot see it: replace the module logger."""

    class Recorder:

        def __init__(self):
            self.records = []

        def _add(self, level):
            return lambda msg, *args, **kwargs: self.records.append((level, msg % args if args else msg))

        def __getattr__(self, name):
            return self._add(name.upper())

        def of(self, level):
            return [m for lv, m in self.records if lv == level]

    rec = Recorder()
    monkeypatch.setattr(aff, "logger", rec)
    return rec


def _pinned_masks(cpus):
    return {tuple(sorted(m)) for _, m in cpus["calls"]}


def test_cpu_list_helpers():
    assert aff.parse_cpu_list("0-2,8,10-11\n") == {0, 1, 2, 8, 10, 11}
    assert aff.format_cpu_list({0, 1, 2, 8, 10, 11}) == "0-2,8,10-11"
    assert aff.format_cpu_list(set()) == "-"


def test_default_only_logs_one_info_line(sysfs, proc, cpus, logs):
    open_device(proc, 1)
    aff.apply_affinity(rank=1, local_rank=1, sysfs=sysfs, proc=str(proc))
    assert cpus["calls"] == [] and cpus["threads"] == []
    assert len(logs.records) == 1
    (line, ) = logs.of("INFO")
    assert "rank=1 local_rank=1 card_pci=0000:1a:00.0 (source: opened-device) numa_node=0 mode=0 cpus=0-31" in line
    assert "(before: 32 cpus)" in line and "torch_threads=" in line


@pytest.mark.parametrize("node", ["accel/accel", "accel/accel_controlD"])
def test_opened_device_identifies_card(sysfs, proc, node):
    open_device(proc, 3, node)
    assert aff.find_opened_card_pci(sysfs, str(proc)) == "0000:b4:00.0"


def test_opened_device_ignores_other_fds_and_missing_proc(sysfs, proc, tmp_path):
    (proc / "self/fd/3").symlink_to("/dev/null")
    (proc / "self/fd/4").symlink_to("/dev/accel/accel9")  # no such sysfs class entry
    assert aff.find_opened_card_pci(sysfs, str(proc)) is None
    assert aff.find_opened_card_pci(sysfs, str(tmp_path / "nope")) is None


def test_several_open_devices_are_not_used(sysfs, proc, logs):
    open_device(proc, 0)
    open_device(proc, 2)
    assert aff.find_opened_card_pci(sysfs, str(proc)) is None
    assert logs.of("WARNING")


def test_mode_1_binds_all_threads_to_opened_card_numa_node(sysfs, proc, cpus, monkeypatch, logs):
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "1")
    open_device(proc, 3)
    aff.apply_affinity(rank=3, local_rank=3, sysfs=sysfs, proc=str(proc))
    assert {pid for pid, _ in cpus["calls"]} == {100, 101, 102}
    assert _pinned_masks(cpus) == {tuple(sorted(NODE1))}
    assert cpus["threads"] == []  # mode 1 shares the cores between ranks: torch's default is kept
    (line, ) = logs.of("INFO")
    assert "numa_node=1 mode=1 cpus=8-15,24-31 (before: 32 cpus) threads_pinned=3/3 torch_threads=99" in line


def test_omp_num_threads_is_respected(sysfs, proc, cpus, monkeypatch):
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    open_device(proc, 0)
    aff.apply_affinity(0, 0, sysfs=sysfs, proc=str(proc))
    assert cpus["calls"] and cpus["threads"] == []


def test_per_thread_errors_are_ignored_and_counted(sysfs, proc, cpus, monkeypatch, logs):
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "1")
    cpus["fail_tids"] = {101}
    open_device(proc, 0)
    aff.apply_affinity(0, 0, sysfs=sysfs, proc=str(proc))
    assert "threads_pinned=2/3" in logs.of("INFO")[0]


def test_all_threads_failing_means_unbound(sysfs, proc, cpus, monkeypatch, logs):
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "1")
    cpus["fail_tids"] = {100, 101, 102}
    open_device(proc, 0)
    aff.apply_affinity(0, 0, sysfs=sysfs, proc=str(proc))
    assert "mode=0" in logs.of("INFO")[0] and cpus["threads"] == []


def test_mode_1_intersects_with_allowed_cpus(sysfs, proc, cpus, monkeypatch):
    cpus["mask"] = {0, 1, 2, 3, 8, 9}
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "1")
    open_device(proc, 0)
    aff.apply_affinity(0, 0, sysfs=sysfs, proc=str(proc))
    assert _pinned_masks(cpus) == {(0, 1, 2, 3)}


@pytest.mark.parametrize("index,expected", [
    (0, {0, 1, 2, 3, 16, 17, 18, 19}),
    (1, {4, 5, 6, 7, 20, 21, 22, 23}),
    (2, {8, 9, 10, 11, 24, 25, 26, 27}),
    (3, {12, 13, 14, 15, 28, 29, 30, 31}),
])
def test_core_mode_gives_disjoint_physical_core_slices(sysfs, proc, cpus, monkeypatch, index, expected):
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "core")
    open_device(proc, index)
    aff.apply_affinity(0, 0, sysfs=sysfs, proc=str(proc))
    assert _pinned_masks(cpus) == {tuple(sorted(expected))}
    assert cpus["threads"] == [4]


def test_core_mode_unequal_cards_per_numa_node(tmp_path, proc, cpus, monkeypatch):
    cards = {
        "0000:19:00.0": (0, 0, 0),
        "0000:1a:00.0": (0, 1, 1),
        "0000:1b:00.0": (0, 2, 2),
        "0000:b3:00.0": (1, 3, 3),
    }
    _make_sysfs(tmp_path / "sys", cards)
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "core")
    sets = []
    for index in range(4):
        cpus["calls"].clear()
        open_device(proc, index)
        aff.apply_affinity(0, 0, sysfs=str(tmp_path / "sys"), proc=str(proc))
        sets.append(set(cpus["calls"][0][1]))
        close_devices(proc)
        cpus["mask"] = set(range(32))
    node0 = sets[:3]
    assert set().union(*node0) == NODE0 and sum(len(s) for s in node0) == len(NODE0)  # disjoint, complete
    assert [len(s) for s in node0] == [4, 6, 6]  # 8 cores split 2/3/3 (hyperthread pairs)
    assert sets[3] == NODE1


def test_core_mode_fewer_cores_than_cards_shares_cores(tmp_path, proc, cpus, monkeypatch, logs):
    _make_sysfs(tmp_path / "sys")
    cpus["mask"] = {0, 16, 1, 17}  # pod cpuset: 2 physical cores of node 0, 2 cards on that node
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "core")
    open_device(proc, 0)
    aff.apply_affinity(0, 0, sysfs=str(tmp_path / "sys"), proc=str(proc))
    assert _pinned_masks(cpus) == {(0, 16)}
    # one core only for two cards: both cards get that core (shared), never an empty set
    cpus["mask"] = {0, 16}
    cpus["calls"].clear()
    aff.apply_affinity(0, 0, sysfs=str(tmp_path / "sys"), proc=str(proc))
    assert _pinned_masks(cpus) == {(0, 16)}
    assert any("shared" in m for m in logs.of("WARNING"))


def test_core_mode_splits_the_allowed_set_not_the_whole_node(sysfs, proc, cpus, monkeypatch):
    # pod cpuset covers only the upper half of the node-0 cores; each of the two node-0 cards must get part of it
    allowed = {4, 5, 6, 7, 20, 21, 22, 23}
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "core")
    results = []
    for index in (0, 1):
        cpus["mask"] = set(allowed)
        cpus["calls"].clear()
        open_device(proc, index)
        aff.apply_affinity(0, 0, sysfs=sysfs, proc=str(proc))
        results.append(set(cpus["calls"][0][1]))
        close_devices(proc)
    assert all(results) and results[0].isdisjoint(results[1]) and results[0] | results[1] == allowed


def test_no_overlap_with_allowed_cpus_stays_unbound(sysfs, proc, cpus, monkeypatch, logs):
    cpus["mask"] = {8, 9}  # card 0 is on node 0
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "core")
    open_device(proc, 0)
    aff.apply_affinity(0, 0, sysfs=sysfs, proc=str(proc))
    assert cpus["calls"] == [] and logs.of("WARNING") and "mode=0" in logs.of("INFO")[0]


@pytest.mark.parametrize("env", [
    {
        "HABANA_VISIBLE_MODULES": "3,5,0,7"
    },
    {
        "HL_VISIBLE_PCI_ADDRESS": "0000:19:00.0,0000:b3:00.0"
    },
    {
        "HABANA_VISIBLE_MODULES": "3,5",
        "HL_VISIBLE_PCI_ADDRESS": "0000:19:00.0,0000:b3:00.0"
    },
    {},
])
def test_env_with_several_entries_identifies_nothing(sysfs, env):
    assert aff.find_env_card_pci(sysfs, env) is None


def test_env_with_a_single_entry(sysfs):
    assert aff.find_env_card_pci(sysfs, {"HABANA_VISIBLE_MODULES": "7"}) == "0000:b4:00.0"
    assert aff.find_env_card_pci(sysfs, {"HL_VISIBLE_PCI_ADDRESS": "0000:19:00.0"}) == "0000:19:00.0"


def test_unknown_single_module_id_falls_back_to_single_pci(sysfs):
    env = {"HABANA_VISIBLE_MODULES": "42", "HL_VISIBLE_PCI_ADDRESS": "0000:b4:00.0"}
    assert aff.find_env_card_pci(sysfs, env) == "0000:b4:00.0"


def test_env_source_is_named_in_the_log(sysfs, proc, cpus, monkeypatch, logs):
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "1")
    monkeypatch.setenv("HABANA_VISIBLE_MODULES", "5")
    aff.apply_affinity(0, 0, sysfs=sysfs, proc=str(proc))  # no device node open
    assert _pinned_masks(cpus) == {tuple(sorted(NODE0))}
    assert any("taken from the environment" in m for m in logs.of("WARNING"))
    assert "(source: env)" in logs.of("INFO")[0]


def test_opened_device_wins_over_disagreeing_env(sysfs, proc, cpus, monkeypatch, logs):
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "1")
    monkeypatch.setenv("HABANA_VISIBLE_MODULES", "3")  # names the node-0 card
    open_device(proc, 3)  # but the process actually holds the node-1 card
    aff.apply_affinity(0, 0, sysfs=sysfs, proc=str(proc))
    assert _pinned_masks(cpus) == {tuple(sorted(NODE1))}
    assert any("using the opened device" in m for m in logs.of("WARNING"))


def test_unidentified_card_warns_and_stays_unbound(sysfs, proc, cpus, monkeypatch, logs):
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "1")
    aff.apply_affinity(0, 0, sysfs=sysfs, proc=str(proc))
    assert cpus["calls"] == []
    assert any("not identified" in m for m in logs.of("WARNING"))
    assert "card_pci=None" in logs.of("INFO")[0] and "mode=0" in logs.of("INFO")[0]


def test_listing_threads_failure_never_raises(sysfs, tmp_path, cpus, monkeypatch, logs):
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "1")
    monkeypatch.setenv("HABANA_VISIBLE_MODULES", "3")
    aff.apply_affinity(0, 0, sysfs=sysfs, proc=str(tmp_path / "nope"))
    assert cpus["calls"] == []
    assert any("binding failed" in m for m in logs.of("WARNING"))
    assert "mode=0" in logs.of("INFO")[-1]


def test_missing_sysfs_never_raises(tmp_path, proc, cpus, monkeypatch):
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "core")
    monkeypatch.setenv("HABANA_VISIBLE_MODULES", "3")
    aff.apply_affinity(0, 0, sysfs=str(tmp_path / "nosys"), proc=str(proc))
    assert cpus["calls"] == []


def test_unknown_mode_is_treated_as_off(sysfs, proc, cpus, monkeypatch, logs):
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "bogus")
    open_device(proc, 0)
    aff.apply_affinity(0, 0, sysfs=sysfs, proc=str(proc))
    assert cpus["calls"] == []
    assert any("unknown VLLM_HPU_BIND_NUMA" in m for m in logs.of("WARNING"))


def test_whole_node_env_list_does_not_pin_without_open_device(sysfs, proc, cpus, monkeypatch, logs):
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "core")
    monkeypatch.setenv("HABANA_VISIBLE_MODULES", "3,5,0,7")
    monkeypatch.setenv("HL_VISIBLE_PCI_ADDRESS", ",".join(CARDS))
    aff.apply_affinity(2, 2, sysfs=sysfs, proc=str(proc))
    assert cpus["calls"] == [] and cpus["threads"] == []
    assert any("not identified" in m for m in logs.of("WARNING"))
    assert "(source: none)" in logs.of("INFO")[0]


def test_mode_0_with_whole_node_env_list_emits_no_warning(sysfs, proc, cpus, monkeypatch, logs):
    monkeypatch.setenv("HABANA_VISIBLE_MODULES", "3,5,0,7")
    monkeypatch.setenv("HL_VISIBLE_PCI_ADDRESS", ",".join(CARDS))
    open_device(proc, 3)
    aff.apply_affinity(0, 0, sysfs=sysfs, proc=str(proc))
    assert logs.of("WARNING") == [] and len(logs.records) == 1
    assert cpus["calls"] == []


def test_mode_0_mismatch_is_logged_at_debug(sysfs, proc, cpus, monkeypatch, logs):
    monkeypatch.setenv("HABANA_VISIBLE_MODULES", "3")
    open_device(proc, 3)
    aff.apply_affinity(0, 0, sysfs=sysfs, proc=str(proc))
    assert any("env says card" in m for m in logs.of("DEBUG")) and logs.of("WARNING") == []


def test_pin_mode_mismatch_is_a_warning(sysfs, proc, cpus, monkeypatch, logs):
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "1")
    monkeypatch.setenv("HABANA_VISIBLE_MODULES", "3")
    open_device(proc, 3)
    aff.apply_affinity(0, 0, sysfs=sysfs, proc=str(proc))
    assert any("env says card" in m for m in logs.of("WARNING"))


@pytest.mark.parametrize("mode,level", [("0", "DEBUG"), ("1", "WARNING")])
def test_several_open_devices_log_level(sysfs, proc, cpus, monkeypatch, logs, mode, level):
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", mode)
    open_device(proc, 0)
    open_device(proc, 1)
    aff.apply_affinity(0, 0, sysfs=sysfs, proc=str(proc))
    assert any("several Habana devices" in m for m in logs.of(level))
    assert not any("several Habana devices" in m for m in logs.of("DEBUG" if level == "WARNING" else "WARNING"))


@pytest.mark.parametrize("mode,level", [("0", "DEBUG"), ("core", "WARNING")])
def test_unknown_module_id_log_level(sysfs, proc, cpus, monkeypatch, logs, mode, level):
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", mode)
    monkeypatch.setenv("HABANA_VISIBLE_MODULES", "42")
    aff.apply_affinity(0, 0, sysfs=sysfs, proc=str(proc))
    assert any("not found in sysfs" in m for m in logs.of(level))
    assert not any("not found in sysfs" in m for m in logs.of("DEBUG" if level == "WARNING" else "WARNING"))


def test_core_mode_with_empty_dev_falls_back_to_all_sysfs_cards(sysfs, proc, cpus, monkeypatch):
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "core")
    open_device(proc, 0)
    aff.apply_affinity(0, 0, sysfs=sysfs, proc=str(proc))  # _DEV is empty here
    assert len(_pinned_masks(cpus)) == 1 and len(next(iter(_pinned_masks(cpus)))) == len(NODE0) // 2


def test_partial_dev_gives_disjoint_slices_to_two_visible_cards(sysfs, proc, tmp_path, cpus, monkeypatch):
    dev = tmp_path / "dev"
    (dev / "accel").mkdir(parents=True)
    for name in ("accel0", "accel1", "accel_controlD0", "accel_controlD1"):  # same-node cards 0 and 1 only
        (dev / "accel" / name).write_text("")
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "core")
    masks = []
    for index in (0, 1):
        close_devices(proc)
        open_device(proc, index)
        cpus["calls"].clear()
        cpus["mask"] = set(range(32))  # every rank is a fresh process
        aff.apply_affinity(index, index, sysfs=sysfs, proc=str(proc), dev=str(dev))
        masks.append(_pinned_masks(cpus))
    first, second = (set(next(iter(m))) for m in masks)
    assert first and second and not first & second and first | second == set(NODE0)


def test_mode_0_mismatch_is_not_a_warning(sysfs, proc, cpus, monkeypatch, logs):
    monkeypatch.setenv("HABANA_VISIBLE_MODULES", "3")  # single entry, node 0
    open_device(proc, 3)  # opened card is on node 1
    aff.apply_affinity(0, 0, sysfs=sysfs, proc=str(proc))
    assert logs.of("WARNING") == [] and "(source: opened-device)" in logs.of("INFO")[0]


@pytest.mark.parametrize("node", ["hl", "hl_controlD"])
def test_legacy_hl_device_nodes(sysfs, proc, node):
    link = os.path.join(sysfs, "class/habanalabs/hl2/device")
    os.makedirs(os.path.dirname(link))
    os.symlink(os.path.join(sysfs, "bus/pci/devices/0000:b3:00.0"), link)
    open_device(proc, 2, f"{node}")
    assert aff.find_opened_card_pci(sysfs, str(proc)) == "0000:b3:00.0"


def test_mode_1_with_four_cards_per_node_keeps_torch_threads(tmp_path, proc, cpus, monkeypatch, logs):
    cards = {f"0000:{0x19 + i:02x}:00.0": (0, i, i) for i in range(4)}
    cards["0000:b3:00.0"] = (1, 4, 4)
    _make_sysfs(tmp_path / "sys", cards)
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "1")
    open_device(proc, 1)
    aff.apply_affinity(0, 0, sysfs=str(tmp_path / "sys"), proc=str(proc))
    assert cpus["threads"] == [] and "torch_threads=99" in logs.of("INFO")[0]


def test_core_mode_never_raises_torch_threads(sysfs, proc, cpus, monkeypatch):
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "core")
    monkeypatch.setattr(aff.torch, "get_num_threads", lambda: 2)  # fewer than the 4 cores of the slice
    open_device(proc, 0)
    aff.apply_affinity(0, 0, sysfs=sysfs, proc=str(proc))
    assert cpus["threads"] == [2]


def test_core_mode_counts_only_cards_visible_in_the_container(sysfs, proc, tmp_path, cpus, monkeypatch):
    dev = tmp_path / "dev"
    (dev / "accel").mkdir(parents=True)
    for name in ("accel0", "accel_controlD0", "accel2", "accel3"):  # card on index 1 is not mapped into the pod
        (dev / "accel" / name).write_text("")
    monkeypatch.setenv("VLLM_HPU_BIND_NUMA", "core")
    open_device(proc, 0)
    aff.apply_affinity(0, 0, sysfs=sysfs, proc=str(proc), dev=str(dev))
    assert _pinned_masks(cpus) == {tuple(sorted(NODE0))}  # alone on node 0, so it gets all its cores
