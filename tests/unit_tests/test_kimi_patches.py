# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the Kimi-K3 patch helpers in vllm_gaudi/patches.py against a stub of the upstream linear module."""

import importlib.util
import logging
import pathlib
import sys
import types

import pytest

_PATCHES = pathlib.Path(__file__).resolve().parents[2] / "vllm_gaudi" / "patches.py"
LOGGER = "vllm.gaudi.kimi_patches"


@pytest.fixture
def patches(monkeypatch):
    # patches.py only needs ``from vllm import envs`` at import time; the rest is imported lazily by the helpers
    vllm = types.ModuleType("vllm")
    vllm.envs = types.ModuleType("vllm.envs")
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.envs", vllm.envs)
    spec = importlib.util.spec_from_file_location("hpu_patches_under_test", _PATCHES)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "_configure_situ_aliases", lambda: "exact")  # needs the whole plugin package
    return mod


def _stub_linear(monkeypatch):

    class KimiMoE:

        def __init__(self, *args, **kwargs):
            self.args = args

    class KimiDecoderLayer:

        def forward(self, x):
            return x

    original_forward = KimiDecoderLayer.forward
    mod = types.ModuleType("vllm.models.kimi_k3.amd.linear")
    mod.KimiMoE, mod.KimiDecoderLayer = KimiMoE, KimiDecoderLayer
    for name in ("vllm.models", "vllm.models.kimi_k3", "vllm.models.kimi_k3.amd"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.models.kimi_k3.amd.linear", mod)
    return mod, original_forward


def test_linear_weights_patch_installs_once_and_leaves_the_decoder_layer_alone(patches, monkeypatch, caplog):
    mod, original_forward = _stub_linear(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        patches._patch_kimi_linear_weights()
        patched_init = mod.KimiMoE.__init__
        patches._patch_kimi_linear_weights()
    assert mod.KimiMoE._patched_weights and mod.KimiMoE.__init__ is patched_init  # idempotent
    assert mod.KimiDecoderLayer.forward is original_forward  # the old pass-through wrapper is gone
    assert mod.KimiMoE(1, 2).args == (1, 2)
    assert patches._KIMI_PATCHES_INSTALLED.count("linear_weights") == 1
    assert caplog.records == []


def test_a_failed_attempt_is_logged_at_debug_only(patches, monkeypatch, caplog):
    # no vllm.models.kimi_k3 at all: the import inside the helper fails (like a circular import in an early attempt)
    monkeypatch.setitem(sys.modules, "vllm.models.kimi_k3.amd.linear", None)
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        patches._patch_kimi_linear_weights()
    (record, ) = caplog.records
    assert record.name == LOGGER and record.levelno == logging.DEBUG
    assert "linear_weights" in record.getMessage() and record.exc_info is not None
    assert "linear_weights" not in patches._KIMI_PATCHES_INSTALLED


def test_a_patch_still_missing_after_the_final_attempt_warns_once_without_traceback(patches, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "vllm.models.kimi_k3.amd.linear", None)
    patches._patch_kimi_linear_weights()  # first attempt
    patches._patch_kimi_linear_weights()  # final attempt
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        patches._kimi_patches_log_failed()
    (record, ) = caplog.records
    assert record.levelno == logging.WARNING and record.exc_info is None
    assert "'linear_weights' was not installed" in record.getMessage()


def test_a_patch_that_succeeds_in_the_final_attempt_does_not_warn(patches, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "vllm.models.kimi_k3.amd.linear", None)
    patches._patch_kimi_linear_weights()  # first attempt: circular import
    _stub_linear(monkeypatch)
    patches._patch_kimi_linear_weights()  # final attempt
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        patches._kimi_patches_log_failed()
    assert caplog.records == []
    assert "linear_weights" in patches._KIMI_PATCHES_INSTALLED


def test_a_patch_without_an_installed_marker_is_cleared_by_ok(patches, caplog):
    patches._KIMI_PATCHES_FAILED["hybrid_config"] = "ImportError: x"
    patches._kimi_patch_ok("hybrid_config")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        patches._kimi_patches_log_failed()
    assert caplog.records == [] and "hybrid_config" not in patches._KIMI_PATCHES_INSTALLED


def test_an_outer_level_failure_cleared_by_the_final_attempt_does_not_warn(patches, monkeypatch, caplog):
    attn_res = types.ModuleType("vllm_gaudi.ops.hpu_kda_pytorch")
    attn_res.attn_res_torch = lambda *args, **kwargs: None
    # first attempt: the outermost import of _patch_kimi_attn_res fails
    monkeypatch.setitem(sys.modules, "vllm_gaudi.ops.hpu_kda_pytorch", None)
    patches._patch_kimi_attn_res()
    assert "attn_res" in patches._KIMI_PATCHES_FAILED
    # final attempt: it imports; the inner upstream modules are missing, which is recorded under their own names
    monkeypatch.setitem(sys.modules, "vllm_gaudi.ops.hpu_kda_pytorch", attn_res)
    for name in ("vllm.models.kimi_k3.amd.ops.attn_res", "vllm.models.kimi_k3.amd.linear"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    for name in ("vllm.models", "vllm.models.kimi_k3", "vllm.models.kimi_k3.amd", "vllm.models.kimi_k3.amd.ops"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    patches._patch_kimi_attn_res()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        patches._kimi_patches_log_failed()
    assert caplog.records == []
    assert {"attn_res_op", "attn_res_linear"} <= set(patches._KIMI_PATCHES_INSTALLED)


def test_installed_patches_are_listed_in_one_info_line(patches, caplog):
    patches._kimi_patch_installed("a")
    patches._kimi_patch_installed("b")
    patches._kimi_patch_installed("a")
    with caplog.at_level(logging.INFO, logger=LOGGER):
        patches._kimi_patches_log_installed()
    assert [r.getMessage() for r in caplog.records] == ["Kimi-K3 patches installed: a, b"]


def _model(module, bases=()):
    cls = type("Model", bases, {"__module__": module})
    return cls()


def test_repeated_attempts_log_nothing_above_debug(patches, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        for _ in range(3):
            patches._kimi_patch_failed("kda_forward")
            patches._kimi_patch_installed("attn_res_op")
    assert caplog.records == []


def test_final_report_is_logged_once_even_if_called_twice(patches, caplog):
    patches._kimi_patch_installed("a")
    model = _model("vllm.models.kimi_k3.model")
    with caplog.at_level(logging.INFO, logger=LOGGER):
        patches.report_kimi_patches(model)
        patches.report_kimi_patches(model)
    assert [r.getMessage() for r in caplog.records] == ["Kimi-K3 patches installed: a"]


def test_a_still_missing_patch_gives_exactly_one_warning(patches, caplog):
    patches._kimi_patch_installed("a")
    patches._kimi_patch_failed("kda_forward")
    patches._kimi_patch_failed("kda_forward")
    model = _model("vllm.models.kimi_k3.model")
    with caplog.at_level(logging.INFO, logger=LOGGER):
        patches.report_kimi_patches(model)
        patches.report_kimi_patches(model)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "kda_forward" in warnings[0].getMessage()


def test_a_non_kimi_model_logs_nothing(patches, caplog):
    patches._kimi_patch_installed("a")
    patches._kimi_patch_failed("b")
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        patches.report_kimi_patches(_model("vllm.model_executor.models.llama"))
    assert caplog.records == []


def test_an_absent_rocm_runner_is_not_applicable_not_missing(patches, caplog):
    name = "moe_runner_delegation[ROCmLatentMoERunner]"
    patches._kimi_patch_failed(name)  # e.g. an earlier circular-import attempt
    patches._kimi_patch_not_applicable(name)
    patches._kimi_patch_installed("a")
    assert name not in patches._KIMI_PATCHES_FAILED
    with caplog.at_level(logging.INFO, logger=LOGGER):
        patches.report_kimi_patches(_model("vllm.models.kimi_k3.model"))
    assert [r.levelno for r in caplog.records] == [logging.INFO]


def test_kimi_subclass_is_detected_through_the_mro(patches, caplog):
    base = type(_model("vllm.models.kimi_k3.amd.model"))
    sub = _model("some.other.module", bases=(base, ))
    patches._kimi_patch_installed("a")
    with caplog.at_level(logging.INFO, logger=LOGGER):
        patches.report_kimi_patches(sub)
    assert [r.getMessage() for r in caplog.records] == ["Kimi-K3 patches installed: a"]


@pytest.mark.parametrize("module", [
    "vllm.model_executor.models.kimi_vl", "vllm.model_executor.models.kimi_k25", "vllm.model_executor.models.kimi_audio"
])
def test_other_kimi_models_get_no_k3_report(patches, caplog, module):
    patches._kimi_patch_failed("kda_forward")
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        patches.report_kimi_patches(_model(module))
    assert not [r for r in caplog.records if r.levelno > logging.DEBUG]


@pytest.mark.parametrize("exc, mod, cls, absent", [
    (ModuleNotFoundError("x", name="m.latent"), "m.latent", "ROCmLatentMoERunner", True),
    (AttributeError("no class"), "m.latent", "ROCmLatentMoERunner", True),
    (ImportError("circular"), "m.latent", "ROCmLatentMoERunner", False),
    (ModuleNotFoundError("x", name="other"), "m.latent", "ROCmLatentMoERunner", False),
    (ModuleNotFoundError("x", name="m.moe_runner"), "m.moe_runner", "MoERunner", False),
    (AttributeError("no class"), "m.moe_runner", "MoERunner", False),
])
def test_only_a_missing_rocm_runner_is_not_applicable(patches, exc, mod, cls, absent):
    assert patches._optional_runner_absent(exc, mod, cls) is absent


class _Cfg:

    def __init__(self, *archs):
        self.architectures = list(archs)


def test_get_model_failure_reports_missing_patches_once_for_kimi(patches, caplog):
    patches._kimi_patch_failed("kda_forward")
    with caplog.at_level(logging.INFO, logger=LOGGER):
        patches.report_kimi_patches_on_load_failure(_Cfg("KimiK3ForConditionalGeneration"))
        patches.report_kimi_patches_on_load_failure(_Cfg("KimiK3ForConditionalGeneration"))
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert "kda_forward" in caplog.records[0].getMessage()


def test_get_model_failure_of_another_architecture_logs_nothing(patches, caplog):
    patches._kimi_patch_failed("kda_forward")
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        patches.report_kimi_patches_on_load_failure(_Cfg("LlamaForCausalLM"))
    assert not [r for r in caplog.records if r.levelno > logging.DEBUG]


def test_config_report_warns_only_for_config_side_patches_of_a_kimi_arch(patches, caplog):
    # the #942 pattern: per-attempt failures that are irrelevant to a process that only builds the config
    for name in ("kda_forward", "kimi_gdn_forward", "linear_weights", "attn_res_op", "attn_res_linear"):
        patches._kimi_patch_failed(name)
    patches._kimi_patch_failed("k3_arch")
    with caplog.at_level(logging.INFO, logger=LOGGER):
        patches._report_kimi_config_patches(_Cfg("LlamaForCausalLM"))
        assert caplog.records == []
        patches._report_kimi_config_patches(_Cfg("KimiK3ForConditionalGeneration"))
        patches._report_kimi_config_patches(_Cfg("KimiK3ForConditionalGeneration"))
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert "k3_arch" in caplog.records[0].getMessage()


def test_config_report_is_silent_when_config_patches_are_installed(patches, caplog):
    patches._kimi_patch_failed("kda_forward")
    patches._kimi_patch_failed("k3_arch")
    patches._kimi_patch_installed("kimi_moe_2d")
    patches._kimi_patch_ok("k3_arch")
    with caplog.at_level(logging.INFO, logger=LOGGER):
        patches._report_kimi_config_patches(_Cfg("KimiK3ForConditionalGeneration"))
    assert caplog.records == []


def test_a_patch_warned_by_the_config_report_is_not_warned_again_by_the_model_report(patches, caplog):
    patches._kimi_patch_failed("k3_arch")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        patches._report_kimi_config_patches(_Cfg("KimiK3ForConditionalGeneration"))
        patches.report_kimi_patches(_model("vllm.models.kimi_k3.model"))
    assert len(caplog.records) == 1
