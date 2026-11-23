"""Builder runtime configuration: every declared DVC parameter must actually
change runtime behavior, and invalid values must be rejected up front.

These tests exercise the pure configuration function and verify the DVC stage
declares and passes the same parameters, so a params.yaml change is not silently
ignored.
"""

from pathlib import Path

import pytest
import yaml

from app.ml.build_config import (
    DEFAULT_LABEL_IGNORE_VALUE,
    DEFAULT_PAIR_WINDOW_DAYS,
    DEFAULT_REBUILD_STALE,
    DEFAULT_S2_CLOUD_MAX,
    DEFAULT_SAR_UNIT,
    BuildConfigError,
    build_runtime_config,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DVC_YAML = REPO_ROOT / "dvc.yaml"
PARAMS_YAML = REPO_ROOT / "params.yaml"


def test_defaults_match_historical_constants():
    config = build_runtime_config()
    assert config.sar_unit == DEFAULT_SAR_UNIT == "dB"
    assert config.s2_cloud_max == DEFAULT_S2_CLOUD_MAX == 12
    assert config.pair_window_days == DEFAULT_PAIR_WINDOW_DAYS == 1
    assert config.label_ignore_value == DEFAULT_LABEL_IGNORE_VALUE == -1.0
    assert config.rebuild_stale is DEFAULT_REBUILD_STALE is False


def test_each_parameter_changes_runtime_config():
    base = build_runtime_config()
    assert build_runtime_config(sar_unit="linear").sar_unit == "linear" != base.sar_unit
    assert build_runtime_config(s2_cloud_max=40).s2_cloud_max == 40 != base.s2_cloud_max
    assert build_runtime_config(pair_window_days=3).pair_window_days == 3 != base.pair_window_days
    assert build_runtime_config(label_ignore_value=-9.0).label_ignore_value == -9.0
    assert build_runtime_config(rebuild_stale=True).rebuild_stale is True


def test_sar_unit_aliases_and_case_normalized():
    assert build_runtime_config(sar_unit="DB").sar_unit == "dB"
    assert build_runtime_config(sar_unit="decibels").sar_unit == "dB"


@pytest.mark.parametrize("bad", ["bananas", "reflectance", ""])
def test_invalid_sar_unit_rejected(bad):
    with pytest.raises(BuildConfigError):
        build_runtime_config(sar_unit=bad)


@pytest.mark.parametrize("bad", [-1, 101, 200, "high"])
def test_invalid_cloud_threshold_rejected(bad):
    with pytest.raises(BuildConfigError):
        build_runtime_config(s2_cloud_max=bad)


@pytest.mark.parametrize("good", [0, 12, 100])
def test_valid_cloud_threshold_accepted(good):
    assert build_runtime_config(s2_cloud_max=good).s2_cloud_max == good


@pytest.mark.parametrize("bad", [-1, -5])
def test_negative_pair_window_rejected(bad):
    with pytest.raises(BuildConfigError):
        build_runtime_config(pair_window_days=bad)


def test_zero_pair_window_allowed():
    assert build_runtime_config(pair_window_days=0).pair_window_days == 0


# --------------------------------------------------------------------------- #
# Strict integer handling: no silent truncation, no booleans-as-ints
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad", [12.9, 1.9, 0.5, -0.5, "12.9", "1.9"])
def test_fractional_cloud_threshold_rejected(bad):
    with pytest.raises(BuildConfigError):
        build_runtime_config(s2_cloud_max=bad)


@pytest.mark.parametrize("bad", [1.9, 3.0001, "2.5"])
def test_fractional_pair_window_rejected(bad):
    with pytest.raises(BuildConfigError):
        build_runtime_config(pair_window_days=bad)


@pytest.mark.parametrize("value", [True, False])
def test_boolean_cloud_threshold_rejected(value):
    # bool is a subclass of int; it must NOT be silently accepted.
    with pytest.raises(BuildConfigError):
        build_runtime_config(s2_cloud_max=value)


@pytest.mark.parametrize("value", [True, False])
def test_boolean_pair_window_rejected(value):
    with pytest.raises(BuildConfigError):
        build_runtime_config(pair_window_days=value)


def test_whole_float_and_numeric_string_accepted():
    assert build_runtime_config(s2_cloud_max=12.0).s2_cloud_max == 12
    assert build_runtime_config(s2_cloud_max="12").s2_cloud_max == 12
    assert build_runtime_config(pair_window_days="3").pair_window_days == 3
    assert build_runtime_config(pair_window_days=0.0).pair_window_days == 0


@pytest.mark.parametrize("bad", ["x", None, "1.5.3"])
def test_invalid_ignore_value_rejected(bad):
    with pytest.raises(BuildConfigError):
        build_runtime_config(label_ignore_value=bad)


@pytest.mark.parametrize("bad", [0.0, 1.0])
def test_ignore_value_colliding_with_class_rejected(bad):
    with pytest.raises(BuildConfigError):
        build_runtime_config(label_ignore_value=bad)


def test_valid_ignore_values_accepted():
    assert build_runtime_config(label_ignore_value=-1.0).label_ignore_value == -1.0
    assert build_runtime_config(label_ignore_value=-9.0).label_ignore_value == -9.0
    assert build_runtime_config(label_ignore_value=255.0).label_ignore_value == 255.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "nan", "inf"])
def test_non_finite_ignore_value_rejected(bad):
    with pytest.raises(BuildConfigError):
        build_runtime_config(label_ignore_value=bad)


def test_boolean_ignore_value_rejected():
    with pytest.raises(BuildConfigError):
        build_runtime_config(label_ignore_value=True)


# --------------------------------------------------------------------------- #
# Strict boolean handling: "false" must never become True
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value", [True, "true", "TRUE", "1", "yes", "on"])
def test_rebuild_stale_true_tokens(value):
    assert build_runtime_config(rebuild_stale=value).rebuild_stale is True


@pytest.mark.parametrize("value", [False, "false", "FALSE", "0", "no", "off"])
def test_rebuild_stale_false_tokens(value):
    assert build_runtime_config(rebuild_stale=value).rebuild_stale is False


def test_rebuild_stale_string_false_does_not_become_true():
    # Regression: bool("false") is True; our validator must return False.
    assert build_runtime_config(rebuild_stale="false").rebuild_stale is False
    assert build_runtime_config(rebuild_stale="0").rebuild_stale is False


@pytest.mark.parametrize("bad", ["maybe", 2, 1.5, None, [], {}])
def test_rebuild_stale_invalid_rejected(bad):
    with pytest.raises(BuildConfigError):
        build_runtime_config(rebuild_stale=bad)


def test_effective_config_recorded_in_dict():
    config = build_runtime_config(sar_unit="linear", s2_cloud_max=40, rebuild_stale=True)
    as_dict = config.as_dict()
    assert as_dict == {
        "sar_unit": "linear",
        "s2_cloud_max": 40,
        "pair_window_days": 1,
        "label_ignore_value": -1.0,
        "rebuild_stale": True,
    }


# --------------------------------------------------------------------------- #
# DVC stage <-> runtime consistency
# --------------------------------------------------------------------------- #

def _dvc_build_stage():
    stage = yaml.safe_load(DVC_YAML.read_text(encoding="utf-8"))["stages"]["build_hakaluki_v04_dataset"]
    return stage


def test_dvc_yaml_and_params_yaml_exist():
    assert DVC_YAML.exists(), "dvc.yaml missing"
    assert PARAMS_YAML.exists(), "params.yaml missing"


def _declared_build_params(stage) -> set[str]:
    """Collect the build.* params a DVC stage lists (list or params.yaml dict)."""
    declared: set[str] = set()
    for entry in stage["params"]:
        if isinstance(entry, str):
            declared.add(entry)
        elif isinstance(entry, dict):
            for value in entry.values():
                declared.update(value if isinstance(value, list) else [value])
    return declared


def test_dvc_declares_all_runtime_parameters():
    stage = _dvc_build_stage()
    declared = _declared_build_params(stage)
    expected = {
        "build.sar_unit",
        "build.s2_cloud_max",
        "build.pair_window_days",
        "build.label_ignore_value",
        "build.rebuild_stale",
    }
    assert expected <= declared, f"dvc stage is missing params: {expected - declared}"


def test_params_yaml_defines_all_build_keys():
    params = yaml.safe_load(PARAMS_YAML.read_text(encoding="utf-8"))
    build = params["build"]
    for key in ("sar_unit", "s2_cloud_max", "pair_window_days", "label_ignore_value", "rebuild_stale"):
        assert key in build, f"params.yaml build.{key} missing"


def test_dvc_cmd_uses_stable_wrapper_not_templating():
    """The stage must not rely on DVC per-parameter command templating (fragile
    for boolean flags). It calls the single wrapper script instead."""
    stage = _dvc_build_stage()
    cmd = stage["cmd"]
    flat = " ".join(str(part) for part in cmd) if isinstance(cmd, list) else str(cmd)
    assert "dvc_build_dataset.py" in flat
    assert "${build." not in flat
    # A dict/arg-form command part must not be used.
    assert not any(isinstance(part, dict) for part in cmd) if isinstance(cmd, list) else True


def test_dvc_stage_deps_include_wrapper_and_builder():
    stage = _dvc_build_stage()
    deps = set(stage["deps"])
    assert "backend/scripts/dvc_build_dataset.py" in deps
    assert "backend/scripts/build_hakaluki_v04_dataset.py" in deps


# --------------------------------------------------------------------------- #
# Expanded-command verification for both boolean values (reproducible without DVC)
# --------------------------------------------------------------------------- #

def _wrapper_argv(rebuild_stale):
    from scripts.dvc_build_dataset import build_argv

    build = {
        "sar_unit": "dB",
        "s2_cloud_max": 12,
        "pair_window_days": 1,
        "label_ignore_value": -1.0,
        "rebuild_stale": rebuild_stale,
    }
    return build_argv(build, out_dir="backend/data/hakaluki_v04_dataset", version="0.4.0",
                      aoi="backend/data/hakaluki_aoi.geojson")


def test_expanded_command_rebuild_stale_false_omits_flag():
    argv = _wrapper_argv(False)
    assert "--rebuild-stale" not in argv
    assert "--sar-unit" in argv and "dB" in argv
    assert "--s2-cloud-max" in argv and "12" in argv
    assert "--pair-window-days" in argv and "1" in argv
    assert "--label-ignore-value" in argv and "-1.0" in argv


def test_expanded_command_rebuild_stale_true_adds_flag():
    argv = _wrapper_argv(True)
    assert "--rebuild-stale" in argv
    assert argv.count("--rebuild-stale") == 1


def test_expanded_command_parses_with_builder_argparse():
    """The generated argv must be accepted by the builder's argparse and yield
    the intended runtime configuration."""
    import importlib

    builder = importlib.import_module("scripts.build_hakaluki_v04_dataset")
    for value in (False, True):
        ns = builder._parse_args(_wrapper_argv(value))
        assert ns.sar_unit == "dB"
        assert ns.s2_cloud_max == 12
        assert ns.pair_window_days == 1
        assert ns.label_ignore_value == -1.0
        assert ns.rebuild_stale is value


def test_wrapper_reads_params_yaml_and_validates(tmp_path):
    from scripts.dvc_build_dataset import load_build_params

    good = tmp_path / "params.yaml"
    good.write_text(
        "build:\n  sar_unit: dB\n  s2_cloud_max: 12\n  pair_window_days: 1\n"
        "  label_ignore_value: -1.0\n  rebuild_stale: false\n",
        encoding="utf-8",
    )
    build = load_build_params(good)
    assert build["sar_unit"] == "dB"

    bad = tmp_path / "bad.yaml"
    bad.write_text("build:\n  sar_unit: bananas\n", encoding="utf-8")
    with pytest.raises(BuildConfigError):
        load_build_params(bad)


def test_wrapper_rejects_missing_build_section(tmp_path):
    from scripts.dvc_build_dataset import load_build_params

    p = tmp_path / "params.yaml"
    p.write_text("train:\n  seed: 42\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_build_params(p)
