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


def test_rebuild_stale_boolean_coerced():
    assert build_runtime_config(rebuild_stale=1).rebuild_stale is True
    assert build_runtime_config(rebuild_stale=0).rebuild_stale is False


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


def test_dvc_declares_all_runtime_parameters():
    stage = _dvc_build_stage()
    declared = set(stage["params"])
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


def test_dvc_cmd_passes_each_parameter_to_script():
    stage = _dvc_build_stage()
    cmd = stage["cmd"]
    # Flatten list/dict command parts into a single string for inspection.
    flat = " ".join(str(part) for part in cmd) if isinstance(cmd, list) else str(cmd)
    assert "build_hakaluki_v04_dataset.py" in flat
    assert "--sar-unit ${build.sar_unit}" in flat
    assert "--s2-cloud-max ${build.s2_cloud_max}" in flat
    assert "--pair-window-days ${build.pair_window_days}" in flat
    assert "--label-ignore-value ${build.label_ignore_value}" in flat


def test_dvc_rebuild_stale_is_conditional_flag():
    """rebuild_stale must be a conditional flag (present only when true), not a
    valued argument that would break argparse store_true."""
    stage = _dvc_build_stage()
    cmd = stage["cmd"]
    flag_entries = [part for part in cmd if isinstance(part, dict) and part.get("arg") == "--rebuild-stale"]
    assert len(flag_entries) == 1, cmd
    assert flag_entries[0]["value"] == "${build.rebuild_stale}"
