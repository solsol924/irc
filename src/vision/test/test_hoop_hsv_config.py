from pathlib import Path
import sys
from types import SimpleNamespace

import yaml


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from hoop_vision import HoopVisionNode  # noqa: E402
from realsense_hsv_calibrator import (  # noqa: E402
    ProfileStore,
    hoop_ros_parameters,
)


class _Logger:
    def __init__(self):
        self.info_messages = []
        self.warning_messages = []

    def info(self, message):
        self.info_messages.append(message)

    def warning(self, message):
        self.warning_messages.append(message)


def test_wrapping_red_profile_is_exported_as_two_hue_ranges():
    values = hoop_ros_parameters(
        {
            "h_low": 165,
            "h_high": 8,
            "s_low": 91,
            "v_low": 72,
        },
        {
            "s_high": 63,
            "v_low": 104,
        },
    )

    assert values == {
        "red_h1_low": 0,
        "red_h1_high": 8,
        "red_h2_low": 165,
        "red_h2_high": 179,
        "red_s_low": 91,
        "red_v_low": 72,
        "white_s_high": 63,
        "white_v_low": 104,
    }


def test_non_wrapping_red_profile_does_not_open_an_unrelated_hue():
    values = hoop_ros_parameters(
        {"h_low": 2, "h_high": 12, "s_low": 80, "v_low": 60},
        {"s_high": 80, "v_low": 80},
    )

    assert values["red_h1_low"] == 2
    assert values["red_h1_high"] == 12
    assert values["red_h2_low"] == 2
    assert values["red_h2_high"] == 12


def test_profile_store_replaces_legacy_hoop_with_red_and_white(tmp_path):
    profile_path = tmp_path / "hsv_profiles.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "version": 3,
                "profiles": {"hoop": {"h_low": 8, "h_high": 60}},
            }
        ),
        encoding="utf-8",
    )

    store = ProfileStore(profile_path)

    assert "hoop" not in store.data["profiles"]
    assert store.get("hoop_red")["h_low"] == 160
    assert store.get("hoop_red")["h_high"] == 10
    assert store.get("hoop_white")["s_high"] == 80
    assert store.data["version"] == 4


def test_hoop_node_loads_exported_yaml(tmp_path):
    config_path = tmp_path / "hoop_hsv.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "hoop_vision": {
                    "ros__parameters": {
                        "red_h1_low": 0,
                        "red_h1_high": 7,
                        "red_h2_low": 168,
                        "red_h2_high": 179,
                        "red_s_low": 93,
                        "red_v_low": 71,
                        "white_s_high": 65,
                        "white_v_low": 102,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    logger = _Logger()
    harness = SimpleNamespace(get_logger=lambda: logger)

    values = HoopVisionNode._load_hsv_defaults(harness, config_path)

    assert values["red_h2_low"] == 168
    assert values["red_s_low"] == 93
    assert values["white_s_high"] == 65
    assert values["white_v_low"] == 102
    assert logger.warning_messages == []
