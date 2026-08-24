from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

from std_msgs.msg import Bool


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from realsense_debug_selector import RealSenseDebugSelector  # noqa: E402


def _selector_harness():
    published = []
    harness = SimpleNamespace(
        ball_enabled=True,
        hoop_enabled=False,
        selected_source="ball",
        hurdle_detected=False,
        hurdle_state_time=0.0,
        state_timeout_sec=0.5,
        debug_timeout_sec=0.25,
        latest_ball_image=None,
        latest_hurdle_image=None,
        latest_hoop_image=None,
        latest_ball_image_time=0.0,
        latest_hurdle_image_time=0.0,
        latest_hoop_image_time=0.0,
        latest_raw_image=None,
        last_debug_output_time=0.0,
        _publish_and_show=published.append,
    )
    harness._active_source = MethodType(
        RealSenseDebugSelector._active_source,
        harness,
    )
    harness._fresh_debug_image = MethodType(
        RealSenseDebugSelector._fresh_debug_image,
        harness,
    )
    return harness, published


def test_ball_to_hoop_handoff_keeps_raw_output_until_new_hoop_debug():
    harness, published = _selector_harness()
    raw_during_gap = object()
    raw_waiting_for_hoop = object()
    raw_after_hoop = object()
    hoop_image = object()

    RealSenseDebugSelector.cb_ball_active(harness, Bool(data=False))
    RealSenseDebugSelector.cb_raw_image(harness, raw_during_gap)
    RealSenseDebugSelector.cb_hoop_active(harness, Bool(data=True))
    RealSenseDebugSelector.cb_raw_image(harness, raw_waiting_for_hoop)
    RealSenseDebugSelector.cb_hoop_image(harness, hoop_image)
    RealSenseDebugSelector.cb_raw_image(harness, raw_after_hoop)

    assert published == [raw_during_gap, raw_waiting_for_hoop, hoop_image]
    assert harness.selected_source == "hoop"


def test_late_ball_debug_is_ignored_after_ball_is_disabled():
    harness, _ = _selector_harness()
    stale_ball = object()

    RealSenseDebugSelector.cb_ball_active(harness, Bool(data=False))
    RealSenseDebugSelector.cb_ball_image(harness, stale_ball)

    assert harness.latest_ball_image is None
    assert harness.latest_ball_image_time == 0.0


def test_reordered_mode_callbacks_keep_the_new_true_source_sticky():
    harness, _ = _selector_harness()

    RealSenseDebugSelector.cb_hoop_active(harness, Bool(data=True))
    RealSenseDebugSelector.cb_ball_active(harness, Bool(data=False))

    assert harness.selected_source == "hoop"
    assert harness._active_source() == "hoop"


def test_each_debug_frame_is_used_once_then_live_raw_resumes():
    harness, published = _selector_harness()
    debug = object()
    raw_one = object()
    raw_two = object()

    RealSenseDebugSelector.cb_ball_image(harness, debug)
    RealSenseDebugSelector.cb_raw_image(harness, raw_one)
    RealSenseDebugSelector.cb_raw_image(harness, raw_two)

    assert published == [debug, raw_two]
