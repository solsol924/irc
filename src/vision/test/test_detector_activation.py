from collections import deque
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

from std_msgs.msg import Bool


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from ball_vision_fusion import BallVisionFusionNode  # noqa: E402
from hoop_vision import HoopVisionNode  # noqa: E402


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


class _Publisher:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def publish(self, msg):
        self.events.append((self.name, bool(msg.data)))


def test_inactive_ball_callbacks_skip_image_and_webcam_processing():
    harness = SimpleNamespace(ball_detection_active=False)

    BallVisionFusionNode.cb_realsense_images(harness, object(), object())
    BallVisionFusionNode.cb_webcam_state(harness, object())


def test_ball_activity_switch_keeps_warm_subscriptions_and_clears_state():
    logger = _Logger()
    color_sub = object()
    depth_sub = object()
    sync = object()
    reset_calls = []
    harness = SimpleNamespace(
        ball_detection_active=True,
        latest_realsense={"realsense_ball_detected": True},
        latest_realsense_time=1.0,
        last_realsense_detection={"realsense_ball_detected": True},
        realsense_lost_frames=0,
        realsense_hold_frames=3,
        latest_webcam={"webcam_ball_detected": True},
        latest_webcam_time=1.0,
        rs_color_sub=color_sub,
        rs_depth_sub=depth_sub,
        rs_sync=sync,
        get_logger=lambda: logger,
        ball_status_publisher=SimpleNamespace(
            _reset_webcam_detection_cycle=lambda: reset_calls.append(True)
        ),
    )
    harness._clear_ball_detection_state = MethodType(
        BallVisionFusionNode._clear_ball_detection_state,
        harness,
    )

    BallVisionFusionNode.cb_ball_active(harness, Bool(data=False))
    assert harness.ball_detection_active is False
    assert harness.latest_realsense is None
    assert harness.latest_webcam is None
    assert (harness.rs_color_sub, harness.rs_depth_sub, harness.rs_sync) == (
        color_sub,
        depth_sub,
        sync,
    )

    BallVisionFusionNode.cb_ball_active(harness, Bool(data=True))
    assert harness.ball_detection_active is True
    assert reset_calls == [True]
    assert (harness.rs_color_sub, harness.rs_depth_sub, harness.rs_sync) == (
        color_sub,
        depth_sub,
        sync,
    )


def test_hoop_activity_switch_keeps_warm_subscriptions_and_clears_history():
    logger = _Logger()
    color_sub = object()
    depth_sub = object()
    sync = object()
    published = []
    harness = SimpleNamespace(
        active=True,
        history=deque([{"detected": True}], maxlen=5),
        last_detection={"detected": True},
        last_detection_time=1.0,
        color_sub=color_sub,
        depth_sub=depth_sub,
        sync=sync,
        get_logger=lambda: logger,
        _publish_state=lambda **kwargs: published.append(kwargs),
    )

    HoopVisionNode.active_callback(harness, Bool(data=False))
    assert harness.active is False
    assert list(harness.history) == []
    assert harness.last_detection is None
    assert len(published) == 1
    assert (harness.color_sub, harness.depth_sub, harness.sync) == (
        color_sub,
        depth_sub,
        sync,
    )

    HoopVisionNode.active_callback(harness, Bool(data=True))
    assert harness.active is True
    assert len(published) == 1
    assert (harness.color_sub, harness.depth_sub, harness.sync) == (
        color_sub,
        depth_sub,
        sync,
    )


def test_ball_in_hand_coordinates_modes_inside_vision_off_before_on():
    events = []
    logger = _Logger()
    harness = SimpleNamespace(
        manage_activity_from_ball_in_hand=True,
        managed_hoop_active=False,
        ball_detection_active=True,
        pub_ball_active=_Publisher("ball", events),
        pub_hoop_active=_Publisher("hoop", events),
        get_logger=lambda: logger,
    )

    def set_ball_processing(msg):
        harness.ball_detection_active = bool(msg.data)

    harness.cb_ball_active = set_ball_processing

    changed = BallVisionFusionNode._set_vision_mode_from_ball_in_hand(
        harness,
        True,
    )

    assert changed is True
    assert harness.ball_detection_active is False
    assert harness.managed_hoop_active is True
    assert events == [("ball", False), ("hoop", True)]

    events.clear()
    changed = BallVisionFusionNode._set_vision_mode_from_ball_in_hand(
        harness,
        True,
    )
    assert changed is False
    assert events == []

    changed = BallVisionFusionNode._set_vision_mode_from_ball_in_hand(
        harness,
        False,
    )
    assert changed is True
    assert harness.ball_detection_active is True
    assert events == [("hoop", False), ("ball", True)]
