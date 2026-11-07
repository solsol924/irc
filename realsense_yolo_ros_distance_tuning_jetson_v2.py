#!/usr/bin/env python3
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


# ============================================================
# Jetson RealSense + YOLO manual tuning
# Distance logic matched to realsense_yolo_detector.py
# ============================================================

MODEL_PATH = "/home/rnd/Desktop/realsense_round3_best.engine"

WIDTH = 640
HEIGHT = 480
FPS = 30
IMGSZ = 640
CONF = 0.25

WINDOW_NAME = "RealSense YOLO + ROS Distance Logic"

# Current 2-class RealSense model
CLASS_NAMES = {
    0: "backboard",
    1: "ball",
}

# Match settings.ini / realsense_yolo_detector.py defaults
DEPTH_SCALE = 0.001
DEPTH_MIN_M = 0.08
DEPTH_MAX_M = 3.0
DEPTH_INNER_RATIO_BALL = 0.45
DEPTH_INNER_RATIO_HOOP = 0.30
MIN_VALID_DEPTH_PIXELS = 5


@dataclass
class Detection:
    name: str
    cls_id: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) * 0.5

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) * 0.5

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


class RealSenseController:
    def __init__(self):
        self.pipeline = None
        self.align = None
        self.color_sensor = None

        # Fallback values identical to the ROS node.
        self.fx = 607.0
        self.fy = 606.0
        self.cx_intr = 325.5
        self.cy_intr = 239.4

    def start(self):
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(
            rs.stream.color,
            WIDTH,
            HEIGHT,
            rs.format.bgr8,
            FPS,
        )
        config.enable_stream(
            rs.stream.depth,
            WIDTH,
            HEIGHT,
            rs.format.z16,
            FPS,
        )

        profile = pipeline.start(config)

        # Use live color-camera intrinsics.
        try:
            color_profile = profile.get_stream(
                rs.stream.color
            ).as_video_stream_profile()
            intr = color_profile.get_intrinsics()

            if intr.fx > 0 and intr.fy > 0:
                self.fx = float(intr.fx)
                self.fy = float(intr.fy)
                self.cx_intr = float(intr.ppx)
                self.cy_intr = float(intr.ppy)

        except Exception as exc:
            print(
                "[WARN] Could not read live color intrinsics; "
                f"using fallback values: {exc}"
            )

        color_sensor = None

        for sensor in profile.get_device().query_sensors():
            try:
                if (
                    sensor.supports(rs.option.exposure)
                    and sensor.supports(rs.option.gain)
                ):
                    color_sensor = sensor

                    if sensor.supports(rs.option.white_balance):
                        break

            except Exception:
                continue

        if color_sensor is None:
            pipeline.stop()
            raise RuntimeError("RealSense RGB sensor not found.")

        self.pipeline = pipeline
        self.align = rs.align(rs.stream.color)
        self.color_sensor = color_sensor

        # Let AE/AWB settle.
        for _ in range(45):
            pipeline.wait_for_frames(timeout_ms=3000)

        print(
            "Color intrinsics: "
            f"fx={self.fx:.2f} fy={self.fy:.2f} "
            f"cx={self.cx_intr:.2f} cy={self.cy_intr:.2f}"
        )

    def stop(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass

        self.pipeline = None

    def get_frames(self):
        if self.pipeline is None:
            return None, None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1500)

            # Same concept as /camera/aligned_depth_to_color/image_raw
            aligned = self.align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                return None, None

            color = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())

            return color, depth_raw

        except Exception:
            return None, None

    # ---------------- Camera tuning ----------------

    def supports(self, option):
        try:
            return (
                self.color_sensor is not None
                and self.color_sensor.supports(option)
            )
        except Exception:
            return False

    def get(self, option):
        if not self.supports(option):
            return None

        try:
            return self.color_sensor.get_option(option)
        except Exception:
            return None

    def set(self, option, value):
        if not self.supports(option):
            return False

        try:
            rng = self.color_sensor.get_option_range(option)
            value = max(rng.min, min(rng.max, value))

            if rng.step > 0:
                value = (
                    round((value - rng.min) / rng.step)
                    * rng.step
                    + rng.min
                )
                value = max(rng.min, min(rng.max, value))

            self.color_sensor.set_option(option, value)
            return True

        except Exception as exc:
            print(f"[SET ERROR] {exc}")
            return False

    def adjust(self, option, direction, fraction):
        if not self.supports(option):
            return False

        try:
            rng = self.color_sensor.get_option_range(option)
            current = self.color_sensor.get_option(option)

            step = max(
                rng.step,
                (rng.max - rng.min) * fraction,
            )

            return self.set(
                option,
                current + direction * step,
            )

        except Exception:
            return False

    def toggle_auto_exposure(self):
        opt = rs.option.enable_auto_exposure
        current = self.get(opt)

        if current is None:
            return None

        target = 0.0 if current >= 0.5 else 1.0
        return bool(target) if self.set(opt, target) else None

    def toggle_auto_white_balance(self):
        opt = rs.option.enable_auto_white_balance
        current = self.get(opt)

        if current is None:
            return None

        target = 0.0 if current >= 0.5 else 1.0
        return bool(target) if self.set(opt, target) else None

    def exposure_down(self):
        ae = self.get(rs.option.enable_auto_exposure)

        if ae is not None and ae >= 0.5:
            return False

        return self.adjust(
            rs.option.exposure,
            -1,
            0.035,
        )

    def exposure_up(self):
        ae = self.get(rs.option.enable_auto_exposure)

        if ae is not None and ae >= 0.5:
            return False

        return self.adjust(
            rs.option.exposure,
            +1,
            0.035,
        )

    def gain_down(self):
        ae = self.get(rs.option.enable_auto_exposure)

        if ae is not None and ae >= 0.5:
            return False

        return self.adjust(
            rs.option.gain,
            -1,
            0.05,
        )

    def gain_up(self):
        ae = self.get(rs.option.enable_auto_exposure)

        if ae is not None and ae >= 0.5:
            return False

        return self.adjust(
            rs.option.gain,
            +1,
            0.05,
        )

    def wb_down(self):
        awb = self.get(rs.option.enable_auto_white_balance)

        if awb is not None and awb >= 0.5:
            return False

        return self.adjust(
            rs.option.white_balance,
            -1,
            0.03,
        )

    def wb_up(self):
        awb = self.get(rs.option.enable_auto_white_balance)

        if awb is not None and awb >= 0.5:
            return False

        return self.adjust(
            rs.option.white_balance,
            +1,
            0.03,
        )

    def info(self):
        return {
            "ae": self.get(rs.option.enable_auto_exposure),
            "exposure": self.get(rs.option.exposure),
            "gain": self.get(rs.option.gain),
            "awb": self.get(rs.option.enable_auto_white_balance),
            "wb": self.get(rs.option.white_balance),
        }


# ============================================================
# Distance logic copied conceptually from realsense_yolo_detector.py
# ============================================================

def depth_m(depth_raw: np.ndarray) -> np.ndarray:
    """Equivalent to _depth_m() for a 16UC1/Z16 depth image."""
    depth = np.asarray(depth_raw, dtype=np.float32)
    return depth * float(DEPTH_SCALE)


def sample_depth(
    depth_image_m: np.ndarray,
    det: Detection,
    inner_ratio: float,
) -> Optional[float]:
    """
    Match _sample_depth():

    1) centered inner ROI inside YOLO bbox
    2) valid depth range filtering
    3) if too few pixels -> retry 11x11 around bbox center
    4) median of valid depth pixels
    """
    h, w = depth_image_m.shape[:2]

    ratio = max(
        0.08,
        min(float(inner_ratio), 0.95),
    )

    half_w = max(
        2,
        int(round(det.width * ratio * 0.5)),
    )
    half_h = max(
        2,
        int(round(det.height * ratio * 0.5)),
    )

    cx = int(round(det.cx))
    cy = int(round(det.cy))

    x1 = max(0, cx - half_w)
    x2 = min(w, cx + half_w + 1)
    y1 = max(0, cy - half_h)
    y2 = min(h, cy + half_h + 1)

    patch = depth_image_m[y1:y2, x1:x2]

    valid = patch[
        np.isfinite(patch)
        & (patch >= DEPTH_MIN_M)
        & (patch <= DEPTH_MAX_M)
    ]

    minimum = max(
        1,
        int(MIN_VALID_DEPTH_PIXELS),
    )

    # Same fallback as the ROS node.
    if valid.size < minimum:
        radius = 5

        x1 = max(0, cx - radius)
        x2 = min(w, cx + radius + 1)
        y1 = max(0, cy - radius)
        y2 = min(h, cy + radius + 1)

        patch = depth_image_m[y1:y2, x1:x2]

        valid = patch[
            np.isfinite(patch)
            & (patch >= DEPTH_MIN_M)
            & (patch <= DEPTH_MAX_M)
        ]

    if valid.size < minimum:
        return None

    return float(np.median(valid))


def xyz(
    px: float,
    py: float,
    z_m: float,
    fx: float,
    fy: float,
    cx_intr: float,
    cy_intr: float,
) -> Tuple[float, float, float]:
    """Match _xyz()."""
    x_m = (
        (px - cx_intr)
        * z_m
        / max(fx, 1e-6)
    )

    y_m = (
        (py - cy_intr)
        * z_m
        / max(fy, 1e-6)
    )

    return (
        float(x_m),
        float(y_m),
        float(z_m),
    )


def distance_3d_cm(
    x_m: float,
    y_m: float,
    z_m: float,
) -> float:
    """Match _distance() * 100."""
    return (
        math.sqrt(
            x_m * x_m
            + y_m * y_m
            + z_m * z_m
        )
        * 100.0
    )


def calculate_detection_distance(
    depth_image_m: np.ndarray,
    det: Detection,
    camera: RealSenseController,
):
    if det.name == "ball":
        inner_ratio = DEPTH_INNER_RATIO_BALL
    else:
        # backboard / hoop
        inner_ratio = DEPTH_INNER_RATIO_HOOP

    z_m = sample_depth(
        depth_image_m,
        det,
        inner_ratio,
    )

    if z_m is None:
        return None

    x_m, y_m, z_m = xyz(
        det.cx,
        det.cy,
        z_m,
        camera.fx,
        camera.fy,
        camera.cx_intr,
        camera.cy_intr,
    )

    distance_cm = distance_3d_cm(
        x_m,
        y_m,
        z_m,
    )

    return {
        "distance_cm": float(distance_cm),
        "center_depth_cm": float(z_m * 100.0),
        "x_m": float(x_m),
        "y_m": float(y_m),
        "z_m": float(z_m),
    }


def draw_status(
    frame,
    controller,
    fps,
    detection_text,
    conf,
):
    out = frame.copy()

    panel = out.copy()
    cv2.rectangle(
        panel,
        (5, 5),
        (520, 160),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(
        panel,
        0.55,
        out,
        0.45,
        0,
        out,
    )

    info = controller.info()

    ae = (
        "N/A"
        if info["ae"] is None
        else ("ON" if info["ae"] >= 0.5 else "OFF")
    )
    awb = (
        "N/A"
        if info["awb"] is None
        else ("ON" if info["awb"] >= 0.5 else "OFF")
    )
    exp = (
        "N/A"
        if info["exposure"] is None
        else f"{info['exposure']:.0f}"
    )
    gain = (
        "N/A"
        if info["gain"] is None
        else f"{info['gain']:.0f}"
    )
    wb = (
        "N/A"
        if info["wb"] is None
        else f"{info['wb']:.0f}"
    )

    rows = [
        f"AE {ae}   EXP {exp}   GAIN {gain}",
        f"AWB {awb}   WB {wb}",
        f"YOLO conf {conf:.2f}   FPS {fps:.1f}",
        f"Detections: {detection_text}",
        "A=AE  [ ]=Exposure  , .=Gain",
        "B=AWB  - ==WB   C/V=conf   Q=quit",
    ]

    y = 27

    for text in rows:
        cv2.putText(
            out,
            text,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        y += 23

    return out


def main():
    global CONF

    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        return

    print(f"Loading model: {MODEL_PATH}")

    model = YOLO(
        MODEL_PATH,
        task="detect",
    )

    camera = RealSenseController()

    try:
        camera.start()

    except Exception as exc:
        print(
            f"[ERROR] RealSense open failed: {exc}"
        )
        return

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )
    cv2.resizeWindow(
        WINDOW_NAME,
        960,
        720,
    )

    smoothed_fps = 0.0

    print()
    print("Distance logic:")
    print(
        "backboard: inner_ratio=0.30 -> median z "
        "-> XYZ -> sqrt(x^2+y^2+z^2)"
    )
    print(
        "ball     : inner_ratio=0.45 -> median z "
        "-> XYZ -> sqrt(x^2+y^2+z^2)"
    )
    print()
    print("Controls:")
    print("A       : Auto Exposure ON/OFF")
    print("[ / ]   : Exposure down / up")
    print(", / .   : Gain down / up")
    print("B       : Auto White Balance ON/OFF")
    print("- / =   : White Balance down / up")
    print("C / V   : YOLO confidence down / up")
    print("Q / ESC : Quit")

    try:
        while True:
            color, depth_raw = camera.get_frames()

            if color is None or depth_raw is None:
                continue

            depth_image_m = depth_m(
                depth_raw
            )

            t0 = time.perf_counter()

            result = model.predict(
                source=color,
                imgsz=IMGSZ,
                conf=CONF,
                device=0,
                verbose=False,
            )[0]

            annotated = color.copy()
            detections_text = []

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(
                        box.cls[0].item()
                    )
                    confidence = float(
                        box.conf[0].item()
                    )

                    name = CLASS_NAMES.get(
                        cls_id,
                        f"class{cls_id}",
                    )

                    x1, y1, x2, y2 = [
                        float(v)
                        for v in box.xyxy[0].tolist()
                    ]

                    det = Detection(
                        name=name,
                        cls_id=cls_id,
                        conf=confidence,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )

                    distance = calculate_detection_distance(
                        depth_image_m,
                        det,
                        camera,
                    )

                    ix1 = max(
                        0,
                        min(
                            WIDTH - 1,
                            int(round(x1)),
                        ),
                    )
                    iy1 = max(
                        0,
                        min(
                            HEIGHT - 1,
                            int(round(y1)),
                        ),
                    )
                    ix2 = max(
                        0,
                        min(
                            WIDTH - 1,
                            int(round(x2)),
                        ),
                    )
                    iy2 = max(
                        0,
                        min(
                            HEIGHT - 1,
                            int(round(y2)),
                        ),
                    )

                    if distance is None:
                        label = (
                            f"{name} {confidence:.2f} "
                            f"distance:N/A"
                        )

                        detections_text.append(
                            f"{name}:{confidence:.2f}/N/A"
                        )

                    else:
                        d_cm = distance["distance_cm"]
                        z_cm = distance["center_depth_cm"]

                        label = (
                            f"{name} {confidence:.2f} "
                            f"D:{d_cm:.1f}cm "
                            f"Z:{z_cm:.1f}cm"
                        )

                        detections_text.append(
                            f"{name}:{confidence:.2f}/"
                            f"D{d_cm:.1f}/Z{z_cm:.1f}cm"
                        )

                    cv2.rectangle(
                        annotated,
                        (ix1, iy1),
                        (ix2, iy2),
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        annotated,
                        label,
                        (
                            ix1,
                            max(20, iy1 - 8),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            elapsed = (
                time.perf_counter()
                - t0
            )

            fps_now = (
                1.0 / elapsed
                if elapsed > 0
                else 0.0
            )

            if smoothed_fps == 0.0:
                smoothed_fps = fps_now
            else:
                smoothed_fps = (
                    0.85 * smoothed_fps
                    + 0.15 * fps_now
                )

            detection_text = (
                ", ".join(
                    detections_text[:3]
                )
                if detections_text
                else "none"
            )

            display = draw_status(
                annotated,
                camera,
                smoothed_fps,
                detection_text,
                CONF,
            )

            cv2.imshow(
                WINDOW_NAME,
                display,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == 255:
                continue

            if key in (
                27,
                ord("q"),
                ord("Q"),
            ):
                break

            if key in (
                ord("a"),
                ord("A"),
            ):
                state = (
                    camera.toggle_auto_exposure()
                )

                print(
                    "Auto Exposure:",
                    "ON"
                    if state
                    else "OFF"
                    if state is not None
                    else "unsupported",
                )

                continue

            if key == ord("["):
                if not camera.exposure_down():
                    print(
                        "Turn Auto Exposure OFF first with A."
                    )
                continue

            if key == ord("]"):
                if not camera.exposure_up():
                    print(
                        "Turn Auto Exposure OFF first with A."
                    )
                continue

            if key == ord(","):
                if not camera.gain_down():
                    print(
                        "Turn Auto Exposure OFF first with A."
                    )
                continue

            if key == ord("."):
                if not camera.gain_up():
                    print(
                        "Turn Auto Exposure OFF first with A."
                    )
                continue

            if key in (
                ord("b"),
                ord("B"),
            ):
                state = (
                    camera.toggle_auto_white_balance()
                )

                print(
                    "Auto White Balance:",
                    "ON"
                    if state
                    else "OFF"
                    if state is not None
                    else "unsupported",
                )

                continue

            if key in (
                ord("-"),
                ord("_"),
            ):
                if not camera.wb_down():
                    print(
                        "Turn Auto White Balance OFF first with B."
                    )
                continue

            if key in (
                ord("="),
                ord("+"),
            ):
                if not camera.wb_up():
                    print(
                        "Turn Auto White Balance OFF first with B."
                    )
                continue

            if key in (
                ord("c"),
                ord("C"),
            ):
                CONF = max(
                    0.05,
                    round(
                        CONF - 0.05,
                        2,
                    ),
                )

                print(
                    f"YOLO confidence: {CONF:.2f}"
                )

                continue

            if key in (
                ord("v"),
                ord("V"),
            ):
                CONF = min(
                    0.95,
                    round(
                        CONF + 0.05,
                        2,
                    ),
                )

                print(
                    f"YOLO confidence: {CONF:.2f}"
                )

                continue

    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
