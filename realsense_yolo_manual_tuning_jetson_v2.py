#!/usr/bin/env python3
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

MODEL_PATH = "/home/rnd/Desktop/realsense_round3_best.engine"

WIDTH = 640
HEIGHT = 480
FPS = 30
IMGSZ = 640
CONF = 0.25
WINDOW_NAME = "RealSense YOLO Manual Tuning"

CLASS_NAMES = {0: "backboard", 1: "ball"}


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


class RealSenseController:
    def __init__(self):
        self.pipeline = None
        self.color_sensor = None

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
        profile = pipeline.start(config)

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
                pass

        if color_sensor is None:
            pipeline.stop()
            raise RuntimeError("RealSense RGB sensor not found.")

        self.pipeline = pipeline
        self.color_sensor = color_sensor

        for _ in range(45):
            pipeline.wait_for_frames(timeout_ms=3000)

    def stop(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.pipeline = None

    def get_frame(self):
        if self.pipeline is None:
            return None
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1500)
            color = frames.get_color_frame()
            if not color:
                return None
            return np.asanyarray(color.get_data())
        except Exception:
            return None

    def supports(self, option):
        try:
            return self.color_sensor is not None and self.color_sensor.supports(option)
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
            value = clamp(value, rng.min, rng.max)
            if rng.step > 0:
                value = round((value - rng.min) / rng.step) * rng.step + rng.min
                value = clamp(value, rng.min, rng.max)
            self.color_sensor.set_option(option, value)
            return True
        except Exception as e:
            print(f"[SET ERROR] {e}")
            return False

    def adjust(self, option, direction, fraction):
        if not self.supports(option):
            return False
        try:
            rng = self.color_sensor.get_option_range(option)
            current = self.color_sensor.get_option(option)
            step = max(rng.step, (rng.max - rng.min) * fraction)
            return self.set(option, current + direction * step)
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
        return self.adjust(rs.option.exposure, -1, 0.035)

    def exposure_up(self):
        ae = self.get(rs.option.enable_auto_exposure)
        if ae is not None and ae >= 0.5:
            return False
        return self.adjust(rs.option.exposure, +1, 0.035)

    def gain_down(self):
        ae = self.get(rs.option.enable_auto_exposure)
        if ae is not None and ae >= 0.5:
            return False
        return self.adjust(rs.option.gain, -1, 0.05)

    def gain_up(self):
        ae = self.get(rs.option.enable_auto_exposure)
        if ae is not None and ae >= 0.5:
            return False
        return self.adjust(rs.option.gain, +1, 0.05)

    def wb_down(self):
        awb = self.get(rs.option.enable_auto_white_balance)
        if awb is not None and awb >= 0.5:
            return False
        return self.adjust(rs.option.white_balance, -1, 0.03)

    def wb_up(self):
        awb = self.get(rs.option.enable_auto_white_balance)
        if awb is not None and awb >= 0.5:
            return False
        return self.adjust(rs.option.white_balance, +1, 0.03)

    def info(self):
        return {
            "ae": self.get(rs.option.enable_auto_exposure),
            "exposure": self.get(rs.option.exposure),
            "gain": self.get(rs.option.gain),
            "awb": self.get(rs.option.enable_auto_white_balance),
            "wb": self.get(rs.option.white_balance),
        }


def draw_status(frame, controller, fps, detection_text, conf):
    out = frame.copy()
    panel = out.copy()
    cv2.rectangle(panel, (5, 5), (430, 155), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.55, out, 0.45, 0, out)

    info = controller.info()

    ae = "N/A" if info["ae"] is None else ("ON" if info["ae"] >= 0.5 else "OFF")
    awb = "N/A" if info["awb"] is None else ("ON" if info["awb"] >= 0.5 else "OFF")
    exp = "N/A" if info["exposure"] is None else f"{info['exposure']:.0f}"
    gain = "N/A" if info["gain"] is None else f"{info['gain']:.0f}"
    wb = "N/A" if info["wb"] is None else f"{info['wb']:.0f}"

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
        print("Edit MODEL_PATH at the top of this file if the engine name is different.")
        return

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH, task="detect")

    camera = RealSenseController()
    try:
        camera.start()
    except Exception as e:
        print(f"[ERROR] RealSense open failed: {e}")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 720)

    smoothed_fps = 0.0

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
            frame = camera.get_frame()
            if frame is None:
                continue

            t0 = time.perf_counter()
            result = model.predict(
                source=frame,
                imgsz=IMGSZ,
                conf=CONF,
                device=0,
                verbose=False,
            )[0]

            annotated = result.plot()

            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    score = float(box.conf[0].item())
                    name = CLASS_NAMES.get(cls_id, f"class{cls_id}")
                    detections.append(f"{name}:{score:.2f}")

            dt = time.perf_counter() - t0
            current_fps = 1.0 / dt if dt > 0 else 0.0
            smoothed_fps = current_fps if smoothed_fps == 0 else 0.85 * smoothed_fps + 0.15 * current_fps

            detection_text = ", ".join(detections[:4]) if detections else "none"

            display = draw_status(
                annotated,
                camera,
                smoothed_fps,
                detection_text,
                CONF,
            )
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue

            if key in (27, ord("q"), ord("Q")):
                break

            if key in (ord("a"), ord("A")):
                state = camera.toggle_auto_exposure()
                print("Auto Exposure:", "ON" if state else "OFF" if state is not None else "unsupported")
                continue

            if key == ord("["):
                if not camera.exposure_down():
                    print("Turn Auto Exposure OFF first with A.")
                continue

            if key == ord("]"):
                if not camera.exposure_up():
                    print("Turn Auto Exposure OFF first with A.")
                continue

            if key == ord(","):
                if not camera.gain_down():
                    print("Turn Auto Exposure OFF first with A.")
                continue

            if key == ord("."):
                if not camera.gain_up():
                    print("Turn Auto Exposure OFF first with A.")
                continue

            if key in (ord("b"), ord("B")):
                state = camera.toggle_auto_white_balance()
                print("Auto White Balance:", "ON" if state else "OFF" if state is not None else "unsupported")
                continue

            if key in (ord("-"), ord("_")):
                if not camera.wb_down():
                    print("Turn Auto White Balance OFF first with B.")
                continue

            if key in (ord("="), ord("+")):
                if not camera.wb_up():
                    print("Turn Auto White Balance OFF first with B.")
                continue

            if key in (ord("c"), ord("C")):
                CONF = max(0.05, round(CONF - 0.05, 2))
                print(f"YOLO confidence: {CONF:.2f}")
                continue

            if key in (ord("v"), ord("V")):
                CONF = min(0.95, round(CONF + 0.05, 2))
                print(f"YOLO confidence: {CONF:.2f}")
                continue

    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
