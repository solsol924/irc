#!/usr/bin/env python3
import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from ultralytics import YOLO


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


class AlignmentDebug(Node):
    def __init__(self, args):
        super().__init__("realsense_alignment_debug")

        self.args = args
        self.bridge = CvBridge()
        self.model = YOLO(args.model, task="detect")

        # Same defaults as the user's current detector until CameraInfo arrives.
        self.fx = 607.0
        self.fy = 606.0
        self.cx_intr = 325.5
        self.cy_intr = 239.4
        self.camera_info_received = False

        self.image_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(
            CameraInfo,
            args.info_topic,
            self.cb_camera_info,
            10,
        )

        self.color_sub = Subscriber(
            self,
            Image,
            args.color_topic,
            qos_profile=self.image_qos,
        )
        self.depth_sub = Subscriber(
            self,
            Image,
            args.depth_topic,
            qos_profile=self.image_qos,
        )

        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10,
            slop=0.05,
        )
        self.sync.registerCallback(self.cb_images)

        self.frame_index = 0
        self.get_logger().info(f"model: {args.model}")
        self.get_logger().info(f"color: {args.color_topic}")
        self.get_logger().info(f"depth: {args.depth_topic}")
        self.get_logger().info(f"info : {args.info_topic}")
        self.get_logger().info(
            "Green = YOLO bbox, Yellow = central 30% ROI, "
            "Red cross = bbox center"
        )

    def cb_camera_info(self, msg: CameraInfo):
        if len(msg.k) < 9:
            return
        fx = float(msg.k[0])
        fy = float(msg.k[4])
        if fx <= 0.0 or fy <= 0.0:
            return

        self.fx = fx
        self.fy = fy
        self.cx_intr = float(msg.k[2])
        self.cy_intr = float(msg.k[5])

        if not self.camera_info_received:
            self.camera_info_received = True
            self.get_logger().info(
                f"CameraInfo fx={self.fx:.2f} fy={self.fy:.2f} "
                f"cx={self.cx_intr:.2f} cy={self.cy_intr:.2f}"
            )

    @staticmethod
    def result_name(result, cls_id: int) -> str:
        names = result.names
        if isinstance(names, dict):
            return str(names.get(cls_id, cls_id))
        try:
            return str(names[cls_id])
        except Exception:
            return str(cls_id)

    def depth_m(self, depth_raw: np.ndarray, encoding: str) -> np.ndarray:
        depth = np.asarray(depth_raw, dtype=np.float32)
        if str(encoding).upper() == "32FC1":
            return depth
        return depth * float(self.args.depth_scale)

    def roi_bounds(self, det: Detection, ratio: float, shape):
        h, w = shape[:2]
        ratio = max(0.08, min(float(ratio), 0.95))

        half_w = max(2, int(round(det.width * ratio * 0.5)))
        half_h = max(2, int(round(det.height * ratio * 0.5)))
        cx = int(round(det.cx))
        cy = int(round(det.cy))

        x1 = max(0, cx - half_w)
        x2 = min(w, cx + half_w + 1)
        y1 = max(0, cy - half_h)
        y2 = min(h, cy + half_h + 1)
        return x1, y1, x2, y2, cx, cy

    def analyze_depth(
        self,
        depth_img_m: np.ndarray,
        det: Detection,
        ratio: float,
    ):
        x1, y1, x2, y2, cx, cy = self.roi_bounds(
            det, ratio, depth_img_m.shape
        )

        patch = depth_img_m[y1:y2, x1:x2]
        valid = patch[
            np.isfinite(patch)
            & (patch >= self.args.depth_min)
            & (patch <= self.args.depth_max)
        ]
        roi_valid = int(valid.size)

        minimum = max(1, int(self.args.min_valid))
        used_fallback = False
        fallback_valid = None
        selected = valid

        if valid.size < minimum:
            used_fallback = True
            radius = 5
            h, w = depth_img_m.shape[:2]
            fx1 = max(0, cx - radius)
            fx2 = min(w, cx + radius + 1)
            fy1 = max(0, cy - radius)
            fy2 = min(h, cy + radius + 1)

            patch2 = depth_img_m[fy1:fy2, fx1:fx2]
            valid2 = patch2[
                np.isfinite(patch2)
                & (patch2 >= self.args.depth_min)
                & (patch2 <= self.args.depth_max)
            ]
            fallback_valid = int(valid2.size)
            selected = valid2

        if selected.size < minimum:
            return {
                "roi": (x1, y1, x2, y2),
                "roi_valid": roi_valid,
                "used_fallback": used_fallback,
                "fallback_valid": fallback_valid,
                "z_m": None,
                "distance_cm": None,
            }

        z_m = float(np.median(selected))
        x_m = (det.cx - self.cx_intr) * z_m / max(self.fx, 1e-6)
        y_m = (det.cy - self.cy_intr) * z_m / max(self.fy, 1e-6)
        distance_cm = math.sqrt(
            x_m * x_m + y_m * y_m + z_m * z_m
        ) * 100.0

        return {
            "roi": (x1, y1, x2, y2),
            "roi_valid": roi_valid,
            "used_fallback": used_fallback,
            "fallback_valid": fallback_valid,
            "z_m": z_m,
            "distance_cm": distance_cm,
        }

    def make_depth_view(self, depth_img_m: np.ndarray) -> np.ndarray:
        clipped = np.clip(depth_img_m, 0.0, self.args.depth_max)
        scaled = np.zeros_like(clipped, dtype=np.uint8)

        valid = (
            np.isfinite(clipped)
            & (clipped >= self.args.depth_min)
            & (clipped <= self.args.depth_max)
        )

        if np.any(valid):
            scaled[valid] = np.clip(
                clipped[valid] / self.args.depth_max * 255.0,
                1,
                255,
            ).astype(np.uint8)

        colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        colored[~valid] = (0, 0, 0)
        return colored

    @staticmethod
    def draw_cross(img, cx, cy, color=(0, 0, 255)):
        cv2.line(img, (cx - 7, cy), (cx + 7, cy), color, 2)
        cv2.line(img, (cx, cy - 7), (cx, cy + 7), color, 2)

    def cb_images(self, color_msg: Image, depth_msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                color_msg, desired_encoding="bgr8"
            )
            depth_raw = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
        except Exception as exc:
            self.get_logger().warning(f"image conversion failed: {exc}")
            return

        depth_img_m = self.depth_m(
            np.asarray(depth_raw), depth_msg.encoding
        )

        if depth_img_m.ndim != 2 or depth_img_m.shape[:2] != frame.shape[:2]:
            self.get_logger().warning(
                f"shape mismatch RGB={frame.shape[:2]} "
                f"DEPTH={depth_img_m.shape[:2]}"
            )
            return

        result = self.model.predict(
            source=frame,
            imgsz=self.args.imgsz,
            conf=self.args.conf,
            device=self.args.device,
            verbose=False,
        )[0]

        backboards = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                name = self.result_name(result, cls_id)
                if name != "backboard":
                    continue

                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = [
                    float(v) for v in box.xyxy[0].tolist()
                ]
                backboards.append(
                    Detection(name, cls_id, conf, x1, y1, x2, y2)
                )

        rgb_view = frame.copy()
        depth_view = self.make_depth_view(depth_img_m)

        self.frame_index += 1

        if not backboards:
            cv2.putText(
                rgb_view, "NO BACKBOARD", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        else:
            det = max(backboards, key=lambda d: d.conf)
            info = self.analyze_depth(
                depth_img_m,
                det,
                self.args.inner_ratio,
            )

            bx1, by1, bx2, by2 = map(
                int, [det.x1, det.y1, det.x2, det.y2]
            )
            rx1, ry1, rx2, ry2 = info["roi"]
            cx = int(round(det.cx))
            cy = int(round(det.cy))

            for view in (rgb_view, depth_view):
                cv2.rectangle(
                    view, (bx1, by1), (bx2, by2), (0, 255, 0), 2
                )
                cv2.rectangle(
                    view, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2
                )
                self.draw_cross(view, cx, cy)

            if info["distance_cm"] is None:
                status = "N/A"
                color = (0, 0, 255)
            else:
                status = f"{info['distance_cm']:.1f}cm"
                color = (0, 255, 0)

            fallback_text = (
                "-"
                if info["fallback_valid"] is None
                else str(info["fallback_valid"])
            )

            line1 = (
                f"BACKBOARD {det.conf:.2f}  D={status}"
            )
            line2 = (
                f"ROI valid={info['roi_valid']}  "
                f"fallback={fallback_text}"
            )

            for view in (rgb_view, depth_view):
                cv2.putText(
                    view, line1, (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2
                )
                cv2.putText(
                    view, line2, (15, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2
                )

            z_text = "N/A" if info["z_m"] is None else f"{info['z_m']*100:.1f}cm"

            print(
                f"[{self.frame_index:06d}] "
                f"conf={det.conf:.3f} "
                f"bbox=({bx1},{by1},{bx2},{by2}) "
                f"ROIvalid={info['roi_valid']} "
                f"fallback={fallback_text} "
                f"z={z_text} "
                f"distance={status}",
                flush=True,
            )

        title_rgb = "RGB + YOLO bbox / central 30% ROI"
        title_depth = "ALIGNED DEPTH at the SAME pixel coordinates"

        cv2.putText(
            rgb_view, title_rgb, (15, rgb_view.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1
        )
        cv2.putText(
            depth_view, title_depth, (15, depth_view.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1
        )

        combined = np.hstack([rgb_view, depth_view])
        cv2.imshow("RGB vs Aligned Depth Debug", combined)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q"), ord("Q")):
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="/home/rnd/Desktop/realsense_round3_best.engine",
    )
    parser.add_argument(
        "--color-topic",
        default="/camera/camera/color/image_raw",
    )
    parser.add_argument(
        "--depth-topic",
        default="/camera/camera/aligned_depth_to_color/image_raw",
    )
    parser.add_argument(
        "--info-topic",
        default="/camera/camera/color/camera_info",
    )

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")

    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--depth-min", type=float, default=0.08)
    parser.add_argument("--depth-max", type=float, default=3.0)
    parser.add_argument("--inner-ratio", type=float, default=0.30)
    parser.add_argument("--min-valid", type=int, default=5)

    args = parser.parse_args()

    rclpy.init()
    node = AlignmentDebug(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
