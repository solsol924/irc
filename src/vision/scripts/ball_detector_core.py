#!/usr/bin/env python3
"""
IRC RealSense 공 검출기에서 공통으로 사용하는 색상 문맥 검사 모듈입니다.

대회용 공은 항상 검은색 원형 받침대 위에 놓입니다. 이 모듈은 실제 운용
노드와 보정 도구의 검출 미리보기에서 받침대 및 바닥을 동일한 방식으로
검증하도록 합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class BallSupportConfig:
    """검은색 공 받침대의 물리적 특성과 색상 문맥에 대한 임곗값입니다."""

    support_diameter_m: float = 0.150
    support_v_max: int = 75
    black_ratio_min: float = 0.30
    edge_black_ratio_min: float = 0.25
    surrounding_ball_ratio_max: float = 0.15
    floor_ratio_max: float = 0.35
    sector_black_ratio_min: float = 0.18
    min_sectors: int = 3
    edge_min_sectors: int = 2
    sector_count: int = 8
    min_visible_fraction: float = 0.45
    edge_min_visible_fraction: float = 0.12
    inner_radius_scale: float = 1.08
    outer_radius_scale: float = 0.95

    def validated(self) -> "BallSupportConfig":
        """마스크 생성에 안전하도록 값의 범위를 제한한 복사본을 반환합니다."""
        return BallSupportConfig(
            support_diameter_m=max(0.001, float(self.support_diameter_m)),
            support_v_max=max(0, min(255, int(self.support_v_max))),
            black_ratio_min=max(0.0, min(1.0, float(self.black_ratio_min))),
            edge_black_ratio_min=max(
                0.0, min(1.0, float(self.edge_black_ratio_min))
            ),
            surrounding_ball_ratio_max=max(
                0.0, min(1.0, float(self.surrounding_ball_ratio_max))
            ),
            floor_ratio_max=max(0.0, min(1.0, float(self.floor_ratio_max))),
            sector_black_ratio_min=max(
                0.0, min(1.0, float(self.sector_black_ratio_min))
            ),
            min_sectors=max(1, int(self.min_sectors)),
            edge_min_sectors=max(1, int(self.edge_min_sectors)),
            sector_count=max(1, int(self.sector_count)),
            min_visible_fraction=max(
                0.0, min(1.0, float(self.min_visible_fraction))
            ),
            edge_min_visible_fraction=max(
                0.0, min(1.0, float(self.edge_min_visible_fraction))
            ),
            inner_radius_scale=max(0.1, float(self.inner_radius_scale)),
            outer_radius_scale=max(0.1, float(self.outer_radius_scale)),
        )


def hsv_range_mask(
    hsv: np.ndarray,
    lower: Tuple[int, int, int],
    upper: Tuple[int, int, int],
) -> np.ndarray:
    """HSV 마스크를 생성하며, 0을 경계로 이어지는 H 범위도 지원합니다."""
    h_low, s_low, v_low = (int(value) for value in lower)
    h_high, s_high, v_high = (int(value) for value in upper)
    h_low = max(0, min(179, h_low))
    h_high = max(0, min(179, h_high))
    s_low = max(0, min(255, s_low))
    s_high = max(0, min(255, s_high))
    v_low = max(0, min(255, v_low))
    v_high = max(0, min(255, v_high))
    if s_low > s_high:
        s_low, s_high = s_high, s_low
    if v_low > v_high:
        v_low, v_high = v_high, v_low

    if h_low <= h_high:
        return cv2.inRange(
            hsv,
            np.array([h_low, s_low, v_low], dtype=np.uint8),
            np.array([h_high, s_high, v_high], dtype=np.uint8),
        )

    low_hue = cv2.inRange(
        hsv,
        np.array([0, s_low, v_low], dtype=np.uint8),
        np.array([h_high, s_high, v_high], dtype=np.uint8),
    )
    high_hue = cv2.inRange(
        hsv,
        np.array([h_low, s_low, v_low], dtype=np.uint8),
        np.array([179, s_high, v_high], dtype=np.uint8),
    )
    return cv2.bitwise_or(low_hue, high_hue)


def black_support_mask(hsv: np.ndarray, support_v_max: int) -> np.ndarray:
    """불안정한 색조의 영향을 피하도록 밝기를 기준으로 검은 받침대 픽셀을 분류합니다."""
    limit = max(0, min(255, int(support_v_max)))
    return cv2.inRange(
        hsv,
        np.array([0, 0, 0], dtype=np.uint8),
        np.array([179, 255, limit], dtype=np.uint8),
    )


def evaluate_ball_support(
    *,
    hsv: np.ndarray,
    ball_color_mask: np.ndarray,
    floor_mask: np.ndarray,
    center: Tuple[float, float],
    detected_radius_px: float,
    expected_ball_radius_px: float,
    depth_m: float,
    focal_x_px: float,
    touches_edge: bool,
    config: BallSupportConfig,
    black_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    공 후보가 검은색 받침대로 둘러싸여 있는지 검증합니다.

    이미지 내부의 픽셀만 계산합니다. 프레임 경계에서는 이미지 밖의 픽셀이
    검은색으로 해석되지 않아야 하므로 특히 중요합니다.
    """
    cfg = config.validated()
    height, width = hsv.shape[:2]
    if (
        height <= 0
        or width <= 0
        or ball_color_mask.shape[:2] != (height, width)
        or floor_mask.shape[:2] != (height, width)
        or (
            black_mask is not None
            and black_mask.shape[:2] != (height, width)
        )
        or not math.isfinite(depth_m)
        or depth_m <= 0.0
        or not math.isfinite(focal_x_px)
        or focal_x_px <= 0.0
    ):
        return _empty_support_result("invalid_input")

    cx, cy = float(center[0]), float(center[1])
    if not math.isfinite(cx) or not math.isfinite(cy):
        return _empty_support_result("invalid_center")

    expected_support_radius = (
        focal_x_px * (cfg.support_diameter_m * 0.5) / depth_m
    )
    inner_radius = max(
        float(expected_ball_radius_px),
        float(detected_radius_px),
    ) * cfg.inner_radius_scale
    outer_radius = expected_support_radius * cfg.outer_radius_scale
    if (
        not math.isfinite(inner_radius)
        or not math.isfinite(outer_radius)
        or inner_radius <= 0.0
        or outer_radius <= inner_radius + 1.0
    ):
        return _empty_support_result("invalid_ring")

    # 받침대 검사에는 이 후보 주변의 원형 영역만 필요합니다. 이후 사용하는 모든
    # 배열을 이 경계 상자 안으로 제한하면 주황색 윤곽선마다 전체 카메라 프레임을
    # 다시 탐색하지 않아도 됩니다.
    x_start = max(
        0,
        min(width, int(math.floor(cx - outer_radius))),
    )
    x_end = max(
        0,
        min(width, int(math.ceil(cx + outer_radius)) + 1),
    )
    y_start = max(
        0,
        min(height, int(math.floor(cy - outer_radius))),
    )
    y_end = max(
        0,
        min(height, int(math.ceil(cy + outer_radius)) + 1),
    )
    if x_end <= x_start or y_end <= y_start:
        return _empty_support_result("no_visible_ring")

    x_coords = (
        np.arange(x_start, x_end, dtype=np.float32)[None, :] - cx
    )
    y_coords = (
        np.arange(y_start, y_end, dtype=np.float32)[:, None] - cy
    )
    dx = x_coords
    dy = y_coords
    distance_sq = dx * dx + dy * dy
    ring = (
        (distance_sq >= inner_radius * inner_radius)
        & (distance_sq <= outer_radius * outer_radius)
    )
    visible_pixels = int(np.count_nonzero(ring))
    theoretical_pixels = math.pi * (
        outer_radius * outer_radius - inner_radius * inner_radius
    )
    visible_fraction = (
        visible_pixels / theoretical_pixels
        if theoretical_pixels > 1e-6
        else 0.0
    )
    visible_fraction = max(0.0, min(1.0, float(visible_fraction)))
    required_visible_fraction = (
        cfg.edge_min_visible_fraction
        if touches_edge
        else cfg.min_visible_fraction
    )
    if visible_pixels <= 0:
        return _empty_support_result("no_visible_ring")

    local_hsv = hsv[y_start:y_end, x_start:x_end]
    local_ball_mask = ball_color_mask[y_start:y_end, x_start:x_end]
    local_floor_mask = floor_mask[y_start:y_end, x_start:x_end]
    if black_mask is None:
        black = black_support_mask(local_hsv, cfg.support_v_max) > 0
    else:
        black = black_mask[y_start:y_end, x_start:x_end] > 0
    # 검은색에 가까울수록 색조가 불안정하므로 어두운 픽셀에서는 색상 프로파일이
    # 겹칠 수 있습니다. 픽셀이 보정된 검은색 상한을 충족하면 빨간색 바닥이나
    # 주황색 번짐으로 중복 계산하지 않습니다.
    ball_like = (local_ball_mask > 0) & ~black
    floor_like = (local_floor_mask > 0) & ~black
    black_ratio = float(np.count_nonzero(black & ring) / visible_pixels)
    surrounding_ball_ratio = float(
        np.count_nonzero(ball_like & ring) / visible_pixels
    )
    floor_ratio = float(np.count_nonzero(floor_like & ring) / visible_pixels)

    # 하나의 그림자나 어두운 프레임 모서리만으로 받침대 조건을 충족해서는 안 됩니다.
    # 따라서 여러 각도 구역에서 검은색 근거가 확인되어야 합니다.
    angles = np.arctan2(dy, dx)
    sector_width = 2.0 * math.pi / cfg.sector_count
    sector_ids = np.floor((angles + math.pi) / sector_width).astype(np.int16)
    sector_ids = np.clip(sector_ids, 0, cfg.sector_count - 1)
    sector_ratios = []
    qualified_sectors = 0
    for sector in range(cfg.sector_count):
        sector_region = ring & (sector_ids == sector)
        sector_pixels = int(np.count_nonzero(sector_region))
        # 잘려서 매우 작아진 조각은 독립적인 받침대 근거로 보지 않습니다.
        if sector_pixels < 8:
            sector_ratios.append(0.0)
            continue
        ratio = float(np.count_nonzero(black & sector_region) / sector_pixels)
        sector_ratios.append(ratio)
        if ratio >= cfg.sector_black_ratio_min:
            qualified_sectors += 1

    required_black_ratio = (
        cfg.edge_black_ratio_min if touches_edge else cfg.black_ratio_min
    )
    required_sectors = (
        cfg.edge_min_sectors if touches_edge else cfg.min_sectors
    )

    failures = []
    if visible_fraction < required_visible_fraction:
        failures.append("support_not_visible")
    if black_ratio < required_black_ratio:
        failures.append("not_enough_black")
    if qualified_sectors < required_sectors:
        failures.append("black_not_distributed")
    if surrounding_ball_ratio > cfg.surrounding_ball_ratio_max:
        failures.append("ball_color_surrounding")
    if floor_ratio > cfg.floor_ratio_max:
        failures.append("floor_surrounding")

    return {
        "accepted": not failures,
        "reason": "ok" if not failures else ",".join(failures),
        "black_ratio": black_ratio,
        "surrounding_ball_ratio": surrounding_ball_ratio,
        "floor_ratio": floor_ratio,
        "qualified_sectors": int(qualified_sectors),
        "required_sectors": int(required_sectors),
        "sector_ratios": sector_ratios,
        "visible_fraction": visible_fraction,
        "required_visible_fraction": float(required_visible_fraction),
        "inner_radius_px": float(inner_radius),
        "outer_radius_px": float(outer_radius),
        "support_radius_px": float(expected_support_radius),
    }


def _empty_support_result(reason: str) -> Dict[str, Any]:
    return {
        "accepted": False,
        "reason": reason,
        "black_ratio": 0.0,
        "surrounding_ball_ratio": 0.0,
        "floor_ratio": 0.0,
        "qualified_sectors": 0,
        "required_sectors": 0,
        "sector_ratios": [],
        "visible_fraction": 0.0,
        "required_visible_fraction": 0.0,
        "inner_radius_px": 0.0,
        "outer_radius_px": 0.0,
        "support_radius_px": 0.0,
    }
