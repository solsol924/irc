#!/usr/bin/env python3
"""
RealSense RGB의 60 Hz anti-flicker와 현장 노출/WB를 고정한다.

실행 순서
1. RealSense 파라미터 서비스가 나타날 때까지 기다린다.
2. 한국/60 Hz 조명에 맞게 power_line_frequency=2를 적용한다.
3. 자동 노출과 자동 white balance를 켠 상태로 현장 조명에 적응시킨다.
4. warmup 시간이 지나면 현재 exposure/gain/white_balance 값을 읽는다.
5. 자동 기능을 끈 뒤 읽어 둔 값을 다시 적용해 HSV 밝기/색이 계속 움직이지
   않게 한다.

power_line_frequency 값은 Linux UVC 규격에서 0=off, 1=50 Hz, 2=60 Hz,
3=auto이다. 사용 중인 RealSense가 [0, 2]만 지원하므로 auto(3)가 아니라
60 Hz(2)를 명시한다.

이 프로세스는 카메라 설정을 한 번 적용하고 종료한다. 공 검출 노드와 프레임
처리를 막거나 로봇에 정지 명령을 보내지 않는다.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, Iterable

import rclpy
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.node import Node
from rclpy.parameter import Parameter, parameter_value_to_python


COLOR_AUTO_PARAMETERS = (
    "rgb_camera.enable_auto_exposure",
    "rgb_camera.enable_auto_white_balance",
)
COLOR_MANUAL_PARAMETERS = (
    "rgb_camera.exposure",
    "rgb_camera.gain",
    "rgb_camera.white_balance",
)


def _service_name(node_name: str, suffix: str) -> str:
    return f"{node_name.rstrip('/')}/{suffix}"


def _get_parameters(
    node: Node,
    client,
    names: Iterable[str],
    timeout_sec: float = 5.0,
) -> Dict[str, Any]:
    requested = list(names)
    request = GetParameters.Request()
    request.names = requested
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
    response = future.result()
    if response is None:
        return {}
    return {
        name: parameter_value_to_python(value)
        for name, value in zip(requested, response.values)
    }


def _set_parameters(
    node: Node,
    client,
    values: Dict[str, Any],
    timeout_sec: float = 5.0,
) -> bool:
    request = SetParameters.Request()
    request.parameters = [
        Parameter(name=name, value=value).to_parameter_msg()
        for name, value in values.items()
        if value is not None
    ]
    if not request.parameters:
        return True
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
    response = future.result()
    if response is None:
        return False
    successful = True
    for parameter, result in zip(request.parameters, response.results):
        if result.successful:
            continue
        successful = False
        node.get_logger().warning(
            f"Could not set {parameter.name}: {result.reason}"
        )
    return successful


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-node", default="/camera")
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    parser.add_argument("--power-line-frequency", type=int, default=2)
    parser.add_argument("--service-wait-seconds", type=float, default=20.0)
    args = parser.parse_args()

    rclpy.init()
    node = Node("realsense_rgb_stabilizer")
    get_client = node.create_client(
        GetParameters,
        _service_name(args.camera_node, "get_parameters"),
    )
    set_client = node.create_client(
        SetParameters,
        _service_name(args.camera_node, "set_parameters"),
    )

    try:
        if not get_client.wait_for_service(
            timeout_sec=max(0.1, args.service_wait_seconds)
        ) or not set_client.wait_for_service(
            timeout_sec=max(0.1, args.service_wait_seconds)
        ):
            node.get_logger().error(
                f"Camera parameter services not found for {args.camera_node}"
            )
            return

        _set_parameters(
            node,
            set_client,
            {
                "rgb_camera.power_line_frequency":
                    int(args.power_line_frequency),
                "rgb_camera.enable_auto_exposure": True,
                "rgb_camera.enable_auto_white_balance": True,
            },
        )
        warmup_seconds = max(0.0, float(args.warmup_seconds))
        node.get_logger().info(
            "RGB auto exposure/WB warmup started: "
            f"{warmup_seconds:.1f}s"
        )
        deadline = time.monotonic() + warmup_seconds
        while rclpy.ok() and time.monotonic() < deadline:
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

        settled = _get_parameters(
            node,
            get_client,
            COLOR_MANUAL_PARAMETERS,
        )
        node.get_logger().info(f"Settled RGB values: {settled}")

        # 자동 기능을 먼저 끄면 UVC 장치가 마지막으로 수렴한 값을 유지한다.
        # 그 뒤 읽어 둔 값을 명시적으로 다시 써서 드라이버/펌웨어별 차이에도
        # 동일한 현장값이 유지되도록 한다.
        _set_parameters(
            node,
            set_client,
            {name: False for name in COLOR_AUTO_PARAMETERS},
        )
        _set_parameters(node, set_client, settled)

        verified = _get_parameters(
            node,
            get_client,
            (
                "rgb_camera.power_line_frequency",
                *COLOR_AUTO_PARAMETERS,
                *COLOR_MANUAL_PARAMETERS,
            ),
        )
        node.get_logger().info(f"Locked RGB parameters: {verified}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
