# RealSense 공 HSV 튜닝 및 적용

캘리브레이터는 필요할 때만 별도 실행합니다. `robot_bringup`은 캘리브레이터를
실행하지 않고, 마지막으로 `S`를 눌러 확정한 공 HSV만 읽습니다.

## 1. RealSense만 실행

```bash
cd ~/irc
source install/setup.bash
ros2 launch robot_bringup vision_stack.launch.py \
  start_webcam:=false \
  start_yolo:=false \
  start_ball:=false \
  start_hurdle:=false \
  start_hoop:=false \
  start_monitor:=false \
  start_selector:=false
```

## 2. 다른 터미널에서 캘리브레이터 실행

```bash
cd ~/irc
source install/setup.bash
ros2 run vision realsense_hsv_calibrator.py
```

공을 선택한 상태에서 `ROI -> SPACE -> A -> D` 순서로 확인합니다. 결과가
괜찮을 때 `S`를 눌러 확정합니다.

공 자동 튜닝은 ROI에서 `V low`를 계산하되 `V high`는 항상 255로 유지하여,
같은 공이 더 밝은 조명을 받았을 때 밝기 상한 때문에 제외되지 않게 합니다.

- `config/hsv_profiles.yaml`: 다음 튜닝 때 이어서 사용할 전체 최신 프로필
- `config/ball_hsv.yaml`: 공 검출 노드가 시작할 때 읽는 최신 확정 H/S/V
- `config/backups/`: `S`로 전체 프로필을 덮어쓰기 전 값

`A`만 누른 값은 생산 설정에 반영되지 않습니다. `D`로 확인한 뒤 `S`를
눌러야 `ball_hsv.yaml`이 갱신됩니다.

## 3. 평소 로봇 실행

```bash
cd ~/irc
source install/setup.bash
ros2 launch robot_bringup robot_bringup.py
```

`ball_vision_fusion.py`가 시작할 때 `config/ball_hsv.yaml`을 한 번 직접
읽습니다. 따라서 캘리브레이터는 실행되지 않고 공 검출 노드만 저장된 HSV로
시작하며, 이 기능은 `src/vision` 폴더 안에서 완결됩니다.
