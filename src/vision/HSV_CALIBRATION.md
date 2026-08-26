# RealSense 레거시 공 HSV 튜닝

이 문서는 RealSense 공 검출을 HSV/OpenCV fallback으로 수동 실행할 때만
사용합니다. 기본 `robot_bringup`은 RealSense 전용 YOLO를 사용하므로 이
HSV 값으로 공이나 hoop를 검출하지 않습니다.

## 1. RealSense만 실행

```bash
cd ~/irc
source install/setup.bash
ros2 launch robot_bringup vision_stack.launch.py \
  start_webcam:=false \
  start_yolo:=false \
  start_ball:=false \
  start_hurdle:=false \
  start_monitor:=false
```

## 2. 다른 터미널에서 캘리브레이터 실행

```bash
cd ~/irc
source install/setup.bash
ros2 run vision realsense_hsv_calibrator.py
```

같은 화면에 공과 받침대가 함께 있어도 됩니다. 각 대상의 작은 영역을
따로 선택해 다음 순서로 샘플을 모읍니다.

1. `B`: 주황색 공 선택 → 공 내부 ROI → `SPACE`를 여러 번 → `A`
2. `K`: 검은 받침대 선택 → 공이 섞이지 않은 검은 부분 ROI를 여러 곳에서
   `SPACE` → `A`
3. `F`: 빨간 바닥 선택 → 받침대가 섞이지 않은 바닥 ROI를 여러 곳에서
   `SPACE` → `A`
4. 다시 `B`로 전환하고 `D`를 눌러 실제 공+받침대 검출 조건을 확인
5. 결과가 괜찮으면 `S`를 눌러 확정

받침대는 검은색이라 Hue가 불안정하므로 `K`의 자동 맞춤은 H/S 평균을
사용하지 않고, 여러 샘플의 V(밝기) 95백분위에 여유값을 더한 검정 상한을
저장합니다. 공·받침대·바닥을 물리적으로 분리해 촬영할 필요는 없습니다.

공 자동 튜닝은 ROI에서 `V low`를 계산하되 `V high`는 항상 255로 유지하여,
같은 공이 더 밝은 조명을 받았을 때 밝기 상한 때문에 제외되지 않게 합니다.

- `config/hsv_profiles.yaml`: 공·받침대·바닥의 전체 최신 프로필
- `config/ball_hsv.yaml`: 공 검출 노드가 시작할 때 읽는 공/받침대/바닥
  확정값과 받침대 판정 기준
- `config/backups/`: `S`로 전체 프로필을 덮어쓰기 전 값

`A`만 누른 값은 생산 설정에 반영되지 않습니다. `D`로 확인한 뒤 `S`를
눌러야 `ball_hsv.yaml`이 갱신됩니다.

## 3. 평소 YOLO 로봇 실행

```bash
cd ~/irc
source install/setup.bash
ros2 launch robot_bringup robot_bringup.py
```

`vision_stack.launch.py`는 `ball_vision_fusion.py`에
`use_realsense_yolo:=true`를 전달합니다. 따라서 평소 실행에서는 레거시
HSV/OpenCV 공 영상 처리를 시작하지 않으며, hoop도 같은 RealSense YOLO
엔진의 `goal/backboard` 결과를 사용합니다.
