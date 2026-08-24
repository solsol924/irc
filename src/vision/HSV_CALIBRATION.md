# RealSense 공·Hoop HSV 튜닝 및 적용

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

같은 화면에 공과 받침대가 함께 있어도 됩니다. 각 대상의 작은 영역을
따로 선택해 다음 순서로 샘플을 모읍니다.

1. `B`: 주황색 공 선택 → 공 내부 ROI → `SPACE`를 여러 번 → `A`
2. `K`: 검은 받침대 선택 → 공이 섞이지 않은 검은 부분 ROI를 여러 곳에서
   `SPACE` → `A`
3. `F`: 빨간 바닥 선택 → 받침대가 섞이지 않은 바닥 ROI를 여러 곳에서
   `SPACE` → `A`
4. 다시 `B`로 전환하고 `D`를 눌러 실제 공+받침대 검출 조건을 확인
5. 결과가 괜찮으면 `S`를 눌러 확정

Hoop는 빨간 테두리와 흰 내부를 반드시 따로 샘플링합니다.

1. `G`: hoop의 빨간 테두리 선택
2. 빨간색만 들어오도록 작은 ROI를 여러 위치·거리·조명에서 선택하고
   각각 `SPACE`
3. `A`를 눌러 빨간색 H/S/V 범위를 자동 계산하고 `D`로 마스크 확인
4. `W`: hoop의 흰 내부 선택
5. 빨간 테두리나 배경이 섞이지 않도록 흰 부분만 여러 번 `SPACE`
6. `A`를 눌러 흰색의 `S high`와 `V low`를 계산하고 `D`로 확인
7. 다시 `G`, `W`를 오가며 마스크를 확인한 뒤 `S`로 저장

빨간색은 OpenCV Hue의 0/179 경계를 자동으로 처리해 두 구간으로
`hoop_hsv.yaml`에 저장합니다. 흰색은 Hue를 사용하지 않고 채도 상한과
밝기 하한을 사용합니다.

받침대는 검은색이라 Hue가 불안정하므로 `K`의 자동 맞춤은 H/S 평균을
사용하지 않고, 여러 샘플의 V(밝기) 95백분위에 여유값을 더한 검정 상한을
저장합니다. 공·받침대·바닥을 물리적으로 분리해 촬영할 필요는 없습니다.

공 자동 튜닝은 ROI에서 `V low`를 계산하되 `V high`는 항상 255로 유지하여,
같은 공이 더 밝은 조명을 받았을 때 밝기 상한 때문에 제외되지 않게 합니다.

- `config/hsv_profiles.yaml`: 공·받침대·바닥·후프의 전체 최신 프로필
- `config/ball_hsv.yaml`: 공 검출 노드가 시작할 때 읽는 공/받침대/바닥
  확정값과 받침대 판정 기준
- `config/hoop_hsv.yaml`: hoop 검출 노드가 시작할 때 읽는 빨간 테두리와
  흰 내부 HSV 확정값
- `config/backups/`: `S`로 전체 프로필을 덮어쓰기 전 값

`A`만 누른 값은 생산 설정에 반영되지 않습니다. `D`로 확인한 뒤 `S`를
눌러야 `ball_hsv.yaml`과 `hoop_hsv.yaml`이 갱신됩니다.

## 3. 평소 로봇 실행

```bash
cd ~/irc
source install/setup.bash
ros2 launch robot_bringup robot_bringup.py
```

`ball_vision_fusion.py`가 시작할 때 `config/ball_hsv.yaml`을 한 번 직접
읽습니다. 모든 RealSense 공 후보는 공 HSV/Depth/형상뿐 아니라 지름
150 mm의 검은 받침대와 빨간 바닥 배제 조건도 통과해야 합니다. 공의 실제
지름 허용 범위는 50~70 mm입니다. 가장자리 후보는 영상 안에 보이는 받침대
영역만 계산하되, 강화된 원형도와 실제 크기 조건을 함께 적용합니다.

`hoop_vision.py`도 시작할 때 `config/hoop_hsv.yaml`을 직접 읽습니다.
따라서 캘리브레이터에서 `S`를 누른 뒤 vision 노드를 다시 시작하면 새
빨강/흰색 기준이 적용됩니다.
