# Door Safety Behavior Classification

## 프로젝트 주제

**자체 구축 영상 데이터셋을 활용한 현관 앞 접근 인물의 위험행동 분류 모델 개발**

---

## 프로젝트 개요

본 프로젝트는 현관 앞 접근 상황을 촬영한 자체 영상 데이터셋을 구축하고, 영상에서 추출한 행동 특징을 기반으로 접근 인물의 행동을 `normal`, `suspicious`, `dangerous` 3개 클래스로 분류하는 모델을 개발하는 것을 목표로 한다.

기존 CCTV나 보안 장비는 주로 영상 기록 또는 단순 움직임 감지에 의존하는 경우가 많다. 본 프로젝트에서는 팀원들이 직접 촬영한 현관 앞 접근 상황 영상을 활용하여, 사람의 체류 시간, 위험구역 진입 비율, 이동 거리, 평균 속도, 방향 전환 횟수 등의 행동 특징을 추출하고 이를 바탕으로 위험행동 분류 모델을 학습한다.

본 프로젝트는 단순히 기존 모델을 가져와 서비스 형태로 연결하는 것이 아니라, 자체 구축 데이터셋을 기반으로 특징을 추출하고 분류 모델을 학습시키는 과정을 포함한다.

---

## 프로젝트 목표

- 현관 앞 접근 상황에 대한 자체 영상 데이터셋 구축
- `normal`, `suspicious`, `dangerous` 3개 클래스 정의 및 라벨링
- YOLO 기반 사람 감지 및 위치 정보 추출
- 사람의 움직임을 바탕으로 행동 특징 추출
- 추출된 특징값을 CSV 데이터셋으로 저장
- RandomForest, SVM 등 머신러닝 기반 위험행동 분류 모델 학습
- 테스트 영상에 대한 위험행동 예측 결과 출력

---

## 행동 클래스 정의

| 클래스 | 의미 | 예시 |
|---|---|---|
| `normal` | 정상적인 접근 또는 방문 행동 | 문 앞을 지나감, 택배를 놓고 이동, 짧게 대기 후 이동 |
| `suspicious` | 의심스러운 접근 행동 | 문 앞 장시간 체류, 반복 배회, 주변을 두리번거림 |
| `dangerous` | 위험 가능성이 높은 행동 | 문 손잡이에 손을 뻗음, 문을 열려고 시도, 강한 노크 또는 위협적인 몸짓 |

---

## 전체 개발 흐름

1. 현관 앞 접근 상황 영상 촬영
2. 영상 데이터를 `normal`, `suspicious`, `dangerous`로 라벨링
3. YOLO를 활용하여 영상 속 사람 감지
4. 사람의 bounding box 및 중심점 좌표 추출
5. 위험구역 진입 여부 및 이동 패턴 분석
6. 영상별 행동 특징 추출
7. `features.csv` 생성
8. `features.csv`와 `labels.csv`를 이용한 분류 모델 학습
9. 테스트 영상에 대한 위험행동 클래스 예측
10. 예측 결과 시각화 및 최종 시연

---

## 주요 기능

- 영상 파일 입력 및 프레임 처리
- YOLO 기반 사람 감지
- 사람 bounding box 좌표 추출
- 사람 중심점 좌표 계산
- 현관 앞 위험구역 설정
- 위험구역 진입 여부 판단
- 체류 시간 및 위험구역 체류 비율 계산
- 이동 거리 및 평균 속도 계산
- 방향 전환 횟수 계산
- 영상별 행동 특징 CSV 저장
- 위험행동 분류 모델 학습 및 평가
- 테스트 영상 예측 결과 출력

---

## 팀원 역할

| 이름 | 담당 |
|---|---|
| 음아인 | 영상 기반 인물 탐지 및 행동 특징 추출 모듈 |
| 문지민 | 자체 영상 데이터셋 구축 및 라벨링 관리 |
| 송시은 | 위험행동 분류 모델 학습 및 평가 |

---

## 역할별 세부 내용

### 음아인 - 영상 기반 인물 탐지 및 행동 특징 추출

- OpenCV 기반 영상 입력 처리
- YOLO 기반 사람 감지
- 사람 bounding box 좌표 추출
- 사람 중심점 계산
- 현관 앞 위험구역 설정
- 위험구역 진입 여부 판단
- 체류 시간, 이동 거리, 평균 속도, 방향 전환 횟수 등 행동 특징 추출
- 영상별 특징값을 `features.csv`로 저장

### 문지민 - 자체 영상 데이터셋 구축 및 라벨링 관리

- `normal`, `suspicious`, `dangerous` 클래스 정의
- 클래스별 촬영 시나리오 작성
- 영상 파일명 규칙 정리
- 팀원 촬영 영상 수합 및 정리
- `labels.csv` 작성
- 데이터셋 구성 및 라벨링 기준 문서화

### 송시은 - 위험행동 분류 모델 학습 및 평가

- `features.csv`와 `labels.csv`를 활용한 학습 데이터 구성
- train/test 데이터 분리
- RandomForest, SVM 등 분류 모델 학습
- `normal`, `suspicious`, `dangerous` 3개 클래스 예측
- 정확도 및 confusion matrix 기반 모델 평가
- 학습된 모델 저장 및 예측 코드 구현

---

## 데이터셋 구성

본 프로젝트에서는 팀원들이 직접 현관 앞 접근 상황을 연기하여 촬영한 영상을 사용한다.

목표 데이터 수는 다음과 같다.

| 클래스 | 영상 개수 |
|---|---:|
| `normal` | 30개 |
| `suspicious` | 30개 |
| `dangerous` | 30개 |
| 총합 | 90개 |

영상 파일명은 아래와 같은 형식을 사용한다.

```text
normal_001.mp4
normal_002.mp4
suspicious_001.mp4
suspicious_002.mp4
dangerous_001.mp4
dangerous_002.mp4
## 촬영 시나리오 예시

### normal

```

클래스 폴더 안에 행동별 하위 폴더가 있어도 된다.

```text
data/videos/normal/1. 문 앞 지나가기/
data/videos/normal/2. 잠깐 섰다가 이동/
data/videos/suspicious/문 앞 10초 이상 서 있기/
data/videos/dangerous/문을 강하게 두드리는 행동/
```

단, `features.csv`의 `video_name`은 파일명만 저장하므로 전체 영상 파일명은 서로 겹치지 않게 관리한다.

## 촬영 시나리오 예시

### normal

- 문 앞을 자연스럽게 지나가기
- 택배를 내려놓고 바로 이동하기
- 초인종을 누르는 척한 뒤 짧게 대기하기
- 휴대폰을 보며 지나가기
- 문 앞에 잠깐 섰다가 바로 이동하기

### suspicious

- 문 앞에서 장시간 머무르기
- 문 앞을 반복적으로 왔다 갔다 하기
- 주변을 계속 두리번거리기
- 문 가까이 갔다가 뒤로 물러나기를 반복하기
- 문 앞에서 휴대폰을 보는 척하며 오래 머무르기

### dangerous

- 문 손잡이에 손을 뻗는 행동
- 문을 열려고 시도하는 행동
- 문 앞에 매우 가까이 붙어 오래 머무르기
- 강하게 노크하는 행동
- 위협적인 팔 동작 또는 몸짓

---

## 특징 추출 항목

각 영상에서 추출할 주요 특징은 다음과 같다.

| 특징명 | 설명 |
|---|---|
| `duration` | 영상 전체 길이 |
| `detected_ratio` | 전체 프레임 중 사람이 감지된 비율 |
| `zone_ratio` | 사람이 위험구역 안에 있었던 비율 |
| `zone_ratio` | 사람이 감지된 프레임 중 위험구역 안에 있었던 비율 |
| `zone_stay_time` | 사람이 위험구역 안에 머문 시간(초) |
| `zone_entry_count` | 위험구역에 진입한 횟수 |
| `move_distance` | 사람 중심점의 총 이동 거리 |
| `avg_speed` | 평균 이동 속도 |
| `direction_change` | 이동 방향이 바뀐 횟수 |
| `max_box_area` | 영상 내 사람 bounding box의 최대 면적 |

---

## CSV 파일 형식

### labels.csv

`labels.csv`는 영상 파일명과 실제 라벨 정보를 저장한다.

```csv
video_name,label,scenario
normal_001.mp4,normal,문 앞을 지나감
suspicious_001.mp4,suspicious,문 앞에서 배회
dangerous_001.mp4,dangerous,문 손잡이에 손을 뻗음
```

### features.csv

`features.csv`는 영상에서 추출한 행동 특징값을 저장한다.

```csv
video_name,duration,detected_ratio,zone_ratio,zone_entry_count,move_distance,avg_speed,direction_change,max_box_area
normal_001.mp4,5.2,0.95,0.12,1,230.5,44.3,1,30240
suspicious_001.mp4,12.4,0.98,0.76,3,610.2,49.2,5,42100
dangerous_001.mp4,10.1,0.97,0.91,1,120.8,11.9,1,58000
`features.csv`는 영상에서 추출한 행동 특징값을 저장한다. `label`은 포함하지 않으며, 이후 `labels.csv`와 `video_name` 기준으로 병합한다.

```csv
video_name,duration,detected_ratio,zone_ratio,zone_stay_time,zone_entry_count,move_distance,avg_speed,direction_change,max_box_area
normal_001.mp4,5.2,0.95,0.12,0.62,1,230.5,44.3,1,30240
suspicious_001.mp4,12.4,0.98,0.76,9.24,3,610.2,49.2,5,42100
dangerous_001.mp4,10.1,0.97,0.91,8.92,1,120.8,11.9,1,58000
```

---

## 프로젝트 폴더 구조

```text
guardians-of-the-door/
│
├── main.py
├── requirements.txt
├── README.md
│
├── modules/
│   ├── __init__.py
│   ├── person_detector.py
│   ├── feature_extractor.py
│   ├── model_trainer.py
│   ├── risk_classifier.py
│   └── visualizer.py
│
├── data/
│   ├── videos/
│   │   ├── normal/
│   │   ├── suspicious/
│   │   └── dangerous/
│   ├── labels.csv
│   └── features.csv
│
├── models/
│   └── risk_model.pkl
│
├── results/
│   ├── screenshots/
│   ├── demo_videos/
│   └── confusion_matrix.png
│
└── docs/
    ├── dataset_description.md
    ├── label_rules.md
    └── model_description.md
```

---

## 설치 방법

필요한 라이브러리는 `requirements.txt`에 정리한다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

주요 사용 라이브러리는 다음과 같다.

```text
opencv-python
ultralytics
numpy
pandas
scikit-learn
matplotlib
joblib
```

---

## 실행 방법

전체 영상 데이터셋에서 행동 특징을 추출하려면 다음 명령어를 실행한다.

```bash
python main.py
```

특정 영상 하나만 테스트하려면 다음과 같이 실행할 수 있다.

```bash
python main.py --video data/videos/normal/normal_001.mp4
```

---

## 최종 산출물

- 자체 구축 영상 데이터셋
- `labels.csv`
- `features.csv`
- 위험행동 분류 모델
- 모델 평가 결과
- 시연 영상 또는 결과 캡처
- 최종 보고서
- 발표 자료

---

python main.py --video "data/videos/normal/normal_001.mp4"
```

YOLO 탐지 결과, 사람 bounding box, 중심점, 위험구역을 화면으로 확인하고 싶으면 `--show` 옵션을 추가한다.

```bash
python main.py --video "data/videos/normal/normal_001.mp4" --show
```

위험구역 좌표는 실행 시 변경할 수 있다.

```bash
python main.py --zone-x1 80 --zone-y1 450 --zone-x2 780 --zone-y2 1450
```

---

## 최종 산출물

- 자체 구축 영상 데이터셋
- `labels.csv`
- `features.csv`
- 위험행동 분류 모델
- 모델 평가 결과
- 시연 영상 또는 결과 캡처
- 최종 보고서
- 발표 자료

---

## 향후 확장 가능성

본 프로젝트에서는 영상 기반 행동 특징 추출 및 위험행동 분류 모델 학습에 집중한다. 향후에는 다음 기능으로 확장할 수 있다.

- 등록 인물 얼굴 인식
- 음성 기반 고성 및 노크 감지
- 실시간 CCTV 스트리밍 적용
- 위험 점수 기반 단계별 알림
- 보호자 알림 또는 신고 연동
