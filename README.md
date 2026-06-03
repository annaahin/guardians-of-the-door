# Door Safety AI

멀티모달 AI 기반 데이트폭력 피해자 보호를 위한 접근 인물 식별 및 위험행동 감지 시스템

## 프로젝트 개요

본 프로젝트는 현관 앞 영상 및 음성 데이터를 기반으로 접근 인물을 감지하고, 장시간 체류, 반복 배회, 고성 등의 위험 신호를 분석하여 사용자에게 위험 단계를 알리는 AI 기반 안전 서비스이다.

## 주요 기능

- 컴퓨터 비전 기반 사람 감지
- 등록 인물 / 미등록 인물 구분
- 현관 앞 장시간 체류 감지
- 반복 방문 및 배회 감지
- 음성 기반 고성 및 위험 신호 감지
- 위험 점수 계산 및 알림 출력

## 팀원 역할

| 이름 | 담당 |
|---|---|
| 음아인 | 인물 탐지, 얼굴 인식, 보행 패턴 및 객체 추적 조사 |
| 문지민 | 음성 인식, 고성 및 위험 음성 신호 분석 |
| 송시은 | 위험행동 감지, 장시간 체류 및 배회 유형 정리 |

## 실행 방법

### 1. 가상환경 생성

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 웹캠으로 사람 감지 실행

```bash
python main.py --source 0
```

### 4. 영상 파일로 실행

```bash
python main.py --source data/videos/sample.mp4
```

## 브랜치 전략

- `main`: 최종 제출용 안정 버전
- `develop`: 팀원이 확인한 기능을 합치는 개발 브랜치
- `feature/person-recognition`: 접근 인물 인식 모듈 작업 브랜치

## 현재 구현 범위

이 브랜치는 음아인 담당 범위만 다룬다.

- `modules/person_detector.py`: YOLO 기반 사람 감지
- `modules/face_recognizer.py`: 등록 얼굴 비교 기능을 붙이기 위한 기본 구조
- `data/registered_faces/`: 등록 인물 얼굴 이미지 저장 위치
- `docs/research.md`: 접근 인물 인식 관련 조사 정리
