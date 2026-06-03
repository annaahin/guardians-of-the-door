# 접근 인물 인식 모듈 조사 정리

## 1. 사람 감지

- 사용 기술: YOLO, OpenCV
- 목적: 현관 앞 영상에서 `person` 클래스만 탐지한다.
- 출력값: 사람 감지 여부, bounding box 좌표, confidence score
- 프로젝트 적용: `modules/person_detector.py`에서 YOLO 결과 중 class id 0만 사용한다.

## 2. 얼굴 인식

- 사용 후보: DeepFace, face_recognition, OpenCV face detector
- 목적: 등록된 얼굴 이미지와 실시간 영상 속 얼굴의 유사도를 비교한다.
- 출력값: 등록 인물 여부, 이름, 유사도 점수
- 예외 처리: 얼굴이 보이지 않으면 얼굴 인식 결과 대신 보행/체류/반복 방문 정보를 보조 판단에 사용한다.

## 3. 체형/보행/반복 출현 보조 판단

- 사용 후보: object tracking, YOLO tracking, DeepSORT, ByteTrack
- 목적: 같은 사람이 반복적으로 등장하는지, 현관 앞에 오래 머무는지 판단한다.
- 프로젝트 적용: 초기 버전에서는 사람 감지 여부를 기반으로 방문 횟수와 체류 시간을 계산한다.
- 확장 방향: 추적 ID를 붙여 같은 사람의 이동 경로와 재등장 여부를 더 정확히 분석한다.
