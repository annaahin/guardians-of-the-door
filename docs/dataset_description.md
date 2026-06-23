# 데이터셋 구성표 (Dataset Description)

본 문서는 모델 학습에 사용되는 자체 구축 영상 데이터셋의 전체 통계 및 분포를 기록하는 문서이다.

## 1. 데이터셋 총괄 요약
- 데이터셋 구축 담당자: 문지민
- 총 영상 개수: 82개 
- 영상 포맷: MOV 및 MP4

## 2. 클래스별 데이터 분포
데이터셋의 클래스 불균형 문제를 방지하기 위해 균등한 비율로 수합됨

| 클래스명 (Class) | 영상 개수 (Count) | 데이터 비율 (%) | 데이터 정량적 특징 요약 |

| **normal** | 32개 |  약 37.5% | 짧은 체류 시간 (대부분 8초 미만), 일방향 이동 동선 |
| **suspicious** | 25개 | 약 31.25% | 긴 체류 시간 (8초 이상), 복도 내 왕복 이동 |
| **dangerous** | 25개 | 약 31.25% | 문 앞 집중 체류 (80% 이상), 도어락/문고리 마찰 발생 |
| **합계** | **82 개** | **100%** | **총 특징 데이터 추출 준비 완료** |

## 3. 디렉토리 구조 (Directory Structure)

osp_project/
├── data/                             # 데이터 관리 폴더
│   ├── videos/                       # 원본 영상 데이터셋
│   │   ├── normal/                   # 일반 상황 영상 (osp_normal_001.mp4 등)
│   │   ├── suspicious/               # 의심 상황 영상 (osp_suspicious_001.mp4 등)
│   │   └── dangerous/                # 위험 상황 영상 (osp_dangerous_001.mp4 등)
│   ├── labels.csv                    # 영상 파일 경로 및 정답 라벨 매핑 파일
│   └── features.csv                  # 영상별로 추출된 14개 행동 특징값 데이터
│
├── modules/                          # 기능별 독립 파이썬 모듈
│   ├── person_detector.py            # OpenCV 및 YOLO 기반 사람 감지/ROI 설정
│   ├── feature_extractor.py          # 체류 시간, 이동 거리 등 특징 추출 및 CSV 저장
│   └── model_trainer.py              # RandomForest 모델 학습 및 성능 평가 담당
│
├── models/                           # 학습 완료된 모델 저장소
│   └── risk_model.pkl                # 최종 학습된 위험행동 분류 모델 가중치 파일
│
├── docs/                             # 프로젝트 및 데이터셋 문서화
│   ├── label_rules.md                # 클래스별 정량적 기준 및 촬영 가이드라인
│   └── dataset_description.md         # 전체 데이터셋 통계 및 환경 기술 문서
│
├── main.py                           # 전체 파이프라인 제어 및 실행 메인 스크립트
└── requirements.txt                  # 개발 환경 구축을 위한 라이브러리 목록