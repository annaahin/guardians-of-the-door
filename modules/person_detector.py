"""
YOLO를 이용해 영상 프레임에서 사람(person)을 감지하는 모듈.

이 파일은 "분류 모델"을 만드는 코드가 아닙니다.
YOLO는 영상 안에서 사람의 위치를 찾기 위한 보조 도구로만 사용합니다.
"""

from ultralytics import YOLO


class PersonDetector:
    """YOLO 모델로 한 프레임 안의 사람 bounding box를 찾는 클래스."""

    # YOLO가 기본으로 사용하는 COCO 데이터셋에서 person 클래스 번호는 0입니다.
    PERSON_CLASS_ID = 0

    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.3):
        """
        Args:
            model_path: 사용할 YOLO 모델 파일 경로입니다. yolov8n.pt는 가볍고 테스트하기 좋습니다.
            confidence_threshold: 이 값보다 낮은 신뢰도의 탐지 결과는 버립니다.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect_people(self, frame):
        """
        한 장의 frame에서 사람을 감지합니다.

        frame이란 영상에서 잘라낸 "한 장의 이미지"입니다.
        예를 들어 30 FPS 영상은 1초에 frame 30장을 보여주는 영상입니다.

        Args:
            frame: OpenCV가 읽어온 이미지 한 장입니다. numpy 배열 형태입니다.

        Returns:
            감지된 사람 정보를 담은 딕셔너리 리스트입니다.
            각 딕셔너리는 bounding box, center point, confidence, box_area를 포함합니다.
        """
        # verbose=False는 YOLO가 프레임마다 긴 로그를 출력하지 않게 합니다.
        results = self.model(frame, verbose=False)
        detections = []

        # results[0].boxes에는 현재 frame에서 감지된 모든 객체의 박스 정보가 들어 있습니다.
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # COCO 클래스 중 person만 사용합니다.
            if class_id != self.PERSON_CLASS_ID:
                continue

            if confidence < self.confidence_threshold:
                continue

            # bounding box는 객체를 둘러싼 사각형입니다.
            # x1, y1은 왼쪽 위 좌표이고 x2, y2는 오른쪽 아래 좌표입니다.
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # center point는 bounding box의 중앙점입니다.
            # 사람의 위치를 하나의 점으로 추적하기 위해 사용합니다.
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            box_area = max(0, x2 - x1) * max(0, y2 - y1)

            detections.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "center_x": center_x,
                    "center_y": center_y,
                    "confidence": confidence,
                    "box_area": box_area,
                }
            )

        # 한 프레임에 여러 명이 잡힐 수 있으므로, 신뢰도가 높은 순서로 정렬합니다.
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        return detections

    def detect_main_person(self, frame):
        """
        현재 프로젝트는 현관 앞 접근 인물 1명을 중심으로 보기 때문에
        가장 confidence가 높은 사람 1명을 대표 인물로 사용합니다.

        Returns:
            사람이 감지되면 딕셔너리 1개, 감지되지 않으면 None을 반환합니다.
        """
        detections = self.detect_people(frame)
        if not detections:
            return None
        return detections[0]
