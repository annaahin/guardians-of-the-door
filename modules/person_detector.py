from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PERSON_CLASS_ID = 0


@dataclass(frozen=True)
class PersonDetection:
    """Detected person result from a video frame."""

    bbox: tuple[int, int, int, int]
    confidence: float
    class_name: str = "person"

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class PersonDetector:
    """YOLO-based person detector.

    This module is the first step of the approach-person recognition flow:
    frame input -> person detection -> bounding boxes/confidence output.
    """

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.5) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required. Install dependencies with: "
                "pip install -r requirements.txt"
            ) from exc

        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame: Any) -> list[PersonDetection]:
        results = self.model(frame, verbose=False)
        detections: list[PersonDetection] = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                score = float(box.conf[0])
                if class_id != PERSON_CLASS_ID or score < self.confidence:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    PersonDetection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=score,
                    )
                )

        return detections

    @staticmethod
    def draw_detections(frame: Any, detections: list[PersonDetection]) -> Any:
        import cv2

        annotated = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            label = f"{detection.class_name} {detection.confidence:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        return annotated
