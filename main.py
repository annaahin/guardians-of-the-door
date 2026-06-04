from __future__ import annotations

import argparse

from modules.face_recognizer import FaceRecognizer
from modules.person_detector import PersonDetector, build_person_detection_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Door Safety AI demo runner"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source. Use 0 for webcam or pass a video file path.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model path or model name.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence score for person detection.",
    )
    parser.add_argument(
        "--enable-face-recognition",
        action="store_true",
        help="Compare detected face with images in data/registered_faces.",
    )
    parser.add_argument(
        "--registered-faces-dir",
        default="data/registered_faces",
        help="Directory containing registered face images.",
    )
    parser.add_argument(
        "--face-threshold",
        type=float,
        default=0.75,
        help="Minimum similarity for registered-face matching.",
    )
    return parser.parse_args()


def open_video_capture(source: str):
    import cv2

    capture_source: int | str
    capture_source = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return capture


def main() -> None:
    import cv2

    args = parse_args()
    detector = PersonDetector(model_path=args.model, confidence=args.confidence)
    face_recognizer = (
        FaceRecognizer(
            registered_faces_dir=args.registered_faces_dir,
            similarity_threshold=args.face_threshold,
        )
        if args.enable_face_recognition
        else None
    )

    capture = open_video_capture(args.source)
    print("Press q to quit.")

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        detections = detector.detect(frame)
        payload = build_person_detection_payload(detections)
        face_result = (
            face_recognizer.recognize_frame(frame)
            if face_recognizer is not None
            else None
        )
        annotated = detector.draw_detections(frame, detections)
        status = (
            f"person_detected={payload['person_detected']} "
            f"count={payload['person_count']}"
        )
        if face_result is not None:
            status += (
                f" registered={face_result.is_registered} "
                f"similarity={face_result.similarity:.2f}"
            )
        cv2.putText(
            annotated,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Door Safety AI", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
