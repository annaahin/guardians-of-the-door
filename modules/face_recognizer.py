from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


FACE_IMAGE_SIZE = (100, 100)


@dataclass(frozen=True)
class FaceMatchResult:
    name: str | None
    similarity: float
    is_registered: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "is_registered": self.is_registered,
            "name": self.name,
            "similarity": round(self.similarity, 4),
        }


@dataclass(frozen=True)
class RegisteredFace:
    name: str
    image_path: Path
    histogram: Any


class FaceRecognizer:
    """OpenCV-based registered-face matcher.

    This is a lightweight baseline for the prototype. It can be replaced with
    DeepFace or face_recognition later without changing the output shape.
    """

    def __init__(
        self,
        registered_faces_dir: str | Path = "data/registered_faces",
        similarity_threshold: float = 0.75,
    ) -> None:
        import cv2

        self.registered_faces_dir = Path(registered_faces_dir)
        self.similarity_threshold = similarity_threshold
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.face_detector.empty():
            raise RuntimeError("Cannot load OpenCV Haar face detector.")
        self.registered_faces = self.load_registered_faces()

    def load_registered_faces(self) -> list[RegisteredFace]:
        import cv2

        registered_faces: list[RegisteredFace] = []
        for image_path in self.list_registered_images():
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            registered_faces.append(
                RegisteredFace(
                    name=self.get_name_from_path(image_path),
                    image_path=image_path,
                    histogram=self.compute_histogram(image),
                )
            )

        return registered_faces

    def recognize(self, face_image: Any) -> FaceMatchResult:
        if not self.registered_faces:
            return FaceMatchResult(name=None, similarity=0.0, is_registered=False)

        query_histogram = self.compute_histogram(face_image)
        best_name: str | None = None
        best_similarity = 0.0

        for registered_face in self.registered_faces:
            similarity = self.compare_histograms(
                query_histogram,
                registered_face.histogram,
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_name = registered_face.name

        is_registered = best_similarity >= self.similarity_threshold
        return FaceMatchResult(
            name=best_name if is_registered else None,
            similarity=best_similarity,
            is_registered=is_registered,
        )

    def recognize_frame(self, frame: Any) -> FaceMatchResult:
        face_crop = self.extract_largest_face(frame)
        if face_crop is None:
            return FaceMatchResult(name=None, similarity=0.0, is_registered=False)
        return self.recognize(face_crop)

    def extract_largest_face(self, image: Any) -> Any | None:
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        return image[y : y + h, x : x + w]

    @staticmethod
    def compute_histogram(face_image: Any) -> Any:
        import cv2

        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, FACE_IMAGE_SIZE)
        gray = cv2.equalizeHist(gray)
        histogram = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(histogram, histogram)
        return histogram

    @staticmethod
    def compare_histograms(query_histogram: Any, registered_histogram: Any) -> float:
        import cv2

        correlation = cv2.compareHist(
            query_histogram,
            registered_histogram,
            cv2.HISTCMP_CORREL,
        )
        return max(0.0, min(1.0, (correlation + 1.0) / 2.0))

    def list_registered_images(self) -> list[Path]:
        if not self.registered_faces_dir.exists():
            return []
        allowed = {".jpg", ".jpeg", ".png"}
        return sorted(
            path
            for path in self.registered_faces_dir.iterdir()
            if path.suffix.lower() in allowed
        )

    @staticmethod
    def get_name_from_path(image_path: Path) -> str:
        return image_path.stem.split("_")[0]
