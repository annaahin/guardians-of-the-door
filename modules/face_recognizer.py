from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FaceMatchResult:
    name: str | None
    similarity: float
    is_registered: bool


class FaceRecognizer:
    """Registered-face matcher scaffold.

    The project can later connect this class to DeepFace, face_recognition,
    or another embedding model. For now it defines the input/output shape used
    by the approach-person module.
    """

    def __init__(self, registered_faces_dir: str | Path = "data/registered_faces") -> None:
        self.registered_faces_dir = Path(registered_faces_dir)

    def recognize(self, face_image) -> FaceMatchResult:
        # TODO: Add face embedding extraction and similarity matching.
        return FaceMatchResult(name=None, similarity=0.0, is_registered=False)

    def list_registered_images(self) -> list[Path]:
        if not self.registered_faces_dir.exists():
            return []
        allowed = {".jpg", ".jpeg", ".png"}
        return sorted(
            path
            for path in self.registered_faces_dir.iterdir()
            if path.suffix.lower() in allowed
        )
