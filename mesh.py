"""
mesh.py
-------
Thin, robust wrapper around the modern MediaPipe Tasks `FaceLandmarker`
API (the legacy `mp.solutions.face_mesh` API is deprecated/removed in
recent mediapipe releases — that's what caused the original
`AttributeError: module 'mediapipe' has no attribute 'solutions'`).

Returns plain Python data (list of faces, each a list of (x, y) pixel
tuples) so the rest of the app never has to touch mediapipe types.
"""

import os
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config
from logger import get_logger

log = get_logger(__name__)


class ModelNotFoundError(FileNotFoundError):
    """Raised when the .task model file is missing, with download help."""


@dataclass
class FaceResult:
    faces: list[list[tuple[int, int]]] = field(default_factory=list)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    def is_empty(self) -> bool:
        return len(self.faces) == 0


class Mesh:
    """Detects facial landmarks in video frames using MediaPipe Tasks."""

    def __init__(
        self,
        model_path=config.MODEL_PATH,
        max_num_faces: int = config.MAX_NUM_FACES,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_presence_confidence: float = config.MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
    ):
        if not os.path.exists(model_path):
            raise ModelNotFoundError(
                f"Face landmarker model not found at: {model_path}\n"
                f"Download it from:\n  {config.MODEL_DOWNLOAD_URL}\n"
                f"and place it next to your project as 'face_landmarker.task'."
            )

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._detector = mp_vision.FaceLandmarker.create_from_options(options)
        self._frame_index = 0
        log.info("Mesh detector initialized (max_num_faces=%d)", max_num_faces)

    def process(self, frame_bgr: np.ndarray) -> FaceResult:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = self._frame_index * 33  # assume ~30fps pacing
        self._frame_index += 1

        result = self._detector.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return FaceResult(faces=[])

        faces = [
            [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]
            for face_landmarks in result.face_landmarks
        ]
        return FaceResult(faces=faces)

    def close(self):
        self._detector.close()
        log.info("Mesh detector closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()