"""
capture.py
----------
Thin wrapper around cv2.VideoCapture: opens the camera, applies the
requested resolution, and exposes a simple read() that always returns
a correctly-sized frame (or None at end of stream / on failure).
"""

import cv2

from logger import get_logger

log = get_logger(__name__)


class Capture:
    def __init__(self, camera_index: int, width: int, height: int):
        self.width = width
        self.height = height
        self._cap = cv2.VideoCapture(camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}.")
        log.info("Camera %d opened at %dx%d", camera_index, width, height)

    def read(self):
        ret, frame = self._cap.read()
        if not ret:
            return None
        return cv2.resize(frame, (self.width, self.height))

    def release(self):
        self._cap.release()
        log.info("Camera released")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()