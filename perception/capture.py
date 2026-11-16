"""OpenCV camera capture boundary for standalone perception."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any

from perception.exceptions import CameraUnavailableError, MissingDependencyError


@dataclass(frozen=True)
class CameraConfig:
    source: int | str = 0
    width_px: int | None = None
    height_px: int | None = None
    backend: int | None = None


class OpenCVCameraCapture:
    """OpenCV VideoCapture wrapper with lazy optional dependency import."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._cv2: Any = None
        self._capture: Any = None

    def open(self) -> None:
        try:
            import cv2
        except ModuleNotFoundError as error:
            raise MissingDependencyError(
                "OpenCV is not installed. Install optional perception dependencies first, "
                "for example with: python -m pip install -r requirements-perception.txt"
            ) from error

        self._cv2 = cv2
        backend = self.config.backend

        if backend is None and platform.system() == "Darwin" and isinstance(self.config.source, int):
            backend = getattr(cv2, "CAP_AVFOUNDATION", None)

        self._capture = (
            cv2.VideoCapture(self.config.source, backend)
            if backend is not None
            else cv2.VideoCapture(self.config.source)
        )

        if self.config.width_px:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width_px)

        if self.config.height_px:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height_px)

        if not self._capture or not self._capture.isOpened():
            self.release()
            raise CameraUnavailableError(
                f"Unable to open camera source {self.config.source!r}. On macOS, check System Settings "
                "→ Privacy & Security → Camera, close other camera users, and verify USB/UVC device access."
            )

    def read(self) -> tuple[bool, object | None]:
        if self._capture is None:
            raise CameraUnavailableError("Camera has not been opened.")

        return self._capture.read()

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
