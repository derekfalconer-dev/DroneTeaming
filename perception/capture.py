"""OpenCV camera capture boundary for standalone perception."""

from __future__ import annotations

import platform
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from perception.exceptions import CameraUnavailableError, MissingDependencyError


@dataclass(frozen=True)
class CaptureFrame:
    ok: bool
    frame: object | None
    capture_frame_sequence: int | None
    received_monotonic_s: float | None
    source_time_s: float | None
    source_time_basis: str
    overwritten_frame_count: int = 0
    eof: bool = False
    error: str | None = None


@dataclass(frozen=True)
class CameraConfig:
    source: int | str = 0
    width_px: int | None = None
    height_px: int | None = None
    backend: int | None = None
    source_mode: str = "auto"


LIVE_SOURCE_MODES = {"live", "camera", "stream"}
FILE_SOURCE_MODES = {"file", "video"}


def resolve_source_mode(source: int | str, requested_mode: str = "auto") -> str:
    if requested_mode != "auto":
        if requested_mode not in {"live", "file"}:
            raise ValueError("--source-mode must be one of: auto, live, file.")
        return requested_mode

    if isinstance(source, int):
        return "live"

    parsed = urlparse(source)
    if parsed.scheme.lower() in {"rtsp", "rtmp", "http", "https"}:
        return "live"

    if Path(source).expanduser().is_file():
        return "file"

    return "live"


class OpenCVCameraCapture:
    """OpenCV VideoCapture wrapper with lazy optional dependency import."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._cv2: Any = None
        self._capture: Any = None
        self._frame_sequence = 0
        self._opened_monotonic_s: float | None = None
        self._source_mode = resolve_source_mode(config.source, config.source_mode)

    def open(self) -> None:
        try:
            import cv2
        except ModuleNotFoundError as error:
            raise MissingDependencyError(
                "OpenCV is not installed. Install optional perception dependencies first, "
                "for example with: python -m pip install -r requirements-perception.txt"
            ) from error

        self._cv2 = cv2
        self._frame_sequence = 0
        self._opened_monotonic_s = time.monotonic()
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

    def read(self) -> CaptureFrame:
        if self._capture is None:
            raise CameraUnavailableError("Camera has not been opened.")

        ok, frame = self._capture.read()
        received_monotonic_s = time.monotonic()
        if not ok or frame is None:
            return CaptureFrame(
                ok=False,
                frame=None,
                capture_frame_sequence=None,
                received_monotonic_s=received_monotonic_s,
                source_time_s=None,
                source_time_basis=self._source_time_basis(),
                eof=self._is_eof(),
            )

        self._frame_sequence += 1
        return CaptureFrame(
            ok=True,
            frame=frame,
            capture_frame_sequence=self._frame_sequence,
            received_monotonic_s=received_monotonic_s,
            source_time_s=self._source_time_s(received_monotonic_s),
            source_time_basis=self._source_time_basis(),
        )

    def _source_time_basis(self) -> str:
        if self._source_mode == "file":
            return "video_presentation_time_or_deterministic_frame_fallback"
        return "monotonic_frame_receipt_time"

    def _source_time_s(self, received_monotonic_s: float) -> float:
        if self._source_mode != "file":
            return received_monotonic_s

        assert self._capture is not None
        position_msec = float(self._capture.get(self._cv2.CAP_PROP_POS_MSEC))
        if position_msec > 0:
            return position_msec / 1000.0

        fps = float(self._capture.get(self._cv2.CAP_PROP_FPS) or 0.0)
        if fps > 0:
            return max(0.0, (self._frame_sequence - 1) / fps)

        return max(0.0, (self._frame_sequence - 1) / 30.0)

    def _is_eof(self) -> bool:
        if self._source_mode != "file" or self._capture is None:
            return False

        frame_count = float(self._capture.get(self._cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        position = float(self._capture.get(self._cv2.CAP_PROP_POS_FRAMES) or 0.0)
        return frame_count > 0 and position >= frame_count

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class LatestFrameCapture:
    """Bounded latest-frame capture for live inputs.

    A single owner thread continuously reads frames, retains only the newest
    pending frame, counts overwritten pending frames, and never runs inference.
    Consumers block without busy polling and consume each selected frame once.
    """

    def __init__(self, capture: OpenCVCameraCapture, *, shutdown_timeout_s: float = 2.0) -> None:
        self.capture = capture
        self.shutdown_timeout_s = shutdown_timeout_s
        self._condition = threading.Condition()
        self._pending_frame: CaptureFrame | None = None
        self._closed = False
        self._worker_error: BaseException | None = None
        self._worker: threading.Thread | None = None
        self._overwritten_frame_count = 0
        self._read_count = 0
        self._started_monotonic_s: float | None = None

    def open(self) -> None:
        self._closed = False
        self._pending_frame = None
        self._worker_error = None
        self._overwritten_frame_count = 0
        self._read_count = 0
        self._started_monotonic_s = time.monotonic()
        self.capture.open()
        self._worker = threading.Thread(target=self._read_loop, name="latest-frame-capture", daemon=True)
        self._worker.start()

    def read(self) -> CaptureFrame:
        with self._condition:
            while self._pending_frame is None and not self._closed and self._worker_error is None:
                self._condition.wait()

            if self._pending_frame is not None:
                frame = self._pending_frame
                self._pending_frame = None
                return frame

            if self._worker_error is not None:
                return CaptureFrame(
                    ok=False,
                    frame=None,
                    capture_frame_sequence=None,
                    received_monotonic_s=time.monotonic(),
                    source_time_s=None,
                    source_time_basis="monotonic_frame_receipt_time",
                    overwritten_frame_count=self._overwritten_frame_count,
                    error=str(self._worker_error),
                )

            return CaptureFrame(
                ok=False,
                frame=None,
                capture_frame_sequence=None,
                received_monotonic_s=time.monotonic(),
                source_time_s=None,
                source_time_basis="monotonic_frame_receipt_time",
                overwritten_frame_count=self._overwritten_frame_count,
                eof=True,
            )

    def release(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=self.shutdown_timeout_s)

        self._worker = None
        if not (worker is not None and worker.is_alive()):
            self.capture.release()

    def capture_rate_hz(self) -> float:
        if self._started_monotonic_s is None:
            return 0.0

        elapsed_s = max(time.monotonic() - self._started_monotonic_s, 0.000001)
        return self._read_count / elapsed_s

    def _read_loop(self) -> None:
        try:
            while True:
                with self._condition:
                    if self._closed:
                        return

                capture_frame = self.capture.read()
                self._read_count += 1

                with self._condition:
                    if capture_frame.ok and self._pending_frame is not None:
                        self._overwritten_frame_count += 1

                    capture_frame = CaptureFrame(
                        ok=capture_frame.ok,
                        frame=capture_frame.frame,
                        capture_frame_sequence=capture_frame.capture_frame_sequence,
                        received_monotonic_s=capture_frame.received_monotonic_s,
                        source_time_s=capture_frame.source_time_s,
                        source_time_basis=capture_frame.source_time_basis,
                        overwritten_frame_count=self._overwritten_frame_count,
                        eof=capture_frame.eof,
                        error=capture_frame.error,
                    )

                    if capture_frame.ok:
                        self._pending_frame = capture_frame
                    else:
                        self._closed = True
                        if self._pending_frame is None:
                            self._pending_frame = capture_frame

                    self._condition.notify_all()

                    if not capture_frame.ok:
                        return
        except BaseException as error:
            with self._condition:
                self._worker_error = error
                self._closed = True
                self._condition.notify_all()
