"""Live OpenCV display for camera perception."""

from __future__ import annotations

from typing import Any

from perception.detector import Detection
from perception.exceptions import MissingDependencyError


class OpenCVDisplay:
    def __init__(self, window_name: str = "Camera perception") -> None:
        try:
            import cv2
        except ModuleNotFoundError as error:
            raise MissingDependencyError(
                "OpenCV display support is unavailable because opencv-python is not installed."
            ) from error

        self.cv2: Any = cv2
        self.window_name = window_name

    def show(self, frame: object, detections: list[Detection]) -> None:
        display_frame = frame.copy() if hasattr(frame, "copy") else frame

        for detection in detections:
            x_min, y_min, x_max, y_max = [int(round(value)) for value in detection.bbox_xyxy]
            label = f"{detection.class_name} {detection.confidence:.2f}"
            if detection.local_track_id is not None:
                label = f"{label} id={detection.local_track_id}"

            self.cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            self.cv2.putText(
                display_frame,
                label,
                (x_min, max(15, y_min - 8)),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                self.cv2.LINE_AA,
            )

        self.cv2.imshow(self.window_name, display_frame)

    def poll_key(self) -> int:
        return int(self.cv2.waitKey(1) & 0xFF)

    def close(self) -> None:
        self.cv2.destroyAllWindows()
