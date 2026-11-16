"""Detection and tracking interfaces for standalone real-camera perception."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from perception.exceptions import MissingDependencyError, ModelLoadError


@dataclass(frozen=True)
class Detection:
    """One image-space detector result."""

    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    class_id: int | None = None
    local_track_id: int | None = None


class Detector(Protocol):
    """Portable detector interface used by the runner and fake tests."""

    def track(self, frame: object) -> list[Detection]:
        """Return detections with nullable persistent local track IDs."""


class UltralyticsYoloTracker:
    """Ultralytics YOLO detector with persistent local tracking enabled."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.25, device: str | None = None) -> None:
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as error:
            raise MissingDependencyError(
                "Ultralytics is not installed. Install optional perception dependencies first, "
                "for example with: python -m pip install -r requirements-perception.txt"
            ) from error

        self.confidence_threshold = confidence_threshold
        self.device = device

        try:
            self.model = YOLO(model_path)
        except Exception as error:  # noqa: BLE001 - preserve actionable context from model loader.
            raise ModelLoadError(
                f"Unable to load YOLO model {model_path!r}. Check that the file exists, "
                "that the model format is supported on this machine, and that weights are not "
                "TensorRT engines generated on a different host."
            ) from error

    def track(self, frame: object) -> list[Detection]:
        try:
            results = self.model.track(
                frame,
                persist=True,
                conf=self.confidence_threshold,
                verbose=False,
                device=self.device,
            )
        except Exception as error:  # noqa: BLE001
            raise ModelLoadError(f"YOLO inference/tracking failed: {error}") from error

        detections: list[Detection] = []

        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            names = getattr(result, "names", None) or getattr(self.model, "names", {}) or {}
            xyxy_values = _to_python_list(getattr(boxes, "xyxy", []))
            confidence_values = _to_python_list(getattr(boxes, "conf", []))
            class_values = _to_python_list(getattr(boxes, "cls", []))
            track_values = _to_python_list(getattr(boxes, "id", []))

            for index, xyxy in enumerate(xyxy_values):
                class_id = _optional_int(_value_at(class_values, index))
                track_id = _optional_int(_value_at(track_values, index))
                class_name = str(names.get(class_id, class_id if class_id is not None else "unknown"))
                confidence = float(_value_at(confidence_values, index, 0.0))

                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=confidence,
                        bbox_xyxy=tuple(float(value) for value in xyxy[:4]),  # type: ignore[arg-type]
                        class_id=class_id,
                        local_track_id=track_id,
                    ),
                )

        return detections


def _to_python_list(value: Any) -> list[Any]:
    if value is None:
        return []

    if hasattr(value, "detach"):
        value = value.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    if hasattr(value, "numpy"):
        value = value.numpy()

    if hasattr(value, "tolist"):
        result = value.tolist()
        return result if isinstance(result, list) else [result]

    return list(value) if isinstance(value, tuple | list) else [value]


def _value_at(values: list[Any], index: int, default: Any = None) -> Any:
    return values[index] if index < len(values) else default


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None
