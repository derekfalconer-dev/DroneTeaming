"""Detection and tracking interfaces for standalone real-camera perception."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from perception.exceptions import MissingDependencyError, ModelLoadError


DEFAULT_SAR_CLASS_NAMES = (
    "person",
    "bicycle",
    "car",
    "van",
    "motorcycle",
    "bus",
    "truck",
    "boat",
    "airplane",
)

LabelProfileName = Literal["coco", "visdrone"]
LABEL_PROFILE_NAMES: tuple[LabelProfileName, ...] = ("coco", "visdrone")


class DetectionClassFilterError(ValueError):
    """Raised when a requested detection class filter cannot be resolved."""


@dataclass(frozen=True)
class LabelProfile:
    name: LabelProfileName
    aliases: Mapping[str, str]

    def normalize_source_name(self, source_name: str) -> str:
        normalized_source_name = _normalize_class_name(source_name)
        return self.aliases.get(normalized_source_name, normalized_source_name)


LABEL_PROFILES: dict[str, LabelProfile] = {
    "coco": LabelProfile(name="coco", aliases={}),
    "visdrone": LabelProfile(
        name="visdrone",
        aliases={
            "pedestrian": "person",
            "people": "person",
            "motor": "motorcycle",
            "awning-tricycle": "awning-tricycle",
            "awning tricycle": "awning-tricycle",
        },
    ),
}


@dataclass(frozen=True)
class Detection:
    """One image-space detector result."""

    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    class_id: int | None = None
    model_class_name: str | None = None
    local_track_id: int | None = None


class Detector(Protocol):
    """Portable detector interface used by the runner and fake tests."""

    def track(self, frame: object) -> list[Detection]:
        """Return detections with nullable persistent local track IDs."""


@dataclass(frozen=True)
class DetectionClassFilter:
    requested_names: tuple[str, ...] | None
    matched_names: tuple[str, ...]
    class_ids: tuple[int, ...]
    available_names: tuple[str, ...]
    unsupported_names: tuple[str, ...]
    matched_source_names: tuple[str, ...] = ()

    @property
    def enabled(self) -> bool:
        return self.requested_names is not None

    @property
    def inference_class_ids(self) -> list[int] | None:
        return list(self.class_ids) if self.enabled else None

    @property
    def panel_label(self) -> str:
        if not self.enabled:
            return "all model classes"

        if not self.matched_names:
            return "none"

        joined_names = ", ".join(self.matched_names)
        return joined_names if len(joined_names) <= 72 else f"{len(self.matched_names)} classes"

    @property
    def startup_label(self) -> str:
        if not self.enabled:
            return "all model classes"

        label = ", ".join(self.matched_names) if self.matched_names else "none"
        if self.matched_source_names:
            source_label = ", ".join(self.matched_source_names)
            label = f"{label} [model classes: {source_label}]"
        if self.unsupported_names:
            label = f"{label} (unsupported by model: {', '.join(self.unsupported_names)})"

        return label

    def allows(self, class_id: int | None, class_name: str) -> bool:
        if not self.enabled:
            return True

        if class_id in self.class_ids:
            return True

        normalized_names = {_normalize_class_name(name) for name in self.matched_names}
        return _normalize_class_name(class_name) in normalized_names


class UltralyticsYoloTracker:
    """Ultralytics YOLO detector with persistent local tracking enabled."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        device: str | None = None,
        class_names: Sequence[str] | None = DEFAULT_SAR_CLASS_NAMES,
        strict_class_names: bool = False,
        inference_image_size: int = 640,
        label_profile: LabelProfileName = "coco",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as error:
            raise MissingDependencyError(
                "Ultralytics is not installed. Install optional perception dependencies first, "
                "for example with: python -m pip install -r requirements-perception.txt"
            ) from error

        if isinstance(inference_image_size, bool) or inference_image_size <= 0:
            raise ValueError("inference_image_size must be a positive integer.")

        self.confidence_threshold = confidence_threshold
        self.device = device
        self.inference_image_size = inference_image_size
        self.label_profile = resolve_label_profile(label_profile)

        try:
            self.model = YOLO(model_path)
        except Exception as error:  # noqa: BLE001 - preserve actionable context from model loader.
            raise ModelLoadError(
                f"Unable to load YOLO model {model_path!r}. Check that the file exists, "
                "that the model format is supported on this machine, and that weights are not "
                "TensorRT engines generated on a different host."
            ) from error

        self.names_by_id = _class_names_by_id(getattr(self.model, "names", {}))
        self.normalized_names_by_id = normalize_class_names_by_profile(
            self.names_by_id,
            self.label_profile,
        )
        self.class_filter = resolve_detection_class_filter(
            self.normalized_names_by_id,
            class_names,
            strict=strict_class_names,
            source_names_by_id=self.names_by_id,
        )

    def track(self, frame: object) -> list[Detection]:
        track_kwargs = {
            "persist": True,
            "conf": self.confidence_threshold,
            "verbose": False,
            "device": self.device,
            "classes": self.class_filter.inference_class_ids,
            "imgsz": self.inference_image_size,
        }

        try:
            results = self.model.track(frame, **track_kwargs)
        except TypeError:
            track_kwargs.pop("classes", None)
            try:
                results = self.model.track(frame, **track_kwargs)
            except Exception as error:  # noqa: BLE001
                raise ModelLoadError(f"YOLO inference/tracking failed: {error}") from error
        except Exception as error:  # noqa: BLE001
            raise ModelLoadError(f"YOLO inference/tracking failed: {error}") from error

        detections: list[Detection] = []

        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            xyxy_values = _to_python_list(getattr(boxes, "xyxy", []))
            confidence_values = _to_python_list(getattr(boxes, "conf", []))
            class_values = _to_python_list(getattr(boxes, "cls", []))
            track_values = _to_python_list(getattr(boxes, "id", []))

            for index, xyxy in enumerate(xyxy_values):
                class_id = _optional_int(_value_at(class_values, index))
                track_id = _optional_int(_value_at(track_values, index))
                model_class_name = str(
                    self.names_by_id.get(class_id, class_id if class_id is not None else "unknown")
                )
                class_name = str(self.normalized_names_by_id.get(class_id, model_class_name))
                confidence = float(_value_at(confidence_values, index, 0.0))

                if not self.class_filter.allows(class_id, class_name):
                    continue

                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=confidence,
                        bbox_xyxy=tuple(float(value) for value in xyxy[:4]),  # type: ignore[arg-type]
                        class_id=class_id,
                        model_class_name=model_class_name,
                        local_track_id=track_id,
                    ),
                )

        return detections

    def filter_detections(self, detections: list[Detection]) -> list[Detection]:
        return [
            detection
            for detection in detections
            if self.class_filter.allows(detection.class_id, detection.class_name)
        ]


def resolve_detection_class_filter(
    available_names_by_id: Mapping[int, str],
    requested_names: Sequence[str] | None,
    *,
    strict: bool = False,
    source_names_by_id: Mapping[int, str] | None = None,
) -> DetectionClassFilter:
    available_names = _unique_in_order(
        available_names_by_id[class_id] for class_id in sorted(available_names_by_id)
    )
    if requested_names is None:
        class_ids = tuple(sorted(available_names_by_id))
        return DetectionClassFilter(
            requested_names=None,
            matched_names=available_names,
            class_ids=class_ids,
            available_names=available_names,
            unsupported_names=(),
            matched_source_names=_source_names_for_ids(class_ids, source_names_by_id),
        )

    name_to_ids: dict[str, list[int]] = {}
    for class_id, name in available_names_by_id.items():
        name_to_ids.setdefault(_normalize_class_name(name), []).append(class_id)

    matched_names: list[str] = []
    class_ids: list[int] = []
    unsupported_names: list[str] = []
    seen_requested_names: set[str] = set()
    seen_class_ids: set[int] = set()

    for requested_name in requested_names:
        normalized_name = _normalize_class_name(requested_name)
        if not normalized_name or normalized_name in seen_requested_names:
            continue

        seen_requested_names.add(normalized_name)
        matched_class_ids = name_to_ids.get(normalized_name, [])
        if not matched_class_ids:
            unsupported_names.append(requested_name)
            continue

        matched_names.append(available_names_by_id[matched_class_ids[0]])
        for class_id in sorted(matched_class_ids):
            if class_id not in seen_class_ids:
                class_ids.append(class_id)
                seen_class_ids.add(class_id)

    if strict and unsupported_names:
        raise DetectionClassFilterError(
            "Unsupported class name(s): "
            f"{', '.join(unsupported_names)}. Available model classes (normalized): {', '.join(available_names)}"
        )

    if strict and not class_ids:
        raise DetectionClassFilterError(
            "Requested class filter did not match any classes in the loaded model. "
            f"Available model classes (normalized): {', '.join(available_names)}"
        )

    return DetectionClassFilter(
        requested_names=tuple(requested_names),
        matched_names=tuple(matched_names),
        class_ids=tuple(class_ids),
        available_names=available_names,
        unsupported_names=tuple(unsupported_names),
        matched_source_names=_source_names_for_ids(class_ids, source_names_by_id),
    )


def resolve_label_profile(label_profile: str) -> LabelProfile:
    try:
        return LABEL_PROFILES[label_profile]
    except KeyError as error:
        raise DetectionClassFilterError(
            f"Unsupported label profile {label_profile!r}. Supported profiles: {', '.join(LABEL_PROFILE_NAMES)}"
        ) from error


def normalize_class_names_by_profile(
    names_by_id: Mapping[int, str],
    label_profile: LabelProfile,
) -> dict[int, str]:
    seen_source_names: dict[str, int] = {}
    normalized_names_by_id: dict[int, str] = {}

    for class_id, source_name in names_by_id.items():
        normalized_source_name = _normalize_class_name(source_name)
        if normalized_source_name in seen_source_names:
            first_class_id = seen_source_names[normalized_source_name]
            raise DetectionClassFilterError(
                "Ambiguous model class metadata: source label "
                f"{source_name!r} appears for class IDs {first_class_id} and {class_id}."
            )

        seen_source_names[normalized_source_name] = class_id
        normalized_names_by_id[class_id] = label_profile.normalize_source_name(source_name)

    return normalized_names_by_id


def _class_names_by_id(value: object) -> dict[int, str]:
    if isinstance(value, Mapping):
        names_by_id: dict[int, str] = {}
        for raw_class_id, raw_name in value.items():
            class_id = _optional_int(raw_class_id)
            if class_id is not None:
                names_by_id[class_id] = str(raw_name)

        return names_by_id

    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return {index: str(name) for index, name in enumerate(value)}

    return {}


def _normalize_class_name(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def _unique_in_order(values: Sequence[str] | object) -> tuple[str, ...]:
    unique_values: list[str] = []
    seen: set[str] = set()
    for value in values:  # type: ignore[union-attr]
        if value not in seen:
            unique_values.append(value)
            seen.add(value)

    return tuple(unique_values)


def _source_names_for_ids(
    class_ids: Sequence[int],
    source_names_by_id: Mapping[int, str] | None,
) -> tuple[str, ...]:
    if source_names_by_id is None:
        return ()

    return tuple(source_names_by_id[class_id] for class_id in class_ids if class_id in source_names_by_id)


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
