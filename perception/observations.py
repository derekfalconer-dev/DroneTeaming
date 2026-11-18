"""Structured image-observation serialization for real-camera perception."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


IMAGE_OBSERVATION_SCHEMA_VERSION = "image-observation-v1"
UNKNOWN_GEOGRAPHIC_POSITION: dict[str, float | str | None] = {
    "status": "unknown",
    "latitude": None,
    "longitude": None,
    "altitude_m": None,
}


def utc_now_iso() -> str:
    """Return a compact UTC ISO-8601 timestamp with explicit Zulu suffix."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class PixelBBox:
    """Axis-aligned bounding box in image pixel coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if any(not isinstance(value, int | float) for value in values):
            raise TypeError("Pixel bounding-box coordinates must be numeric.")

        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("Pixel bounding-box maximums must be greater than or equal to minimums.")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ImageObservation:
    """Image-space camera detection observation.

    Track IDs are scoped only to ``camera_session_id``. They are never
    shared-world identities, and these observations intentionally do not satisfy
    simulator interfaces that require geographic position.

    ``detection_confidence`` is the detector-reported score for the image
    detection. It is useful for ranking and display, but detector scores are not
    calibrated probabilities that the detection is correct.

    ``appearance_attributes`` is an additive schema extension point. Current
    upper-clothing color attributes may include a ``diagnostics`` object with
    raw color evidence, explicit denominators, thresholds, rejection conditions,
    and the heuristic quality calculation. No-sample appearance quality is
    serialized as ``null``; analyzed samples with a calculated zero quality
    retain numeric ``0.0``. Readers should ignore unknown nested fields to
    preserve forward compatibility.
    """

    schema_version: str
    node_id: str
    camera_id: str
    camera_session_id: str
    frame_sequence: int
    timestamp_utc: str
    timestamp_provenance: str
    image_width_px: int
    image_height_px: int
    bbox_px: PixelBBox
    class_name: str
    detection_confidence: float
    local_track_id: int | None
    track_id_scope: str
    scoped_local_track_id: str | None
    geographic_position: dict[str, float | str | None]
    appearance_attributes: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox_px"] = self.bbox_px.to_dict()
        payload["geographic_position"] = dict(self.geographic_position)
        if self.appearance_attributes is None:
            payload.pop("appearance_attributes", None)
        else:
            payload["appearance_attributes"] = dict(self.appearance_attributes)
        return payload

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def serialize_observation_jsonl(observation: ImageObservation) -> str:
    return f"{observation.to_json_line()}\n"
