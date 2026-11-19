"""Observation-adapter boundary for image-only camera detections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from perception.detector import Detection
from perception.exceptions import UnlocatedObservationError
from perception.observations import (
    IMAGE_OBSERVATION_SCHEMA_VERSION,
    UNKNOWN_GEOGRAPHIC_POSITION,
    ImageObservation,
    PixelBBox,
    utc_now_iso,
)


@dataclass(frozen=True)
class FrameMetadata:
    node_id: str
    camera_id: str
    camera_session_id: str
    frame_sequence: int
    image_width_px: int
    image_height_px: int
    timestamp_utc: str
    timestamp_provenance: str = "system_utc_capture_time"


class ImageObservationAdapter:
    """Convert detector results into JSONL-ready image observations.

    This adapter deliberately does not create simulator ``Observation`` objects,
    because those require geographic position. Geographic fields are explicitly
    unknown until a future localization component adds valid coordinates.
    """

    def observations_for_frame(
        self,
        detections: list[Detection],
        metadata: FrameMetadata,
    ) -> list[ImageObservation]:
        return [self.observation_from_detection(detection, metadata) for detection in detections]

    def observation_from_detection(
        self,
        detection: Detection,
        metadata: FrameMetadata,
        appearance_attributes: dict[str, Any] | None = None,
    ) -> ImageObservation:
        bbox = PixelBBox(*detection.bbox_xyxy)
        scoped_track_id = (
            f"{metadata.camera_session_id}:{detection.local_track_id}"
            if detection.local_track_id is not None
            else None
        )

        return ImageObservation(
            schema_version=IMAGE_OBSERVATION_SCHEMA_VERSION,
            node_id=metadata.node_id,
            camera_id=metadata.camera_id,
            camera_session_id=metadata.camera_session_id,
            frame_sequence=metadata.frame_sequence,
            timestamp_utc=metadata.timestamp_utc,
            timestamp_provenance=metadata.timestamp_provenance,
            image_width_px=metadata.image_width_px,
            image_height_px=metadata.image_height_px,
            bbox_px=bbox,
            class_name=detection.class_name,
            detection_confidence=float(detection.confidence),
            local_track_id=detection.local_track_id,
            track_id_scope="camera_session",
            scoped_local_track_id=scoped_track_id,
            geographic_position=dict(UNKNOWN_GEOGRAPHIC_POSITION),
            model_class_id=detection.class_id,
            model_class_name=detection.model_class_name or detection.class_name,
            appearance_attributes=appearance_attributes,
        )


def frame_metadata_from_frame(
    *,
    frame: object,
    node_id: str,
    camera_id: str,
    camera_session_id: str,
    frame_sequence: int,
    timestamp_utc: str | None = None,
    timestamp_provenance: str = "system_utc_capture_time",
) -> FrameMetadata:
    shape = getattr(frame, "shape", None)

    if not shape or len(shape) < 2:
        raise ValueError("Captured frame must expose image shape as (height, width[, channels]).")

    image_height_px = int(shape[0])
    image_width_px = int(shape[1])

    return FrameMetadata(
        node_id=node_id,
        camera_id=camera_id,
        camera_session_id=camera_session_id,
        frame_sequence=frame_sequence,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        timestamp_utc=timestamp_utc or utc_now_iso(),
        timestamp_provenance=timestamp_provenance,
    )


def require_geographic_position(observation: ImageObservation) -> None:
    """Reject image observations at boundaries that require geolocation."""

    position = observation.geographic_position
    if (
        position.get("status") != "known"
        or position.get("latitude") is None
        or position.get("longitude") is None
    ):
        raise UnlocatedObservationError(
            "Image observation has unknown geographic position and must not be "
            "fed into simulator/geospatial observation interfaces."
        )
