"""Standalone real-camera perception support.

This package is intentionally separate from the simulator's geographic
observation pipeline. Camera detections emitted here are image observations with
unknown geographic position unless a future localization component explicitly
adds a georeference.
"""

from perception.adapter import FrameMetadata, ImageObservationAdapter, require_geographic_position
from perception.detector import Detection
from perception.observations import ImageObservation, PixelBBox

__all__ = [
    "Detection",
    "FrameMetadata",
    "ImageObservation",
    "ImageObservationAdapter",
    "PixelBBox",
    "require_geographic_position",
]
