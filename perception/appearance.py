"""Lightweight appearance-attribute extraction for camera detections.

This module keeps OpenCV/NumPy imports lazy so simulation-only code can import
perception packages without computer-vision dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from perception.detector import Detection
from perception.exceptions import MissingDependencyError


UPPER_CLOTHING_COLOR_ATTRIBUTE = "upper_clothing_color"
UPPER_CLOTHING_COLOR_METHOD = "opencv_hsv_upper_torso_heuristic"
UPPER_CLOTHING_COLOR_METHOD_VERSION = "upper-clothing-color-heuristic-v1"
UPPER_CLOTHING_COLOR_QUALITY_DEFINITION = (
    "Uncalibrated heuristic score derived from dominant HSV palette share and "
    "separation from the next palette color in a fixed upper-body crop; it is "
    "not a probability and must not be treated as identity evidence."
)


class AppearanceAnalyzer(Protocol):
    """Interface for image appearance models.

    Future implementations can replace the HSV heuristic with learned clothing
    attributes or embeddings without changing runner/display wiring.
    """

    def analyze_frame(
        self,
        frame: object,
        detections: list[Detection],
    ) -> list[dict[str, dict[str, object]] | None]:
        ...


@dataclass(frozen=True)
class SamplingRegion:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def to_dict(self) -> dict[str, int]:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }


class NoOpAppearanceAnalyzer:
    def analyze_frame(
        self,
        frame: object,
        detections: list[Detection],
    ) -> list[dict[str, dict[str, object]] | None]:
        return [None for _ in detections]


class OpenCVUpperClothingColorAnalyzer:
    """Estimate dominant upper-clothing color for sufficiently visible people.

    The sampling region is a conservative fixed fraction of the person bounding
    box. A person box does not prove torso visibility, so ambiguous crops remain
    unknown rather than forcing a garment-color label.
    """

    minimum_person_box_width_px = 24
    minimum_person_box_height_px = 64
    minimum_sampling_width_px = 16
    minimum_sampling_height_px = 18
    maximum_crop_dimension_px = 96
    minimum_person_aspect_ratio = 1.15
    minimum_valid_pixel_fraction = 0.35
    minimum_dominant_share = 0.55
    minimum_winner_gap = 0.20

    def __init__(self) -> None:
        try:
            import cv2
        except ModuleNotFoundError as error:
            raise MissingDependencyError(
                "Appearance color estimation requires opencv-python for HSV conversion."
            ) from error

        try:
            import numpy
        except ModuleNotFoundError as error:
            raise MissingDependencyError(
                "Appearance color estimation requires numpy for bounded crop analysis."
            ) from error

        self.cv2: Any = cv2
        self.numpy: Any = numpy

    def analyze_frame(
        self,
        frame: object,
        detections: list[Detection],
    ) -> list[dict[str, dict[str, object]] | None]:
        return [self._analyze_detection(frame, detection) for detection in detections]

    def _analyze_detection(
        self,
        frame: object,
        detection: Detection,
    ) -> dict[str, dict[str, object]] | None:
        if detection.class_name.strip().lower() != "person":
            return None

        if not isinstance(frame, self.numpy.ndarray):
            return self._unknown("unsupported_frame_type")

        shape = getattr(frame, "shape", ())
        if len(shape) < 2:
            return self._unknown("unsupported_frame_type")

        image_height_px = int(shape[0])
        image_width_px = int(shape[1])
        clipped_bbox = self._clip_bbox(detection.bbox_xyxy, image_width_px, image_height_px)
        if clipped_bbox is None:
            return self._unknown("crop_too_small")

        x_min, y_min, x_max, y_max = clipped_bbox
        box_width = x_max - x_min
        box_height = y_max - y_min
        if (
            box_width < self.minimum_person_box_width_px
            or box_height < self.minimum_person_box_height_px
        ):
            return self._unknown("crop_too_small")

        if box_height / max(float(box_width), 1.0) < self.minimum_person_aspect_ratio:
            return self._unknown("insufficient_torso_visibility")

        sampling_region = self._torso_sampling_region(x_min, y_min, x_max, y_max)
        if sampling_region is None:
            return self._unknown("crop_too_small")

        crop = frame[
            sampling_region.y_min : sampling_region.y_max,
            sampling_region.x_min : sampling_region.x_max,
        ]
        if crop.size == 0:
            return self._unknown("crop_too_small", sampling_region=sampling_region)

        resized_crop = self._resize_crop(crop)
        color_result = self._dominant_palette_color(resized_crop)
        if color_result["value"] == "unknown":
            return self._unknown(
                str(color_result["reason"]),
                sampling_region=sampling_region,
                quality_score=float(color_result["quality_score"]),
                quality_label=str(color_result["quality_label"]),
            )

        return {
            UPPER_CLOTHING_COLOR_ATTRIBUTE: {
                "attribute_name": UPPER_CLOTHING_COLOR_ATTRIBUTE,
                "value": color_result["value"],
                "method": UPPER_CLOTHING_COLOR_METHOD,
                "method_version": UPPER_CLOTHING_COLOR_METHOD_VERSION,
                "sampling_region_px": sampling_region.to_dict(),
                "quality_score": color_result["quality_score"],
                "quality_label": color_result["quality_label"],
                "quality_definition": UPPER_CLOTHING_COLOR_QUALITY_DEFINITION,
                "reason": None,
            }
        }

    def _clip_bbox(
        self,
        bbox_xyxy: tuple[float, float, float, float],
        image_width_px: int,
        image_height_px: int,
    ) -> tuple[int, int, int, int] | None:
        x_min = max(0, min(image_width_px, int(round(bbox_xyxy[0]))))
        y_min = max(0, min(image_height_px, int(round(bbox_xyxy[1]))))
        x_max = max(0, min(image_width_px, int(round(bbox_xyxy[2]))))
        y_max = max(0, min(image_height_px, int(round(bbox_xyxy[3]))))

        if x_max <= x_min or y_max <= y_min:
            return None

        return x_min, y_min, x_max, y_max

    def _torso_sampling_region(
        self,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
    ) -> SamplingRegion | None:
        box_width = x_max - x_min
        box_height = y_max - y_min

        region = SamplingRegion(
            x_min=x_min + int(round(box_width * 0.22)),
            y_min=y_min + int(round(box_height * 0.34)),
            x_max=x_min + int(round(box_width * 0.78)),
            y_max=y_min + int(round(box_height * 0.68)),
        )
        if (
            region.x_max - region.x_min < self.minimum_sampling_width_px
            or region.y_max - region.y_min < self.minimum_sampling_height_px
        ):
            return None

        return region

    def _resize_crop(self, crop: object) -> object:
        height_px, width_px = crop.shape[:2]
        largest_dimension = max(height_px, width_px)
        if largest_dimension <= self.maximum_crop_dimension_px:
            return crop

        scale = self.maximum_crop_dimension_px / largest_dimension
        resized_size = (max(1, int(width_px * scale)), max(1, int(height_px * scale)))
        return self.cv2.resize(crop, resized_size, interpolation=self.cv2.INTER_AREA)

    def _dominant_palette_color(self, bgr_crop: object) -> Mapping[str, object]:
        bgr_crop = self._as_bgr_crop(bgr_crop)
        hsv_crop = self.cv2.cvtColor(bgr_crop, self.cv2.COLOR_BGR2HSV)
        hue = hsv_crop[:, :, 0].astype(self.numpy.int16)
        saturation = hsv_crop[:, :, 1].astype(self.numpy.int16)
        value = hsv_crop[:, :, 2].astype(self.numpy.int16)
        total_pixels = int(hue.size)

        masks = {
            "black": value < 45,
            "white": (saturation < 35) & (value > 205),
            "gray": (saturation < 35) & (value >= 45) & (value <= 205),
            "brown": (hue >= 8) & (hue <= 24) & (saturation >= 45) & (value >= 45) & (value < 160),
            "red": ((hue <= 9) | (hue >= 170)) & (saturation >= 55) & (value >= 45),
            "orange": (hue >= 10) & (hue <= 24) & (saturation >= 55) & (value >= 160),
            "yellow": (hue >= 25) & (hue <= 34) & (saturation >= 55) & (value >= 70),
            "green": (hue >= 35) & (hue <= 84) & (saturation >= 55) & (value >= 45),
            "cyan": (hue >= 85) & (hue <= 99) & (saturation >= 55) & (value >= 45),
            "blue": (hue >= 100) & (hue <= 129) & (saturation >= 55) & (value >= 45),
            "purple": (hue >= 130) & (hue <= 154) & (saturation >= 55) & (value >= 45),
            "pink": (hue >= 155) & (hue <= 169) & (saturation >= 55) & (value >= 70),
        }
        counts = {
            color_name: int(self.numpy.count_nonzero(mask))
            for color_name, mask in masks.items()
        }
        valid_pixels = sum(counts.values())
        if total_pixels == 0 or valid_pixels / total_pixels < self.minimum_valid_pixel_fraction:
            return {
                "value": "unknown",
                "reason": "insufficient_evidence",
                "quality_score": 0.0,
                "quality_label": "unavailable",
            }

        ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        top_color, top_count = ranked[0]
        second_count = ranked[1][1] if len(ranked) > 1 else 0
        top_share = top_count / valid_pixels
        winner_gap = (top_count - second_count) / valid_pixels
        quality_score = round(max(0.0, min(1.0, (top_share * 0.7) + (winner_gap * 0.3))), 2)

        if top_share < self.minimum_dominant_share or winner_gap < self.minimum_winner_gap:
            return {
                "value": "unknown",
                "reason": "insufficient_evidence",
                "quality_score": quality_score,
                "quality_label": "low",
            }

        return {
            "value": top_color,
            "reason": None,
            "quality_score": quality_score,
            "quality_label": self._quality_label(quality_score),
        }

    def _as_bgr_crop(self, crop: object) -> object:
        shape = getattr(crop, "shape", ())
        if len(shape) == 2:
            return self.cv2.cvtColor(crop, self.cv2.COLOR_GRAY2BGR)
        if len(shape) == 3 and shape[2] == 4:
            return self.cv2.cvtColor(crop, self.cv2.COLOR_BGRA2BGR)
        return crop

    def _quality_label(self, quality_score: float) -> str:
        if quality_score >= 0.78:
            return "high"
        if quality_score >= 0.55:
            return "moderate"
        return "low"

    def _unknown(
        self,
        reason: str,
        *,
        sampling_region: SamplingRegion | None = None,
        quality_score: float = 0.0,
        quality_label: str = "unavailable",
    ) -> dict[str, dict[str, object]]:
        attribute: dict[str, object] = {
            "attribute_name": UPPER_CLOTHING_COLOR_ATTRIBUTE,
            "value": "unknown",
            "method": UPPER_CLOTHING_COLOR_METHOD,
            "method_version": UPPER_CLOTHING_COLOR_METHOD_VERSION,
            "sampling_region_px": sampling_region.to_dict() if sampling_region is not None else None,
            "quality_score": round(quality_score, 2),
            "quality_label": quality_label,
            "quality_definition": UPPER_CLOTHING_COLOR_QUALITY_DEFINITION,
            "reason": reason,
        }
        return {UPPER_CLOTHING_COLOR_ATTRIBUTE: attribute}


def create_default_appearance_analyzer() -> AppearanceAnalyzer:
    try:
        return OpenCVUpperClothingColorAnalyzer()
    except MissingDependencyError:
        return NoOpAppearanceAnalyzer()
