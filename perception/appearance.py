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
UPPER_CLOTHING_COLOR_METHOD_VERSION = "upper-clothing-color-heuristic-v2"
UPPER_CLOTHING_COLOR_QUALITY_DEFINITION = (
    "Uncalibrated heuristic score derived from dominant HSV palette share and "
    "separation from the next palette color in a fixed upper-body crop; it is "
    "not a probability and must not be treated as identity evidence."
)
UPPER_CLOTHING_COLOR_QUALITY_FORMULA = (
    "round(clamp((top_share_of_valid_color_pixels * 0.7) + "
    "(winner_gap_of_valid_color_pixels * 0.3), 0.0, 1.0), 2)"
)
PIXEL_COLOR_ANALYZER_VERSION = "hsv-pixel-color-rules-v2"
PIXEL_COLOR_RULE_RATIONALE = (
    "Dark pixels with only weak saturation are classified as neutral before hue "
    "rules are considered, because hue is unstable in low-light nearly neutral "
    "pixels. Genuinely chromatic dark colors remain available when saturation is "
    "strong enough to support the hue assignment."
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


class PixelColorAnalyzer:
    """Classify sampled BGR pixels into a small HSV color palette.

    Object-specific code decides which pixels to sample. This analyzer only
    handles pixel color classification, evidence aggregation, and diagnostics.
    """

    minimum_valid_pixel_fraction = 0.35
    minimum_dominant_share = 0.55
    minimum_winner_gap = 0.20
    dark_value_threshold = 90
    black_value_threshold = 45
    dark_neutral_saturation_threshold = 70
    chromatic_saturation_threshold = 55
    dark_chromatic_saturation_threshold = 80

    def __init__(self, *, cv2: object, numpy: object) -> None:
        self.cv2: Any = cv2
        self.numpy: Any = numpy

    def analyze(
        self,
        bgr_region: object,
        *,
        validity_mask: object | None = None,
    ) -> Mapping[str, object]:
        bgr_region = self._as_bgr_crop(bgr_region)
        shape = getattr(bgr_region, "shape", ())
        if len(shape) < 2:
            return self._unavailable_result("unsupported_sample_type")

        total_pixels = int(shape[0]) * int(shape[1])
        if total_pixels == 0:
            return self._unavailable_result("empty_sample")

        hsv_region = self.cv2.cvtColor(bgr_region, self.cv2.COLOR_BGR2HSV)
        hue = hsv_region[:, :, 0].astype(self.numpy.int16)
        saturation = hsv_region[:, :, 1].astype(self.numpy.int16)
        value = hsv_region[:, :, 2].astype(self.numpy.int16)

        analyzed_mask = self.numpy.ones(hue.shape, dtype=bool)
        if validity_mask is not None:
            analyzed_mask = self.numpy.asarray(validity_mask, dtype=bool)
            if analyzed_mask.shape != hue.shape:
                raise ValueError("Pixel-color validity mask must match the sampled region shape.")

        labels = self.numpy.full(hue.shape, "excluded", dtype=object)
        self._assign(labels, analyzed_mask & (value < self.black_value_threshold), "black")
        self._assign(
            labels,
            analyzed_mask
            & (labels == "excluded")
            & (value < self.dark_value_threshold)
            & (saturation < self.dark_neutral_saturation_threshold),
            "black",
        )
        self._assign(
            labels,
            analyzed_mask & (labels == "excluded") & (saturation < 35) & (value > 205),
            "white",
        )
        self._assign(
            labels,
            analyzed_mask
            & (labels == "excluded")
            & (saturation < 35)
            & (value >= self.dark_value_threshold)
            & (value <= 205),
            "gray",
        )
        self._assign(
            labels,
            analyzed_mask
            & (labels == "excluded")
            & (hue >= 8)
            & (hue <= 24)
            & (saturation >= 45)
            & (value >= self.black_value_threshold)
            & (value < 160),
            "brown",
        )

        chromatic_support = (
            analyzed_mask
            & (labels == "excluded")
            & (value >= self.black_value_threshold)
            & (
                ((value >= self.dark_value_threshold) & (saturation >= self.chromatic_saturation_threshold))
                | ((value < self.dark_value_threshold) & (saturation >= self.dark_chromatic_saturation_threshold))
            )
        )
        self._assign(labels, chromatic_support & ((hue <= 9) | (hue >= 170)), "red")
        self._assign(labels, chromatic_support & (labels == "excluded") & (hue >= 10) & (hue <= 24) & (value >= 160), "orange")
        self._assign(labels, chromatic_support & (labels == "excluded") & (hue >= 25) & (hue <= 34) & (value >= 70), "yellow")
        self._assign(labels, chromatic_support & (labels == "excluded") & (hue >= 35) & (hue <= 84), "green")
        self._assign(labels, chromatic_support & (labels == "excluded") & (hue >= 85) & (hue <= 99), "cyan")
        self._assign(labels, chromatic_support & (labels == "excluded") & (hue >= 100) & (hue <= 129), "blue")
        self._assign(labels, chromatic_support & (labels == "excluded") & (hue >= 130) & (hue <= 154), "purple")
        self._assign(labels, chromatic_support & (labels == "excluded") & (hue >= 155) & (hue <= 169) & (value >= 70), "pink")

        palette = (
            "black",
            "white",
            "gray",
            "brown",
            "red",
            "orange",
            "yellow",
            "green",
            "cyan",
            "blue",
            "purple",
            "pink",
        )
        counts = {
            color_name: int(self.numpy.count_nonzero(labels == color_name))
            for color_name in palette
        }
        analyzed_pixels = int(self.numpy.count_nonzero(analyzed_mask))
        invalid_pixels = total_pixels - analyzed_pixels
        valid_pixels = sum(counts.values())
        excluded_pixels = max(0, analyzed_pixels - valid_pixels)
        valid_pixel_fraction = valid_pixels / total_pixels if total_pixels > 0 else 0.0
        ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        top_color, top_count = ranked[0] if ranked else ("unknown", 0)
        second_count = ranked[1][1] if len(ranked) > 1 else 0
        top_share = top_count / valid_pixels if valid_pixels > 0 else 0.0
        winner_gap = (top_count - second_count) / valid_pixels if valid_pixels > 0 else 0.0
        quality_score = (
            round(max(0.0, min(1.0, (top_share * 0.7) + (winner_gap * 0.3))), 2)
            if valid_pixels > 0
            else None
        )
        quality_label = self._quality_label(quality_score) if quality_score is not None else "unavailable"
        failed_valid_fraction = valid_pixel_fraction < self.minimum_valid_pixel_fraction
        failed_dominance = top_share < self.minimum_dominant_share
        failed_winner_gap = winner_gap < self.minimum_winner_gap
        rejection_reason = (
            "insufficient_evidence"
            if failed_valid_fraction or failed_dominance or failed_winner_gap
            else None
        )
        diagnostics = self._sample_diagnostics(
            counts=counts,
            ranked=ranked,
            total_pixels=total_pixels,
            analyzed_pixels=analyzed_pixels,
            invalid_pixels=invalid_pixels,
            valid_pixels=valid_pixels,
            excluded_pixels=excluded_pixels,
            valid_pixel_fraction=valid_pixel_fraction,
            top_color=top_color,
            top_count=top_count,
            second_count=second_count,
            top_share=top_share,
            winner_gap=winner_gap,
            quality_score=quality_score,
            quality_label=quality_label,
            rejection_reason=rejection_reason,
            rejection_conditions={
                "valid_fraction_below_minimum": failed_valid_fraction,
                "dominant_share_below_minimum": failed_dominance,
                "winner_gap_below_minimum": failed_winner_gap,
            },
        )

        if rejection_reason is not None:
            return {
                "value": "unknown",
                "reason": rejection_reason,
                "quality_score": quality_score,
                "pixel_color_analyzer_version": PIXEL_COLOR_ANALYZER_VERSION,
                "color_rule_rationale": PIXEL_COLOR_RULE_RATIONALE,
                "quality_label": quality_label,
                "diagnostics": diagnostics,
            }

        return {
            "value": top_color,
            "reason": None,
            "quality_score": quality_score,
            "quality_label": quality_label,
            "diagnostics": diagnostics,
        }

    def _assign(self, labels: object, mask: object, color_name: str) -> None:
        labels[mask] = color_name

    def _sample_diagnostics(
        self,
        *,
        counts: Mapping[str, int],
        ranked: list[tuple[str, int]],
        total_pixels: int,
        analyzed_pixels: int,
        invalid_pixels: int,
        valid_pixels: int,
        excluded_pixels: int,
        valid_pixel_fraction: float,
        top_color: str,
        top_count: int,
        second_count: int,
        top_share: float,
        winner_gap: float,
        quality_score: float | None,
        quality_label: str,
        rejection_reason: str | None,
        rejection_conditions: Mapping[str, bool],
    ) -> dict[str, object]:
        return {
            "method": UPPER_CLOTHING_COLOR_METHOD,
            "method_version": UPPER_CLOTHING_COLOR_METHOD_VERSION,
            "pixel_color_analyzer_version": PIXEL_COLOR_ANALYZER_VERSION,
            "color_rule_rationale": PIXEL_COLOR_RULE_RATIONALE,
            "color_evidence": {
                "sample_analyzed": True,
                "candidate_fraction_denominator": "total_sample_pixels",
                "valid_fraction_denominator": "total_sample_pixels",
                "total_sample_pixels": total_pixels,
                "analyzed_sample_pixel_count": analyzed_pixels,
                "invalid_sample_pixel_count": invalid_pixels,
                "valid_color_pixel_count": valid_pixels,
                "excluded_pixel_count": excluded_pixels,
                "valid_pixel_fraction": round(valid_pixel_fraction, 4),
                "palette_counts": dict(counts),
                "top_candidates": [
                    {
                        "color": color_name,
                        "pixel_count": count,
                        "fraction_of_sample": round(count / total_pixels, 4)
                        if total_pixels > 0
                        else None,
                        "fraction_of_valid_color_pixels": round(count / valid_pixels, 4)
                        if valid_pixels > 0
                        else None,
                    }
                    for color_name, count in ranked[:3]
                ],
            },
            "decision": {
                "method": UPPER_CLOTHING_COLOR_METHOD,
                "method_version": UPPER_CLOTHING_COLOR_METHOD_VERSION,
                "pixel_color_analyzer_version": PIXEL_COLOR_ANALYZER_VERSION,
                "accepted": rejection_reason is None,
                "accepted_color_label": top_color if rejection_reason is None else None,
                "rejection_reason": rejection_reason,
                "quality_score": quality_score,
                "quality_label": quality_label,
                "quality_formula": UPPER_CLOTHING_COLOR_QUALITY_FORMULA,
                "quality_inputs": {
                    "top_color": top_color,
                    "top_color_pixel_count": top_count,
                    "second_color_pixel_count": second_count,
                    "valid_color_pixel_count": valid_pixels,
                    "top_share_of_valid_color_pixels": round(top_share, 4),
                    "winner_gap_of_valid_color_pixels": round(winner_gap, 4),
                },
                "thresholds": self._decision_thresholds(),
                "rejection_conditions": dict(rejection_conditions),
            },
        }

    def _unavailable_result(self, reason: str) -> Mapping[str, object]:
        return {
            "value": "unknown",
            "reason": reason,
            "quality_score": None,
            "quality_label": "unavailable",
            "diagnostics": {
                "method": UPPER_CLOTHING_COLOR_METHOD,
                "method_version": UPPER_CLOTHING_COLOR_METHOD_VERSION,
                "pixel_color_analyzer_version": PIXEL_COLOR_ANALYZER_VERSION,
                "color_rule_rationale": PIXEL_COLOR_RULE_RATIONALE,
                "color_evidence": {
                    "sample_analyzed": False,
                    "unavailable_reason": reason,
                },
                "decision": {
                    "method": UPPER_CLOTHING_COLOR_METHOD,
                    "method_version": UPPER_CLOTHING_COLOR_METHOD_VERSION,
                    "pixel_color_analyzer_version": PIXEL_COLOR_ANALYZER_VERSION,
                    "accepted": False,
                    "accepted_color_label": None,
                    "rejection_reason": reason,
                    "quality_score": None,
                    "quality_label": "unavailable",
                    "quality_formula": UPPER_CLOTHING_COLOR_QUALITY_FORMULA,
                    "thresholds": self._decision_thresholds(),
                    "rejection_conditions": {
                        "no_sample_analyzed": True,
                    },
                },
            },
        }

    def _decision_thresholds(self) -> dict[str, float | int]:
        return {
            "minimum_valid_pixel_fraction": self.minimum_valid_pixel_fraction,
            "minimum_dominant_share": self.minimum_dominant_share,
            "minimum_winner_gap": self.minimum_winner_gap,
            "dark_value_threshold": self.dark_value_threshold,
            "black_value_threshold": self.black_value_threshold,
            "dark_neutral_saturation_threshold": self.dark_neutral_saturation_threshold,
            "chromatic_saturation_threshold": self.chromatic_saturation_threshold,
            "dark_chromatic_saturation_threshold": self.dark_chromatic_saturation_threshold,
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
        self.pixel_color_analyzer = PixelColorAnalyzer(cv2=cv2, numpy=numpy)

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
            return {
                UPPER_CLOTHING_COLOR_ATTRIBUTE: {
                    "attribute_name": UPPER_CLOTHING_COLOR_ATTRIBUTE,
                    "value": "unknown",
                    "method": UPPER_CLOTHING_COLOR_METHOD,
                    "method_version": UPPER_CLOTHING_COLOR_METHOD_VERSION,
                    "sampling_region_px": sampling_region.to_dict(),
                    "quality_score": color_result["quality_score"],
                    "quality_label": color_result["quality_label"],
                    "quality_definition": UPPER_CLOTHING_COLOR_QUALITY_DEFINITION,
                    "reason": color_result["reason"],
                    "diagnostics": color_result["diagnostics"],
                }
            }

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
                "diagnostics": color_result["diagnostics"],
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
        return self.pixel_color_analyzer.analyze(bgr_crop)

    def _unavailable_diagnostics(
        self,
        reason: str,
        *,
        quality_score: float | None = None,
        quality_label: str = "unavailable",
    ) -> dict[str, object]:
        return {
            "method": UPPER_CLOTHING_COLOR_METHOD,
            "method_version": UPPER_CLOTHING_COLOR_METHOD_VERSION,
            "color_evidence": {
                "sample_analyzed": False,
                "unavailable_reason": reason,
            },
            "decision": {
                "method": UPPER_CLOTHING_COLOR_METHOD,
                "method_version": UPPER_CLOTHING_COLOR_METHOD_VERSION,
                "accepted": False,
                "accepted_color_label": None,
                "rejection_reason": reason,
                "quality_score": quality_score,
                "quality_label": quality_label,
                "quality_formula": UPPER_CLOTHING_COLOR_QUALITY_FORMULA,
                "thresholds": self._decision_thresholds(),
                "rejection_conditions": {
                    "no_sample_analyzed": True,
                },
            },
        }

    def _decision_thresholds(self) -> dict[str, float | int]:
        return {
            "minimum_valid_pixel_fraction": self.minimum_valid_pixel_fraction,
            "minimum_dominant_share": self.minimum_dominant_share,
            "minimum_winner_gap": self.minimum_winner_gap,
            "minimum_person_box_width_px": self.minimum_person_box_width_px,
            "minimum_person_box_height_px": self.minimum_person_box_height_px,
            "minimum_sampling_width_px": self.minimum_sampling_width_px,
            "minimum_sampling_height_px": self.minimum_sampling_height_px,
            "minimum_person_aspect_ratio": self.minimum_person_aspect_ratio,
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
        quality_score: float | None = None,
        quality_label: str = "unavailable",
    ) -> dict[str, dict[str, object]]:
        quality_score = round(quality_score, 2) if quality_score is not None else None
        attribute: dict[str, object] = {
            "attribute_name": UPPER_CLOTHING_COLOR_ATTRIBUTE,
            "value": "unknown",
            "method": UPPER_CLOTHING_COLOR_METHOD,
            "method_version": UPPER_CLOTHING_COLOR_METHOD_VERSION,
            "sampling_region_px": sampling_region.to_dict() if sampling_region is not None else None,
            "quality_score": quality_score,
            "quality_label": quality_label,
            "quality_definition": UPPER_CLOTHING_COLOR_QUALITY_DEFINITION,
            "reason": reason,
            "diagnostics": self._unavailable_diagnostics(
                reason,
                quality_score=quality_score,
                quality_label=quality_label,
            ),
        }
        return {UPPER_CLOTHING_COLOR_ATTRIBUTE: attribute}


def create_default_appearance_analyzer() -> AppearanceAnalyzer:
    try:
        return OpenCVUpperClothingColorAnalyzer()
    except MissingDependencyError:
        return NoOpAppearanceAnalyzer()
