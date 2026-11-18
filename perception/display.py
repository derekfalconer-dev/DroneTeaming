"""Live OpenCV display for camera perception."""

from __future__ import annotations

import math
import textwrap
from collections.abc import Mapping, Sequence
from typing import Any

from perception.detector import Detection
from perception.exceptions import MissingDependencyError
from perception.observations import ImageObservation


class OpenCVDisplay:
    def __init__(self, window_name: str = "Camera perception") -> None:
        try:
            import cv2
        except ModuleNotFoundError as error:
            raise MissingDependencyError(
                "OpenCV display support is unavailable because opencv-python is not installed."
            ) from error

        try:
            import numpy
        except ModuleNotFoundError as error:
            raise MissingDependencyError(
                "OpenCV display support is unavailable because numpy is not installed."
            ) from error

        self.cv2: Any = cv2
        self.numpy: Any = numpy
        self.window_name = window_name
        self.panel_width_px = 420
        self.max_window_width_px = 1280
        self.max_window_height_px = 800

    def show(self, frame: object, detections: list[Detection]) -> None:
        self.cv2.imshow(self.window_name, self._annotated_frame(frame, detections))

    def show_observation_panel(
        self,
        frame: object,
        detections: list[Detection],
        observations: list[ImageObservation],
        context: Mapping[str, object],
    ) -> None:
        display_frame = frame.copy() if hasattr(frame, "copy") else frame
        display_frame = self._as_bgr_frame(display_frame)
        video_frame, coordinate_scale = self._resize_to_fit(
            display_frame,
            max_width_px=self.max_window_width_px - self.panel_width_px,
            max_height_px=self.max_window_height_px,
        )
        annotated_video_frame = self._annotated_frame(
            video_frame,
            detections,
            coordinate_scale=coordinate_scale,
        )
        if bool(context.get("appearance_debug", False)):
            self._annotate_appearance_sampling_regions(
                annotated_video_frame,
                observations,
                coordinate_scale=coordinate_scale,
            )
        panel = self._panel_frame(annotated_video_frame.shape[0], observations, context)
        combined_frame = self.numpy.hstack((annotated_video_frame, panel))
        self.cv2.imshow(self.window_name, combined_frame)

    def _annotated_frame(
        self,
        frame: object,
        detections: list[Detection],
        *,
        coordinate_scale: float = 1.0,
    ) -> object:
        display_frame = frame.copy() if hasattr(frame, "copy") else frame
        display_frame = self._as_bgr_frame(display_frame)

        for detection in detections:
            x_min, y_min, x_max, y_max = [
                int(round(value * coordinate_scale)) for value in detection.bbox_xyxy
            ]
            label = f"{detection.class_name} {detection.confidence:.2f}"
            if detection.local_track_id is not None:
                label = f"{label} id={detection.local_track_id}"

            self.cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            self.cv2.putText(
                display_frame,
                label,
                (x_min, max(15, y_min - 8)),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                self.cv2.LINE_AA,
            )

        return display_frame

    def _as_bgr_frame(self, frame: object) -> object:
        shape = getattr(frame, "shape", ())
        if len(shape) == 2:
            return self.cv2.cvtColor(frame, self.cv2.COLOR_GRAY2BGR)
        if len(shape) == 3 and shape[2] == 4:
            return self.cv2.cvtColor(frame, self.cv2.COLOR_BGRA2BGR)
        return frame

    def _resize_to_fit(self, frame: object, *, max_width_px: int, max_height_px: int) -> tuple[object, float]:
        height_px, width_px = frame.shape[:2]
        scale = min(max_width_px / width_px, max_height_px / height_px, 1.0)
        if scale >= 1.0:
            return frame, 1.0

        resized_size = (max(1, int(width_px * scale)), max(1, int(height_px * scale)))
        return self.cv2.resize(frame, resized_size, interpolation=self.cv2.INTER_AREA), scale

    def _panel_frame(
        self,
        height_px: int,
        observations: list[ImageObservation],
        context: Mapping[str, object],
    ) -> object:
        panel = self.numpy.full((height_px, self.panel_width_px, 3), 24, dtype=self.numpy.uint8)
        lines = [
            "Camera perception",
            f"Source: {context.get('source_label', 'unknown')}",
            f"Node: {context.get('node_id', 'unknown')}",
            f"Camera: {context.get('camera_id', 'unknown')}",
            f"Class filter: {context.get('class_filter_label', 'all model classes')}",
            f"Model: {context.get('model_identifier') or 'unknown'}",
            f"Inference request: {self._format_inference_request(context.get('requested_inference_size_px'))}",
            f"Original frame: {self._format_frame_dimensions(context)}",
            f"Session: {context.get('camera_session_id', 'unknown')}",
            f"FPS: {float(context.get('processing_fps', 0.0)):.1f}",
            f"Frame: {context.get('frame_sequence', 'unknown')}",
            f"Received UTC: {context.get('frame_timestamp_utc', 'unknown')}",
            f"Observations total: {context.get('observations_written', 0)}",
            "Geographic position: unknown",
            "",
            "Current observations:",
        ]

        lines.extend(
            self._observation_lines(
                observations,
                appearance_debug=bool(context.get("appearance_debug", False)),
            )
            if observations
            else ["  none"]
        )

        y_px = 26
        for index, line in enumerate(lines):
            color = (255, 255, 255) if index == 0 else (210, 235, 210)
            scale = 0.62 if index == 0 else 0.46
            thickness = 2 if index == 0 else 1
            self.cv2.putText(
                panel,
                line,
                (14, y_px),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness,
                self.cv2.LINE_AA,
            )
            y_px += 24 if index == 0 else 20
            if y_px > height_px - 12:
                break

        return panel

    def _format_inference_request(self, value: object) -> str:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return "unknown; tensor shape may differ"

        requested_size = float(value)
        if not math.isfinite(requested_size) or requested_size <= 0:
            return "unknown; tensor shape may differ"

        return f"imgsz={int(round(requested_size))}px requested; tensor shape may differ"

    def _format_frame_dimensions(self, context: Mapping[str, object]) -> str:
        width_px = context.get("frame_width_px")
        height_px = context.get("frame_height_px")
        if (
            isinstance(width_px, bool)
            or isinstance(height_px, bool)
            or not isinstance(width_px, int | float)
            or not isinstance(height_px, int | float)
            or not math.isfinite(float(width_px))
            or not math.isfinite(float(height_px))
            or width_px <= 0
            or height_px <= 0
        ):
            return "unknown"

        return f"{int(round(float(width_px)))}x{int(round(float(height_px)))} px"

    def _observation_lines(
        self,
        observations: list[ImageObservation],
        *,
        appearance_debug: bool = False,
    ) -> list[str]:
        lines: list[str] = []
        maximum_observations = 5
        maximum_panel_lines = 22

        for observation in observations[:maximum_observations]:
            if len(lines) >= maximum_panel_lines:
                break

            payload = observation.to_dict()
            lines.append(self._format_observation_line(payload))
            for detail_line in self._format_appearance_lines(
                payload,
                appearance_debug=appearance_debug,
            ):
                if len(lines) >= maximum_panel_lines:
                    lines.append("... additional details hidden")
                    return lines
                lines.append(detail_line)

        if len(observations) > maximum_observations and len(lines) < maximum_panel_lines:
            lines.append(f"... {len(observations) - maximum_observations} more")

        return lines

    def _format_observation_line(self, payload: Mapping[str, object]) -> str:
        class_name = self._format_class_name(self._find_value(payload, ("class_name", "object_class", "label")))
        track_id = self._format_track_id(self._find_value(payload, ("local_track_id", "track_id", "source_track_id")))
        detection_score = self._format_detection_score(
            self._find_value(payload, ("detection_confidence", "confidence", "score"))
        )
        return f"{class_name} | Track {track_id} | Detection score {detection_score}"

    def _format_appearance_lines(
        self,
        payload: Mapping[str, object],
        *,
        appearance_debug: bool = False,
    ) -> list[str]:
        appearance_attributes = payload.get("appearance_attributes")
        if not isinstance(appearance_attributes, Mapping):
            return []

        attribute = appearance_attributes.get("upper_clothing_color")
        if not isinstance(attribute, Mapping):
            return []

        lines: list[str] = []
        color_value = str(attribute.get("value") or "unknown")
        quality_label = str(attribute.get("quality_label") or "unavailable")
        lines.extend(
            self._wrap_panel_text(
                f"  Upper clothing: {color_value} | Color quality: {quality_label}"
            )
        )

        reason = attribute.get("reason")
        if color_value == "unknown" and reason:
            lines.extend(self._wrap_panel_text(f"  Reason: {reason}"))

        if appearance_debug:
            lines.extend(self._format_appearance_debug_lines(attribute))

        return lines

    def _format_appearance_debug_lines(self, attribute: Mapping[str, object]) -> list[str]:
        diagnostics = attribute.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            return self._format_legacy_appearance_debug_lines(attribute)

        lines: list[str] = []
        evidence = diagnostics.get("color_evidence")
        decision = diagnostics.get("decision")

        if isinstance(evidence, Mapping) and evidence.get("sample_analyzed") is True:
            lines.extend(self._format_sampling_region_lines(attribute))
            candidates = evidence.get("top_candidates")
            if isinstance(candidates, Sequence) and not isinstance(candidates, str | bytes):
                candidate_parts: list[str] = []
                for candidate in candidates[:3]:
                    if not isinstance(candidate, Mapping):
                        continue

                    color = candidate.get("color")
                    fraction = candidate.get("fraction_of_sample")
                    if (
                        color is not None
                        and not isinstance(fraction, bool)
                        and isinstance(fraction, int | float)
                        and math.isfinite(float(fraction))
                    ):
                        candidate_parts.append(f"{color} {float(fraction) * 100:.1f}%")

                if candidate_parts:
                    lines.extend(
                        self._wrap_panel_text(
                            f"  Top colors: {', '.join(candidate_parts)} of sample"
                        )
                    )
        elif isinstance(evidence, Mapping):
            reason = evidence.get("unavailable_reason")
            if reason:
                lines.extend(self._wrap_panel_text(f"  Evidence unavailable: {reason}"))

        if isinstance(decision, Mapping):
            lines.extend(self._format_decision_debug_lines(decision, attribute))
        else:
            lines.extend(self._format_legacy_quality_lines(attribute))

        return lines

    def _format_sampling_region_lines(self, attribute: Mapping[str, object]) -> list[str]:
        region = attribute.get("sampling_region_px")
        if not isinstance(region, Mapping):
            return []

        values = [
            region.get("x_min"),
            region.get("y_min"),
            region.get("x_max"),
            region.get("y_max"),
        ]
        if not all(isinstance(value, int | float) for value in values):
            return []

        x_min, y_min, x_max, y_max = [int(round(float(value))) for value in values]
        width_px = max(0, x_max - x_min)
        height_px = max(0, y_max - y_min)
        return self._wrap_panel_text(
            f"  Sample: x={x_min} y={y_min} w={width_px} h={height_px} px original"
        )

    def _format_decision_debug_lines(
        self,
        decision: Mapping[str, object],
        attribute: Mapping[str, object],
    ) -> list[str]:
        lines: list[str] = []
        quality_score = decision.get("quality_score", attribute.get("quality_score"))
        thresholds = decision.get("thresholds")

        if isinstance(quality_score, bool):
            return lines

        if isinstance(quality_score, int | float) and math.isfinite(float(quality_score)):
            quality_text = f"{float(quality_score):.2f}"
        elif quality_score is None:
            quality_text = "unavailable"
        else:
            quality_text = None

        if quality_text is not None:
            threshold_text = self._format_decision_threshold_text(thresholds)
            if threshold_text:
                lines.extend(
                    self._wrap_panel_text(
                        f"  Quality score: {quality_text}; thresholds: {threshold_text}"
                    )
                )
            else:
                lines.extend(self._wrap_panel_text(f"  Quality score: {quality_text}"))

        rejection_reason = decision.get("rejection_reason", attribute.get("reason"))
        lines.extend(self._wrap_panel_text(f"  Rejection: {rejection_reason or 'none'}"))
        return lines

    def _format_decision_threshold_text(self, thresholds: object) -> str:
        if not isinstance(thresholds, Mapping):
            return ""

        threshold_parts: list[str] = []
        for key, label in (
            ("minimum_valid_pixel_fraction", "valid"),
            ("minimum_dominant_share", "top"),
            ("minimum_winner_gap", "gap"),
        ):
            value = thresholds.get(key)
            if (
                not isinstance(value, bool)
                and isinstance(value, int | float)
                and math.isfinite(float(value))
            ):
                threshold_parts.append(f"{label}>={float(value) * 100:.0f}%")

        return ", ".join(threshold_parts)

    def _format_legacy_appearance_debug_lines(self, attribute: Mapping[str, object]) -> list[str]:
        return [
            *self._format_sampling_region_lines(attribute),
            *self._format_legacy_quality_lines(attribute),
        ]

    def _format_legacy_quality_lines(self, attribute: Mapping[str, object]) -> list[str]:
        quality_score = attribute.get("quality_score")
        if (
            not isinstance(quality_score, bool)
            and isinstance(quality_score, int | float)
            and math.isfinite(float(quality_score))
        ):
            return self._wrap_panel_text(f"  Quality score: {float(quality_score):.2f}")

        return []

    def _wrap_panel_text(self, text: str, *, maximum_chars: int = 54) -> list[str]:
        leading_space_count = len(text) - len(text.lstrip(" "))
        prefix = text[:leading_space_count]
        body = text.strip()
        wrapped = textwrap.wrap(
            body,
            width=max(8, maximum_chars - leading_space_count),
            break_long_words=True,
            break_on_hyphens=False,
        )
        if not wrapped:
            return [prefix.rstrip()]

        return [f"{prefix}{line}" for line in wrapped]

    def _annotate_appearance_sampling_regions(
        self,
        frame: object,
        observations: list[ImageObservation],
        *,
        coordinate_scale: float,
    ) -> None:
        for observation in observations:
            payload = observation.to_dict()
            appearance_attributes = payload.get("appearance_attributes")
            if not isinstance(appearance_attributes, Mapping):
                continue

            attribute = appearance_attributes.get("upper_clothing_color")
            if not isinstance(attribute, Mapping):
                continue

            region = attribute.get("sampling_region_px")
            if not isinstance(region, Mapping):
                continue

            values = [
                region.get("x_min"),
                region.get("y_min"),
                region.get("x_max"),
                region.get("y_max"),
            ]
            if not all(isinstance(value, int | float) for value in values):
                continue

            x_min, y_min, x_max, y_max = [
                int(round(float(value) * coordinate_scale)) for value in values
            ]
            self.cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 255), 1)

    def _format_class_name(self, value: object | None) -> str:
        if value is None:
            return "Unknown"

        text = " ".join(str(value).replace("_", " ").split())
        if not text:
            return "Unknown"

        return f"{text[:1].upper()}{text[1:]}"

    def _format_track_id(self, value: object | None) -> str:
        if value is None or value == "":
            return "unavailable"

        return str(value)

    def _format_detection_score(self, value: object | None) -> str:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return "unavailable"

        score = float(value)
        if not math.isfinite(score):
            return "unavailable"

        return f"{int(round(score * 100))}%"

    def _find_value(self, value: object, keys: Sequence[str]) -> object | None:
        if isinstance(value, Mapping):
            for key in keys:
                if key in value:
                    return value[key]
            for nested_value in value.values():
                found = self._find_value(nested_value, keys)
                if found is not None:
                    return found
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            for nested_value in value:
                found = self._find_value(nested_value, keys)
                if found is not None:
                    return found

        return None

    def _format_bbox(self, payload: Mapping[str, object]) -> str:
        bbox = self._find_bbox(payload)
        if bbox is None:
            return "unknown"

        if isinstance(bbox, Mapping):
            values = [
                bbox.get("x_min_px", bbox.get("x_min", bbox.get("left"))),
                bbox.get("y_min_px", bbox.get("y_min", bbox.get("top"))),
                bbox.get("x_max_px", bbox.get("x_max", bbox.get("right"))),
                bbox.get("y_max_px", bbox.get("y_max", bbox.get("bottom"))),
            ]
            if all(isinstance(value, int | float) for value in values):
                return ",".join(str(int(round(float(value)))) for value in values)

        if isinstance(bbox, Sequence) and not isinstance(bbox, str | bytes) and len(bbox) >= 4:
            values = bbox[:4]
            if all(isinstance(value, int | float) for value in values):
                return ",".join(str(int(round(float(value)))) for value in values)

        return "unknown"

    def _find_bbox(self, value: object) -> object | None:
        if isinstance(value, Mapping):
            for key in ("pixel_bbox", "bbox_px", "bbox_xyxy", "bbox"):
                if key in value:
                    return value[key]
            for nested_value in value.values():
                found = self._find_bbox(nested_value)
                if found is not None:
                    return found
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            for nested_value in value:
                found = self._find_bbox(nested_value)
                if found is not None:
                    return found

        return None

    def poll_key(self) -> int:
        return int(self.cv2.waitKey(1) & 0xFF)

    def close(self) -> None:
        self.cv2.destroyAllWindows()
