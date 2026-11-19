"""Standalone camera perception runner orchestration."""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Protocol

from perception.adapter import ImageObservationAdapter, frame_metadata_from_frame
from perception.appearance import AppearanceAnalyzer, create_default_appearance_analyzer
from perception.capture import CaptureFrame
from perception.contact_lifecycle import (
    ContactConfirmationConfig,
    ContactDetectionEvidence,
    ContactFrame,
    ContactLifecycleManager,
)
from perception.detector import Detection, Detector
from perception.observations import ImageObservation


QUIT_KEYS = {ord("q"), ord("Q"), 27}


class Capture(Protocol):
    def open(self) -> None:
        ...

    def read(self) -> tuple[bool, object | None]:
        ...

    def release(self) -> None:
        ...


class Display(Protocol):
    def show(self, frame: object, detections: list[Detection]) -> None:
        ...

    def poll_key(self) -> int:
        ...

    def close(self) -> None:
        ...


class ObservationWriter(Protocol):
    def write_observations(self, observations: list[ImageObservation]) -> None:
        ...

    def close(self) -> None:
        ...


class ContactEventWriter(Protocol):
    def write_events(self, events: list[Mapping[str, object]]) -> None:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class RunnerConfig:
    node_id: str
    camera_id: str
    camera_session_id: str
    timestamp_provenance: str = "system_utc_capture_time"
    max_frames: int | None = None
    source_label: str = "unknown"
    appearance_debug: bool = False
    class_filter_label: str = "all model classes"
    label_profile: str = "coco"
    active_class_label: str = "all model classes"
    requested_inference_size_px: int | None = None
    model_identifier: str | None = None
    source_mode: str = "auto"

    @classmethod
    def create(
        cls,
        *,
        node_id: str,
        camera_id: str,
        camera_session_id: str | None = None,
        timestamp_provenance: str = "system_utc_capture_time",
        max_frames: int | None = None,
        source_label: str = "unknown",
        appearance_debug: bool = False,
        class_filter_label: str = "all model classes",
        label_profile: str = "coco",
        active_class_label: str = "all model classes",
        requested_inference_size_px: int | None = None,
        model_identifier: str | None = None,
        source_mode: str = "auto",
    ) -> "RunnerConfig":
        return cls(
            node_id=node_id,
            camera_id=camera_id,
            camera_session_id=camera_session_id or f"{camera_id}-{uuid.uuid4().hex[:12]}",
            timestamp_provenance=timestamp_provenance,
            max_frames=max_frames,
            source_label=source_label,
            appearance_debug=appearance_debug,
            class_filter_label=class_filter_label,
            label_profile=label_profile,
            active_class_label=active_class_label,
            requested_inference_size_px=requested_inference_size_px,
            model_identifier=model_identifier,
            source_mode=source_mode,
        )


@dataclass(frozen=True)
class RunnerResult:
    frames_processed: int
    observations_written: int
    stopped_reason: str


class CameraPerceptionRunner:
    def __init__(
        self,
        *,
        config: RunnerConfig,
        capture: Capture,
        detector: Detector,
        writer: ObservationWriter,
        adapter: ImageObservationAdapter | None = None,
        display: Display | None = None,
        appearance_analyzer: AppearanceAnalyzer | None = None,
        contact_manager: ContactLifecycleManager | None = None,
        contact_writer: ContactEventWriter | None = None,
        contact_config: ContactConfirmationConfig | None = None,
    ) -> None:
        self.config = config
        self.capture = capture
        self.detector = detector
        self.writer = writer
        self.adapter = adapter or ImageObservationAdapter()
        self.display = display
        self.appearance_analyzer = (
            appearance_analyzer
            if appearance_analyzer is not None
            else create_default_appearance_analyzer()
        )
        self.contact_manager = contact_manager or ContactLifecycleManager(contact_config)
        self.contact_writer = contact_writer

    def run(self) -> RunnerResult:
        frames_processed = 0
        observations_written = 0
        stopped_reason = "completed"

        try:
            self.capture.open()

            run_started_at = time.monotonic()

            while self.config.max_frames is None or frames_processed < self.config.max_frames:
                capture_frame = self._normalize_capture_frame(self.capture.read())

                if not capture_frame.ok or capture_frame.frame is None:
                    stopped_reason = "completed" if capture_frame.eof else "capture_failure"
                    break

                frame = capture_frame.frame
                frame_sequence = frames_processed + 1
                metadata = frame_metadata_from_frame(
                    frame=frame,
                    node_id=self.config.node_id,
                    camera_id=self.config.camera_id,
                    camera_session_id=self.config.camera_session_id,
                    frame_sequence=frame_sequence,
                    timestamp_provenance=self.config.timestamp_provenance,
                )
                inference_started_monotonic_s = time.monotonic()
                detections = self._filter_detections(self.detector.track(frame))
                appearance_attributes = (
                    self.appearance_analyzer.analyze_frame(frame, detections)
                    if detections
                    else []
                )
                observations = self.adapter.observations_for_frame(detections, metadata)
                observations = self._attach_appearance_attributes(observations, appearance_attributes)
                self.writer.write_observations(observations)
                observations_written += len(observations)
                contact_events = self.contact_manager.process_frame(
                    ContactFrame(
                        node_id=self.config.node_id,
                        camera_id=self.config.camera_id,
                        camera_session_id=self.config.camera_session_id,
                        processed_frame_sequence=frame_sequence,
                        capture_frame_sequence=capture_frame.capture_frame_sequence,
                        timestamp_utc=metadata.timestamp_utc,
                        source_time_s=self._source_time_s(capture_frame),
                        source_time_basis=capture_frame.source_time_basis,
                        image_width_px=metadata.image_width_px,
                        image_height_px=metadata.image_height_px,
                    ),
                    self._contact_evidence(detections),
                )
                if self.contact_writer is not None and contact_events:
                    self.contact_writer.write_events(contact_events)
                inference_completed_monotonic_s = time.monotonic()
                frames_processed += 1

                if self.display is not None:
                    panel_show = getattr(self.display, "show_observation_panel", None)
                    if callable(panel_show):
                        elapsed_seconds = max(time.monotonic() - run_started_at, 0.000001)
                        panel_show(
                            frame,
                            detections,
                            observations,
                            {
                                "source_label": self.config.source_label,
                                "node_id": self.config.node_id,
                                "camera_id": self.config.camera_id,
                                "camera_session_id": self.config.camera_session_id,
                                "processing_fps": frames_processed / elapsed_seconds,
                                "frame_sequence": frame_sequence,
                                "frame_timestamp_utc": metadata.timestamp_utc,
                                "observations_written": observations_written,
                                "appearance_debug": self.config.appearance_debug,
                                "class_filter_label": self.config.class_filter_label,
                                "label_profile": self.config.label_profile,
                                "active_class_label": self.config.active_class_label,
                                "requested_inference_size_px": self.config.requested_inference_size_px,
                                "model_identifier": self.config.model_identifier,
                                "frame_width_px": metadata.image_width_px,
                                "frame_height_px": metadata.image_height_px,
                                "source_mode": self.config.source_mode,
                                "capture_frame_sequence": capture_frame.capture_frame_sequence,
                                "source_time_s": capture_frame.source_time_s,
                                "source_time_basis": capture_frame.source_time_basis,
                                "capture_rate_hz": self._capture_rate_hz(),
                                "overwritten_frame_count": capture_frame.overwritten_frame_count,
                                "image_age_at_inference_start_s": self._image_age_s(
                                    capture_frame,
                                    inference_started_monotonic_s,
                                ),
                                "image_age_at_result_completion_s": self._image_age_s(
                                    capture_frame,
                                    inference_completed_monotonic_s,
                                ),
                                "contact_counts": self.contact_manager.snapshot_counts(),
                                "contacts": self.contact_manager.snapshot_contacts(),
                                "contact_coasting_display_max_elapsed_s": self.contact_manager.config.coasting_max_elapsed_s,
                            },
                        )
                    else:
                        self.display.show(frame, detections)

                    if self.display.poll_key() in QUIT_KEYS:
                        stopped_reason = "user_requested"
                        break

            if (
                stopped_reason == "completed"
                and self.config.max_frames is not None
                and frames_processed >= self.config.max_frames
            ):
                stopped_reason = "max_frames_reached"
        except KeyboardInterrupt:
            stopped_reason = "interrupted"
        finally:
            self.capture.release()
            self.writer.close()

            if self.contact_writer is not None:
                self.contact_writer.close()

            if self.display is not None:
                self.display.close()

        return RunnerResult(
            frames_processed=frames_processed,
            observations_written=observations_written,
            stopped_reason=stopped_reason,
        )

    def _normalize_capture_frame(self, value: object) -> CaptureFrame:
        if isinstance(value, CaptureFrame):
            return value

        if isinstance(value, tuple) and len(value) >= 2:
            ok = bool(value[0])
            frame = value[1]
            now = time.monotonic()
            return CaptureFrame(
                ok=ok,
                frame=frame,
                capture_frame_sequence=None,
                received_monotonic_s=now,
                source_time_s=now,
                source_time_basis="monotonic_frame_receipt_time_fallback",
                eof=False,
            )

        raise TypeError("capture.read() must return CaptureFrame or (ok, frame).")

    def _source_time_s(self, capture_frame: CaptureFrame) -> float:
        if capture_frame.source_time_s is not None:
            return capture_frame.source_time_s
        if capture_frame.received_monotonic_s is not None:
            return capture_frame.received_monotonic_s
        return time.monotonic()

    def _image_age_s(self, capture_frame: CaptureFrame, now_monotonic_s: float) -> float | None:
        if capture_frame.received_monotonic_s is None:
            return None
        return max(0.0, now_monotonic_s - capture_frame.received_monotonic_s)

    def _capture_rate_hz(self) -> float:
        capture_rate = getattr(self.capture, "capture_rate_hz", None)
        if callable(capture_rate):
            return float(capture_rate())
        return 0.0

    def _contact_evidence(self, detections: list[Detection]) -> list[ContactDetectionEvidence]:
        return [
            ContactDetectionEvidence(
                local_track_id=detection.local_track_id,
                class_name=detection.class_name,
                confidence=detection.confidence,
                bbox_xyxy=detection.bbox_xyxy,
                model_class_id=detection.class_id,
                model_class_name=detection.model_class_name,
            )
            for detection in detections
        ]

    def _filter_detections(self, detections: list[Detection]) -> list[Detection]:
        filter_detections = getattr(self.detector, "filter_detections", None)
        if callable(filter_detections):
            return list(filter_detections(detections))

        return detections

    def _attach_appearance_attributes(
        self,
        observations: list[ImageObservation],
        appearance_attributes: list[dict[str, dict[str, object]] | None],
    ) -> list[ImageObservation]:
        if not appearance_attributes:
            return observations

        updated_observations: list[ImageObservation] = []
        for index, observation in enumerate(observations):
            attributes = appearance_attributes[index] if index < len(appearance_attributes) else None
            if attributes is not None and isinstance(observation, ImageObservation):
                updated_observations.append(replace(observation, appearance_attributes=attributes))
            else:
                updated_observations.append(observation)

        return updated_observations
