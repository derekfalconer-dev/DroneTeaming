"""Standalone camera perception runner orchestration."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, replace
from typing import Protocol

from perception.adapter import ImageObservationAdapter, frame_metadata_from_frame
from perception.appearance import AppearanceAnalyzer, create_default_appearance_analyzer
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

    def run(self) -> RunnerResult:
        frames_processed = 0
        observations_written = 0
        stopped_reason = "completed"

        try:
            self.capture.open()

            run_started_at = time.monotonic()

            while self.config.max_frames is None or frames_processed < self.config.max_frames:
                ok, frame = self.capture.read()

                if not ok or frame is None:
                    stopped_reason = "capture_failure"
                    break

                frame_sequence = frames_processed + 1
                metadata = frame_metadata_from_frame(
                    frame=frame,
                    node_id=self.config.node_id,
                    camera_id=self.config.camera_id,
                    camera_session_id=self.config.camera_session_id,
                    frame_sequence=frame_sequence,
                    timestamp_provenance=self.config.timestamp_provenance,
                )
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
                            },
                        )
                    else:
                        self.display.show(frame, detections)

                    if self.display.poll_key() in QUIT_KEYS:
                        stopped_reason = "user_requested"
                        break
        except KeyboardInterrupt:
            stopped_reason = "interrupted"
        finally:
            self.capture.release()
            self.writer.close()

            if self.display is not None:
                self.display.close()

        return RunnerResult(
            frames_processed=frames_processed,
            observations_written=observations_written,
            stopped_reason=stopped_reason,
        )

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
