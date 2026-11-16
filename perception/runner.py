"""Standalone camera perception runner orchestration."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol

from perception.adapter import ImageObservationAdapter, frame_metadata_from_frame
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

    @classmethod
    def create(
        cls,
        *,
        node_id: str,
        camera_id: str,
        camera_session_id: str | None = None,
        timestamp_provenance: str = "system_utc_capture_time",
        max_frames: int | None = None,
    ) -> "RunnerConfig":
        return cls(
            node_id=node_id,
            camera_id=camera_id,
            camera_session_id=camera_session_id or f"{camera_id}-{uuid.uuid4().hex[:12]}",
            timestamp_provenance=timestamp_provenance,
            max_frames=max_frames,
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
    ) -> None:
        self.config = config
        self.capture = capture
        self.detector = detector
        self.writer = writer
        self.adapter = adapter or ImageObservationAdapter()
        self.display = display

    def run(self) -> RunnerResult:
        frames_processed = 0
        observations_written = 0
        stopped_reason = "completed"

        try:
            self.capture.open()

            while self.config.max_frames is None or frames_processed < self.config.max_frames:
                ok, frame = self.capture.read()

                if not ok or frame is None:
                    stopped_reason = "capture_failure"
                    break

                frame_sequence = frames_processed + 1
                detections = self.detector.track(frame)
                metadata = frame_metadata_from_frame(
                    frame=frame,
                    node_id=self.config.node_id,
                    camera_id=self.config.camera_id,
                    camera_session_id=self.config.camera_session_id,
                    frame_sequence=frame_sequence,
                    timestamp_provenance=self.config.timestamp_provenance,
                )
                observations = self.adapter.observations_for_frame(detections, metadata)
                self.writer.write_observations(observations)
                observations_written += len(observations)
                frames_processed += 1

                if self.display is not None:
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
