"""Temporal contact confirmation for standalone camera perception.

These settings are experimental. Defaults favor a short confirmation delay:
a contact confirms after 3 detection-hit frames within the last 5 processed
images, while elapsed-time limits prevent stale evidence from confirming later.

Persistent false positives can still become confirmed if the detector/tracker
keeps producing consistent evidence. Stationary status is intentionally not
used to reject people or vehicles.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any


CONTACT_STATE_TENTATIVE = "tentative"
CONTACT_STATE_CONFIRMED = "confirmed"
CONTACT_STATE_COASTING = "coasting"
CONTACT_STATE_LOST = "lost"
CONTACT_STATE_EXPIRED = "expired"


@dataclass(frozen=True)
class ContactConfirmationConfig:
    confirmation_hits: int = 3
    confirmation_window_size: int = 5
    confirmation_max_elapsed_s: float = 2.0
    coasting_max_elapsed_s: float = 1.0
    expiration_elapsed_s: float = 5.0
    class_evidence_window_size: int = 7
    class_change_hits: int = 3
    update_interval_s: float = 1.0

    def __post_init__(self) -> None:
        if self.confirmation_hits < 1:
            raise ValueError("confirmation_hits must be at least 1.")
        if self.confirmation_window_size < 1:
            raise ValueError("confirmation_window_size must be at least 1.")
        if self.confirmation_hits > self.confirmation_window_size:
            raise ValueError("confirmation_hits must be less than or equal to confirmation_window_size.")
        for name in (
            "confirmation_max_elapsed_s",
            "coasting_max_elapsed_s",
            "expiration_elapsed_s",
            "update_interval_s",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.coasting_max_elapsed_s > self.expiration_elapsed_s:
            raise ValueError("coasting_max_elapsed_s must be less than or equal to expiration_elapsed_s.")
        if self.class_evidence_window_size < 1:
            raise ValueError("class_evidence_window_size must be at least 1.")
        if self.class_change_hits < 1:
            raise ValueError("class_change_hits must be at least 1.")


@dataclass(frozen=True)
class ContactDetectionEvidence:
    local_track_id: int | str | None
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    model_class_id: int | None = None
    model_class_name: str | None = None


@dataclass(frozen=True)
class ContactFrame:
    node_id: str
    camera_id: str
    camera_session_id: str
    processed_frame_sequence: int
    capture_frame_sequence: int | None
    timestamp_utc: str
    source_time_s: float
    source_time_basis: str
    image_width_px: int
    image_height_px: int


@dataclass
class ContactTrack:
    node_id: str
    camera_id: str
    camera_session_id: str
    local_track_id: int | str
    generation: int
    first_processed_frame_sequence: int
    first_capture_frame_sequence: int | None
    first_source_time_s: float
    first_timestamp_utc: str
    last_observed_processed_frame_sequence: int
    last_observed_capture_frame_sequence: int | None
    last_observed_source_time_s: float
    last_observed_timestamp_utc: str
    current_detection_class: str
    stabilized_class: str
    last_bbox_xyxy: tuple[float, float, float, float]
    last_confidence: float
    source_time_basis: str
    state: str = CONTACT_STATE_TENTATIVE
    hit_frame_sequences: deque[int] = field(default_factory=deque)
    class_evidence: deque[str] = field(default_factory=deque)
    last_event_source_time_s: float | None = None

    @property
    def contact_ref(self) -> str:
        return f"{self.camera_session_id}:{self.local_track_id}:{self.generation}"

    @property
    def scoped_local_track_id(self) -> str:
        return f"{self.camera_session_id}:{self.local_track_id}"


class ContactLifecycleManager:
    """Track tentative, confirmed, coasting, lost, and expired contacts.

    Evidence association uses only local tracker IDs scoped by camera session.
    Detections without tracker IDs are intentionally left unassociated.
    """

    def __init__(self, config: ContactConfirmationConfig | None = None) -> None:
        self.config = config or ContactConfirmationConfig()
        self._contacts: dict[tuple[str, str, str, int | str], ContactTrack] = {}
        self._generations: dict[tuple[str, str, str, int | str], int] = {}
        self._last_source_time_s: float | None = None
        self._last_source_time_basis: str | None = None

    def process_frame(
        self,
        frame: ContactFrame,
        detections: list[ContactDetectionEvidence],
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        source_time_s = self._monotonic_source_time(frame.source_time_s)
        if self._has_time_discontinuity(source_time_s, frame.source_time_basis):
            events.extend(self._expire_all(frame, reason="source_time_discontinuity"))
            self._contacts.clear()

        self._last_source_time_s = source_time_s
        self._last_source_time_basis = frame.source_time_basis

        detections_by_track_id = self._deduplicate_detections(detections)
        active_keys = set(self._contacts.keys())
        frame_keys = {
            (frame.node_id, frame.camera_id, frame.camera_session_id, local_track_id)
            for local_track_id in detections_by_track_id
        }

        new_keys = frame_keys - active_keys
        for key in new_keys:
            detection = detections_by_track_id[key[3]]
            contact = self._new_contact(frame, detection, source_time_s)
            self._contacts[key] = contact
            events.append(
                self._event_payload(
                    contact,
                    frame,
                    event_type="created",
                    bbox_designation="measured",
                    detection=detection,
                )
            )
            if self._has_confirmation_evidence(contact, frame.processed_frame_sequence):
                previous_state = contact.state
                contact.state = CONTACT_STATE_CONFIRMED
                events.append(
                    self._event_payload(
                        contact,
                        frame,
                        event_type="state_transition",
                        previous_state=previous_state,
                        bbox_designation="measured",
                        detection=detection,
                    )
                )
                contact.last_event_source_time_s = source_time_s

        for key in list(self._contacts):
            if key in new_keys:
                continue
            contact = self._contacts[key]
            detection = detections_by_track_id.get(contact.local_track_id)
            if detection is not None:
                events.extend(self._record_detection(contact, frame, detection, source_time_s))
            else:
                transition_event = self._record_miss(contact, frame, source_time_s)
                if transition_event is not None:
                    events.append(transition_event)

            if contact.state == CONTACT_STATE_EXPIRED:
                del self._contacts[key]

        return events

    def snapshot_counts(self) -> dict[str, int]:
        counts = {CONTACT_STATE_TENTATIVE: 0, CONTACT_STATE_CONFIRMED: 0, CONTACT_STATE_COASTING: 0}
        for contact in self._contacts.values():
            if contact.state in counts:
                counts[contact.state] += 1
        return counts

    def snapshot_contacts(self) -> list[dict[str, Any]]:
        return [self._contact_summary(contact) for contact in self._contacts.values()]

    def _deduplicate_detections(
        self,
        detections: list[ContactDetectionEvidence],
    ) -> dict[int | str, ContactDetectionEvidence]:
        detections_by_track_id: dict[int | str, ContactDetectionEvidence] = {}
        for detection in detections:
            if detection.local_track_id is None:
                continue

            existing = detections_by_track_id.get(detection.local_track_id)
            if existing is None or detection.confidence > existing.confidence:
                detections_by_track_id[detection.local_track_id] = detection

        return detections_by_track_id

    def _new_contact(
        self,
        frame: ContactFrame,
        detection: ContactDetectionEvidence,
        source_time_s: float,
    ) -> ContactTrack:
        assert detection.local_track_id is not None
        key = (frame.node_id, frame.camera_id, frame.camera_session_id, detection.local_track_id)
        generation = self._generations.get(key, 0) + 1
        self._generations[key] = generation
        contact = ContactTrack(
            node_id=frame.node_id,
            camera_id=frame.camera_id,
            camera_session_id=frame.camera_session_id,
            local_track_id=detection.local_track_id,
            generation=generation,
            first_processed_frame_sequence=frame.processed_frame_sequence,
            first_capture_frame_sequence=frame.capture_frame_sequence,
            first_source_time_s=source_time_s,
            first_timestamp_utc=frame.timestamp_utc,
            last_observed_processed_frame_sequence=frame.processed_frame_sequence,
            last_observed_capture_frame_sequence=frame.capture_frame_sequence,
            last_observed_source_time_s=source_time_s,
            last_observed_timestamp_utc=frame.timestamp_utc,
            current_detection_class=detection.class_name,
            stabilized_class=detection.class_name,
            last_bbox_xyxy=detection.bbox_xyxy,
            last_confidence=detection.confidence,
            source_time_basis=frame.source_time_basis,
        )
        contact.hit_frame_sequences.append(frame.processed_frame_sequence)
        contact.class_evidence.append(detection.class_name)
        contact.last_event_source_time_s = source_time_s
        return contact

    def _record_detection(
        self,
        contact: ContactTrack,
        frame: ContactFrame,
        detection: ContactDetectionEvidence,
        source_time_s: float,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        previous_state = contact.state
        previous_stabilized_class = contact.stabilized_class
        previous_observed_processed_frame_sequence = contact.last_observed_processed_frame_sequence

        contact.current_detection_class = detection.class_name
        contact.last_bbox_xyxy = detection.bbox_xyxy
        contact.last_confidence = detection.confidence
        contact.last_observed_processed_frame_sequence = frame.processed_frame_sequence
        contact.last_observed_capture_frame_sequence = frame.capture_frame_sequence
        contact.last_observed_source_time_s = source_time_s
        contact.last_observed_timestamp_utc = frame.timestamp_utc
        if frame.processed_frame_sequence != previous_observed_processed_frame_sequence:
            contact.hit_frame_sequences.append(frame.processed_frame_sequence)
            contact.class_evidence.append(detection.class_name)
        self._trim_contact_windows(contact, frame.processed_frame_sequence)

        if contact.state == CONTACT_STATE_TENTATIVE:
            if self._has_confirmation_evidence(contact, frame.processed_frame_sequence):
                contact.state = CONTACT_STATE_CONFIRMED
        elif contact.state in {CONTACT_STATE_COASTING, CONTACT_STATE_LOST}:
            contact.state = CONTACT_STATE_CONFIRMED

        self._update_stabilized_class(contact)

        if contact.state != previous_state:
            events.append(
                self._event_payload(
                    contact,
                    frame,
                    event_type="state_transition",
                    previous_state=previous_state,
                    bbox_designation="measured",
                    detection=detection,
                )
            )

        if contact.stabilized_class != previous_stabilized_class:
            events.append(
                self._event_payload(
                    contact,
                    frame,
                    event_type="class_transition",
                    previous_stabilized_class=previous_stabilized_class,
                    bbox_designation="measured",
                    detection=detection,
                )
            )

        if (
            contact.state == CONTACT_STATE_CONFIRMED
            and contact.last_event_source_time_s is not None
            and source_time_s - contact.last_event_source_time_s >= self.config.update_interval_s
        ):
            events.append(
                self._event_payload(
                    contact,
                    frame,
                    event_type="rate_limited_update",
                    bbox_designation="measured",
                    detection=detection,
                )
            )

        if events:
            contact.last_event_source_time_s = source_time_s

        return events

    def _record_miss(
        self,
        contact: ContactTrack,
        frame: ContactFrame,
        source_time_s: float,
    ) -> dict[str, Any] | None:
        previous_state = contact.state
        age_since_first = max(0.0, source_time_s - contact.first_source_time_s)
        age_since_observed = max(0.0, source_time_s - contact.last_observed_source_time_s)

        if contact.state == CONTACT_STATE_TENTATIVE:
            if age_since_first > self.config.confirmation_max_elapsed_s:
                contact.state = CONTACT_STATE_EXPIRED
                return self._event_payload(
                    contact,
                    frame,
                    event_type="expired",
                    previous_state=previous_state,
                    reason="confirmation_time_limit",
                    bbox_designation="last_seen",
                )
            return None

        if age_since_observed > self.config.expiration_elapsed_s:
            contact.state = CONTACT_STATE_EXPIRED
            return self._event_payload(
                contact,
                frame,
                event_type="expired",
                previous_state=previous_state,
                reason="expiration_time_limit",
                bbox_designation="last_seen",
            )

        if age_since_observed > self.config.coasting_max_elapsed_s:
            contact.state = CONTACT_STATE_LOST
        else:
            contact.state = CONTACT_STATE_COASTING

        if contact.state != previous_state:
            return self._event_payload(
                contact,
                frame,
                event_type="state_transition",
                previous_state=previous_state,
                bbox_designation="last_seen",
            )

        return None

    def _has_confirmation_evidence(self, contact: ContactTrack, frame_sequence: int) -> bool:
        age_since_first = max(0.0, contact.last_observed_source_time_s - contact.first_source_time_s)
        if age_since_first > self.config.confirmation_max_elapsed_s:
            return False

        minimum_frame = frame_sequence - self.config.confirmation_window_size + 1
        hit_count = sum(1 for hit_frame in contact.hit_frame_sequences if hit_frame >= minimum_frame)
        return hit_count >= self.config.confirmation_hits

    def _update_stabilized_class(self, contact: ContactTrack) -> None:
        if not contact.class_evidence:
            return

        counts = Counter(contact.class_evidence)
        current_count = counts.get(contact.stabilized_class, 0)
        candidate, candidate_count = counts.most_common(1)[0]
        if contact.stabilized_class == candidate:
            return

        if candidate_count >= self.config.class_change_hits and candidate_count > current_count:
            contact.stabilized_class = candidate

    def _trim_contact_windows(self, contact: ContactTrack, frame_sequence: int) -> None:
        minimum_frame = frame_sequence - self.config.confirmation_window_size + 1
        while contact.hit_frame_sequences and contact.hit_frame_sequences[0] < minimum_frame:
            contact.hit_frame_sequences.popleft()

        while len(contact.class_evidence) > self.config.class_evidence_window_size:
            contact.class_evidence.popleft()

    def _expire_all(self, frame: ContactFrame, *, reason: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for contact in self._contacts.values():
            previous_state = contact.state
            contact.state = CONTACT_STATE_EXPIRED
            events.append(
                self._event_payload(
                    contact,
                    frame,
                    event_type="expired",
                    previous_state=previous_state,
                    reason=reason,
                    bbox_designation="last_seen",
                )
            )
        return events

    def _event_payload(
        self,
        contact: ContactTrack,
        frame: ContactFrame,
        *,
        event_type: str,
        bbox_designation: str,
        detection: ContactDetectionEvidence | None = None,
        previous_state: str | None = None,
        previous_stabilized_class: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "contact-event-v1",
            "event_type": event_type,
            "lifecycle_state": contact.state,
            "previous_lifecycle_state": previous_state,
            "reason": reason,
            "node_id": contact.node_id,
            "camera_id": contact.camera_id,
            "camera_session_id": contact.camera_session_id,
            "local_track_id": contact.local_track_id,
            "scoped_local_track_id": contact.scoped_local_track_id,
            "contact_ref": contact.contact_ref,
            "current_detection_class": detection.class_name if detection is not None else None,
            "stabilized_class": contact.stabilized_class,
            "previous_stabilized_class": previous_stabilized_class,
            "detection_confidence": detection.confidence if detection is not None else None,
            "model_class_id": detection.model_class_id if detection is not None else None,
            "model_class_name": detection.model_class_name if detection is not None else None,
            "evidence_frames": {
                "first_processed_frame_sequence": contact.first_processed_frame_sequence,
                "last_observed_processed_frame_sequence": contact.last_observed_processed_frame_sequence,
                "event_processed_frame_sequence": frame.processed_frame_sequence,
                "first_capture_frame_sequence": contact.first_capture_frame_sequence,
                "last_observed_capture_frame_sequence": contact.last_observed_capture_frame_sequence,
                "event_capture_frame_sequence": frame.capture_frame_sequence,
                "confirmation_window_size": self.config.confirmation_window_size,
                "confirmation_hits_required": self.config.confirmation_hits,
                "recent_hit_frames": list(contact.hit_frame_sequences),
            },
            "times": {
                "time_basis": contact.source_time_basis,
                "first_observed_source_time_s": contact.first_source_time_s,
                "last_observed_source_time_s": contact.last_observed_source_time_s,
                "event_source_time_s": frame.source_time_s,
                "first_observed_utc": contact.first_timestamp_utc,
                "last_observed_utc": contact.last_observed_timestamp_utc,
                "event_utc": frame.timestamp_utc,
            },
            "bbox_px": {
                "designation": bbox_designation,
                "x_min": contact.last_bbox_xyxy[0],
                "y_min": contact.last_bbox_xyxy[1],
                "x_max": contact.last_bbox_xyxy[2],
                "y_max": contact.last_bbox_xyxy[3],
            },
            "geographic_position": {
                "status": "unknown",
                "latitude": None,
                "longitude": None,
                "altitude_m": None,
            },
        }
        return payload

    def _contact_summary(self, contact: ContactTrack) -> dict[str, Any]:
        return {
            "contact_ref": contact.contact_ref,
            "local_track_id": contact.local_track_id,
            "scoped_local_track_id": contact.scoped_local_track_id,
            "lifecycle_state": contact.state,
            "current_detection_class": contact.current_detection_class,
            "stabilized_class": contact.stabilized_class,
            "last_observed_processed_frame_sequence": contact.last_observed_processed_frame_sequence,
            "last_observed_source_time_s": contact.last_observed_source_time_s,
            "source_time_basis": contact.source_time_basis,
            "bbox_px": {
                "designation": "measured" if contact.state in {CONTACT_STATE_TENTATIVE, CONTACT_STATE_CONFIRMED} else "last_seen",
                "x_min": contact.last_bbox_xyxy[0],
                "y_min": contact.last_bbox_xyxy[1],
                "x_max": contact.last_bbox_xyxy[2],
                "y_max": contact.last_bbox_xyxy[3],
            },
        }

    def _has_time_discontinuity(self, source_time_s: float, source_time_basis: str) -> bool:
        if self._last_source_time_s is None:
            return False
        if source_time_basis != self._last_source_time_basis:
            return True
        return source_time_s < self._last_source_time_s

    def _monotonic_source_time(self, source_time_s: float) -> float:
        if self._last_source_time_s is None:
            return source_time_s
        return max(source_time_s, self._last_source_time_s)
