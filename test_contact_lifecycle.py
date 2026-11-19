import unittest

from perception.contact_lifecycle import (
    CONTACT_STATE_CONFIRMED,
    CONTACT_STATE_EXPIRED,
    CONTACT_STATE_TENTATIVE,
    ContactConfirmationConfig,
    ContactDetectionEvidence,
    ContactFrame,
    ContactLifecycleManager,
)


def frame(sequence: int, source_time_s: float, *, session: str = "session-a") -> ContactFrame:
    return ContactFrame(
        node_id="node-1",
        camera_id="camera-1",
        camera_session_id=session,
        processed_frame_sequence=sequence,
        capture_frame_sequence=sequence + 100,
        timestamp_utc=f"2026-11-18T00:00:{sequence:02d}.000Z",
        source_time_s=source_time_s,
        source_time_basis="test_source_time",
        image_width_px=640,
        image_height_px=480,
    )


def detection(track_id=7, class_name="person", confidence=0.9) -> ContactDetectionEvidence:
    return ContactDetectionEvidence(
        local_track_id=track_id,
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy=(10.0, 20.0, 110.0, 220.0),
        model_class_id=0,
        model_class_name=class_name,
    )


class ContactLifecycleTest(unittest.TestCase):
    def test_single_frame_contact_remains_tentative_then_expires(self) -> None:
        manager = ContactLifecycleManager(
            ContactConfirmationConfig(confirmation_max_elapsed_s=0.5, expiration_elapsed_s=2.0)
        )

        events = manager.process_frame(frame(1, 0.0), [detection()])

        self.assertEqual("created", events[0]["event_type"])
        self.assertEqual(CONTACT_STATE_TENTATIVE, events[0]["lifecycle_state"])
        self.assertEqual({"tentative": 1, "confirmed": 0, "coasting": 0}, manager.snapshot_counts())

        events = manager.process_frame(frame(2, 0.6), [])

        self.assertEqual("expired", events[0]["event_type"])
        self.assertEqual(CONTACT_STATE_EXPIRED, events[0]["lifecycle_state"])
        self.assertEqual("last_seen", events[0]["bbox_px"]["designation"])
        self.assertEqual({"tentative": 0, "confirmed": 0, "coasting": 0}, manager.snapshot_counts())

    def test_three_of_five_evidence_confirms_contact_and_duplicates_count_once(self) -> None:
        manager = ContactLifecycleManager()
        manager.process_frame(frame(1, 0.0), [detection(), detection(confidence=0.1)])
        self.assertEqual(CONTACT_STATE_TENTATIVE, manager.snapshot_contacts()[0]["lifecycle_state"])
        manager.process_frame(frame(2, 0.1), [])
        events = manager.process_frame(frame(3, 0.2), [detection()])
        self.assertFalse(any(event["lifecycle_state"] == CONTACT_STATE_CONFIRMED for event in events))
        self.assertEqual(CONTACT_STATE_TENTATIVE, manager.snapshot_contacts()[0]["lifecycle_state"])

        events = manager.process_frame(frame(4, 0.3), [detection(), detection(confidence=0.2)])

        confirmed_events = [event for event in events if event["lifecycle_state"] == CONTACT_STATE_CONFIRMED]
        self.assertEqual(1, len(confirmed_events))
        self.assertEqual("state_transition", confirmed_events[0]["event_type"])
        self.assertEqual([1, 3, 4], confirmed_events[0]["evidence_frames"]["recent_hit_frames"])

    def test_empty_images_advance_coasting_and_expiration(self) -> None:
        manager = ContactLifecycleManager(
            ContactConfirmationConfig(
                confirmation_hits=1,
                confirmation_window_size=1,
                coasting_max_elapsed_s=0.5,
                expiration_elapsed_s=1.0,
            )
        )
        manager.process_frame(frame(1, 0.0), [detection()])

        events = manager.process_frame(frame(2, 0.25), [])
        self.assertEqual("coasting", events[0]["lifecycle_state"])
        self.assertEqual("last_seen", events[0]["bbox_px"]["designation"])

        events = manager.process_frame(frame(3, 1.1), [])
        self.assertEqual("expired", events[0]["event_type"])

    def test_time_limits_work_independently_of_frame_rate(self) -> None:
        manager = ContactLifecycleManager(
            ContactConfirmationConfig(
                confirmation_hits=3,
                confirmation_window_size=5,
                confirmation_max_elapsed_s=0.25,
            )
        )
        manager.process_frame(frame(1, 0.0), [detection()])
        manager.process_frame(frame(2, 0.2), [detection()])
        events = manager.process_frame(frame(3, 0.4), [detection()])

        self.assertFalse(any(event["lifecycle_state"] == CONTACT_STATE_CONFIRMED for event in events))

    def test_class_smoothing_resists_one_frame_flip_and_accepts_sustained_change(self) -> None:
        manager = ContactLifecycleManager(
            ContactConfirmationConfig(
                confirmation_hits=1,
                confirmation_window_size=1,
                class_evidence_window_size=5,
                class_change_hits=3,
            )
        )
        manager.process_frame(frame(1, 0.0), [detection(class_name="person")])
        manager.process_frame(frame(2, 0.1), [detection(class_name="car")])
        self.assertEqual("person", manager.snapshot_contacts()[0]["stabilized_class"])

        manager.process_frame(frame(3, 0.2), [detection(class_name="car")])
        events = manager.process_frame(frame(4, 0.3), [detection(class_name="car")])

        self.assertEqual("car", manager.snapshot_contacts()[0]["stabilized_class"])
        self.assertTrue(any(event["event_type"] == "class_transition" for event in events))

    def test_tracker_id_reuse_after_expiration_and_new_sessions_do_not_inherit_evidence(self) -> None:
        manager = ContactLifecycleManager(
            ContactConfirmationConfig(
                confirmation_hits=2,
                confirmation_window_size=2,
                confirmation_max_elapsed_s=0.5,
            )
        )
        manager.process_frame(frame(1, 0.0), [detection(track_id=7)])
        expiration_events = manager.process_frame(frame(2, 0.6), [])
        self.assertEqual("expired", expiration_events[0]["event_type"])
        self.assertEqual(CONTACT_STATE_EXPIRED, expiration_events[0]["lifecycle_state"])

        events = manager.process_frame(frame(3, 0.7), [detection(track_id=7)])
        created_events = [event for event in events if event["event_type"] == "created"]

        self.assertEqual(1, len(created_events))
        self.assertTrue(created_events[0]["contact_ref"].endswith(":7:2"))
        self.assertEqual(CONTACT_STATE_TENTATIVE, created_events[0]["lifecycle_state"])
        self.assertEqual([3], created_events[0]["evidence_frames"]["recent_hit_frames"])

        events = manager.process_frame(frame(1, 0.0, session="session-b"), [detection(track_id=7)])

        self.assertTrue(events[0]["contact_ref"].startswith("session-b:7:1"))
        self.assertEqual(CONTACT_STATE_TENTATIVE, events[0]["lifecycle_state"])

    def test_detections_without_tracker_ids_remain_unassociated(self) -> None:
        manager = ContactLifecycleManager()

        events = manager.process_frame(frame(1, 0.0), [detection(track_id=None)])

        self.assertEqual([], events)
        self.assertEqual([], manager.snapshot_contacts())


if __name__ == "__main__":
    unittest.main()
