import json
import unittest

from perception.adapter import FrameMetadata, ImageObservationAdapter, require_geographic_position
from perception.detector import Detection
from perception.exceptions import UnlocatedObservationError
from perception.runner import CameraPerceptionRunner, RunnerConfig


class FakeFrame:
    shape = (480, 640, 3)

    def copy(self):
        return self


class FakeCapture:
    def __init__(self, read_results):
        self.read_results = list(read_results)
        self.opened = False
        self.released = False

    def open(self):
        self.opened = True

    def read(self):
        if not self.read_results:
            return False, None

        return self.read_results.pop(0)

    def release(self):
        self.released = True


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections
        self.frames_seen = 0

    def track(self, frame):
        self.frames_seen += 1
        return list(self.detections)


class FakeWriter:
    def __init__(self):
        self.observations = []
        self.closed = False

    def write_observations(self, observations):
        self.observations.extend(observations)

    def close(self):
        self.closed = True


class FakeDisplay:
    def __init__(self, keys):
        self.keys = list(keys)
        self.closed = False
        self.frames_shown = 0

    def show(self, frame, detections):
        self.frames_shown += 1

    def poll_key(self):
        if not self.keys:
            return -1

        return self.keys.pop(0)

    def close(self):
        self.closed = True


class CameraPerceptionTest(unittest.TestCase):
    def observation_for(self, session_id="session-a", local_track_id=7):
        adapter = ImageObservationAdapter()
        metadata = FrameMetadata(
            node_id="node-1",
            camera_id="camera-0",
            camera_session_id=session_id,
            frame_sequence=3,
            image_width_px=640,
            image_height_px=480,
            timestamp_utc="2026-11-16T12:00:00.000Z",
        )
        return adapter.observation_from_detection(
            Detection(
                class_name="person",
                confidence=0.91,
                bbox_xyxy=(10.0, 20.0, 110.0, 220.0),
                class_id=0,
                local_track_id=local_track_id,
            ),
            metadata,
        )

    def test_observation_serializes_unknown_geolocation_without_zero_coordinates(self):
        observation = self.observation_for()
        payload = observation.to_dict()
        line_payload = json.loads(observation.to_json_line())

        self.assertEqual("image-observation-v1", payload["schema_version"])
        self.assertEqual({"status": "unknown", "latitude": None, "longitude": None, "altitude_m": None}, payload["geographic_position"])
        self.assertNotIn("east_m", payload)
        self.assertNotIn("north_m", payload)
        self.assertNotIn("latitude", payload)
        self.assertNotIn("longitude", payload)
        self.assertEqual(payload, line_payload)

    def test_track_ids_are_scoped_to_camera_session(self):
        first = self.observation_for("session-a", 7)
        second = self.observation_for("session-b", 7)

        self.assertEqual(7, first.local_track_id)
        self.assertEqual(7, second.local_track_id)
        self.assertEqual("camera_session", first.track_id_scope)
        self.assertEqual("camera_session", second.track_id_scope)
        self.assertEqual("session-a:7", first.scoped_local_track_id)
        self.assertEqual("session-b:7", second.scoped_local_track_id)
        self.assertNotEqual(first.scoped_local_track_id, second.scoped_local_track_id)

    def test_local_track_id_can_be_null(self):
        observation = self.observation_for(local_track_id=None)

        self.assertIsNone(observation.local_track_id)
        self.assertIsNone(observation.scoped_local_track_id)

    def test_unlocated_observation_rejected_at_geographic_boundary(self):
        observation = self.observation_for()

        with self.assertRaises(UnlocatedObservationError):
            require_geographic_position(observation)

    def test_runner_cleans_up_on_user_quit(self):
        capture = FakeCapture([(True, FakeFrame()), (True, FakeFrame())])
        writer = FakeWriter()
        display = FakeDisplay([ord("q")])
        runner = CameraPerceptionRunner(
            config=RunnerConfig.create(node_id="node-1", camera_id="camera-0"),
            capture=capture,
            detector=FakeDetector([Detection("person", 0.9, (1, 2, 3, 4), local_track_id=1)]),
            writer=writer,
            display=display,
        )

        result = runner.run()

        self.assertEqual("user_requested", result.stopped_reason)
        self.assertEqual(1, result.frames_processed)
        self.assertEqual(1, result.observations_written)
        self.assertTrue(capture.released)
        self.assertTrue(writer.closed)
        self.assertTrue(display.closed)

    def test_runner_cleans_up_on_capture_failure(self):
        capture = FakeCapture([(False, None)])
        writer = FakeWriter()
        display = FakeDisplay([])
        runner = CameraPerceptionRunner(
            config=RunnerConfig.create(node_id="node-1", camera_id="camera-0"),
            capture=capture,
            detector=FakeDetector([]),
            writer=writer,
            display=display,
        )

        result = runner.run()

        self.assertEqual("capture_failure", result.stopped_reason)
        self.assertEqual(0, result.frames_processed)
        self.assertEqual(0, result.observations_written)
        self.assertTrue(capture.released)
        self.assertTrue(writer.closed)
        self.assertTrue(display.closed)


if __name__ == "__main__":
    unittest.main()
