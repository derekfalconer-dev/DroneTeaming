import unittest

from perception.display import OpenCVDisplay
from perception.runner import CameraPerceptionRunner, RunnerConfig


class FakeFrame:
    shape = (480, 640, 3)


class FakeCapture:
    def __init__(self) -> None:
        self.released = False
        self.read_count = 0

    def open(self) -> None:
        pass

    def read(self) -> tuple[bool, object | None]:
        self.read_count += 1
        if self.read_count == 1:
            return True, FakeFrame()
        return False, None

    def release(self) -> None:
        self.released = True


class FakeDetector:
    def track(self, frame: object) -> list[object]:
        return []


MISSING = object()


class FakeObservation:
    def __init__(
        self,
        *,
        local_track_id: object | None = "track-1",
        class_name: str = "person",
        detection_confidence: object = 0.91,
        appearance_attributes: dict[str, object] | None = None,
    ) -> None:
        self.local_track_id = local_track_id
        self.class_name = class_name
        self.detection_confidence = detection_confidence
        self.appearance_attributes = appearance_attributes

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "local_track_id": self.local_track_id,
            "class_name": self.class_name,
            "bbox_px": {
                "x_min": 10,
                "y_min": 20,
                "x_max": 30,
                "y_max": 40,
            },
        }
        if self.detection_confidence is not MISSING:
            payload["detection_confidence"] = self.detection_confidence
        if self.appearance_attributes is not None:
            payload["appearance_attributes"] = self.appearance_attributes

        return payload


class FakeAdapter:
    def __init__(self) -> None:
        self.observations = [FakeObservation()]

    def observations_for_frame(self, detections: list[object], metadata: object) -> list[FakeObservation]:
        return self.observations


class FakeWriter:
    def __init__(self) -> None:
        self.observation_batches: list[list[FakeObservation]] = []
        self.closed = False

    def write_observations(self, observations: list[FakeObservation]) -> None:
        self.observation_batches.append(observations)

    def close(self) -> None:
        self.closed = True


class FakeDisplay:
    def __init__(self) -> None:
        self.observations: list[FakeObservation] | None = None
        self.context: dict[str, object] | None = None
        self.closed = False

    def show(self, frame: object, detections: list[object]) -> None:
        raise AssertionError("runner should prefer show_observation_panel when available")

    def show_observation_panel(
        self,
        frame: object,
        detections: list[object],
        observations: list[FakeObservation],
        context: dict[str, object],
    ) -> None:
        self.observations = observations
        self.context = context

    def poll_key(self) -> int:
        return -1

    def close(self) -> None:
        self.closed = True


class LivePanelRunnerTest(unittest.TestCase):
    def test_runner_passes_jsonl_observations_and_context_to_display_panel(self) -> None:
        capture = FakeCapture()
        adapter = FakeAdapter()
        writer = FakeWriter()
        display = FakeDisplay()

        runner = CameraPerceptionRunner(
            config=RunnerConfig.create(
                node_id="node-a",
                camera_id="camera-a",
                camera_session_id="session-a",
                max_frames=1,
                source_label="0",
                requested_inference_size_px=1280,
                model_identifier="test-model.pt",
            ),
            capture=capture,
            detector=FakeDetector(),
            writer=writer,
            adapter=adapter,
            display=display,
        )

        result = runner.run()

        self.assertEqual(result.frames_processed, 1)
        self.assertEqual(result.observations_written, 1)
        self.assertIs(writer.observation_batches[0], adapter.observations)
        self.assertIs(display.observations, adapter.observations)
        self.assertIsNotNone(display.context)
        self.assertEqual(display.context["source_label"], "0")
        self.assertEqual(display.context["node_id"], "node-a")
        self.assertEqual(display.context["camera_id"], "camera-a")
        self.assertEqual(display.context["class_filter_label"], "all model classes")
        self.assertEqual(display.context["label_profile"], "coco")
        self.assertEqual(display.context["active_class_label"], "all model classes")
        self.assertEqual(display.context["camera_session_id"], "session-a")
        self.assertEqual(display.context["frame_sequence"], 1)
        self.assertEqual(display.context["observations_written"], 1)
        self.assertEqual(display.context["requested_inference_size_px"], 1280)
        self.assertEqual(display.context["model_identifier"], "test-model.pt")
        self.assertEqual(display.context["frame_width_px"], 640)
        self.assertEqual(display.context["frame_height_px"], 480)
        self.assertFalse(display.context["appearance_debug"])
        self.assertIn("frame_timestamp_utc", display.context)
        self.assertIn("processing_fps", display.context)
        self.assertTrue(capture.released)
        self.assertTrue(writer.closed)
        self.assertTrue(display.closed)

    def test_runner_passes_empty_observation_list_to_clear_current_panel_rows(self) -> None:
        capture = FakeCapture()
        adapter = FakeAdapter()
        adapter.observations = []
        writer = FakeWriter()
        display = FakeDisplay()

        runner = CameraPerceptionRunner(
            config=RunnerConfig.create(
                node_id="node-a",
                camera_id="camera-a",
                camera_session_id="session-a",
                max_frames=1,
                source_label="0",
            ),
            capture=capture,
            detector=FakeDetector(),
            writer=writer,
            adapter=adapter,
            display=display,
        )

        result = runner.run()

        self.assertEqual(result.frames_processed, 1)
        self.assertEqual(result.observations_written, 0)
        self.assertEqual([], writer.observation_batches[0])
        self.assertEqual([], display.observations)

    def test_runner_passes_label_profile_and_active_classes_to_display_panel(self) -> None:
        capture = FakeCapture()
        adapter = FakeAdapter()
        writer = FakeWriter()
        display = FakeDisplay()

        runner = CameraPerceptionRunner(
            config=RunnerConfig.create(
                node_id="node-a",
                camera_id="camera-a",
                camera_session_id="session-a",
                max_frames=1,
                label_profile="visdrone",
                active_class_label="person, car, van",
            ),
            capture=capture,
            detector=FakeDetector(),
            writer=writer,
            adapter=adapter,
            display=display,
        )

        runner.run()

        self.assertIsNotNone(display.context)
        self.assertEqual(display.context["label_profile"], "visdrone")
        self.assertEqual(display.context["active_class_label"], "person, car, van")

    def test_display_formats_inference_request_as_requested_size_not_tensor_shape(self) -> None:
        display = object.__new__(OpenCVDisplay)

        self.assertEqual(
            "imgsz=1280px requested; tensor shape may differ",
            display._format_inference_request(1280),
        )
        self.assertEqual(
            "3840x2160 px",
            display._format_frame_dimensions({"frame_width_px": 3840, "frame_height_px": 2160}),
        )

    def test_display_observation_lines_include_serialized_clothing_color_attributes(self) -> None:
        display = object.__new__(OpenCVDisplay)

        lines = display._observation_lines(
            [
                FakeObservation(
                    local_track_id=35,
                    class_name="person",
                    detection_confidence=0.94,
                    appearance_attributes={
                        "upper_clothing_color": {
                            "value": "red",
                            "quality_label": "moderate",
                            "reason": None,
                        }
                    },
                ),
                FakeObservation(
                    local_track_id=36,
                    class_name="person",
                    detection_confidence=0.88,
                    appearance_attributes={
                        "upper_clothing_color": {
                            "value": "unknown",
                            "quality_label": "unavailable",
                            "reason": "crop_too_small",
                        }
                    },
                ),
            ]
        )

        self.assertEqual(
            lines,
            [
                "Person | Track 35 | Detection score 94%",
                "  Upper clothing: red | Color quality: moderate",
                "Person | Track 36 | Detection score 88%",
                "  Upper clothing: unknown | Color quality: unavailable",
                "  Reason: crop_too_small",
            ],
        )

    def test_display_wraps_long_appearance_text_without_right_edge_clipping(self) -> None:
        display = object.__new__(OpenCVDisplay)

        lines = display._observation_lines(
            [
                FakeObservation(
                    local_track_id=35,
                    class_name="person",
                    detection_confidence=0.94,
                    appearance_attributes={
                        "upper_clothing_color": {
                            "value": "unknown",
                            "quality_label": "unavailable because crop evidence is mixed",
                            "reason": "insufficient_evidence_due_to_background_contamination",
                        }
                    },
                ),
            ]
        )

        self.assertGreater(len(lines), 3)
        self.assertTrue(all(len(line) <= 54 for line in lines[1:]))
        self.assertTrue(any(line.startswith("  Reason:") for line in lines))

    def test_display_debug_appearance_lines_include_sampling_region_dimensions_and_quality(self) -> None:
        display = object.__new__(OpenCVDisplay)

        lines = display._observation_lines(
            [
                FakeObservation(
                    local_track_id=35,
                    class_name="person",
                    detection_confidence=0.94,
                    appearance_attributes={
                        "upper_clothing_color": {
                            "value": "red",
                            "quality_label": "moderate",
                            "quality_score": 0.73,
                            "sampling_region_px": {
                                "x_min": 32,
                                "y_min": 88,
                                "x_max": 88,
                                "y_max": 156,
                            },
                            "reason": None,
                            "diagnostics": {
                                "color_evidence": {
                                    "sample_analyzed": True,
                                    "top_candidates": [
                                        {"color": "red", "fraction_of_sample": 0.5},
                                        {"color": "blue", "fraction_of_sample": 0.25},
                                        {"color": "green", "fraction_of_sample": 0.1},
                                    ],
                                },
                                "decision": {
                                    "quality_score": 0.73,
                                    "rejection_reason": None,
                                    "thresholds": {
                                        "minimum_valid_pixel_fraction": 0.35,
                                        "minimum_dominant_share": 0.55,
                                        "minimum_winner_gap": 0.2,
                                    },
                                },
                            },
                        }
                    },
                ),
            ],
            appearance_debug=True,
        )

        self.assertEqual(
            lines,
            [
                "Person | Track 35 | Detection score 94%",
                "  Upper clothing: red | Color quality: moderate",
                "  Sample: x=32 y=88 w=56 h=68 px original",
                "  Top colors: red 50.0%, blue 25.0%, green 10.0% of",
                "  sample",
                "  Quality score: 0.73; thresholds: valid>=35%,",
                "  top>=55%, gap>=20%",
                "  Rejection: none",
            ],
        )

    def test_display_debug_appearance_lines_show_unavailable_evidence_reason(self) -> None:
        display = object.__new__(OpenCVDisplay)

        lines = display._observation_lines(
            [
                FakeObservation(
                    local_track_id=35,
                    class_name="person",
                    detection_confidence=0.94,
                    appearance_attributes={
                        "upper_clothing_color": {
                            "value": "unknown",
                            "quality_label": "unavailable",
                            "quality_score": None,
                            "sampling_region_px": None,
                            "reason": "crop_too_small",
                            "diagnostics": {
                                "color_evidence": {
                                    "sample_analyzed": False,
                                    "unavailable_reason": "crop_too_small",
                                },
                                "decision": {
                                    "quality_score": None,
                                    "rejection_reason": "crop_too_small",
                                    "thresholds": {
                                        "minimum_valid_pixel_fraction": 0.35,
                                        "minimum_dominant_share": 0.55,
                                        "minimum_winner_gap": 0.2,
                                    },
                                },
                            },
                        }
                    },
                ),
            ],
            appearance_debug=True,
        )

        self.assertEqual(
            lines,
            [
                "Person | Track 35 | Detection score 94%",
                "  Upper clothing: unknown | Color quality: unavailable",
                "  Reason: crop_too_small",
                "  Evidence unavailable: crop_too_small",
                "  Quality score: unavailable; thresholds: valid>=35%,",
                "  top>=55%, gap>=20%",
                "  Rejection: crop_too_small",
            ],
        )

    def test_display_observation_lines_use_detection_confidence(self) -> None:
        display = object.__new__(OpenCVDisplay)

        lines = display._observation_lines(
            [
                FakeObservation(local_track_id=35, class_name="person", detection_confidence=0.94),
                FakeObservation(local_track_id=7, class_name="potted plant", detection_confidence=0.67),
                FakeObservation(local_track_id=0, class_name="vehicle", detection_confidence=0.0),
                FakeObservation(local_track_id=None, class_name="", detection_confidence=MISSING),
            ]
        )

        self.assertEqual(
            lines,
            [
                "Person | Track 35 | Detection score 94%",
                "Potted plant | Track 7 | Detection score 67%",
                "Vehicle | Track 0 | Detection score 0%",
                "Unknown | Track unavailable | Detection score unavailable",
            ],
        )


if __name__ == "__main__":
    unittest.main()
