import json
import sys
import types
import unittest
from unittest.mock import patch

from perception.adapter import FrameMetadata, ImageObservationAdapter, require_geographic_position
from perception.appearance import (
    UPPER_CLOTHING_COLOR_ATTRIBUTE,
    UPPER_CLOTHING_COLOR_METHOD,
    UPPER_CLOTHING_COLOR_METHOD_VERSION,
    OpenCVUpperClothingColorAnalyzer,
    PixelColorAnalyzer,
)
from perception.detector import Detection, DetectionClassFilterError, UltralyticsYoloTracker
from perception.display import OpenCVDisplay
from perception.exceptions import MissingDependencyError, UnlocatedObservationError
from perception.runner import CameraPerceptionRunner, RunnerConfig
import run_perception


class FakeFrame:
    shape = (480, 640, 3)

    def copy(self):
        return self


class Fake4KFrame:
    shape = (2160, 3840, 3)

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


class FakeBoxes:
    def __init__(self, rows):
        self.xyxy = [row["bbox"] for row in rows]
        self.conf = [row["confidence"] for row in rows]
        self.cls = [row["class_id"] for row in rows]
        self.id = [row["track_id"] for row in rows]


class FakeUltralyticsResult:
    def __init__(self, rows):
        self.boxes = FakeBoxes(rows)


class FakeUltralyticsModel:
    def __init__(self, *, rows, names, supports_class_filter=True):
        self.rows = list(rows)
        self.names = names
        self.supports_class_filter = supports_class_filter
        self.track_calls = []

    def track(self, frame, **kwargs):
        self.track_calls.append(kwargs)
        class_ids = kwargs.get("classes")
        if class_ids is not None and not self.supports_class_filter:
            raise TypeError("classes argument is not supported")

        rows = self.rows
        if class_ids is not None:
            allowed_class_ids = set(class_ids)
            rows = [row for row in rows if row["class_id"] in allowed_class_ids]

        return [FakeUltralyticsResult(rows)]


class RecordingAppearanceAnalyzer:
    def __init__(self):
        self.calls = []

    def analyze_frame(self, frame, detections):
        self.calls.append(list(detections))
        return [None for _ in detections]


class RunPerceptionCliTest(unittest.TestCase):
    def test_imgsz_defaults_to_640_and_accepts_custom_positive_integer(self):
        with patch.object(sys, "argv", ["run_perception.py"]):
            self.assertEqual(640, run_perception.parse_args().imgsz)

        with patch.object(sys, "argv", ["run_perception.py", "--imgsz", "1280"]):
            self.assertEqual(1280, run_perception.parse_args().imgsz)

    def test_imgsz_rejects_non_positive_values(self):
        for value in ("0", "-1"):
            with self.subTest(value=value):
                with patch.object(sys, "argv", ["run_perception.py", "--imgsz", value]):
                    with self.assertRaises(SystemExit):
                        run_perception.parse_args()

    def test_label_profile_defaults_to_coco_and_accepts_visdrone(self):
        with patch.object(sys, "argv", ["run_perception.py"]):
            self.assertEqual("coco", run_perception.parse_args().label_profile)

        with patch.object(sys, "argv", ["run_perception.py", "--label-profile", "visdrone"]):
            self.assertEqual("visdrone", run_perception.parse_args().label_profile)


class DetectionClassFilteringTest(unittest.TestCase):
    def setUp(self):
        self.names = {
            0: "person",
            2: "car",
            5: "laptop",
            7: "potted plant",
            10: "boat",
            12: "cat",
            99: "truck",
        }
        self.rows = [
            {"class_id": 0, "confidence": 0.91, "bbox": (1.0, 2.0, 3.0, 4.0), "track_id": 10},
            {"class_id": 5, "confidence": 0.82, "bbox": (5.0, 6.0, 7.0, 8.0), "track_id": 11},
            {"class_id": 7, "confidence": 0.73, "bbox": (9.0, 10.0, 11.0, 12.0), "track_id": 12},
            {"class_id": 12, "confidence": 0.64, "bbox": (13.0, 14.0, 15.0, 16.0), "track_id": 13},
            {"class_id": 99, "confidence": 0.77, "bbox": (17.0, 18.0, 19.0, 20.0), "track_id": 14},
            {"class_id": 10, "confidence": 0.86, "bbox": (21.0, 22.0, 23.0, 24.0), "track_id": 15},
        ]

    def tracker_for(self, model, **kwargs):
        fake_ultralytics = types.SimpleNamespace(YOLO=lambda model_path: model)
        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            return UltralyticsYoloTracker("fake.pt", **kwargs)

    def test_noncontiguous_model_ids_resolve_by_name_not_position(self):
        model = FakeUltralyticsModel(rows=[], names=self.names)

        tracker = self.tracker_for(model)

        self.assertEqual((0, 2, 99, 10), tracker.class_filter.class_ids)
        tracker.track(object())
        self.assertEqual([0, 2, 99, 10], model.track_calls[0]["classes"])
        self.assertEqual(640, model.track_calls[0]["imgsz"])
        self.assertEqual("coco", tracker.label_profile.name)

    def test_custom_imgsz_is_passed_to_ultralytics_track(self):
        model = FakeUltralyticsModel(rows=[], names=self.names)
        tracker = self.tracker_for(model, inference_image_size=1280)

        tracker.track(object())

        self.assertEqual(1280, model.track_calls[0]["imgsz"])

    def test_default_sar_filter_suppresses_laptop_potted_plant_and_animals(self):
        model = FakeUltralyticsModel(
            rows=self.rows,
            names=self.names,
            supports_class_filter=False,
        )
        tracker = self.tracker_for(model)

        detections = tracker.track(object())

        self.assertEqual(["person", "truck", "boat"], [detection.class_name for detection in detections])
        self.assertEqual(2, len(model.track_calls))
        self.assertEqual(640, model.track_calls[0]["imgsz"])
        self.assertEqual(640, model.track_calls[1]["imgsz"])
        self.assertIn("classes", model.track_calls[0])
        self.assertNotIn("classes", model.track_calls[1])

    def test_allowed_detections_retain_original_model_fields(self):
        model = FakeUltralyticsModel(rows=self.rows, names=self.names)
        tracker = self.tracker_for(model)

        detections = tracker.track(object())
        truck = next(detection for detection in detections if detection.class_name == "truck")

        self.assertEqual(99, truck.class_id)
        self.assertEqual(0.77, truck.confidence)
        self.assertEqual((17.0, 18.0, 19.0, 20.0), truck.bbox_xyxy)
        self.assertEqual(14, truck.local_track_id)

    def test_custom_filter_all_classes_invalid_names_and_empty_results(self):
        custom_model = FakeUltralyticsModel(rows=self.rows, names=self.names)
        custom_tracker = self.tracker_for(
            custom_model,
            class_names=("laptop",),
            strict_class_names=True,
        )

        self.assertEqual(["laptop"], [detection.class_name for detection in custom_tracker.track(object())])
        self.assertEqual([5], custom_model.track_calls[0]["classes"])

        all_model = FakeUltralyticsModel(rows=self.rows, names=self.names)
        all_tracker = self.tracker_for(all_model, class_names=None)

        self.assertEqual(
            ["person", "laptop", "potted plant", "cat", "truck", "boat"],
            [detection.class_name for detection in all_tracker.track(object())],
        )
        self.assertIsNone(all_model.track_calls[0]["classes"])
        self.assertEqual(640, all_model.track_calls[0]["imgsz"])

        with self.assertRaisesRegex(DetectionClassFilterError, "Available model classes"):
            self.tracker_for(
                FakeUltralyticsModel(rows=[], names=self.names),
                class_names=("spaceship",),
                strict_class_names=True,
            )

        empty_tracker = self.tracker_for(FakeUltralyticsModel(rows=[], names=self.names))
        self.assertEqual([], empty_tracker.track(object()))

    def test_visdrone_reordered_ids_normalize_without_assuming_coco_ids(self):
        names = {
            4: "truck",
            9: "pedestrian",
            1: "motor",
            7: "people",
            2: "van",
        }
        rows = [
            {"class_id": 9, "confidence": 0.91, "bbox": (1.0, 2.0, 3.0, 4.0), "track_id": 10},
            {"class_id": 7, "confidence": 0.82, "bbox": (5.0, 6.0, 7.0, 8.0), "track_id": 11},
            {"class_id": 1, "confidence": 0.73, "bbox": (9.0, 10.0, 11.0, 12.0), "track_id": 12},
            {"class_id": 2, "confidence": 0.64, "bbox": (13.0, 14.0, 15.0, 16.0), "track_id": 13},
        ]
        model = FakeUltralyticsModel(rows=rows, names=names)
        tracker = self.tracker_for(model, label_profile="visdrone")

        detections = tracker.track(object())

        self.assertCountEqual([9, 1, 7, 2, 4], model.track_calls[0]["classes"])
        self.assertEqual(
            ["person", "person", "motorcycle", "van"],
            [detection.class_name for detection in detections],
        )
        self.assertEqual(
            ["pedestrian", "people", "motor", "van"],
            [detection.model_class_name for detection in detections],
        )

    def test_visdrone_person_filter_selects_pedestrian_and_people_source_ids(self):
        names = {
            10: "car",
            20: "people",
            30: "pedestrian",
            40: "motor",
        }
        rows = [
            {"class_id": 10, "confidence": 0.91, "bbox": (1.0, 2.0, 3.0, 4.0), "track_id": 10},
            {"class_id": 20, "confidence": 0.82, "bbox": (5.0, 6.0, 7.0, 8.0), "track_id": 11},
            {"class_id": 30, "confidence": 0.73, "bbox": (9.0, 10.0, 11.0, 12.0), "track_id": 12},
            {"class_id": 40, "confidence": 0.64, "bbox": (13.0, 14.0, 15.0, 16.0), "track_id": 13},
        ]
        model = FakeUltralyticsModel(rows=rows, names=names)
        tracker = self.tracker_for(
            model,
            label_profile="visdrone",
            class_names=("person",),
            strict_class_names=True,
        )

        detections = tracker.track(object())

        self.assertEqual((20, 30), tracker.class_filter.class_ids)
        self.assertEqual([20, 30], model.track_calls[0]["classes"])
        self.assertEqual(["person", "person"], [detection.class_name for detection in detections])

    def test_visdrone_van_remains_distinct_and_motorcycle_filter_selects_motor(self):
        names = {
            1: "van",
            2: "truck",
            3: "motor",
        }

        van_tracker = self.tracker_for(
            FakeUltralyticsModel(
                rows=[{"class_id": 1, "confidence": 0.9, "bbox": (1.0, 2.0, 3.0, 4.0), "track_id": 1}],
                names=names,
            ),
            label_profile="visdrone",
            class_names=("van",),
            strict_class_names=True,
        )
        self.assertEqual(["van"], [detection.class_name for detection in van_tracker.track(object())])
        self.assertEqual([1], van_tracker.model.track_calls[0]["classes"])

        motorcycle_tracker = self.tracker_for(
            FakeUltralyticsModel(
                rows=[{"class_id": 3, "confidence": 0.8, "bbox": (5.0, 6.0, 7.0, 8.0), "track_id": 2}],
                names=names,
            ),
            label_profile="visdrone",
            class_names=("motorcycle",),
            strict_class_names=True,
        )
        self.assertEqual(
            ["motorcycle"],
            [detection.class_name for detection in motorcycle_tracker.track(object())],
        )
        self.assertEqual([3], motorcycle_tracker.model.track_calls[0]["classes"])

    def test_explicit_unavailable_and_empty_class_filters_report_useful_errors(self):
        names = {1: "pedestrian", 2: "people"}

        with self.assertRaisesRegex(DetectionClassFilterError, r"Available model classes \(normalized\)"):
            self.tracker_for(
                FakeUltralyticsModel(rows=[], names=names),
                label_profile="visdrone",
                class_names=("boat",),
                strict_class_names=True,
            )

        with self.assertRaisesRegex(ValueError, "at least one"):
            run_perception.parse_class_names(" , ")

        with patch.object(sys, "argv", ["run_perception.py", "--classes", " , "]):
            with self.assertRaises(SystemExit):
                run_perception.parse_args()

    def test_ambiguous_duplicate_model_label_metadata_is_rejected(self):
        with self.assertRaisesRegex(DetectionClassFilterError, "Ambiguous model class metadata"):
            self.tracker_for(
                FakeUltralyticsModel(rows=[], names={1: "car", 2: "car"}),
                label_profile="visdrone",
            )

    def test_filtered_empty_frames_skip_appearance_work_and_emit_no_observations(self):
        capture = FakeCapture([(True, FakeFrame())])
        writer = FakeWriter()
        appearance_analyzer = RecordingAppearanceAnalyzer()
        runner = CameraPerceptionRunner(
            config=RunnerConfig.create(node_id="node-1", camera_id="camera-0", max_frames=1),
            capture=capture,
            detector=FakeDetector([]),
            writer=writer,
            appearance_analyzer=appearance_analyzer,
        )

        result = runner.run()

        self.assertEqual("completed", result.stopped_reason)
        self.assertEqual(1, result.frames_processed)
        self.assertEqual(0, result.observations_written)
        self.assertEqual([], appearance_analyzer.calls)
        self.assertEqual([], writer.observations)


class AppearanceDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        try:
            self.analyzer = OpenCVUpperClothingColorAnalyzer()
        except MissingDependencyError as error:
            self.skipTest(str(error))

    def test_upper_clothing_analyzer_initializes_composed_pixel_analyzer(self):
        self.assertIsInstance(self.analyzer.pixel_color_analyzer, PixelColorAnalyzer)
        self.assertIs(self.analyzer.pixel_color_analyzer.cv2, self.analyzer.cv2)
        self.assertIs(self.analyzer.pixel_color_analyzer.numpy, self.analyzer.numpy)

    def test_dominant_palette_color_delegates_to_composed_pixel_analyzer(self):
        class RecordingPixelAnalyzer:
            def __init__(self):
                self.samples = []

            def analyze(self, bgr_region):
                self.samples.append(bgr_region)
                return {
                    "value": "red",
                    "reason": None,
                    "quality_score": 0.8,
                    "quality_label": "high",
                    "diagnostics": {},
                }

        pixel_analyzer = RecordingPixelAnalyzer()
        sample = object()
        self.analyzer.pixel_color_analyzer = pixel_analyzer

        result = self.analyzer._dominant_palette_color(sample)

        self.assertEqual("red", result["value"])
        self.assertEqual([sample], pixel_analyzer.samples)

    def synthetic_crop(self, color_counts):
        pixels = []
        for hsv_color, count in color_counts:
            color_pixel = self.analyzer.numpy.array([[hsv_color]], dtype=self.analyzer.numpy.uint8)
            bgr_color = self.analyzer.cv2.cvtColor(color_pixel, self.analyzer.cv2.COLOR_HSV2BGR)[0, 0]
            pixels.extend([bgr_color.tolist() for _ in range(count)])

        return self.analyzer.numpy.array(pixels, dtype=self.analyzer.numpy.uint8).reshape((10, 10, 3))

    def assert_top_candidates(self, result, expected):
        self.assertEqual(
            expected,
            [
                (
                    candidate["color"],
                    candidate["pixel_count"],
                    candidate["fraction_of_sample"],
                )
                for candidate in result["diagnostics"]["color_evidence"]["top_candidates"]
            ],
        )

    def test_color_evidence_reports_known_proportions_and_excluded_pixels(self):
        crop = self.synthetic_crop(
            [
                ((0, 255, 255), 50),
                ((110, 255, 255), 25),
                ((60, 255, 255), 10),
                ((90, 40, 100), 15),
            ]
        )

        result = self.analyzer._dominant_palette_color(crop)
        diagnostics = result["diagnostics"]
        evidence = diagnostics["color_evidence"]
        decision = diagnostics["decision"]

        self.assertEqual("red", result["value"])
        self.assertEqual("red", decision["accepted_color_label"])
        self.assertEqual(0.5, result["quality_score"])
        self.assertEqual(0.5, decision["quality_score"])
        self.assertTrue(evidence["sample_analyzed"])
        self.assertEqual("total_sample_pixels", evidence["candidate_fraction_denominator"])
        self.assertEqual("total_sample_pixels", evidence["valid_fraction_denominator"])
        self.assertEqual(100, evidence["total_sample_pixels"])
        self.assertEqual(85, evidence["valid_color_pixel_count"])
        self.assertEqual(15, evidence["excluded_pixel_count"])
        self.assertEqual(100, evidence["analyzed_sample_pixel_count"])
        self.assertEqual(0, evidence["invalid_sample_pixel_count"])
        self.assertEqual(0.85, evidence["valid_pixel_fraction"])
        self.assertEqual(
            [
                {"color": "red", "pixel_count": 50, "fraction_of_sample": 0.5, "fraction_of_valid_color_pixels": 0.5882},
                {"color": "blue", "pixel_count": 25, "fraction_of_sample": 0.25, "fraction_of_valid_color_pixels": 0.2941},
                {"color": "green", "pixel_count": 10, "fraction_of_sample": 0.1, "fraction_of_valid_color_pixels": 0.1176},
            ],
            evidence["top_candidates"],
        )

    def test_evidence_is_retained_for_rejected_ambiguous_samples(self):
        crop = self.synthetic_crop(
            [
                ((0, 255, 255), 35),
                ((110, 255, 255), 30),
                ((60, 255, 255), 20),
                ((90, 40, 100), 15),
            ]
        )

        result = self.analyzer._dominant_palette_color(crop)
        diagnostics = result["diagnostics"]
        evidence = diagnostics["color_evidence"]
        decision = diagnostics["decision"]

        self.assertEqual("unknown", result["value"])
        self.assertEqual("insufficient_evidence", result["reason"])
        self.assertTrue(evidence["sample_analyzed"])
        self.assertEqual(100, evidence["total_sample_pixels"])
        self.assertEqual(85, evidence["valid_color_pixel_count"])
        self.assertEqual(15, evidence["excluded_pixel_count"])
        self.assertEqual("red", evidence["top_candidates"][0]["color"])
        self.assertIsNone(decision["accepted_color_label"])
        self.assertFalse(decision["accepted"])
        self.assertEqual("insufficient_evidence", decision["rejection_reason"])
        self.assertTrue(decision["rejection_conditions"]["dominant_share_below_minimum"])
        self.assertTrue(decision["rejection_conditions"]["winner_gap_below_minimum"])

    def test_no_sample_cases_have_unavailable_evidence_without_fabricated_counts(self):
        frame = self.analyzer.numpy.zeros((50, 50, 3), dtype=self.analyzer.numpy.uint8)

        attributes = self.analyzer.analyze_frame(
            frame,
            [Detection("person", 0.9, (5.0, 5.0, 20.0, 40.0), local_track_id=1)],
        )[0]
        attribute = attributes[UPPER_CLOTHING_COLOR_ATTRIBUTE]
        evidence = attribute["diagnostics"]["color_evidence"]

        self.assertEqual("unknown", attribute["value"])
        self.assertIsNone(attribute["quality_score"])
        self.assertEqual("crop_too_small", attribute["reason"])
        self.assertFalse(evidence["sample_analyzed"])
        self.assertEqual("crop_too_small", evidence["unavailable_reason"])
        self.assertNotIn("total_sample_pixels", evidence)
        self.assertNotIn("valid_color_pixel_count", evidence)
        self.assertNotIn("excluded_pixel_count", evidence)
        self.assertNotIn("top_candidates", evidence)

    def test_diagnostics_reflect_method_thresholds_and_quality_formula(self):
        crop = self.synthetic_crop(
            [
                ((0, 255, 255), 50),
                ((110, 255, 255), 25),
                ((60, 255, 255), 10),
                ((90, 40, 100), 15),
            ]
        )

        result = self.analyzer._dominant_palette_color(crop)
        diagnostics = result["diagnostics"]
        decision = diagnostics["decision"]

        self.assertEqual(UPPER_CLOTHING_COLOR_METHOD, diagnostics["method"])
        self.assertEqual(UPPER_CLOTHING_COLOR_METHOD_VERSION, diagnostics["method_version"])
        self.assertEqual(UPPER_CLOTHING_COLOR_METHOD, decision["method"])
        self.assertEqual(UPPER_CLOTHING_COLOR_METHOD_VERSION, decision["method_version"])
        self.assertEqual(
            "round(clamp((top_share_of_valid_color_pixels * 0.7) + "
            "(winner_gap_of_valid_color_pixels * 0.3), 0.0, 1.0), 2)",
            decision["quality_formula"],
        )
        self.assertEqual(0.55, decision["thresholds"]["minimum_dominant_share"])
        self.assertEqual(0.2, decision["thresholds"]["minimum_winner_gap"])
        self.assertEqual(0.35, decision["thresholds"]["minimum_valid_pixel_fraction"])
        self.assertFalse(decision["rejection_conditions"]["valid_fraction_below_minimum"])
        self.assertFalse(decision["rejection_conditions"]["dominant_share_below_minimum"])
        self.assertFalse(decision["rejection_conditions"]["winner_gap_below_minimum"])

    def test_pixel_analyzer_handles_neutral_black_and_gray_brightness_levels(self):
        crop = self.synthetic_crop(
            [
                ((0, 0, 20), 34),
                ((0, 20, 70), 33),
                ((0, 20, 130), 33),
            ]
        )

        result = self.analyzer._dominant_palette_color(crop)

        self.assertEqual("black", result["value"])
        self.assertEqual("moderate", result["quality_label"])
        self.assert_top_candidates(
            result,
            [
                ("black", 67, 0.67),
                ("gray", 33, 0.33),
                ("white", 0, 0.0),
            ],
        )

    def test_pixel_analyzer_treats_mild_blue_cast_dark_neutral_as_black(self):
        crop = self.synthetic_crop(
            [
                ((110, 60, 60), 60),
                ((0, 20, 70), 25),
                ((110, 100, 60), 15),
            ]
        )

        result = self.analyzer._dominant_palette_color(crop)

        self.assertEqual("black", result["value"])
        self.assert_top_candidates(
            result,
            [
                ("black", 85, 0.85),
                ("blue", 15, 0.15),
                ("white", 0, 0.0),
            ],
        )

    def test_pixel_analyzer_preserves_clearly_chromatic_navy(self):
        crop = self.synthetic_crop(
            [
                ((110, 120, 60), 80),
                ((0, 20, 70), 20),
            ]
        )

        result = self.analyzer._dominant_palette_color(crop)

        self.assertEqual("blue", result["value"])
        self.assertEqual("blue", result["diagnostics"]["decision"]["accepted_color_label"])
        self.assert_top_candidates(
            result,
            [
                ("blue", 80, 0.8),
                ("black", 20, 0.2),
                ("white", 0, 0.0),
            ],
        )

    def test_pixel_analyzer_preserves_dark_red_and_green_when_saturated(self):
        red_crop = self.synthetic_crop([((0, 120, 60), 80), ((0, 20, 70), 20)])
        green_crop = self.synthetic_crop([((60, 120, 60), 80), ((0, 20, 70), 20)])

        self.assertEqual("red", self.analyzer._dominant_palette_color(red_crop)["value"])
        self.assertEqual("green", self.analyzer._dominant_palette_color(green_crop)["value"])

    def test_pixel_analyzer_keeps_mixed_black_blue_ambiguous(self):
        crop = self.synthetic_crop(
            [
                ((0, 20, 70), 45),
                ((110, 120, 60), 40),
                ((0, 20, 130), 15),
            ]
        )

        result = self.analyzer._dominant_palette_color(crop)
        decision = result["diagnostics"]["decision"]

        self.assertEqual("unknown", result["value"])
        self.assertEqual("insufficient_evidence", result["reason"])
        self.assertIsNone(decision["accepted_color_label"])
        self.assertTrue(decision["rejection_conditions"]["dominant_share_below_minimum"])
        self.assertTrue(decision["rejection_conditions"]["winner_gap_below_minimum"])
        self.assert_top_candidates(
            result,
            [
                ("black", 45, 0.45),
                ("blue", 40, 0.4),
                ("gray", 15, 0.15),
            ],
        )

    def test_pixel_analyzer_reports_all_pixels_excluded_without_invented_quality(self):
        crop = self.synthetic_crop([((90, 40, 100), 100)])

        result = self.analyzer._dominant_palette_color(crop)
        evidence = result["diagnostics"]["color_evidence"]
        decision = result["diagnostics"]["decision"]

        self.assertEqual("unknown", result["value"])
        self.assertIsNone(result["quality_score"])
        self.assertEqual("unavailable", result["quality_label"])
        self.assertTrue(evidence["sample_analyzed"])
        self.assertEqual(100, evidence["total_sample_pixels"])
        self.assertEqual(0, evidence["valid_color_pixel_count"])
        self.assertEqual(100, evidence["excluded_pixel_count"])
        self.assertIsNone(decision["quality_score"])
        self.assertEqual("unavailable", decision["quality_label"])

    def test_pixel_analyzer_reports_empty_samples_as_no_sample_analyzed(self):
        empty_crop = self.analyzer.numpy.zeros((0, 0, 3), dtype=self.analyzer.numpy.uint8)

        result = self.analyzer._dominant_palette_color(empty_crop)
        evidence = result["diagnostics"]["color_evidence"]

        self.assertEqual("unknown", result["value"])
        self.assertEqual("empty_sample", result["reason"])
        self.assertIsNone(result["quality_score"])
        self.assertFalse(evidence["sample_analyzed"])
        self.assertNotIn("total_sample_pixels", evidence)

    def test_pixel_analyzer_uses_bgr_input_and_fraction_accounting(self):
        crop = self.analyzer.numpy.zeros((10, 10, 3), dtype=self.analyzer.numpy.uint8)
        crop[:6, :] = (0, 0, 255)
        crop[6:, :] = (255, 0, 0)

        result = self.analyzer._dominant_palette_color(crop)
        evidence = result["diagnostics"]["color_evidence"]

        self.assertEqual("red", result["value"])
        self.assertEqual(100, evidence["total_sample_pixels"])
        self.assertEqual(100, evidence["valid_color_pixel_count"])
        self.assert_top_candidates(
            result,
            [
                ("red", 60, 0.6),
                ("blue", 40, 0.4),
                ("black", 0, 0.0),
            ],
        )

    def test_pixel_analyzer_accepts_optional_validity_mask_for_reusable_sampling(self):
        crop = self.synthetic_crop([((0, 255, 255), 50), ((110, 255, 255), 50)])
        mask = self.analyzer.numpy.zeros((10, 10), dtype=bool)
        mask[:5, :] = True

        result = PixelColorAnalyzer(cv2=self.analyzer.cv2, numpy=self.analyzer.numpy).analyze(
            crop,
            validity_mask=mask,
        )
        evidence = result["diagnostics"]["color_evidence"]

        self.assertEqual("red", result["value"])
        self.assertEqual(100, evidence["total_sample_pixels"])
        self.assertEqual(50, evidence["analyzed_sample_pixel_count"])
        self.assertEqual(50, evidence["invalid_sample_pixel_count"])
        self.assertEqual(50, evidence["valid_color_pixel_count"])
        self.assertEqual(0, evidence["excluded_pixel_count"])
        self.assertEqual(0.5, evidence["valid_pixel_fraction"])

    def test_jsonl_and_panel_use_matching_diagnostic_values(self):
        crop = self.synthetic_crop(
            [
                ((0, 255, 255), 50),
                ((110, 255, 255), 25),
                ((60, 255, 255), 10),
                ((90, 40, 100), 15),
            ]
        )
        color_result = self.analyzer._dominant_palette_color(crop)
        adapter = ImageObservationAdapter()
        metadata = FrameMetadata(
            node_id="node-1",
            camera_id="camera-0",
            camera_session_id="session-a",
            frame_sequence=3,
            image_width_px=640,
            image_height_px=480,
            timestamp_utc="2026-11-16T12:00:00.000Z",
        )
        observation = adapter.observation_from_detection(
            Detection("person", 0.91, (10.0, 20.0, 110.0, 220.0), class_id=0, local_track_id=7),
            metadata,
            {
                UPPER_CLOTHING_COLOR_ATTRIBUTE: {
                    "value": color_result["value"],
                    "quality_label": color_result["quality_label"],
                    "quality_score": color_result["quality_score"],
                    "sampling_region_px": {"x_min": 32, "y_min": 88, "x_max": 88, "y_max": 156},
                    "reason": color_result["reason"],
                    "diagnostics": color_result["diagnostics"],
                }
            },
        )
        payload = json.loads(observation.to_json_line())
        display = object.__new__(OpenCVDisplay)

        lines = display._observation_lines([observation], appearance_debug=True)

        evidence = payload["appearance_attributes"][UPPER_CLOTHING_COLOR_ATTRIBUTE]["diagnostics"]["color_evidence"]
        self.assertEqual(0.5, evidence["top_candidates"][0]["fraction_of_sample"])
        self.assertIn("  Top colors: red 50.0%, blue 25.0%, green 10.0% of", lines)
        self.assertIn("  sample", lines)
        self.assertIn("  Quality score: 0.50; thresholds: valid>=35%,", lines)
        self.assertIn("  top>=55%, gap>=20%", lines)
        self.assertIn("  Rejection: none", lines)


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
                model_class_name="person",
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

    def test_original_model_labels_survive_observation_serialization(self):
        adapter = ImageObservationAdapter()
        metadata = FrameMetadata(
            node_id="node-1",
            camera_id="camera-0",
            camera_session_id="session-a",
            frame_sequence=3,
            image_width_px=640,
            image_height_px=480,
            timestamp_utc="2026-11-16T12:00:00.000Z",
        )
        observation = adapter.observation_from_detection(
            Detection(
                class_name="person",
                confidence=0.91,
                bbox_xyxy=(10.0, 20.0, 110.0, 220.0),
                class_id=7,
                model_class_name="people",
                local_track_id=42,
            ),
            metadata,
        )

        payload = json.loads(observation.to_json_line())

        self.assertEqual("person", payload["class_name"])
        self.assertEqual(7, payload["model_class_id"])
        self.assertEqual("people", payload["model_class_name"])

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

    def test_observation_serializes_optional_appearance_attributes(self):
        adapter = ImageObservationAdapter()
        metadata = FrameMetadata(
            node_id="node-1",
            camera_id="camera-0",
            camera_session_id="session-a",
            frame_sequence=3,
            image_width_px=640,
            image_height_px=480,
            timestamp_utc="2026-11-16T12:00:00.000Z",
        )
        observation = adapter.observation_from_detection(
            Detection(
                class_name="person",
                confidence=0.91,
                bbox_xyxy=(10.0, 20.0, 110.0, 220.0),
                class_id=0,
                local_track_id=7,
            ),
            metadata,
            {
                "upper_clothing_color": {
                    "attribute_name": "upper_clothing_color",
                    "value": "red",
                    "method": "opencv_hsv_upper_torso_heuristic",
                    "method_version": "upper-clothing-color-heuristic-v1",
                    "sampling_region_px": {"x_min": 32, "y_min": 88, "x_max": 88, "y_max": 156},
                    "quality_score": 0.73,
                    "quality_label": "moderate",
                    "quality_definition": "uncalibrated test score",
                    "reason": None,
                }
            },
        )

        payload = observation.to_dict()
        line_payload = json.loads(observation.to_json_line())

        self.assertEqual("red", payload["appearance_attributes"]["upper_clothing_color"]["value"])
        self.assertEqual(
            {"x_min": 32, "y_min": 88, "x_max": 88, "y_max": 156},
            payload["appearance_attributes"]["upper_clothing_color"]["sampling_region_px"],
        )
        self.assertEqual(payload, line_payload)

    def test_unlocated_observation_rejected_at_geographic_boundary(self):
        observation = self.observation_for()

        with self.assertRaises(UnlocatedObservationError):
            require_geographic_position(observation)

    def test_runner_preserves_original_frame_coordinates_with_independent_inference_size(self):
        capture = FakeCapture([(True, Fake4KFrame())])
        writer = FakeWriter()
        runner = CameraPerceptionRunner(
            config=RunnerConfig.create(
                node_id="node-1",
                camera_id="camera-0",
                max_frames=1,
                requested_inference_size_px=640,
                model_identifier="test-model.pt",
            ),
            capture=capture,
            detector=FakeDetector(
                [
                    Detection(
                        "person",
                        0.9,
                        (1000.0, 500.0, 1500.0, 1600.0),
                        class_id=0,
                        local_track_id=1,
                    )
                ]
            ),
            writer=writer,
        )

        result = runner.run()

        self.assertEqual("completed", result.stopped_reason)
        self.assertEqual(1, result.observations_written)
        payload = writer.observations[0].to_dict()
        self.assertEqual(3840, payload["image_width_px"])
        self.assertEqual(2160, payload["image_height_px"])
        self.assertEqual(
            {"x_min": 1000.0, "y_min": 500.0, "x_max": 1500.0, "y_max": 1600.0},
            payload["bbox_px"],
        )

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
