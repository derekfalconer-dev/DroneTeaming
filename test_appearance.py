import unittest

from perception.appearance import OpenCVUpperClothingColorAnalyzer
from perception.detector import Detection


try:
    import cv2
    import numpy
except ModuleNotFoundError:
    cv2 = None
    numpy = None


@unittest.skipIf(cv2 is None or numpy is None, "OpenCV/NumPy are not installed")
class UpperClothingColorHeuristicTest(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = OpenCVUpperClothingColorAnalyzer()

    def person_detection(self, bbox=(10.0, 5.0, 90.0, 150.0)) -> Detection:
        return Detection(
            class_name="person",
            confidence=0.94,
            bbox_xyxy=bbox,
            class_id=0,
            local_track_id=35,
        )

    def attribute_for(self, frame, detection: Detection | None = None) -> dict[str, object]:
        result = self.analyzer.analyze_frame(frame, [detection or self.person_detection()])[0]
        self.assertIsNotNone(result)
        return result["upper_clothing_color"]

    def solid_bgr_frame(self, bgr: tuple[int, int, int]):
        return numpy.full((160, 100, 3), bgr, dtype=numpy.uint8)

    def test_red_hue_wraparound_is_classified_as_red(self) -> None:
        hsv = numpy.full((160, 100, 3), (179, 255, 220), dtype=numpy.uint8)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        attribute = self.attribute_for(frame)

        self.assertEqual("red", attribute["value"])
        self.assertIn(attribute["quality_label"], {"moderate", "high"})
        self.assertIsNone(attribute["reason"])

    def test_neutral_gray_is_classified_without_forcing_chromatic_color(self) -> None:
        attribute = self.attribute_for(self.solid_bgr_frame((128, 128, 128)))

        self.assertEqual("gray", attribute["value"])
        self.assertIsNone(attribute["reason"])

    def test_mixed_ambiguous_crop_remains_unknown(self) -> None:
        frame = self.solid_bgr_frame((0, 0, 255))
        frame[:, 50:] = (255, 0, 0)

        attribute = self.attribute_for(frame)

        self.assertEqual("unknown", attribute["value"])
        self.assertEqual("insufficient_evidence", attribute["reason"])

    def test_tiny_person_crop_returns_unknown(self) -> None:
        attribute = self.attribute_for(
            self.solid_bgr_frame((0, 0, 255)),
            self.person_detection((10.0, 10.0, 20.0, 40.0)),
        )

        self.assertEqual("unknown", attribute["value"])
        self.assertEqual("crop_too_small", attribute["reason"])

    def test_image_boundary_clipping_reports_sampling_region_in_original_coordinates(self) -> None:
        attribute = self.attribute_for(
            self.solid_bgr_frame((0, 0, 255)),
            self.person_detection((-20.0, 0.0, 80.0, 150.0)),
        )

        self.assertEqual("red", attribute["value"])
        self.assertEqual(
            {"x_min": 18, "y_min": 51, "x_max": 62, "y_max": 102},
            attribute["sampling_region_px"],
        )

    def test_unsupported_object_class_has_no_appearance_attributes(self) -> None:
        result = self.analyzer.analyze_frame(
            self.solid_bgr_frame((0, 0, 255)),
            [Detection("dog", 0.9, (10.0, 5.0, 90.0, 150.0), class_id=16, local_track_id=1)],
        )

        self.assertIsNone(result[0])


if __name__ == "__main__":
    unittest.main()
