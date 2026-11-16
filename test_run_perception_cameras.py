import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import run_perception


class RunPerceptionCameraListingTest(unittest.TestCase):
    def test_camera_names_from_system_profiler_reports_names_only(self) -> None:
        payload = {
            "SPCameraDataType": [
                {
                    "_name": "FaceTime HD Camera",
                    "spcamera_model-id": "example",
                },
                {
                    "_name": "USB Camera",
                    "spcamera_unique-id": "example-2",
                },
            ],
        }

        self.assertEqual(
            run_perception._camera_names_from_system_profiler(payload),
            ["FaceTime HD Camera", "USB Camera"],
        )

    def test_list_cameras_exits_before_regular_runner_setup(self) -> None:
        with (
            patch("run_perception.discover_reliable_camera_names", return_value=["Camera A"]),
            patch("sys.argv", ["run_perception.py", "--list-cameras"]),
            redirect_stdout(io.StringIO()) as stdout,
        ):
            result = run_perception.main()

        self.assertEqual(result, 0)
        self.assertEqual(stdout.getvalue(), "Camera A\n")


if __name__ == "__main__":
    unittest.main()
