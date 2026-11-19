import threading
import time
import unittest

from perception.capture import CaptureFrame, LatestFrameCapture


class FakeSequentialCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.opened = False
        self.released = False
        self.read_entered = threading.Event()
        self.allow_read = threading.Event()

    def open(self):
        self.opened = True
        self.allow_read.set()

    def read(self):
        self.read_entered.set()
        self.allow_read.wait(timeout=1)
        if not self.frames:
            return CaptureFrame(
                ok=False,
                frame=None,
                capture_frame_sequence=None,
                received_monotonic_s=time.monotonic(),
                source_time_s=None,
                source_time_basis="test",
                eof=True,
            )
        return self.frames.pop(0)

    def release(self):
        self.released = True


class LatestFrameCaptureTest(unittest.TestCase):
    def test_latest_frame_replacement_has_bounded_storage_and_matching_metadata(self) -> None:
        frames = [
            CaptureFrame(True, "frame-1", 1, 10.0, 10.0, "test"),
            CaptureFrame(True, "frame-2", 2, 11.0, 11.0, "test"),
            CaptureFrame(True, "frame-3", 3, 12.0, 12.0, "test"),
            CaptureFrame(False, None, None, 13.0, None, "test", eof=True),
        ]
        capture = LatestFrameCapture(FakeSequentialCapture(frames))
        capture.open()
        time.sleep(0.05)

        selected = capture.read()

        self.assertEqual("frame-3", selected.frame)
        self.assertEqual(3, selected.capture_frame_sequence)
        self.assertEqual(12.0, selected.source_time_s)
        self.assertEqual(2, selected.overwritten_frame_count)
        capture.release()

    def test_video_like_sequential_capture_retains_all_frames_in_order_without_wrapper(self) -> None:
        capture = FakeSequentialCapture(
            [
                CaptureFrame(True, "frame-1", 1, 1.0, 1.0, "file"),
                CaptureFrame(True, "frame-2", 2, 2.0, 2.0, "file"),
                CaptureFrame(False, None, None, 3.0, None, "file", eof=True),
            ]
        )
        capture.open()

        self.assertEqual("frame-1", capture.read().frame)
        self.assertEqual("frame-2", capture.read().frame)
        self.assertTrue(capture.read().eof)

    def test_eof_errors_and_shutdown_do_not_hang(self) -> None:
        capture = LatestFrameCapture(FakeSequentialCapture([]), shutdown_timeout_s=0.2)
        capture.open()

        selected = capture.read()

        self.assertFalse(selected.ok)
        self.assertTrue(selected.eof)
        capture.release()


if __name__ == "__main__":
    unittest.main()
