#!/usr/bin/env python3
"""Run standalone real-camera YOLO perception and write JSONL image observations."""

from __future__ import annotations

import argparse
import sys

from perception.capture import CameraConfig, OpenCVCameraCapture
from perception.detector import UltralyticsYoloTracker
from perception.display import OpenCVDisplay
from perception.exceptions import CameraUnavailableError, MissingDependencyError, ModelLoadError, PerceptionError
from perception.runner import CameraPerceptionRunner, RunnerConfig
from perception.writer import JsonlObservationWriter


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone real-camera perception: OpenCV capture + Ultralytics YOLO tracking + JSONL image observations.",
    )
    parser.add_argument("--source", default="0", help="OpenCV camera source. Use 0 for the built-in Mac camera or another index/path/URL for USB/UVC sources.")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model path/name. Avoid machine-specific TensorRT engines for portable runs.")
    parser.add_argument("--output", default="observations.jsonl", help="JSONL output path for image observations.")
    parser.add_argument("--node-id", default="camera-node-local")
    parser.add_argument("--camera-id", default="camera-0")
    parser.add_argument("--camera-session-id", default=None, help="Optional explicit session ID; defaults to a generated ID scoped to this camera run.")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="Optional Ultralytics device string. On Intel macOS, CPU is the portable default.")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--no-display", action="store_true", help="Disable live OpenCV display.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional bounded run for smoke tests.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        capture = OpenCVCameraCapture(
            CameraConfig(
                source=parse_source(args.source),
                width_px=args.width,
                height_px=args.height,
            ),
        )
        detector = UltralyticsYoloTracker(
            args.model,
            confidence_threshold=args.confidence,
            device=args.device,
        )
        writer = JsonlObservationWriter(args.output)
        display = None if args.no_display else OpenCVDisplay()
        runner = CameraPerceptionRunner(
            config=RunnerConfig.create(
                node_id=args.node_id,
                camera_id=args.camera_id,
                camera_session_id=args.camera_session_id,
                max_frames=args.max_frames,
            ),
            capture=capture,
            detector=detector,
            writer=writer,
            display=display,
        )
        result = runner.run()
    except MissingDependencyError as error:
        print(f"Missing optional perception dependency: {error}", file=sys.stderr)
        return 2
    except CameraUnavailableError as error:
        print(f"Camera unavailable: {error}", file=sys.stderr)
        return 3
    except ModelLoadError as error:
        print(f"Model loading/inference failure: {error}", file=sys.stderr)
        return 4
    except PerceptionError as error:
        print(f"Perception failure: {error}", file=sys.stderr)
        return 5

    print(
        "Perception stopped: "
        f"{result.stopped_reason}; frames={result.frames_processed}; observations={result.observations_written}; "
        f"output={args.output}"
    )
    return 0 if result.stopped_reason in {"completed", "user_requested", "interrupted"} else 6


if __name__ == "__main__":
    raise SystemExit(main())
