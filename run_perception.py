#!/usr/bin/env python3
"""Run standalone real-camera YOLO perception and write JSONL image observations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from perception.capture import CameraConfig, LatestFrameCapture, OpenCVCameraCapture, resolve_source_mode
from perception.contact_lifecycle import ContactConfirmationConfig
from perception.detector import DEFAULT_SAR_CLASS_NAMES, LABEL_PROFILE_NAMES, DetectionClassFilterError
from perception.display import OpenCVDisplay
from perception.exceptions import CameraUnavailableError, MissingDependencyError, ModelLoadError, PerceptionError
from perception.runner import CameraPerceptionRunner, RunnerConfig
from perception.writer import JsonlContactEventWriter, JsonlObservationWriter


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def parse_class_names(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names:
        raise ValueError("--classes must include at least one comma-separated model class name.")

    return names


def parse_positive_float(value: str) -> float:
    try:
        parsed_value = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a positive duration in seconds.") from error

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive duration in seconds.")

    return parsed_value


def parse_positive_int(value: str) -> int:
    try:
        parsed_value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--imgsz must be a positive integer.") from error

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("--imgsz must be a positive integer.")

    return parsed_value


def _camera_names_from_system_profiler(payload: dict[str, object]) -> list[str]:
    names: list[str] = []

    def collect(value: object) -> None:
        if isinstance(value, dict):
            name = value.get("_name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())

            for nested_value in value.values():
                collect(nested_value)
        elif isinstance(value, list):
            for nested_value in value:
                collect(nested_value)

    collect(payload.get("SPCameraDataType", []))

    unique_names: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in seen:
            unique_names.append(name)
            seen.add(name)

    return unique_names


def discover_reliable_camera_names() -> list[str]:
    if sys.platform != "darwin":
        return []

    try:
        result = subprocess.run(
            ["system_profiler", "-json", "SPCameraDataType"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    if result.returncode != 0:
        return []

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    if not isinstance(payload, dict):
        return []

    return _camera_names_from_system_profiler(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone perception: OpenCV capture + Ultralytics YOLO tracking + JSONL image observations.",
        epilog=(
            "Temporal contact confirmation is experimental. Defaults confirm after 3 detection-hit images "
            "within a 5-image window, trading a short delay for reduced one-frame false contacts. "
            "Live latest-frame mode keeps only the newest pending image; file replay mode is sequential and deterministic."
        ),
    )
    parser.add_argument("--source", default="0", help="OpenCV camera source index/path/URL. OpenCV index ordering is backend-dependent; index 0 is not guaranteed to be the built-in camera.")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model path/name. Avoid machine-specific TensorRT engines for portable runs.")
    parser.add_argument("--output", default="observations.jsonl", help="JSONL output path for raw image observations.")
    parser.add_argument("--contact-output", default=None, help="Optional JSONL output path for temporal contact lifecycle events. Raw observation JSONL is unchanged.")
    parser.add_argument("--node-id", default="camera-node-local")
    parser.add_argument("--camera-id", default="camera-0")
    parser.add_argument("--camera-session-id", default=None, help="Optional explicit session ID; defaults to a generated ID scoped to this camera run.")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--label-profile", choices=LABEL_PROFILE_NAMES, default="coco", help="Class-label profile used to normalize model labels before filtering and serialization.")
    class_filter_group = parser.add_mutually_exclusive_group()
    class_filter_group.add_argument("--classes", default=None, help="Comma-separated model class names to keep. Names are resolved against the loaded model; numeric class IDs are not accepted.")
    class_filter_group.add_argument("--all-classes", action="store_true", help="Disable SAR class filtering for debugging and emit every model class.")
    parser.add_argument("--imgsz", type=parse_positive_int, default=640, help="Requested Ultralytics inference image size. This is independent of capture resolution and display resizing.")
    parser.add_argument("--device", default=None, help="Optional Ultralytics device string. On Intel macOS, CPU is the portable default.")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--no-display", action="store_true", help="Disable live OpenCV display.")
    parser.add_argument("--appearance-debug", action="store_true", help="Overlay heuristic clothing-color sampling regions and show clothing-color diagnostic evidence in the live display.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional bounded run for smoke tests.")
    parser.add_argument("--source-mode", choices=("auto", "live", "file"), default="auto", help="Source timing/playback mode. Auto treats camera indices and streams as live and existing local paths as sequential files.")
    parser.add_argument("--confirmation-hits", type=parse_positive_int, default=3, help="Experimental contact confirmation hits required within the confirmation window.")
    parser.add_argument("--confirmation-window", type=parse_positive_int, default=5, help="Experimental processed-image window size for contact confirmation.")
    parser.add_argument("--confirmation-max-seconds", type=parse_positive_float, default=2.0, help="Experimental maximum elapsed source time allowed before an unconfirmed tentative contact expires.")
    parser.add_argument("--coasting-seconds", type=parse_positive_float, default=1.0, help="Experimental elapsed source time to mark a missed confirmed contact as coasting before lost/expired.")
    parser.add_argument("--expiration-seconds", type=parse_positive_float, default=5.0, help="Experimental elapsed source time after last observation before a contact expires.")
    parser.add_argument("--contact-update-seconds", type=parse_positive_float, default=1.0, help="Experimental rate limit for repeated confirmed-contact update events.")
    parser.add_argument("--list-cameras", action="store_true", help="List reliably discoverable camera names without loading the YOLO model, then exit.")
    args = parser.parse_args()

    if args.confirmation_hits > args.confirmation_window:
        parser.error("--confirmation-hits must be less than or equal to --confirmation-window.")

    if args.coasting_seconds > args.expiration_seconds:
        parser.error("--coasting-seconds must be less than or equal to --expiration-seconds.")

    if args.classes is not None:
        try:
            parse_class_names(args.classes)
        except ValueError as error:
            parser.error(str(error))

    return args


def main() -> int:
    args = parse_args()

    if args.list_cameras:
        camera_names = discover_reliable_camera_names()
        if camera_names:
            for camera_name in camera_names:
                print(camera_name)
        else:
            print("No camera names reliably discoverable on this system.")
        return 0

    from perception.detector import UltralyticsYoloTracker

    class_names = (
        None
        if args.all_classes
        else parse_class_names(args.classes)
        if args.classes is not None
        else DEFAULT_SAR_CLASS_NAMES
    )

    try:
        source = parse_source(args.source)
        source_mode = resolve_source_mode(source, args.source_mode)
        base_capture = OpenCVCameraCapture(
            CameraConfig(
                source=source,
                width_px=args.width,
                height_px=args.height,
                source_mode=source_mode,
            ),
        )
        capture = LatestFrameCapture(base_capture) if source_mode == "live" else base_capture
        detector = UltralyticsYoloTracker(
            args.model,
            confidence_threshold=args.confidence,
            device=args.device,
            class_names=class_names,
            strict_class_names=args.classes is not None,
            inference_image_size=args.imgsz,
            label_profile=args.label_profile,
        )
        print(f"Model: {args.model}")
        print(f"Label profile: {args.label_profile}")
        print(
            f"Requested inference size: imgsz={args.imgsz}px "
            "(Ultralytics may resize/pad internally; original image coordinates are preserved)."
        )
        print("Original frame dimensions: measured per frame and shown in the live panel.")
        print(f"Supported active classes: {detector.class_filter.startup_label}")
        if detector.class_filter.unsupported_names:
            print(
                "Unsupported requested SAR categories for this model: "
                f"{', '.join(detector.class_filter.unsupported_names)}. "
                "Filtering cannot add classes that are absent from the loaded model."
            )
        writer = JsonlObservationWriter(args.output)
        contact_writer = JsonlContactEventWriter(args.contact_output) if args.contact_output else None
        display = None if args.no_display else OpenCVDisplay()
        runner = CameraPerceptionRunner(
            config=RunnerConfig.create(
                node_id=args.node_id,
                camera_id=args.camera_id,
                camera_session_id=args.camera_session_id,
                max_frames=args.max_frames,
                source_label=args.source,
                appearance_debug=args.appearance_debug,
                class_filter_label=detector.class_filter.panel_label,
                label_profile=args.label_profile,
                active_class_label=detector.class_filter.panel_label,
                requested_inference_size_px=args.imgsz,
                model_identifier=args.model,
                source_mode=source_mode,
            ),
            contact_config=ContactConfirmationConfig(
                confirmation_hits=args.confirmation_hits,
                confirmation_window_size=args.confirmation_window,
                confirmation_max_elapsed_s=args.confirmation_max_seconds,
                coasting_max_elapsed_s=args.coasting_seconds,
                expiration_elapsed_s=args.expiration_seconds,
                update_interval_s=args.contact_update_seconds,
            ),
            capture=capture,
            detector=detector,
            writer=writer,
            display=display,
            contact_writer=contact_writer,
        )
        result = runner.run()
    except MissingDependencyError as error:
        print(f"Missing optional perception dependency: {error}", file=sys.stderr)
        return 2
    except CameraUnavailableError as error:
        print(f"Camera unavailable: {error}", file=sys.stderr)
        return 3
    except DetectionClassFilterError as error:
        print(f"Class filter error: {error}", file=sys.stderr)
        return 4
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
