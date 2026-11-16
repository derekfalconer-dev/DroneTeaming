# Standalone camera perception increment

This increment adds a real-camera path that is intentionally separate from the
simulation/geographic observation pipeline.

## Scope

Implemented:

- OpenCV camera capture from a configurable source.
- Built-in Mac camera or USB/UVC camera indices supported by OpenCV.
- Ultralytics YOLO detection with persistent local tracking.
- Live display with class, confidence, bounding box, and local track ID.
- JSONL image observations.
- Explicitly unknown geographic position.
- Clean shutdown on `q`, Escape, Ctrl-C, or capture failure.

Deferred:

- Appearance attributes.
- Embeddings.
- Cross-camera identity association.
- Shared-world identity.
- Simulator/geospatial integration.

## Optional dependency setup

Perception dependencies are optional and separate from simulation dependencies:

```bash
python -m pip install -r requirements-perception.txt
```

For Intel macOS, CPU inference is the portable default. The optional dependency
file pins NumPy below 2 because the currently selected PyTorch Intel macOS
wheel set is built against the NumPy 1.x ABI. Do not assume TensorRT engines
transfer between machines.

## Launch

Built-in Mac camera:

```bash
python run_perception.py --source 0 --model yolov8n.pt --output observations.jsonl
```

USB/UVC camera:

```bash
python run_perception.py --source 1 --model yolov8n.pt --output observations-usb.jsonl
```

Disable display for bounded smoke runs:

```bash
python run_perception.py --source 0 --model yolov8n.pt --output observations.jsonl --no-display --max-frames 50
```

## JSONL output

Each line is one image observation with schema version
`image-observation-v1`. The payload includes:

- node ID
- camera ID
- camera session ID
- frame sequence
- UTC timestamp
- timestamp provenance
- image width/height
- pixel bounding box
- class name
- detection confidence
- nullable local track ID
- track scope of `camera_session`
- explicitly unknown geographic position

Example geographic position field:

```json
{"status":"unknown","latitude":null,"longitude":null,"altitude_m":null}
```

The adapter deliberately does not produce simulator observations because those
interfaces require geographic position.

## macOS camera permissions

If the camera cannot be opened:

1. Open System Settings.
2. Go to Privacy & Security.
3. Open Camera.
4. Allow the terminal or IDE running Python.
5. Fully quit and reopen that terminal or IDE after granting permission.
6. Close other apps that may be using the camera.
7. Re-run the command.

## Local hardware acceptance test

1. Confirm branch and baseline before edits.
2. Install optional perception dependencies.
3. Run `python run_perception.py --source 0 --model yolov8n.pt --output observations.jsonl`.
4. Verify a live window appears.
5. Verify detections show class, confidence, box, and track ID when tracking is available.
6. Press `q` and confirm the process exits cleanly.
7. Confirm `observations.jsonl` contains JSONL records with unknown geographic position and session-scoped track IDs.
