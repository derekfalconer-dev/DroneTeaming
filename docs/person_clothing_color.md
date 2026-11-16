# Person upper-clothing color heuristic

The standalone camera pipeline emits an optional `appearance_attributes.upper_clothing_color`
object for person detections. This attribute is produced per observation only;
it is not temporally smoothed, not associated across identities, and a local
track ID must not turn an uncertain color into a fact.

## Method

`upper-clothing-color-heuristic-v1` uses a lightweight OpenCV/NumPy HSV palette
heuristic on the original, unannotated frame pixels. For person detections, it
clips the detection box to the image and samples a fixed central upper-body
fraction. Crops are resized before analysis to bound processing cost.

The documented palette handles red hue wraparound and separates neutral colors
using saturation and brightness:

- black
- white
- gray
- brown
- red
- orange
- yellow
- green
- cyan
- blue
- purple
- pink

The `quality_score` is an explicitly uncalibrated heuristic derived from the
dominant palette share and separation from the next palette color. It is not a
probability, not identity evidence, and is intentionally separate from
`detection_confidence`.

## Unknown results

The pipeline preserves uncertainty instead of forcing a color. Unknown results
include a `reason`, such as:

- `crop_too_small`
- `insufficient_torso_visibility`
- `insufficient_evidence`
- `unsupported_frame_type`

A person bounding box does not reliably identify the torso. Close-ups,
face-only views, partial bodies, overhead/aerial angles, background
contamination, shadows, lighting, camera white balance, reflective materials,
and mixed clothing colors can all make this heuristic unreliable.

## Manual test

Run the standalone camera pipeline with display enabled and observe:

1. A person wearing several solid upper-clothing colors, including red.
2. A person wearing neutral clothing such as black, white, or gray.
3. A person wearing mixed or patterned upper clothing.
4. A face-only close-up, which should remain unknown rather than claiming hair,
   skin, or background as clothing.
5. A person moving farther from the camera until the crop is too small and the
   attribute becomes unknown.

Use `--appearance-debug` to overlay the heuristic sampling region. The live
panel should show the same clothing-color value and quality label that are
serialized to JSONL.
