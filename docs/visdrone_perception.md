# VisDrone label profile for standalone camera perception

VisDrone is a dataset, not a checkpoint format. To use the standalone perception
pipeline with VisDrone labels, provide a separately acquired or trained
Ultralytics-compatible YOLO checkpoint through the existing `--model` option.

Example with a placeholder local checkpoint path:

```bash
python run_perception.py --model ./models/visdrone-yolo.pt --label-profile visdrone --classes person,car,van
```

The `visdrone` label profile reads class IDs and source names from the loaded
checkpoint metadata. It does not assume COCO class IDs. The profile normalizes
these VisDrone source labels before filtering and observation serialization:

- `pedestrian` -> `person`
- `people` -> `person`
- `motor` -> `motorcycle`

`car`, `van`, `truck`, `bus`, `bicycle`, `tricycle`, and `awning-tricycle`
remain distinct categories. The pipeline does not invent labels that are absent
from the checkpoint, such as boat, airplane, building, or drone. If requested
classes are unavailable, startup validation reports the supported normalized
classes from the checkpoint.
