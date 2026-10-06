> Historical experiment, retained for provenance. The supported workflow is YOLO26m; see [README](../../README.md).

# Fine-tuning the existing YOLOE checkpoint

The current local reference detector uses `models/yoloe-26s-seg.pt`, seeded with
a boxed cash photo from `data/reference.json`. This is pretrained YOLOE 26 Small
with a visual prompt. The original cash-training command instead initializes
YOLO26 Nano from `models/yolo26n.pt`. The paired-video harness also evaluated a
continuation of the older cash model. Neither experiment fine-tuned YOLOE.

## Recommended first YOLOE experiment

Keep the pretrained/reference YOLOE model as the baseline. Start a separate cash
detector from the same YOLOE Small pretrained weights, retain the pretrained image
features, and adapt its classification head. This is the linear-probing route
provided by Ultralytics `YOLOEPETrainer`, with explicit freezing of feature
layers and a verified trainable-parameter list. It is a proposed experiment, not
an implemented or evaluated training backend in this project.

Ultralytics 8.4.171, installed in the working camera environment, includes the
trainer and the matching `yoloe-26s.yaml` detection architecture. The official
detection fine-tuning recipe builds that architecture and loads weights from
`yoloe-26s-seg.pt`. It uses the existing bounding-box labels; no segmentation masks
are needed. Simply passing a YOLOE segmentation checkpoint to the current ordinary
YOLO training function does not select this training route.

In this installed version, the trainer removes the visual-prompt encoder and
fuses text class embeddings into a fixed classification head. This makes a cash
detector specialized to the dataset class list. It does not preserve the existing
reference-photo prompting interface. The experiment needs a dedicated training
adapter and a matching candidate-loading path; the current trained loader also
expects the legacy `money_spread` class name.

1. Audit tight cash boxes and no-cash negatives. Preserve new recordings together
   as one session and collect fresh independent validation/test sessions.
2. Build a detection-only model from matching YOLOE Small pretrained weights.
   Use the explicit YOLOE fine-tuning trainer and a semantic cash class name when
   preparing its text embedding; document compatibility with the app's class naming.
3. Save a candidate separately. Compare it against pretrained reference YOLOE on
   identical held-out frames, including missed cash, false alarms and speed.
4. Select confidence on validation only. Promote only after the candidate passes
   the checks and demonstrably improves the useful camera behavior.

The existing labels have AI-assisted and legacy review provenance. More epochs
cannot compensate for incorrect boxes, missing cash labels or limited recording
diversity. No default model has changed, and no YOLOE fine-tuning was run by the
media organization task.

## Options

| Option | What changes | Tradeoff |
| --- | --- | --- |
| Keep pretrained YOLOE and improve the reference photo | Reference conditioning only | Fastest, but not learning from the videos |
| YOLOE detection-head fine-tuning | Specialized cash classification from reviewed boxes | Recommended first training comparison |
| Broader YOLOE training | More pretrained components adapt | More compute and greater overfitting risk on this small dataset |

Official sources: [YOLOE fine-tuning](https://docs.ultralytics.com/models/yoloe/#fine-tuning-on-your-dataset)
and [YOLOEPETrainer](https://docs.ultralytics.com/reference/models/yolo/yoloe/train/#ultralytics.models.yolo.yoloe.train.YOLOEPETrainer).
The implementation was checked against the installed 8.4.171 source rather than
assuming newer documentation applies unchanged.
