# Evaluate YOLO26m cash checkpoints

Use this after local or Colab training, before choosing a different camera checkpoint.
Evaluation compares existing cash checkpoints; it does not train or promote them.
Install the locked vision environment with `make setup` for actual inference.
Core scoring and regression tests work without vision packages or model downloads.

## Use the existing reviewed export

Keep the dataset and export from the candidate's training run. Both evaluation
splits need reviewed cash and negative frames, and each physical recording session
must belong to exactly one split. Related videos must share a manifest group.

```sh
.venv/bin/dmotion evaluate --dataset data/training \
  --splits data/yolo/export-report.json \
  --baseline /path/to/current-cash.pt --candidate /path/to/candidate.pt \
  --device cpu
```

Paths are relative to the configuration's directory; absolute paths can reference
preserved local artifacts from another checkout. `--baseline` defaults to the
configured detector checkpoint. `--candidate` and `--splits` are required.
The config/default image size is overridden by `--image-size` (default 640), and
`--device` defaults to `auto`; use `mps` for the Apple GPU or `cpu` for the processor.
Use a cash-trained YOLO26m checkpoint, with the `cash` or legacy `money_spread` class.
The existing detector checks class names; this command does not certify the
checkpoint's architecture or earlier training data.

An export report is accepted only when its reviewed record IDs, hashes, groups,
statuses and boxes match the dataset. Exported frames from one group cannot span
multiple splits. If labels have changed since export, review the change and prepare
a new planned experiment instead of quietly reusing an incompatible export.

## Supply an explicit session plan

An explicit JSON map is useful for preserved snapshots or fresh sessions:

```json
{
  "session-training": "train",
  "session-validation": "val",
  "session-test": "test"
}
```

Replace the example keys with the actual manifest `group` values. Assign every
reviewed group exactly once; omitted/extra groups and unknown splits fail.
Validation and test each require positive and negative frames. A map describes
the caller's intended split; it cannot prove that an external/older checkpoint
never trained on those recordings. Use the original training provenance to verify
that separately. Do not reassign trained-on recordings to test.

All unreviewed frames, including empty AI suggestions, block evaluation. Explicit
`excluded` frames stay excluded. The frozen manifest retains reviewer metadata
when available, but the tool cannot establish that review was performed by a human
or that the boxes are accurate.

## Calibration and gates

The default grid is 0.05 through 0.95 in steps of 0.05. Repeat `--threshold` to
supply a grid **before** inspecting test results. Each checkpoint is scored
independently against reviewed validation boxes, with greedy matching in descending
confidence order at IoU >= 0.5. Each ground-truth box can match once. Duplicate,
oversized, or wrongly placed boxes count as false positives. IoU is configurable
with `--iou-threshold`; it measures overlap in original image pixels.

Default acceptance limits:

| Metric | Limit |
| --- | ---: |
| Box precision | >= 0.80 |
| Box recall | >= 0.50 |
| Negative frames with any detection | <= 0.05 |

The selected threshold maximizes F1 among passing validation thresholds. Ties
prefer higher recall, then a higher confidence threshold. If none pass, the report
shows the best failed operating point and test inference is skipped for both models.
Otherwise both models are tested using their own frozen validation thresholds.
The baseline can fail validation while remaining useful for comparison.

These are frame gates, not a statistical confidence guarantee. For example, zero
false detections in a handful of negative frames does not establish reliability.
Box recall differs from the fraction of cash frames found: a frame with two notes
can be counted as found while one note is missed.

Gate options are `--min-precision`, `--min-recall`, and `--max-false-alarm-rate`.
Set acceptance criteria before running the experiment; do not weaken them after
seeing a failed result. Thresholds, gate settings and IoU are retained in reports.

## Evidence and exit status

A new run goes under `outputs/evaluation/<UTC-time>-<id>/`. `--output` selects a
new directory; an existing directory is refused so previous evidence survives.
The source dataset/configuration/checkpoints are preserved. Evaluation copies only
validation/test images into a frozen dataset and temporarily snapshots both
checkpoints before either is used. Checkpoint copies are removed when the run ends.

| Artifact | Contents |
| --- | --- |
| `source-manifest.json`, `splits.json` | Original manifest and exact group plan |
| `dataset/` | Frozen reviewed validation/test images and manifest |
| `calibration.json` | Checkpoint hashes and validation selections saved before test inference |
| `predictions.json` | Validation predictions and test predictions only if test ran |
| `report.json`, `report.md` | Gate verdict, comparison, provenance, runtime versions and limits |
| `run.json` | Preparing/testing/complete/failed status; an error never becomes a completed report |

Prediction timings exclude image decoding. Validation performs one unscored warm-up
call, recorded separately; later timed calls report mean, median and p95 on the
chosen device. These measure detector wall time, not preview FPS or audio delay.

Exit 0 means a report was completed, including a failed candidate. Add
`--require-pass` for exit 2 when the candidate fails frame gates. Input/inference
errors return exit 1 through the CLI. A partial failed run is retained for diagnosis;
use a new output directory when retrying.

The default `--test-context previously-inspected` makes the existing project
benchmark's limits explicit. Use `--test-context fresh` only for new independent
sessions you kept out of training and tuning. This is a caller declaration, not a
verified guarantee. Neither verdict automatically changes the active model.
After inspecting test results, subsequent tuning needs new test sessions.

Before choosing a candidate, also test the real webcam and audio: cash presentations,
empty hands, cards/receipts, lighting/distance, latency, stale results, removing cash,
and rearming. A passed frame gate does not substitute for those checks.
