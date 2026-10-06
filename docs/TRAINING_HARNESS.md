# Training and comparison harness

The harness keeps related recordings together, freezes the reviewed dataset and
split plan, trains a separate candidate, calibrates its confidence on validation
frames, and measures it on separate test recordings. It never replaces
`models/money-spread.pt`. Core checks require no vision packages or camera.

## 1. Import a paired recording session

After installing the existing vision/labeling dependencies, sample both videos
into the same group. Use a new session name for each separate recording session:

```sh
.venv/bin/dmotion harness-import data/money-visible.mov data/money-not-visible.mov \
  --session my-recording-session --interval 3 --dataset data/experiment
```

Every extracted frame starts unreviewed. Names are hints, not labels: a positive
video can contain moments with no visible cash. The same session name prevents
paired poses/backgrounds from leaking between train, validation and test.
Unchanged frames are deduplicated. Previously imported frames are not silently
regrouped; use a fresh dataset or put all related existing groups in the same split.

Three seconds is a starting sampling interval. Lower it for brief gestures or
small datasets, but adjacent frames are highly correlated. More frames from one
recording do not provide more independent test sessions.

## 2. Choose a labeling teacher and review its suggestions

Grounding DINO Tiny is the existing default. Base is now selectable, with greater
memory/download requirements; it is not guaranteed to improve cash labeling.
Both IDs are paired with immutable revisions in `dmotion.teachers`: Tiny
`a2bb814dd30d776dcf7e30523b00659f4f141c71`, Base
`12bdfa3120f3e7ec7b434d90674b3396eccf88eb`. The same pin is passed to the processor
and model through the existing Grounding DINO adapter, and retained in reports:


```sh
.venv/bin/dmotion auto-label --dataset data/experiment --engine grounding \
  --grounding-model tiny --confidence 0.2 --prompt banknotes \
  --output outputs/experiment/dino-tiny

# Optional comparison; requires downloading the Base model on first use.
.venv/bin/dmotion auto-label --dataset data/experiment --engine grounding \
  --grounding-model base --confidence 0.2 --prompt banknotes \
  --output outputs/experiment/dino-base
```

YOLOE can use an explicitly boxed cash reference. Obtain the official
`yoloe-26s-seg.pt` weights and pass their local path:

```sh
.venv/bin/dmotion auto-label --dataset data/experiment --engine yoloe \
  --model models/yoloe-26s-seg.pt --reference data/reference.json \
  --confidence 0.2 --output outputs/experiment/yoloe
```

The reference JSON contains `version: 1`, `image` (relative to that JSON),
`box: [x1, y1, x2, y2]` in original pixels, and the image's `sha256`.
Its contents are verified before initializing YOLOE. It uses this explicit cash
image as the reference, not the first camera frame. The live camera reference
feature is separate from this offline labeling adapter.

Run multiple teachers **before reviewing** if you want to compare them. Each
output directory keeps its own `predictions.json`, contact sheets and report;
the labeler shows the most recently generated suggestions. Reviewed labels are
never replaced by automatic labeling.

```sh
.venv/bin/dmotion label --dataset data/experiment
```

Correct boxes around physical cash. Mark empty hands, shirts, phones (including
printed cases), controllers, bottles and furniture as **No cash**. Exclude
ambiguous or unusable frames. A teacher finding no boxes does not make a negative
label. Review provenance records the reviewer and time; imported older labels
may lack attribution. An agent's visual check is AI-assisted review, not an
independent human annotation audit.

After reviewing, compare the saved teacher drafts on the same frames:

```sh
.venv/bin/dmotion harness-labelers --dataset data/experiment \
  outputs/experiment/dino-tiny/predictions.json \
  outputs/experiment/yoloe/predictions.json \
  --confidence 0.2 --output outputs/experiment/teachers-compared.json
```

This rejects unreviewed frames, mismatched frame sets, altered image hashes and
incompatible provenance. Excluded frames are omitted. It measures box precision,
recall and false alarms on negative frames; teacher confidence values are not
comparable measures of accuracy. Save outputs to a new directory for each batch.

## 3. Freeze whole-session splits

Use `data/experiment/manifest.json` to identify group names. Save a JSON object
mapping **every reviewed group** to exactly one of `train`, `val`, `test`:

```json
{
  "my-recording-session": "train",
  "another-independent-session": "val",
  "third-independent-session": "test"
}
```

Each split must contain reviewed cash and negative examples. The new paired
videos are only one session: they cannot alone supply independent training,
validation and test sets. Add earlier reviewed independent recordings or record
additional sessions. Keep related videos under the same group, even if their
filenames differ. Group names and actual content still require judgment; the
harness cannot infer that different files share one session.

## 4. Train and compare a candidate

With an existing baseline checkpoint and a reviewed dataset:

```sh
.venv/bin/dmotion harness-train --dataset data/experiment \
  --splits data/experiment-splits.json --baseline models/money-spread.pt \
  --epochs 50 --patience 12 --image-size 640 --device mps \
  --output outputs/experiment-run
```

To continue learning from an existing cash checkpoint, add
`--initial-model models/money-spread.pt`. This preserves its learned weights and
records their hash in the run. The initial checkpoint is copied before training.
Its earlier data history may overlap validation/test groups: such a run is a
development comparison, not evidence of independent generalization. The report
sets `initial_model_data_overlap_unknown` and prevents `promotion_eligible` from
becoming true even if the measured performance gate passes.

Use `--device cpu` if Apple GPU support is unavailable. This continues using the
project's pretrained YOLO26 Nano detector; it does not fine-tune the labeling
teachers. A new/empty output directory is required. Any unreviewed input blocks
the run, including frames with no AI box. A private snapshot contains the images,
reviewed labels and hashes; browser edits cannot change an experiment in progress.

The default quality gate requires precision >= 0.8, recall >= 0.5 and false alarms
on <= 5% of negative frames. Override with `--min-precision`, `--min-recall` and
`--max-false-alarm-rate` when the application needs different tradeoffs. The
threshold grid is 0.1, 0.175, 0.25, 0.35, 0.5, 0.65, 0.8. Each model chooses its
threshold on validation only. If no threshold passes, the report explicitly
records failure and uses the best validation F1 for diagnostic testing. The test
set never chooses the threshold or training checkpoint. A gate failure cannot
replace the active model because this harness never promotes models automatically.

Matching requires one-to-one box overlap (IoU >= 0.5): a large box around a person
counts as a false positive even if it contains cash. Reports measure sampled
image inference; they do not measure real-time sound triggers or temporal tracking.

## Outputs and testing

`report.json` records the candidate path, baseline/candidate comparisons, threshold,
gate status and test command. `dataset/`, `splits.json`, `baseline.pt`, `model/yolo/`,
model hashes, calibration files and predictions preserve the run's evidence.
Training logs/checkpoints remain in the existing `outputs/training/` directory.
A failed/interrupted run retains `run.json` with the last stage.

Use the exact candidate path and validation-selected threshold from the report:

```sh
.venv/bin/dmotion run --mode trained \
  --model outputs/experiment-run/model/candidate.pt --confidence 0.5
```

This tests the candidate without changing the default checkpoint. Check empty
hands and objects first, then cash at several distances. A historical baseline
may already have seen the held-out recordings, so the comparison is diagnostic;
its performance is not necessarily an unbiased baseline estimate. After looking
at test results, use fresh independent recordings for future model changes.
Passing unit tests does not establish recognition accuracy. Small recording sets
and AI-assisted labels cannot establish general reliability.

Datasets, weights, outputs and recordings are local artifacts and must stay out
of Git commits. Put new media in `data/`, not the repository root.

For actual results and limitations on the new paired recordings, see
[the 4 October experiment](TRAINING_EXPERIMENT_2026-10-04.md).
