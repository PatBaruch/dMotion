# Fine-tune YOLOE Small on reviewed cash recordings

`train-yoloe` creates a separate cash detector from the selected
`models/yoloe-26s-seg.pt` checkpoint. It loads that checkpoint into the matching
`yoloe-26s.yaml` detection architecture and uses `YOLOEPETrainer`. Reviewed
bounding boxes are sufficient; segmentation masks are not required.

This first experiment freezes the image features and box-regression layers,
training only the terminal classification convolutions. A runtime audit rejects
unexpected trainable parameters. Actual optimizer updates, model identity and
checkpoint SHA-256 revision, package version, and source data are retained.

This route produces a fixed cash detector. It removes visual-reference prompting;
the pretrained reference model should remain available as a separate baseline.
It does not prove that fine-tuning improves recognition on a webcam.

## Review and combine data

Finish human review in each dataset before training. No-cash frames need explicit
negative labels. Pending images cause an error rather than becoming negatives.
Skipped images remain excluded. The run snapshots reviewed records, deduplicates
identical bytes, preserves recording groups and source attribution, and rejects
conflicting labels or groups on duplicate images. Original datasets are unchanged.

```sh
.venv/bin/dmotion train-yoloe \
  --dataset data/training data/training-six-videos data/harness-training-20261004 \
  --epochs 30 --patience 8 --device auto
```

Use only the dataset paths you actually have. Prepare the selected checkpoint
before running this command. The pinned Ultralytics package also needs its
MobileCLIP text encoder when constructing the semantic `cash` class embedding;
the first run may download that component. Existing core checks do not install
vision dependencies or download weights.

## Keep evaluation sessions fixed

By default, export assigns complete groups reproducibly, using seed 42. For a
comparison across repeated runs, supply `--split-file data/yoloe-splits.json`.
This JSON object must assign every reviewed group exactly once to `train`, `val`,
or `test`. Each split must contain cash and no-cash examples. Assign all related
recordings to the same group before running; a split file cannot infer which
separate videos were recorded during the same session.

The chosen split map and immutable reviewed-data snapshot are retained in the
run folder. Training uses only the training split; validation selects the best
checkpoint. The command deliberately leaves test evaluation to a later comparison
after confidence has been selected on validation data.

## Inspect the candidate

Outputs are saved under `outputs/yoloe-training/<run>/`, or beneath the parent
directory supplied with `--output`. They include `candidate.pt`, its JSON report,
the reviewed snapshot, split map, exported labels, and training checkpoints/logs.
The command never replaces `models/money-spread.pt` or the reference checkpoint.

After calibrating a confidence threshold on validation frames, use the candidate:

```sh
.venv/bin/dmotion run --mode trained \
  --model outputs/yoloe-training/<run>/candidate.pt --confidence <validated-threshold>
```

The trained adapter accepts the single semantic `cash` class as well as legacy
`money_spread`. Compare candidate and reference baseline on the same held-out
sessions: tight-box precision/recall, false-positive frames without cash, misses,
and latency. Frame false positives do not directly measure the temporal audio
trigger. Run a fresh hardware rehearsal before choosing a demo model.

The initial route was checked against locked Ultralytics 8.4.171. See
[official YOLOE training documentation](https://docs.ultralytics.com/models/yoloe/#train-usage).
