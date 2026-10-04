# Paired-video experiment — 4 October 2026

The harness ran on the supplied `Money-visible.mov` (108.5 seconds) and
`money-not-visible.mov` (90.1 seconds), with Ultralytics 8.4.171, PyTorch 2.14.1,
Python 3.11.15 and Apple M2 GPU inference/training. This is a development experiment,
not a claim that the detector is ready for reliable use.

## Data and review

Sampling every three seconds produced 37 and 31 frames respectively. All 68 were
inspected by Codex, with explicitly attributed **AI-assisted visual review**;
this is not an independent human annotation audit. The resulting new examples
were 31 cash frames (38 boxes), 34 negative frames, and three excluded ambiguous
partial-cash frames. Physical cash alone was boxed. Shirts, hands, furniture,
a phone with a printed case, a bottle, a tissue box and a controller were negatives.

The original dataset was preserved. A separate experiment dataset contained 333
reviewed images after exclusions. Both new recordings shared one session group:

| Split | Images | Cash frames | Negative frames | Groups |
| --- | ---: | ---: | ---: | ---: |
| Train | 247 | 126 | 121 | 9 |
| Validation | 30 | 13 | 17 | 1 |
| Test | 56 | 31 | 25 | 2 |

The validation recording was `money_spread_girl.mp4`. Test recordings were
`money_spread_bed_1.mp4` and `money_spread_fan_1.mp4`. All images and labels were
snapshotted with hashes before training. Older reviewed labels were reused;
review provenance is incomplete for those legacy records.

## Labeling teacher comparison

Both teachers were compared at confidence 0.2 on the same 65 reviewed new frames.
Excluded frames were omitted. One-to-one box matching required IoU >= 0.5.

| Teacher | Box precision | Box recall | Negative frames with false alarms |
| --- | ---: | ---: | ---: |
| Grounding DINO Tiny | 13.0% | 94.7% | 34/34 |
| YOLOE 26 Small with the existing cash reference | 75.0% | 15.8% | 1/34 |

DINO found cash but proposed many boxes on the shirt and background. The existing
YOLOE reference missed most cash in this different lighting/pose session. Neither
teacher's raw outputs were silently accepted as training truth. DINO Base was
implemented as a selectable option but was not downloaded or evaluated in this run.

## Training and detector comparison

The first candidate initialized from general `yolo26n.pt`: 50 epochs requested,
12-epoch patience, early stop after 22 epochs, best checkpoint at epoch 10, and
1,363 optimizer updates at a positive learning rate. It produced no detections
on validation at any tested threshold >= 0.1 and failed the quality gate.

A second candidate continued from the existing cash checkpoint: 15 epochs
requested, five-epoch patience, early stop after seven epochs, best checkpoint at
epoch two, and 433 positive-learning-rate updates. No model architecture default
was changed. Each detector's confidence threshold was selected on validation.

| Detector | Selected confidence | Test box precision | Test box recall | Test negative false alarms |
| --- | ---: | ---: | ---: | ---: |
| Existing checkpoint | 0.65 | 88.5% | 51.1% | 0/25 |
| Fresh candidate | 0.8 | 0% | 0% | 0/25 |
| Continued candidate | 0.175 | 35.5% | 48.9% | 0/25 |

Both candidates failed the gate (precision >= 80%, recall >= 50%, negative-frame
false alarm rate <= 5%, on validation and test). The existing model also did not
pass the validation gate. **The active model was preserved**, verified by its SHA256.

The historical checkpoint may have seen these recordings previously. Its continued
candidate inherits that possible overlap. The test set was inspected for both
experiments; further changes need fresh independent test recordings. These numbers
must not be presented as an unbiased comparison of generalization.

## Replay of the new session

At the same confidence 0.175, on the 65 reviewed new-session frames:

| Detector | Correctly localized cash frames | Negative frames with false alarms |
| --- | ---: | ---: |
| Existing checkpoint | 11/31 | 33/34 |
| Continued candidate | 7/31 | 1/34 |

The continuation reduced false alarms substantially on this training session,
but missed more cash. A denser replay at approximately 2.9 frames/second produced
99 detection frames out of 316 samples in the money video, and 17 out of 262 in
the video without money (6.5% false alarms). Detection counts in the positive
video are not recall measurements because not every timestamp has a reviewed box.
The replay videos are annotated locally and contain no audio.

## Evidence and how to test

Local artifacts in the original checkout:

- `data/harness-training-20261004/manifest.json`: attributed reviewed labels.
- `outputs/harness-preparation-20261004/`: teacher reports, reviewed previews,
  split plan, visual-review notes and training logs.
- `outputs/harness-run-20261004/report.json`: fresh candidate comparison.
- `outputs/harness-continued-20261004/report.json`: continuation comparison.
- `outputs/harness-continued-20261004/replay-report.json`: session replay metrics.
- `outputs/harness-continued-20261004/replay-*.mp4`: annotated replay videos.

To test the experimental continued candidate from the original project folder:

```sh
.venv/bin/dmotion run --mode trained \
  --model outputs/harness-continued-20261004/model/candidate.pt \
  --confidence 0.175 --device mps
```

Use `--device cpu` if needed. The default checkpoint remains unchanged. Check
empty hands/shirt first, then cash near and far. This candidate remains unreliable.
Review the agent-assisted labels in the existing browser labeler:

```sh
.venv/bin/dmotion label --dataset data/harness-training-20261004
```

Core verification: `make check` passed with 212 tests, without optional vision
packages or hardware. Real inference, training and saved-video replay ran separately.
Live webcam interaction and audio playback were not checked in this experiment.
Recordings, labels, weights and generated outputs are excluded from the PR.
