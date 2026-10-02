# Training run — 3 October 2026

The camera's trained mode now loads a model trained on the reviewed video frames
and original starter photos. Fans, stacks, and single banknotes are positive cash
examples; the internal class remains `money_spread`.

```sh
.venv/bin/dmotion run --mode trained
```

Close an already-running camera window with Q before restarting so it loads the
new weights.

## Current hard-negative retraining

The active model is `spread-20261002-224505-ee790d`. It was trained from 95
reviewed photos after adding 48 cash-free crops and four webcam regression crops
from the false detections shown during testing. The local dataset now has two
additional full webcam examples (one positive with a tight cash box and one
negative) queued for the next retraining run. The group-preserving export used
by this model contains:

| Split | Images | Positive | Negative | Cash boxes |
| --- | ---: | ---: | ---: | ---: |
| Training | 42 | 16 | 26 | 27 |
| Validation | 29 | 13 | 16 | 13 |
| Test | 24 | 14 | 10 | 23 |

The run completed 94 epochs and applied 1,033 optimizer updates at a positive
learning rate. The saved model's untouched test metrics are precision 0.286,
recall 0.522, and mAP50 0.296. These numbers are deliberately reported because
the dataset is still small and the test scenes are difficult.

Validation-only confidence calibration found that `0.175` matched all 13
positive images and produced no detections on the 16 negative images. That value
is now the trained-mode default; `--confidence` overrides it for a one-off test.
On the four supplied webcam screenshots, the new setting removed the two
cash-free face/body detections and detected the held-cash screenshot. One darker
close-up cash screenshot remains a miss, so another webcam session is still
needed before relying on the detector.

## Earlier video-only run

All 45 video frames were visually reviewed. Duplicate and oversized AI boxes
were corrected; eight blurred or ambiguous frames were excluded. The final
dataset contains 41 positive photos and two negative photos, from nine source
groups. Original images and detailed review decisions are preserved locally.

| Split | Images | Cash boxes | Groups |
| --- | ---: | ---: | ---: |
| Training | 16 | 25 | 6 |
| Validation | 13 | 13 | 1 |
| Test | 14 | 23 | 2 |

Complete videos stay in one split. The original euro photo is a training example.
Both negative photos are in training; validation and test contain no negatives,
so these results cannot establish the false-alarm rate on new scenes.

The command was:

```sh
.venv/bin/dmotion train --epochs 100 --patience 30 --device mps
```

Training stopped after 92 epochs, selected the best validation checkpoint, and
applied 367 optimizer steps at positive learning rates. Image size and batch size
were 640 and 4. Runtime: Ultralytics 8.4.171, PyTorch 2.14.1, Apple M2 GPU.

## Saved-image checks

At the camera's default confidence of 0.25 and box overlap threshold of 0.50:

| Split | Images with at least one correctly located cash detection | Matched cash boxes | Extra or unmatched predictions |
| --- | ---: | ---: | ---: |
| Training positives | 14 / 14 | 25 / 25 | 7 |
| Validation | 13 / 13 | 13 / 13 | 10 |
| Test | 9 / 14 | 11 / 23 | 11 |

The euro photo produced one cash detection with score 0.8838, verified on CPU and
the Mac GPU. The two training negatives produced no detections. These are saved
photo checks, not a live webcam accuracy measurement. Extra boxes and missed
spreads remain, especially in the wider bed-video scenes.

Model: `models/money-spread.pt`. SHA-256:
`b70068d84f569d03d51294e35fd572729fef834d298dcc2debff297ec2c8aee4`.
Run: `spread-20261002-221931-3ef49f` (run IDs use UTC).

The model's metadata is in `models/money-spread.json`. Exact labels, metrics and
checkpoints are preserved in `outputs/training/spread-20261002-221931-3ef49f/`.
Individual review decisions and saved-image predictions are in
`outputs/review-20261003/`. The previous starter model is backed up there too.
