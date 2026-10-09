# dMotion

[![Checks](https://github.com/PatBaruch/dMotion/actions/workflows/checks.yml/badge.svg?branch=main)](https://github.com/PatBaruch/dMotion/actions/workflows/checks.yml)

dMotion is a local Python app that recognizes visible cash through a webcam,
draws a white box around it, and plays an alert. Fans of notes, stacks, and single
bills all count as cash. It uses OpenCV for the camera, **YOLO26m** for detection,
and Pygame for audio. It can also test photos, collect recordings, help you review
labels, and train a new cash detector.

YOLO26m is the supported model family. YOLO-World prompts, YOLOE reference photos,
YOLOE training, and Grounding DINO are retired. Different trained YOLO26m
checkpoints use the same commands; choose one with `--model` or `config.local.toml`.
Detection is experimental: evaluate misses, false alarms, and speed on new sessions
before relying on a checkpoint.

**Weights and datasets are not bundled.** A fresh clone can check its camera and
audio immediately after setup. Real cash detection requires your trained cash
checkpoint or the training steps below. The generic pretrained `yolo26m.pt` is
the starting point for training; it is not a ready-made cash detector.

## Contents

- [Install and check the camera](#install-and-check-the-camera)
- [Use a trained checkpoint](#use-a-trained-checkpoint)
- [Test photos and choose another checkpoint](#test-photos-and-choose-another-checkpoint)
- [Collect and review a dataset](#collect-and-review-a-dataset)
- [Train YOLO26m and evaluate the candidate](#train-yolo26m-and-evaluate-the-candidate)
- [Configuration and tuning](#configuration-and-tuning)
- [Commands and project files](#commands-and-project-files)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

## Install and check the camera

Use Python **3.11, 3.12, or 3.13**, Git, and `make`. The shell shortcuts target
macOS and Linux with a desktop GUI. On macOS, command-line tools provide Git and
make (`xcode-select --install` if they are missing); install a supported Python
before setup. Internet access is needed for initial dependencies and pretrained
weights. CUDA requires compatible GPU drivers; Apple Silicon can use MPS.

```sh
git clone --branch main https://github.com/PatBaruch/dMotion.git
cd dMotion
make setup
make doctor
make demo
```

`make setup` installs pinned uv 0.12.22 in `.tools/` and the dependencies in
`uv.lock` in `.venv/`. It does not replace your system Python. To select a specific
interpreter, run `DMOTION_PYTHON=/path/to/python3.11 make setup`.
Run commands from the repository root.

`doctor` lists dependencies, the configured checkpoint path, and the audio asset.
A missing cash checkpoint is expected in a fresh clone. In the demo window,
press **T**: a simulated white box and beep verify camera, display, and audio.
Press **Q** to quit. Demo mode does not load a model or measure recognition.

On macOS, give Camera access to the app launching dMotion under
**System Settings → Privacy & Security → Camera**, then restart that app if needed.
`make sound` tests audio separately. A generated beep works without an audio file;
put your own clip at `assets/motion_detected.wav` to replace it.

## Use a trained checkpoint

Use a YOLO26m detection checkpoint trained on a single class named `cash` or the
legacy `money_spread`. Both names represent visible cash. A generic multiclass
checkpoint is rejected rather than sounding the alert for unrelated objects.

If someone supplies a cash checkpoint, place it at `models/cash-yolo26m.pt`:

```sh
mkdir -p models
cp /path/to/your/cash-best.pt models/cash-yolo26m.pt
make prepare
make run
```

Replace example paths with your actual file paths. `prepare` loads that checkpoint
and runs inference on a blank image without using the camera or audio. It does
not download a cash model or train one. `run` opens the webcam. After setup and
checkpoint selection you can also double-click `start.command` on macOS.

Hold cash clearly in view, remove it, and then show empty hands, cards, receipts,
and your shirt. Check a fan, stack, and single note. Boxes should follow cash, and
cash-free scenes should stay quiet. Camera controls:

| Key | Action |
| --- | --- |
| Q / Escape | Quit |
| M | Toggle audio |
| T | Test audio; simulate a box in demo mode |
| S | Save an unannotated test photo in `data/photos/` |

By default, three consecutive positive model results play the alert. Holding cash
in view does not replay it continuously. Remove it for one second to rearm; a
three-second cooldown also applies. The overlay shows processed-frame count,
inference time, device, and whether results are fresh. One background worker
keeps old frames from building up; results older than two seconds are discarded.

## Test photos and choose another checkpoint

Keep original photos in `data/photos/`. A photo test prints labels, scores,
pixel boxes, and inference time, and saves an annotated copy under `outputs/`.
It does not play the alert.

```sh
make image IMAGE="data/photos/cash.jpg"
.venv/bin/dmotion image data/photos/cash.jpg --show
```

To test a candidate or a checkpoint returned by an external training run:

```sh
.venv/bin/dmotion prepare --model /path/to/candidate.pt
.venv/bin/dmotion image data/photos/cash.jpg --model /path/to/candidate.pt --show
.venv/bin/dmotion run --model /path/to/candidate.pt --confidence 0.55
```

`0.55` is a starting setting, not a universal optimum. Select confidence for each
checkpoint on validation sessions. A larger or newer filename does not establish
better accuracy. Keep checkpoints and their SHA-256 hashes/reports together.
`--model` selects the exact file; it does not rename, retrain, or promote it.
The old `--mode money` and `--mode trained` options are equivalent compatibility
aliases and preserve your configured checkpoint.

For a persistent local selection:

```sh
cp config.toml config.local.toml
# Edit detector.model and detector.confidence in config.local.toml.
.venv/bin/dmotion run --config config.local.toml
```

Paths inside a config are relative to that config's directory. `--model` follows
the same rule. `config.local.toml` is ignored by Git. Older local configs must
remove unused `prompts`, `reference`, `reference_confidence`, and
`trained_confidence` entries and set `backend = "trained"`.

## Collect and review a dataset

The workflow is **collect/import → review → split sessions → train → validate →
test → select a checkpoint**. Images and labels stay local during these commands.
Original files are copied into the dataset; its `manifest.json` stores review
status, pixel boxes, source hashes, and session groups.

```sh
mkdir -p data/videos data/photos
make collect
.venv/bin/dmotion collect --seconds 20 --interval 1 --kind negative
.venv/bin/dmotion import data/videos/session-one.mov --interval 1
.venv/bin/dmotion import data/photos --group independent-photo-session
make label
make dataset
```

Capture cash at different distances and angles, including fans, stacks, and
single bills. Include empty hands, shirts, cards, phones, receipts, and cash
moving out of frame. `--kind positive/negative` describes a recording's intent;
all captured frames still start **unreviewed**.

The labeler opens a local browser page at `127.0.0.1`:

| Control | Review decision |
| --- | --- |
| Save cash + next | Save one tight box per visible cash bundle or separate bill |
| No cash + next | Explicitly save a negative with no boxes |
| Skip + next | Exclude a blurry, ambiguous, or irrelevant frame |
| Clear boxes / Undo box | Correct boxes before saving |
| Previous / Next / Next unreviewed | Navigate and check progress |
| Finish labeling | Finish the session and stop the local server |

Keep boxes around the notes rather than the person or shirt. Save the decision
before moving on. An empty unreviewed frame is never automatically a negative.
Revisit a previously saved frame to correct it.

If you already have a cash-trained YOLO26m checkpoint, it can suggest boxes:

```sh
.venv/bin/dmotion auto-label --model /path/to/cash-checkpoint.pt --confidence 0.2
make label
```

Suggestions remain drafts, including frames with no detections. Review missed
cash as well as incorrect boxes. Existing reviewed records and pending drafts
are preserved. Rerunning after interruption rebuilds the complete report and
contact sheets from saved drafts without repeating their inference. Reports and
contact sheets go to `outputs/video-autolabel/`. Open the sheet paths listed in
`report.json`; each published set has its own folder under `contact-sheets/`.
`predictions.json` saves draft progress; `report.json` describes the last published set.
Rebuilds keep the previous set intact until the new report is saved, then remove
obsolete generated pages after partial human review.
After the final image is reviewed, rerun `auto-label` to publish zero pending
records and remove the obsolete sheets. An empty rebuild needs no model or vision dependencies.
Rebuilds reconcile reviews saved during inference before publishing their audit.
While sheets are being rebuilt, a new review save can wait until publication finishes.
No separate labeling model or text encoder is required. Training refuses pending
AI suggestions until you explicitly choose cash, no cash, or skip.

Video import keeps each recording together. **Related videos from one physical
session must also stay in the same split.** The manifest's `group` is the source
of truth; inspect and align related recording groups before export. The video
import command does not accept `--group`. Independent sessions need distinct
groups, so adjacent frames cannot appear in training and evaluation.

`make fetch` optionally downloads the starter list in
`examples/money-spread-sources.json`; downloads still need review. Webcam sessions
matching your intended use are more useful than many nearly identical frames.

## Train YOLO26m and evaluate the candidate

Review at least **three independent groups containing cash**, with cash-free
examples spread across those sessions. Export needs separate training,
validation, and test groups. Check `make dataset` first. Small datasets can check
the workflow but do not establish useful accuracy.

```sh
make build-dataset
.venv/bin/dmotion train --epochs 30 --patience 10 --batch 2 --device auto
```

Training initializes from `models/yolo26m.pt`, downloading the generic pretrained
YOLO26m checkpoint if necessary. It fine-tunes for the reviewed cash class; it
does not resume your currently running training job. A run updates weights only
from the training split; validation chooses the best checkpoint. `batch` also
sets gradient accumulation's nominal batch size so small datasets receive actual
optimizer updates. A run with no updates at a positive learning rate is rejected.

Candidates are saved in a unique `outputs/yolo26m-training/cash-.../` folder:

```text
weights/best.pt           Best checkpoint selected using validation
candidate.pt             Separate candidate for inspection
candidate.json           Model identity, checkpoint hashes and run settings
training-info.json       Same run report
dataset-provenance.json  Reviewed records and group assignments
results.csv / args.yaml  Ultralytics training evidence
```

`make train` never overwrites the live `models/cash-yolo26m.pt`. It deliberately
leaves the test split unevaluated. Choose confidence on validation frames first,
then freeze it and assess the candidate once on held-out test sessions. For a
repeatable model-level check after freezing the threshold:

```sh
.venv/bin/yolo detect val model=/path/to/candidate.pt data=data/yolo/dataset.yaml split=test conf=0.55 imgsz=640 device=cpu
.venv/bin/dmotion run --model /path/to/candidate.pt --confidence 0.55
```

Replace `0.55` with your selected value. Inspect tight-box precision/recall,
cash-free false-positive frames, missed cash, and inference time. Ultralytics mAP
alone does not measure the repeated-frame audio trigger. Rehearse new webcam
conditions too. Keep test sessions out of retraining; once used to guide changes,
they are no longer a blind test. Preserve exports and reports before rebuilding.

When you choose a candidate for normal use, set `detector.model` to its path in
`config.local.toml` and retain its JSON report. Alternatively copy the selected
candidate and report to `models/cash-yolo26m.pt` and `models/cash-yolo26m.json`.
Use a separate output folder for each external/Colab training run, download its
checkpoint/report, and select it with the same `--model` commands. The CLI here
trains locally; it does not create or manage a cloud training job.

## Configuration and tuning

Command-line overrides apply for one invocation:

```sh
.venv/bin/dmotion run --confidence 0.65 --device cpu --camera 1 --mute
.venv/bin/dmotion image data/photos/cash.jpg --image-size 640 --device mps
```

| Config setting | Purpose |
| --- | --- |
| `detector.model` | Exact cash checkpoint file |
| `detector.confidence` | Minimum score; lower values can increase misses caught and false alarms |
| `detector.image_size` | Detail versus processing time; use a multiple of 32 |
| `detector.device` | `auto`, `cpu`, `mps` for Apple GPU, or `cuda:0` |
| `camera.index`, `width`, `height`, `mirror` | Camera and preview settings |
| `camera.max_result_age_seconds` | Discard stale inference rather than display old boxes |
| `trigger.consecutive_hits`, `reset_seconds`, `cooldown_seconds` | Alert confirmation and rearming |
| `audio.enabled`, `file`, `volume` | Sound settings, with generated-beep fallback |
| `overlay.solid_box` | Filled white rectangle instead of outline |

False alarms need reviewed examples and evaluation, not just more epochs.
Save a problematic frame with **S**, then import that file, review it, and include
its physical recording group in a later training run. Retain fresh sessions for
an honest comparison.

## Commands and project files

| Command | Purpose |
| --- | --- |
| `make setup`, `make doctor` | Install locked dependencies / inspect setup |
| `make demo`, `make sound` | Camera/display demo / audio test |
| `make prepare`, `make run`, `make image IMAGE=...` | Warm up, camera detection, photo detection |
| `make collect`, `dmotion import`, `make fetch` | Capture, import, or download examples |
| `make auto-label`, `make label`, `make dataset` | YOLO26m drafts, human review, counts |
| `make build-dataset`, `make train` | Export grouped labels / train a separate candidate |
| `make organize-media ARGS='--dry-run'` | Preview sorting loose media |
| `make organize-media` | Move loose media with a checksum journal |
| `make check`, `make audit`, `make package-check` | Core tests/security, dependency audit, package smoke test |

Make targets for inference, collection, training and labeling accept `ARGS`, for
example `make run ARGS='--config config.local.toml'`. The exact options are in
`.venv/bin/dmotion --help` and `.venv/bin/dmotion COMMAND --help`.

```text
src/dmotion/          Runtime, data review and YOLO26m training code
tests/               Tests requiring no camera, audio device or model download
scripts/             Setup, checks, publication and media tools
scripts/macos/       Finder shortcuts for collection, review and training
docs/                Developer notes, file map and reliability requirements
docs/archive/        Historical model experiments, clearly marked inactive
examples/            Optional download source list
config.toml          Shared YOLO26m defaults
config.local.toml    Your checkpoint selection (ignored)
assets/              Optional alert audio
models/              Selected cash checkpoint and generic YOLO26m initialization
data/videos/         Original recordings
data/photos/         Original photos and camera snapshots
data/training/       Dataset-owned image copies and review manifest
data/yolo/           Exported images, labels and group report
outputs/             Candidates, training runs, comparisons and annotated previews
```

Keep datasets, checkpoints, reports and source provenance backed up separately;
Git excludes them. Media organization preserves contents and labels. Old run
folders may still contain datasets used by current experiments: inspect their
references before moving them. Retired source remains available through Git
history; historical notes are in `docs/archive/`.

## Troubleshooting

| Problem | Try |
| --- | --- |
| Python too old/new | Use Python 3.11–3.13; set `DMOTION_PYTHON` for setup |
| No cash checkpoint | Supply `--model`, select it in local config, or collect/review/train; use demo first |
| Unexpected config fields | Migrate old prompt/reference config as described above |
| Checkpoint has other classes | Use your single-class cash-trained YOLO26m, not generic COCO weights |
| Camera cannot open | Check permission, close other camera apps, try `--camera 1` |
| MPS/device error | Retry inference or training with `--device cpu` |
| Training runs out of memory | Use `--batch 1`; close other GPU-heavy apps |
| Too few groups/pending labels | Review cash/no-cash/skip decisions and add independent sessions |
| Cash missed / clothing triggers | Review tight boxes and domain-matched negatives; calibrate on validation |
| Boxes absent with frame count rising | Inspect scores, detail, timing, and stale-result warnings |
| No audio | `make sound`, T, output volume/device, and audio file path |
| First training launch slow | Pretrained download and warm-up need time; later runs reuse the cache |

## Development

For core development without heavy vision packages, install
[uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
uv sync --locked --python 3.11
make setup-workflow
make check
```

If setup has already installed uv locally, use `.tools/bin/uv` instead of `uv`.
Core-only sync removes optional vision packages from that environment; use
`make setup` before a camera/model check. Keep dependency changes in both
`pyproject.toml` and `uv.lock`. Core tests cover software behavior; they do not
prove model accuracy or hardware readiness.

The call path is `cli.py → config.py → app.py → detector.py → monitor.py → trigger.py`
with `audio.py` playing the alert. Training is `cli.py → dataset.py → training.py`;
labeling uses `collect.py`, `autolabel.py`, and `labeler.py`.
See the [developer file map](docs/FILE_LAYOUT.md), [development guide](docs/DEVELOPMENT.md),
[Git workflow](docs/GIT_WORKFLOW.md), and [reliability gates](docs/AI_RELIABILITY.md).
Model loading and training capture a private checkpoint snapshot and record the
SHA-256 of those exact bytes, so replacing the source checkpoint during a run
cannot misidentify its weights. Snapshots are cleaned up after loading or training;
training keeps its snapshot available until the run finishes.

The pipeline runs tests, security scans, package checks, and PR policy. It does not
request or require AI code review. Disable the separate repository automatic-review
toggle in [Codex settings](https://chatgpt.com/codex/settings/code-review) to prevent
account-triggered reviews from consuming tokens. Existing review findings still
need resolution before merging.

Features use PRs into `develop`; `main` holds released code. Model/default changes
need an explicit user choice. No publication license has been selected; review
dependency and data terms before distribution.
