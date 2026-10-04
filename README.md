# dMotion

[![Checks](https://github.com/PatBaruch/dMotion/actions/workflows/checks.yml/badge.svg?branch=main)](https://github.com/PatBaruch/dMotion/actions/workflows/checks.yml)

AI-assisted changes use automated tests, security checks, package verification,
and documented pull requests. See [the reliability evidence and limits](docs/AI_RELIABILITY.md).

A local laptop experiment: show cash to the webcam, get a white box around the
banknotes, and hear an alert. Fans, stacks, and single bills count as cash.
Python, OpenCV, Ultralytics YOLO, and Pygame.

**Detection is experimental.** The pretrained model guesses from text prompts.
The training workflow below teaches a separate detector to recognize displayed
cash using examples you review. Its internal class name remains `money_spread`
to keep existing models and commands compatible.

## Start on this laptop

The project uses Python 3.11 and an isolated `.venv`. Setup installs a pinned uv
dependency manager into `.tools`; it does not change your system Python.

```sh
make setup
make prepare
make run
```

After setup you can also double-click `start.command` in Finder. The first model
preparation downloads detector and text-encoder weights (several hundred MB in
total). Subsequent runs reuse local files. `make prepare` verifies model inference
without opening your camera or playing audio.

On macOS, allow Camera access for the app running the command (Terminal or Codex)
under **System Settings → Privacy & Security → Camera**. Restart that app if needed.

## Test the basic experience

```sh
make diagnose # Real model + common objects; stand in view to check detection.
make demo    # Camera + manual box; no model needed. Press T to simulate.
make sound   # Play the configured alert once.
make doctor  # Check config, dependencies, and asset paths.
```

You can also double-click `diagnose.command` in Finder, or run
`.venv/bin/dmotion run --mode check`. Check mode uses the same pretrained detector
with prompts for person, cell phone, cup, bottle, and book at confidence `0.25`.
It draws boxes from real model results. Detecting you or one of these objects
confirms that the camera-to-model pipeline works. If check mode works but money
mode misses your cash, the money prompts or pretrained model are the likely issue.
Money mode remains the default: use `make run` or `--mode money` to return to it.

For a more sensitive cash test, close check mode with Q, then double-click
`test-money.command` in Finder (or run `make test-money`). It uses the original
model with the prompt `paper money` at confidence `0.10`, image size `640`, on CPU.
The larger image raised the uploaded euro photo's score from `0.115` to `0.170`;
live detection can still miss, and it can trigger on unrelated objects.
Hold cash clearly in view for two seconds, remove it, then show empty hands and
cards to check for false alarms. Try a fan, stack, and single bill separately.
Press T to test sound and
S to save a missed example. This shortcut uses the original text-prompt model.

The overlay shows model warm-up, whether the model is running, and the number of
processed frames. After warm-up, it distinguishes no objects, detected objects,
and results that are too slow to display. A rising frame count with no objects
means inference is running even when nothing matches the prompts. In demo mode,
the manual box checks the display and sound only.

In the camera window:

| Key | Action |
| --- | --- |
| Q / Escape | Quit |
| M | Toggle sound |
| T | Test sound; in demo mode also show a simulated box |
| S | Save a raw test photo in `data/` |

A generated beep works immediately. To use the actual "motion detected" clip,
put it at `assets/motion_detected.wav`. The clip is not bundled.

## Test a photo first

Put a photo in `data/photos/`, then run:

```sh
make image IMAGE="data/photos/spread.jpg"
# Optional preview and prompt overrides:
.venv/bin/dmotion image data/photos/spread.jpg --show --prompt "a fan of banknotes"
# Check a photo containing a person, phone, cup, bottle, or book:
.venv/bin/dmotion image data/photos/person.jpg --mode check --show
```

The annotated copy goes into `outputs/`. The command also prints detection labels,
scores, coordinates, and inference time. It does not play the alert.
You can verify model loading and inference without a camera using
`.venv/bin/dmotion prepare --mode check`.

## Tune detection

Edit `config.toml`, or copy it to `config.local.toml` for untracked experiments:

```sh
cp config.toml config.local.toml
.venv/bin/dmotion run --config config.local.toml
.venv/bin/dmotion run --confidence 0.15 --prompt "banknotes"
.venv/bin/dmotion run --device cpu --camera 1 --mute
```

- Lower confidence accepts more guesses; increase it if you get false triggers.
- `--mode trained` uses the separately calibrated `trained_confidence = 0.175`.
  Pass `--confidence` to override that value for one test run.
- `--prompt` replaces the defaults; repeat it to test multiple descriptions.
- `--device cpu` runs inference on the processor. `--device mps` uses the Apple GPU;
  `--device auto` selects the available accelerator. CPU is a compatibility option,
  not an accuracy setting.
- `--image-size 640` overrides `image_size`; use a multiple of 32. Larger images can
  expose more detail but take longer to process.
- `overlay.solid_box = true` covers detections with a filled white rectangle.
- `trigger` settings control confirmation, absence before rearming, and cooldown.
- Paths in a config file are relative to that file's folder.

The audio fires after three consecutive positive model results. Holding cash in
view does not replay it continuously. Remove it for at least one second to rearm;
a three-second cooldown also applies. One background inference worker keeps the
preview responsive and avoids queuing old frames. Results older than two seconds
are discarded; increase `camera.max_result_age_seconds` if your device is slower.

## Train it to recognize displayed cash

To create labels from recorded videos with AI, see the
[automatic video labeling workflow](docs/TRAINING.md#use-ai-to-suggest-boxes-from-videos).
`make auto-label` proposes cash boxes for review; the labeler accepts money fans,
stacks, and single bills for the broader cash-display task. AI drafts do not enter
training until accepted, and previously reviewed pictures are preserved.

Training means showing the model photos and marking where the cash is.
Downloading photos alone does not teach it anything: review each photo and draw
one tight box around each cash bundle or single bill. For a fan or stack, include
all its banknotes in one box. Mark photos without cash as negative examples,
including cards, receipts, phones, and empty hands.

```sh
make fetch    # Download the included starter image sources for review.
make collect  # Capture a short webcam session as photos, roughly one per second.
make label    # Open the local labeling page in your browser.
make dataset  # Show how many photos have been reviewed.
make train    # Build separate training/validation/test sets and train the model.
make trained  # Test the trained model with the webcam.
```

The reviewed photos are in `data/training/images/`. Their statuses and pixel
boxes are recorded in `data/training/manifest.json`. The browser labeler is the
easiest way to see the photo and its box together:

```sh
.venv/bin/dmotion label --dataset data/training
```

After `make train`, the exported copies are in `data/yolo/images/train/`,
`data/yolo/images/val/`, and `data/yolo/images/test/`; the matching normalized
YOLO label files are in the folders with the same names under `data/yolo/labels/`.

You can also double-click `collect.command`, `label.command`, `train.command`,
and `trained.command` in Finder. Collection and downloads create **unreviewed**
photos; they are excluded from training until you label them. Google examples are
a small starter set. Add your own euros, lighting, camera angles, and backgrounds
to make the model useful on your laptop.

The local dataset currently contains 97 reviewed photos: 44 positive cash photos
and 53 cash-free photos. It includes the original photos, reviewed video frames,
manually checked cash-free crops from the false-alarm screenshots, and two full
webcam examples queued for the next retraining run. Eight ambiguous or blurry
records remain excluded. Original images and review decisions are preserved
locally. This remains a small experiment that needs more varied webcam sessions.

The training setup now updates weights every batch and records actual optimizer
updates. It refuses to replace the model if there were no updates at a positive
learning rate. Each run records the exact photos, splits and learning updates.
`make trained` loads the latest completed model; `make test-money` uses the
original prompt detector.

The active hard-negative model was trained and checked on 3 October 2026 using
95 of those reviewed photos. On its validation split it matched all 13 cash
images and produced no boxes on 16 cash-free images at the calibrated threshold.
It still misses some wider or darker held-out scenes, and its box can include
the person when the cash is small. Two full webcam examples are queued for the
next retraining run. Run `make trained` and see the [saved training results](docs/TRAINING_RUN_2026-10-03.md)
for exact checks.

Record at least three separate sessions containing cash, plus sessions without
cash. The workflow keeps each session together when splitting the data,
so neighboring frames do not appear in both training and testing. Aim initially
for roughly 100–200 varied positive photos and a similar number of negatives;
that is a starting target, not an accuracy guarantee. Training explains what is
missing if there are too few reviewed groups.

`make trained` needs a successfully trained `models/money-spread.pt`. `make run`
still runs the original prompt model, and `make diagnose` checks common objects.
Follow the [step-by-step training guide](docs/TRAINING.md) for labeling, importing
videos, adding download sources, and interpreting the results.

When a shirt or another object triggers falsely, press `S` in the camera window
to save that exact frame, press `Q`, and import the newest saved photo as a
negative example:

```sh
# Organize loose uploads and camera snapshots first.
make organize-media
latest=$(ls -t data/photos/test-*.jpg | head -1)
.venv/bin/dmotion import "$latest" --dataset data/training --group shirt-false-positive
.venv/bin/dmotion label --dataset data/training
```

Choose **No cash + next** for that frame. Saving the photo alone does not change
the model; it must be reviewed as a negative and included in a later training run.

## Maintenance

```sh
make check   # Lint, formatting, and automated core tests
make format  # Apply formatting fixes
```

Dependencies are declared in `pyproject.toml` and pinned, including transitive
dependencies, in `uv.lock`. Commit both files when changing dependencies. See
[development notes](docs/DEVELOPMENT.md) for the Git workflow and dependency updates.
GitHub Actions runs the same core checks when the repository is hosted on GitHub.

Implementation agents automatically document completed features, run checks,
commit task files, push a `feature/<name>` branch, and create or update a PR into
`develop`. You do not need to request those completion steps each time. See the
[automatic Git workflow](docs/GIT_WORKFLOW.md) for the branch rules, local hooks,
and one-time Codex review setting. Merging remains a separate decision.

```text
src/dmotion/        App, configuration, detector, alert, trigger logic
tests/             Core tests that do not need a camera or downloaded models
config.toml        Shared defaults
assets/            Your alert clip
models/            Detector weights (ignored)
weights/           Text-encoder weights (ignored)
data/videos/       Original user recordings (ignored)
data/photos/       Original test photos and screenshots (ignored)
data/training/     Dataset-owned frame copies and reviewed labels (ignored)
outputs/           Annotated results (ignored)
```

Inference and training stay on the laptop. During normal detection, S saves a
single photo when you request it; the separate collection command saves a short
sequence for labeling. Weights, photos, outputs, environments, and caches are
excluded from version control.

## Troubleshooting

| Problem | Try |
| --- | --- |
| Cannot open camera | Check macOS permission, close other camera apps, try `--camera 1` |
| First launch is slow | Run `make prepare`; initial downloads and warm-up take time |
| No money detection | Run `make diagnose` and stand in view; if it detects you, test a clear close-up money photo, different prompts, or lower confidence |
| Cards or background objects trigger | Add reviewed examples of those objects with no cash boxes, then train again |
| Preview works but boxes never appear | Check model status and processed-frame count; if results are too slow, try a smaller `image_size` or increase the maximum result age |
| MPS error | Run with `--device cpu` |
| No audio | Run `make sound`, check volume/output device, verify your WAV file |
| System Python is too old | Install Python 3.11–3.13 or set `DMOTION_PYTHON=/path/to/python` for setup |

Model API reference: [Ultralytics YOLO-World](https://docs.ultralytics.com/models/yolo-world/).

For paired-video training, selectable labeling teachers and candidate comparisons,
see [Training harness](docs/TRAINING_HARNESS.md).
New uploads can be placed in `data/videos/` or `data/photos/` directly.
`make organize-media` sorts loose media from the root and `data/`, preserving
filenames, checksums and existing labels. See [file layout](docs/FILE_LAYOUT.md).
The [YOLOE fine-tuning note](docs/YOLOE_FINETUNING.md) explains how the existing
reference model differs from the current custom-training model.
