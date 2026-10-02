# dMotion

A local laptop experiment: show a fan of banknotes to the webcam, get a white box
around detected cash, and hear an alert. Python, OpenCV, Ultralytics YOLO, and Pygame.

**Detection is experimental.** The pretrained model uses text prompts and may
detect a single banknote or closed stack too. It has not been trained to distinguish
the money-spread gesture. The training workflow below teaches a separate detector
what a money spread looks like using examples you review.

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
model with the prompt `paper money` at confidence `0.10` on CPU. This setting found
the fan in the uploaded euro photo; it can also trigger on unrelated objects or
single notes. Hold the fan clearly in view for two seconds, remove it, then show
empty hands and a single note to check for false alarms. Press T to test sound and
S to save a missed example. This shortcut does not use the six-photo starter model.

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

Put a photo in `data/`, then run:

```sh
make image IMAGE="data/spread.jpg"
# Optional preview and prompt overrides:
.venv/bin/dmotion image data/spread.jpg --show --prompt "a fan of banknotes"
# Check a photo containing a person, phone, cup, bottle, or book:
.venv/bin/dmotion image data/person.jpg --mode check --show
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
- `--prompt` replaces the defaults; repeat it to test multiple descriptions.
- `device = "auto"` uses CUDA, Apple MPS, or CPU in that order. Try CPU if MPS fails.
- `image_size` must be a multiple of 32. A smaller size can improve speed but miss details.
- `overlay.solid_box = true` covers detections with a filled white rectangle.
- `trigger` settings control confirmation, absence before rearming, and cooldown.
- Paths in a config file are relative to that file's folder.

The audio fires after three consecutive positive model results. Holding cash in
view does not replay it continuously. Remove it for at least one second to rearm;
a three-second cooldown also applies. One background inference worker keeps the
preview responsive and avoids queuing old frames. Results older than two seconds
are discarded; increase `camera.max_result_age_seconds` if your device is slower.

## Train it to recognize money spreads

Training means showing the model photos and marking where the money fan is.
Downloading photos alone does not teach it anything: review each photo and draw
one box around the **whole fan of banknotes**. Mark photos without a fan as
negative examples, including single notes, closed stacks, cards, and empty hands.

```sh
make fetch    # Download the included starter image sources for review.
make collect  # Capture a short webcam session as photos, roughly one per second.
make label    # Open the local labeling page in your browser.
make dataset  # Show how many photos have been reviewed.
make train    # Build separate training/validation/test sets and train the model.
make trained  # Test the trained model with the webcam.
```

You can also double-click `collect.command`, `label.command`, `train.command`,
and `trained.command` in Finder. Collection and downloads create **unreviewed**
photos; they are excluded from training until you label them. Google examples are
a small starter set. Add your own euros, lighting, camera angles, and backgrounds
to make the model useful on your laptop.

The starter dataset on this laptop has six reviewed photos: three Google/Pinterest
money spreads, your euro photo, and two photos without money. This is enough to
check that the training pipeline runs. A model trained on this tiny set is an
experiment; it still needs more varied examples and webcam testing.

Record at least three separate sessions containing money spreads, plus sessions
without them. The workflow keeps each session together when splitting the data,
so neighboring frames do not appear in both training and testing. Aim initially
for roughly 100–200 varied positive photos and a similar number of negatives;
that is a starting target, not an accuracy guarantee. Training explains what is
missing if there are too few reviewed groups.

`make trained` needs a successfully trained `models/money-spread.pt`. `make run`
still runs the original prompt model, and `make diagnose` checks common objects.
Follow the [step-by-step training guide](docs/TRAINING.md) for labeling, importing
videos, adding download sources, and interpreting the results.

## Maintenance

```sh
make check   # Lint, formatting, and automated core tests
make format  # Apply formatting fixes
```

Dependencies are declared in `pyproject.toml` and pinned, including transitive
dependencies, in `uv.lock`. Commit both files when changing dependencies. See
[development notes](docs/DEVELOPMENT.md) for the Git workflow and dependency updates.
GitHub Actions runs the same core checks when the repository is hosted on GitHub.

```text
src/dmotion/        App, configuration, detector, alert, trigger logic
tests/             Core tests that do not need a camera or downloaded models
config.toml        Shared defaults
assets/            Your alert clip
models/            Detector weights (ignored)
weights/           Text-encoder weights (ignored)
data/              Local testing photos and reviewed training data (ignored)
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
| Single notes trigger | Label single notes as negatives and train the custom spread detector |
| Preview works but boxes never appear | Check model status and processed-frame count; if results are too slow, try a smaller `image_size` or increase the maximum result age |
| MPS error | Run with `--device cpu` |
| No audio | Run `make sound`, check volume/output device, verify your WAV file |
| System Python is too old | Install Python 3.11–3.13 or set `DMOTION_PYTHON=/path/to/python` for setup |

Model API reference: [Ultralytics YOLO-World](https://docs.ultralytics.com/models/yolo-world/).
