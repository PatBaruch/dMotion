# dMotion

A local laptop experiment: show a fan of banknotes to the webcam, get a white box
around detected cash, and hear an alert. Python, OpenCV, YOLO-World, and Pygame.

**Detection is experimental.** The pretrained model uses text prompts and may
detect a single banknote or closed stack too. It has not been trained to distinguish
the money-spread gesture. Start with your own photos to find out whether it is useful.

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
make demo    # Camera + manual box; no model needed. Press T to simulate.
make sound   # Play the configured alert once.
make doctor  # Check config, dependencies, and asset paths.
```

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
```

The annotated copy goes into `outputs/`. The command also prints detection labels,
scores, coordinates, and inference time. It does not play the alert.

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
data/              Local testing photos (ignored)
outputs/           Annotated results (ignored)
```

Inference stays on the laptop. Webcam frames are not recorded automatically; S
saves a single photo when you request it. Weights, photos, outputs, environments,
and caches are excluded from version control.

## Troubleshooting

| Problem | Try |
| --- | --- |
| Cannot open camera | Check macOS permission, close other camera apps, try `--camera 1` |
| First launch is slow | Run `make prepare`; initial downloads and warm-up take time |
| No money detection | Use a clear close-up photo, test different prompts or lower confidence |
| Single notes trigger | This model detects cash; add a custom spread dataset for that distinction |
| Preview works but boxes never appear | Check inference time against the configured maximum result age |
| MPS error | Run with `--device cpu` |
| No audio | Run `make sound`, check volume/output device, verify your WAV file |
| System Python is too old | Install Python 3.11–3.13 or set `DMOTION_PYTHON=/path/to/python` for setup |

Model API reference: [Ultralytics YOLO-World](https://docs.ultralytics.com/models/yolo-world/).
