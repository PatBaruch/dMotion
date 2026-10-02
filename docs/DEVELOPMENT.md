# Development

## Git

This folder has its own repository. The initial prototype is committed on `main`.
Create a short-lived branch for each change:

```sh
git switch -c codex/improve-detection
make check
git add src tests config.toml
git commit -m "Improve detection confirmation"
```

Use focused commits that explain the change. Keep model binaries, photos, caches,
and personal settings out of commits. The repository has no remote configured;
hosting it on GitHub can be added later with your chosen repository and visibility.

## Dependencies

`make setup` installs exactly the versions in `uv.lock`. Avoid installing untracked
packages directly into `.venv`. To add or update a dependency intentionally:

```sh
.tools/bin/uv add --optional vision package-name --cache-dir .cache/uv
.tools/bin/uv lock --upgrade-package ultralytics --cache-dir .cache/uv
make setup
make check
make prepare
git add pyproject.toml uv.lock
```

CLIP is pinned to a specific commit in the Ultralytics-maintained repository so
prompt encoding does not silently install an untracked package at runtime.
Ultralytics automatic package installation is disabled by the app.

For core-only development or CI:

```sh
.tools/bin/uv sync --locked --cache-dir .cache/uv
make check
```

This omits the `vision` extra. Run `make setup` again before testing the camera.

## Architecture

- `config.py`: typed TOML settings and validation.
- `detector.py`: model loading, device selection, box/score conversion.
- `trigger.py`: confirmation, absence reset, and cooldown state machine.
- `audio.py`: preloaded sound with a generated-beep fallback.
- `app.py`: camera lifecycle, one in-flight inference, overlay, photo output.
- `cli.py`: commands, overrides, diagnostics, readable errors.

Core tests cover missed/brief detections, held cash, rearming, cooldown, invalid
settings, path resolution, and the audio fallback format. Before changing the
detector, manually test a real spread, empty hands, a single bill, cards/paper,
and removal/reappearance. Keep different recording sessions separate if you later
train a custom model.

## Next improvement

If prompt-based detection is insufficient, label one box around each entire cash
fan in a diverse dataset, include negative examples, and fine-tune a detector.
Replace the model adapter while preserving the camera, audio, and trigger logic.

## Licensing

No publication license has been selected for this project. Ultralytics code and
weights have their own terms; consult its [licensing page](https://www.ultralytics.com/license)
when choosing how to distribute a future product. Other dependencies retain their
respective licenses.
