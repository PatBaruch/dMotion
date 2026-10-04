# Development

## Git

The remote is [PatBaruch/dMotion](https://github.com/PatBaruch/dMotion).
`main` holds released code; `develop` integrates completed features. New work uses
`feature/<name>` from `origin/develop`, preferably in its own worktree. Releases
use `release/<version>`; urgent release fixes use `hotfix/<name>`.

Agents automatically update documentation, run checks, commit task files, push
the branch, and create or update a pull request. The full process, one-time setup,
and remaining review settings are in [Git workflow](GIT_WORKFLOW.md). No separate
instruction to run tests or publish a completed feature is needed.

Keep model binaries, photos, caches, and personal settings out of commits. Existing
changes belonging to other tasks must be preserved, not included in feature PRs.

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

## Diagnostic testing

Use `make diagnose`, `diagnose.command`, or `.venv/bin/dmotion run --mode check`
to check real inference with common objects. Check mode changes prompts to person,
cell phone, cup, bottle, and book, and sets confidence to `0.25`; it uses the same
detector and camera path as default money mode. `image` and `prepare` also accept
`--mode check` for tests without live camera input.

The overlay exposes warm-up, model-running status, processed-frame count, and
no-objects, objects, or too-slow results. A detected person verifies the pipeline,
while unsuccessful money detection can still reflect the pretrained model's
limitations. `T` checks sound and `S` saves a raw photo for repeatable testing.
Demo mode uses a simulated box and does not verify model recognition. Automated
checks do not establish camera hardware behavior or real-money accuracy; record
those results separately when testing on the laptop.

## Custom training workflow

The [training guide](TRAINING.md) covers collection, local annotation, imports,
downloads, dataset building, and custom detector training. `--mode trained` selects
the `money_spread` model while preserving the camera, audio, and trigger logic.

Review status and session groups are part of the dataset, not incidental UI state.
Never infer a negative label from an unreviewed image. Keep recording sessions
together across train, validation, and test splits; splitting adjacent frames
randomly can produce misleading results. Validate bounding boxes before export.

Datasets and trained weights are excluded from Git. Keep repeatable training
settings and code in version control, and back up the source dataset and useful
training runs separately. Core checks must continue to run without camera
hardware, downloaded weights, or the optional vision dependencies.

## Licensing

No publication license has been selected for this project. Ultralytics code and
weights have their own terms; consult its [licensing page](https://www.ultralytics.com/license)
when choosing how to distribute a future product. Other dependencies retain their
respective licenses.

## Training harness

`dmotion.harness` freezes reviewed manifests, applies explicit group splits, trains
through `train_model(candidate_directory=..., evaluate_test=False)`, calibrates on
validation predictions and evaluates separate test groups. Candidates never
replace the active camera weights. `score_predictions` uses one-to-one IoU matching
and separately counts false alarm frames. `compare_label_drafts` compares stored
teacher suggestions only after review, on identical frame IDs and hashes.

`dmotion.teachers.ReferenceTeacher` is an optional YOLOE labeling adapter with an
explicit hashed reference image; Grounding DINO Tiny/Base use the existing lazy
Transformers adapter. Core tests fake optional dependencies and require no model
downloads, webcam, GPU or audio. Real model/video experiments are separate evidence.
Usage and reproducibility requirements are in `docs/TRAINING_HARNESS.md`.
