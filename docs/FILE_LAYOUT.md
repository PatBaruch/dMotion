# Project file map

Start with [README](../README.md) for setup, commands and the supported YOLO26m workflow.

| Files | Developer responsibility |
| --- | --- |
| `src/dmotion/cli.py`, `config.py` | Command parsing, overrides, typed settings and path validation |
| `app.py`, `detector.py` | Webcam/photo lifecycle and loading a cash checkpoint |
| `monitor.py`, `trigger.py`, `audio.py` | Fresh inference, confirmation/rearming, audio |
| `collect.py`, `download.py` | Original-media import, capture and optional downloads |
| `dataset.py`, `labeler.py`, `autolabel.py` | Review truth, session splits, human labels and YOLO26m drafts |
| `training.py` | YOLO26m initialization, optimizer evidence and separate candidate output |
| `tests/` | Core behavior, integrity, model-selection and workflow checks |
| `scripts/` | Setup, media organization, security/package checks and Git workflow |
| `scripts/macos/`, `start.command` | Finder entry points; commands run from the repository root |
| `docs/archive/` | Historical experiments kept out of active instructions |

## Local artifacts

| Path | Contents |
| --- | --- |
| `models/cash-yolo26m.pt` and `.json` | A user-selected cash checkpoint and its report |
| `models/yolo26m.pt` | Generic pretrained initialization for training |
| `models/archive/` | Retired local weights/text encoders with restoration journals |
| `data/videos/`, `data/photos/` | Original recordings, photos and snapshots |
| `data/training/images/`, `manifest.json` | Dataset-owned images, explicit review decisions and groups |
| `data/yolo/` | Grouped train/val/test export; preserve it before rebuilding |
| `outputs/yolo26m-training/` | Unique training runs, candidate weights and reports |
| `outputs/video-autolabel/` | Draft predictions, current `report.json`, and its immutable sheet set under `contact-sheets/` |
| `outputs/media-imports/` | Checksum/path-move journals |
| `outputs/` | Other evaluation evidence, external/Colab candidates and photo previews |

These artifacts are ignored by Git. Back them up separately. Historical run
paths are intentionally stable when newer experiments depend on their datasets
or evaluation helpers. A historical model name does not mean its whole folder
is disposable. Never move a running experiment's inputs/checkpoints.

## Organize loose media

```sh
make organize-media ARGS='--dry-run'
make organize-media
```

The organizer considers only loose video/photo files in the root and immediately
inside `data/`. It keeps names and bytes intact, refuses collisions/symlinked
destinations, verifies SHA-256, and records moves under `outputs/media-imports/`.
It never traverses datasets or outputs, changes reviewed labels, or trains.
Moves require the same filesystem. If interrupted, inspect its journal for a
partial batch and rerun for the remaining loose files. Dry-run writes nothing.
