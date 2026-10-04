# Local media layout

Provided videos belong in `data/videos/`, and provided photos/screenshots belong
in `data/photos/`. Keep filenames intact so recording provenance still matches.
Drop new media into these folders directly, or put it in the repository root and
run the organizer. The camera's S key still writes loose test photos in `data/`;
the same organizer moves those into `data/photos/`.

```text
data/
  videos/                       Original recordings
  photos/                       Original test photos and screenshots
  reference.json                Cash-reference configuration
  training/                     Original reviewed dataset
    images/                     Dataset-owned copies of frames
    manifest.json               Labels, status, recording groups and provenance
  harness-training-20261004/     Separate paired-video experiment dataset
  yolo/                         Exported images and normalized training labels
models/                         Downloaded and active detector checkpoints
weights/                        Cached text-encoder weights
assets/                         Alert sound
outputs/                        Reports, candidate models and annotated previews
  media-imports/                File-move/checksum journals
```

## Organize uploads

From the repository root, preview first:

```sh
.venv/bin/python scripts/organize_media.py --dry-run
.venv/bin/python scripts/organize_media.py
```

After this feature is merged, `make organize-media` runs the same organizer.
The standalone script also accepts `--root /path/to/dMotion` to select a checkout.

Only loose MOV/MP4/M4V/AVI/MKV/WebM videos and
JPG/JPEG/PNG/WebP/HEIC/HEIF photos in the root or immediately inside `data/` are
considered, with case-insensitive extensions. It does not recurse into datasets,
outputs, model directories or assets. It preserves filenames and file contents,
checks SHA256 after each move, and records original/new paths in a local journal.
A collision stops planning before any media moves; it never overwrites another file.
File bytes do not change, and existing labels/session groups are not edited.

Moves use an exclusive hard link followed by unlinking the old name, so the source
and destination must share a filesystem. An interrupted/failed batch can be partial:
check `outputs/media-imports/` and the paths recorded there, fix the error, and rerun
for the remaining loose files. The report retains successfully completed moves.
Dry-run creates no folders or journal.

Historical experiment reports retain their original paths and hashes as recorded
at the time. Use the move journal to find those recordings at their current paths.
Dataset frame copies and the reference image stay in place, so labeling and the
reference detector remain usable after organizing the originals. Rerunning import
uses the same image hashes and existing review rules; organization never triggers
import, annotation or training automatically.

Videos, photos, labels, weights and outputs stay local and are excluded from the
feature PR. Back them up separately from Git.

## Organization performed on 4 October 2026

The local checkout was organized: six original videos and eight original photos
were moved, including `Money-visible.mov` and `money-not-visible.mov`. All 14
post-move SHA256 checks matched. Both dataset manifests, the reference definition,
active cash weights and pretrained YOLOE weights retained their checksums. The
reference-image checksum still validates, and a repeat run found no remaining moves.
The local move journal is `outputs/media-imports/20261004-134313-8cd6107e.json`,
with a companion verification report. These local media artifacts are not in Git.
