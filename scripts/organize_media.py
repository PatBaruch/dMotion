"""Move loose project media into data/videos and data/photos with a checksum journal."""

import argparse
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from filelock import FileLock

VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm"}
PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def plan_moves(root: Path) -> list[dict]:
    """Inspect only loose media; do not descend into datasets or generated outputs."""
    moves = []
    for directory in (root, root / "data"):
        if directory.is_symlink():
            raise ValueError(f"Refusing a symlinked media directory: {directory}")
        if not directory.exists():
            continue
        for source in sorted(directory.iterdir()):
            if source.is_symlink() or not source.is_file():
                continue
            suffix = source.suffix.lower()
            if suffix in VIDEO_SUFFIXES:
                folder = "videos"
            elif suffix in PHOTO_SUFFIXES:
                folder = "photos"
            else:
                continue
            target = root / "data" / folder / source.name
            if any(parent.is_symlink() for parent in (target.parent, root / "data")):
                raise ValueError(f"Refusing a symlinked destination: {target}")
            if target.exists() or any(
                move["to"] == str(target.relative_to(root)) for move in moves
            ):
                raise ValueError(f"Destination already exists; nothing moved: {target}")
            if target.parent.exists() and not target.parent.is_dir():
                raise ValueError(f"Destination folder is not a directory: {target.parent}")
            moves.append(
                {
                    "from": str(source.relative_to(root)),
                    "to": str(target.relative_to(root)),
                    "bytes": source.stat().st_size,
                    "sha256": digest(source),
                    "status": "pending",
                }
            )
    return moves


def write_journal(path: Path, report: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".media-")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def organize_media(root: Path, *, dry_run: bool = False) -> dict:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Project folder does not exist: {root}")
    if dry_run:
        return {"root": str(root), "stage": "preview", "moves": plan_moves(root)}
    cache = root / ".cache"
    cache.mkdir(exist_ok=True)
    with FileLock(cache / "media-organizer.lock"):
        moves = plan_moves(root)
        report = {"version": 1, "root": str(root), "stage": "complete", "moves": moves}
        if not moves:
            return report
        output = root / "outputs" / "media-imports"
        if not output.resolve().is_relative_to(root):
            raise ValueError("Media journal must stay inside the project")
        output.mkdir(parents=True, exist_ok=True)
        journal = output / f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid4().hex[:8]}.json"
        report.update(stage="moving", journal=str(journal))
        write_journal(journal, report)
        try:
            for move in moves:
                source, target = root / move["from"], root / move["to"]
                target.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive creation: never replace an existing file, even if it
                # appeared after planning. Both paths must share a filesystem.
                os.link(source, target, follow_symlinks=False)
                try:
                    if digest(target) != move["sha256"]:
                        raise ValueError(f"Source changed during organization: {source}")
                    source.unlink()
                except (OSError, ValueError):
                    target.unlink(missing_ok=True)
                    raise
                move["status"] = "moved"
                write_journal(journal, report)
        except (OSError, ValueError) as error:
            report.update(stage="partial", error=str(error))
            write_journal(journal, report)
            raise
        report["stage"] = "complete"
        write_journal(journal, report)
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dry-run", action="store_true", help="Preview without moving files")
    args = parser.parse_args()
    try:
        report = organize_media(args.root, dry_run=args.dry_run)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Media organization stopped: {error}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
