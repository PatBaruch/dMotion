"""Capture model bytes once so loading and provenance use the same checkpoint."""

import hashlib
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def checkpoint_snapshot(source: Path, cache: Path) -> Iterator[tuple[Path, str]]:
    """Copy one open file into a private temporary snapshot and hash those exact bytes.

    Atomic replacement of the original path cannot change the opened file. The
    snapshot remains available for lazy library reloads until the caller exits.
    Never use a hard link here: in-place changes would affect both paths.
    """
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="checkpoint-", dir=cache) as temporary:
        snapshot = Path(temporary) / source.name
        digest = hashlib.sha256()
        with source.open("rb") as original, snapshot.open("xb") as copied:
            for chunk in iter(lambda: original.read(1024 * 1024), b""):
                copied.write(chunk)
                digest.update(chunk)
        yield snapshot, digest.hexdigest()
