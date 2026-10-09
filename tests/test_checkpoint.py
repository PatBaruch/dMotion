import hashlib
from pathlib import Path

import pytest

from dmotion.checkpoint import checkpoint_snapshot


@pytest.mark.parametrize("failure", [False, True])
def test_snapshot_has_exact_digest_survives_original_changes_and_is_cleaned(tmp_path, failure):
    source = tmp_path / "cash.pt"
    original = b"original model bytes" * 100000
    source.write_bytes(original)
    captured = None
    try:
        with checkpoint_snapshot(source, tmp_path / "cache") as (captured, digest):
            assert captured != source and captured.name == source.name
            replacement = tmp_path / "replacement.pt"
            replacement.write_bytes(b"new selected model")
            replacement.replace(source)
            source.write_bytes(b"in-place update")
            assert captured.read_bytes() == original
            assert digest == hashlib.sha256(captured.read_bytes()).hexdigest()
            if failure:
                raise RuntimeError("loader failed")
    except RuntimeError:
        assert failure
    assert captured is not None and not captured.exists()
    assert list((tmp_path / "cache").iterdir()) == []


def test_missing_source_does_not_leave_a_partial_snapshot(tmp_path):
    cache = tmp_path / "cache"
    with pytest.raises(FileNotFoundError):
        with checkpoint_snapshot(Path(tmp_path / "missing.pt"), cache):
            pytest.fail("Missing checkpoint was accepted")
    assert list(cache.iterdir()) == []
