"""Media moves preserve bytes, labels and history, and never overwrite files."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

source = Path(__file__).resolve().parents[1] / "scripts" / "organize_media.py"
spec = importlib.util.spec_from_file_location("organize_media", source)
media = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = media
spec.loader.exec_module(media)


def test_organizes_loose_media_preserving_content_and_dataset_truth(tmp_path):
    (tmp_path / "data/training/images").mkdir(parents=True)
    (tmp_path / "Money-visible.mov").write_bytes(b"original video")
    (tmp_path / "data/PHOTO.JPG").write_bytes(b"original photo")
    dataset = tmp_path / "data/training/images/labeled.jpg"
    dataset.write_bytes(b"dataset frame")
    manifest = tmp_path / "data/training/manifest.json"
    manifest.write_text('{"records": []}')
    reference = tmp_path / "data/reference.json"
    reference.write_text('{"image": "training/images/labeled.jpg"}')
    report = media.organize_media(tmp_path)
    assert report["stage"] == "complete"
    assert (tmp_path / "data/videos/Money-visible.mov").read_bytes() == b"original video"
    assert (tmp_path / "data/photos/PHOTO.JPG").read_bytes() == b"original photo"
    assert not (tmp_path / "Money-visible.mov").exists()
    assert dataset.read_bytes() == b"dataset frame"
    assert manifest.read_text() == '{"records": []}'
    assert reference.read_text() == '{"image": "training/images/labeled.jpg"}'
    saved = json.loads(Path(report["journal"]).read_text())
    assert saved == report
    assert all(item["status"] == "moved" for item in saved["moves"])
    for item in saved["moves"]:
        assert media.digest(tmp_path / item["to"]) == item["sha256"]
    assert media.organize_media(tmp_path)["moves"] == []


def test_preview_creates_no_directories_or_journal(tmp_path):
    (tmp_path / "new.mov").write_bytes(b"video")
    report = media.organize_media(tmp_path, dry_run=True)
    assert report["stage"] == "preview"
    assert report["moves"][0]["to"] == "data/videos/new.mov"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["new.mov"]


def test_collision_preflights_every_move_before_changing_files(tmp_path):
    (tmp_path / "data/videos").mkdir(parents=True)
    (tmp_path / "a.mov").write_bytes(b"first")
    (tmp_path / "z.mov").write_bytes(b"second")
    (tmp_path / "data/videos/z.mov").write_bytes(b"existing")
    with pytest.raises(ValueError, match="already exists"):
        media.organize_media(tmp_path)
    assert (tmp_path / "a.mov").read_bytes() == b"first"
    assert (tmp_path / "z.mov").read_bytes() == b"second"
    assert (tmp_path / "data/videos/z.mov").read_bytes() == b"existing"
    assert not (tmp_path / "data/videos/a.mov").exists()


def test_duplicate_names_in_root_and_legacy_data_are_not_merged(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "a.mov").write_bytes(b"one")
    (tmp_path / "data/a.mov").write_bytes(b"two")
    with pytest.raises(ValueError, match="already exists"):
        media.organize_media(tmp_path)
    assert (tmp_path / "a.mov").exists()
    assert (tmp_path / "data/a.mov").exists()


@pytest.mark.parametrize("folder", ["data", "data/videos"])
def test_symlinked_destination_is_rejected(tmp_path, folder):
    project, external = tmp_path / "project", tmp_path / "external"
    project.mkdir()
    external.mkdir()
    (project / "a.mov").write_bytes(b"video")
    target = project / folder
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked"):
        media.organize_media(project)
    assert (project / "a.mov").exists()
    assert list(external.iterdir()) == []


def test_symlinked_sources_and_non_media_files_are_left_alone(tmp_path):
    target = tmp_path / "untouched.txt"
    target.write_bytes(b"source")
    (tmp_path / "alias.mov").symlink_to(target)
    (tmp_path / "config.toml").write_text("[camera]")
    assert media.organize_media(tmp_path)["moves"] == []
    assert (tmp_path / "alias.mov").is_symlink()
    assert target.read_bytes() == b"source"


def test_racing_destination_does_not_overwrite_or_remove_source(tmp_path, monkeypatch):
    (tmp_path / "a.mov").write_bytes(b"source")
    original_link = media.os.link

    def racing_link(source, target, **kwargs):
        target.write_bytes(b"another writer")
        return original_link(source, target, **kwargs)

    monkeypatch.setattr(media.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        media.organize_media(tmp_path)
    assert (tmp_path / "a.mov").read_bytes() == b"source"
    assert (tmp_path / "data/videos/a.mov").read_bytes() == b"another writer"


def test_source_change_after_planning_preserves_source(tmp_path, monkeypatch):
    source = tmp_path / "a.mov"
    source.write_bytes(b"original")
    original_plan = media.plan_moves

    def changing_plan(root):
        result = original_plan(root)
        source.write_bytes(b"changed")
        return result

    monkeypatch.setattr(media, "plan_moves", changing_plan)
    with pytest.raises(ValueError, match="Source changed"):
        media.organize_media(tmp_path)
    assert source.read_bytes() == b"changed"
    assert not (tmp_path / "data/videos/a.mov").exists()


def test_partial_failure_journals_completed_moves_and_keeps_remaining_source(tmp_path, monkeypatch):
    (tmp_path / "a.mov").write_bytes(b"first")
    (tmp_path / "b.mov").write_bytes(b"second")
    original_link = media.os.link

    def fail_second(source, target, **kwargs):
        if source.name == "b.mov":
            raise OSError("simulated different filesystem")
        return original_link(source, target, **kwargs)

    monkeypatch.setattr(media.os, "link", fail_second)
    with pytest.raises(OSError, match="different filesystem"):
        media.organize_media(tmp_path)
    report = json.loads(next((tmp_path / "outputs/media-imports").glob("*.json")).read_text())
    assert report["stage"] == "partial"
    assert [move["status"] for move in report["moves"]] == ["moved", "pending"]
    assert (tmp_path / "data/videos/a.mov").read_bytes() == b"first"
    assert (tmp_path / "b.mov").read_bytes() == b"second"
    assert not (tmp_path / "data/videos/b.mov").exists()


def test_cli_preview_uses_explicit_root(tmp_path, monkeypatch, capsys):
    (tmp_path / "a.mov").write_bytes(b"video")
    monkeypatch.setattr(sys, "argv", ["organize_media.py", "--root", str(tmp_path), "--dry-run"])
    assert media.main() == 0
    assert json.loads(capsys.readouterr().out)["stage"] == "preview"
    assert (tmp_path / "a.mov").exists()
