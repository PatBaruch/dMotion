import json

import pytest

from dmotion.download import read_sources


def test_sources_reject_non_network_urls_before_any_download(tmp_path):
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps([{"url": "file:///private/file.jpg"}]))
    with pytest.raises(ValueError, match="http"):
        read_sources(sources)


def test_sources_validate_all_entries_and_apply_limit(tmp_path):
    sources = tmp_path / "sources.json"
    entries = [{"url": f"https://example.org/{i}.jpg", "group": "same-shoot"} for i in range(3)]
    sources.write_text(json.dumps(entries))
    assert read_sources(sources, limit=2) == entries[:2]
    entries[-1]["source"] = 123
    sources.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="source"):
        read_sources(sources, limit=1)
