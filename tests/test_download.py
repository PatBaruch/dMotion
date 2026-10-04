import json

import pytest

from dmotion.download import ImageRedirectHandler, read_sources, validate_image_url


@pytest.mark.parametrize("url", ["file:///private/a.jpg", "ftp://example.org/a", "https:///a"])
def test_invalid_url_schemes_and_missing_hosts_are_rejected(url):
    with pytest.raises(ValueError, match="http"):
        validate_image_url(url)


def test_redirect_cannot_switch_to_a_local_file_or_ftp():
    from urllib.request import Request

    handler = ImageRedirectHandler()
    for url in ["file:///private/a.jpg", "ftp://example.org/a.jpg"]:
        with pytest.raises(ValueError, match="http"):
            handler.redirect_request(Request("https://example.org/a"), None, 302, "", {}, url)
    redirect = handler.redirect_request(
        Request("https://example.org/a"), None, 302, "", {}, "https://example.org/b"
    )
    assert redirect.full_url == "https://example.org/b"


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
