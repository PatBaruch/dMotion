"""Download a curated list of image URLs, keeping all labels unreviewed."""

import hashlib
import io
import json
import logging
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from dmotion.dataset import Dataset

logger = logging.getLogger(__name__)
MAX_BYTES = 15 * 1024 * 1024


def validate_image_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Image URLs must start with http:// or https:// and include a host")


class ImageRedirectHandler(HTTPRedirectHandler):
    """Keep redirects within the same permitted network protocols."""

    def redirect_request(self, request, response, code, message, headers, newurl):
        validate_image_url(newurl)
        return super().redirect_request(request, response, code, message, headers, newurl)


def read_sources(path: Path, limit: int = 20) -> list[dict]:
    if not 1 <= limit <= 500:
        raise ValueError("Download limit must be between 1 and 500")
    entries = json.loads(path.read_text())
    if not isinstance(entries, list):
        raise ValueError("Sources must be a JSON list of {url, source, group} objects")
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("url"), str):
            raise ValueError("Each source must contain an image URL")
        validate_image_url(entry["url"])
        for field in ("source", "group"):
            if field in entry and not isinstance(entry[field], str):
                raise ValueError(f"Source {field} must be text")
    return entries[:limit]


def fetch_images(directory: Path, sources: Path, *, limit: int = 20) -> dict:
    from PIL import Image, ImageOps

    entries = read_sources(sources, limit)
    dataset = Dataset(directory)
    report = {"downloaded": [], "failed": []}
    with tempfile.TemporaryDirectory(prefix="dmotion-download-") as temporary:
        for index, entry in enumerate(entries):
            try:
                request = Request(
                    entry["url"], headers={"User-Agent": "dMotion/0.1 dataset collector"}
                )
                with build_opener(ImageRedirectHandler()).open(request, timeout=20) as response:
                    payload = response.read(MAX_BYTES + 1)
                if len(payload) > MAX_BYTES:
                    raise ValueError("Image exceeds the 15 MB download limit")
                with Image.open(io.BytesIO(payload)) as original:
                    if original.width * original.height > 25_000_000:
                        raise ValueError("Image exceeds the 25 megapixel limit")
                    image = ImageOps.exif_transpose(original).convert("RGB")
                    image.thumbnail((1600, 1600))
                    if min(image.size) < 120:
                        raise ValueError("Image is too small for useful training")
                    target = Path(temporary) / f"image-{index}.jpg"
                    image.save(target, quality=92)
                    group = entry.get("group") or hashlib.sha256(entry["url"].encode()).hexdigest()
                    record = dataset.add_image(
                        target,
                        group=group,
                        source=entry.get("source") or entry["url"],
                        width=image.width,
                        height=image.height,
                    )
                report["downloaded"].append({"id": record["id"], **entry})
                logger.info("Collected %s (%s)", record["id"], entry.get("source", entry["url"]))
            except Exception as exc:
                report["failed"].append({**entry, "error": str(exc)})
                logger.warning("Could not download %s: %s", entry["url"], exc)
    (directory / "download-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Collected {len(report['downloaded'])}; failed {len(report['failed'])}. Run make label.")
    return report
