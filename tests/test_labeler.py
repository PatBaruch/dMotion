"""Check local label writes cannot bypass authentication or dataset validation."""

import io
import json
from types import SimpleNamespace

import pytest

from dmotion.dataset import Dataset
from dmotion.labeler import _handler, _review_item


@pytest.fixture
def label_server(tmp_path):
    dataset = Dataset(tmp_path / "dataset")
    photo = tmp_path / "photo.png"
    photo.write_bytes(b"test image contents")
    record = dataset.add_image(photo, group="one-session", width=100, height=60)
    server = SimpleNamespace(server_address=("127.0.0.1", 12345))
    handler = _handler(dataset, "secret-test-token")

    class MemoryConnection:
        def __init__(self, contents):
            self.input = io.BytesIO(contents)
            self.output = io.BytesIO()

        def makefile(self, mode, _buffering=-1):
            assert mode == "rb"
            return self.input

        def sendall(self, contents):
            self.output.write(contents)

    def request(method, path, value=None, *, token="secret-test-token", origin=None, host=None):
        body = json.dumps(value).encode("utf-8") if value is not None else b""
        headers = {
            "Host": "127.0.0.1:12345",
            "X-Review-Token": token,
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }
        if origin is not None:
            headers["Origin"] = origin
        if host is not None:
            headers["Host"] = host
        head = f"{method} {path} HTTP/1.0\r\n"
        head += "".join(f"{key}: {value}\r\n" for key, value in headers.items())
        connection = MemoryConnection(head.encode("ascii") + b"\r\n" + body)
        handler(connection, ("127.0.0.1", 12346), server)
        response_head, content = connection.output.getvalue().split(b"\r\n\r\n", 1)
        status = int(response_head.split(b" ")[1])
        return status, content

    return dataset, record, request


def test_review_requires_token_origin_and_host(label_server):
    dataset, record, request = label_server
    review = {"id": record["id"], "status": "negative", "boxes": []}
    assert request("POST", "/api/review", review, token="wrong")[0] == 403
    assert request("POST", "/api/review", review, origin="https://example.com")[0] == 403
    assert request("POST", "/api/review", review, host="untrusted.example")[0] == 403
    assert dataset.get_record(record["id"])["status"] == "unreviewed"


def test_review_rejects_invalid_boxes_without_changing_labels(label_server):
    dataset, record, request = label_server
    review = {"id": record["id"], "status": "positive", "boxes": [[0, 0, 101, 60]]}
    assert request("POST", "/api/review", review)[0] == 400
    review["boxes"] = []
    assert request("POST", "/api/review", review)[0] == 400
    assert dataset.get_record(record["id"])["status"] == "unreviewed"
    review["boxes"] = [[5, 8, 90, 55]]
    status, content = request("POST", "/api/review", review)
    assert status == 200
    assert json.loads(content)["boxes"] == [[5, 8, 90, 55]]
    assert dataset.get_record(record["id"])["status"] == "positive"


def test_image_endpoint_only_serves_manifest_records_with_token(label_server):
    _dataset, record, request = label_server
    assert request("GET", f"/api/image/{record['id']}")[0] == 403
    status, content = request("GET", f"/api/image/{record['id']}?token=secret-test-token")
    assert status == 200
    assert content == b"test image contents"
    assert request("GET", "/api/image/..%2F..%2Fconfig.toml?token=secret-test-token")[0] == 404
    assert request("GET", "/api/items", token="")[0] == 403


def _suggestion(boxes):
    return {
        "boxes": boxes,
        "scores": [0.8] * len(boxes),
        "labels": ["paper money"] * len(boxes),
        "model": "yolov8s-worldv2.pt",
        "prompts": ["paper money"],
        "created_at": "2026-10-02T09:00:00+00:00",
    }


def test_suggestion_prefills_only_the_unsaved_review_draft():
    record = {"status": "unreviewed", "boxes": [], "suggestion": _suggestion([[5, 8, 90, 55]])}
    item = _review_item(record)
    assert item["review_boxes"] == [[5, 8, 90, 55]]
    assert item["status"] == "unreviewed"
    assert item["boxes"] == []
    item["review_boxes"][0][0] = 10
    assert record["suggestion"]["boxes"] == [[5, 8, 90, 55]]
    assert item["suggestion"]["boxes"] == [[5, 8, 90, 55]]


@pytest.mark.parametrize(
    ("status", "boxes"),
    [("positive", [[10, 12, 70, 48]]), ("negative", []), ("excluded", [])],
)
def test_ai_suggestions_never_replace_saved_review_boxes(status, boxes):
    record = {
        "status": status,
        "boxes": boxes,
        "suggestion": _suggestion([[5, 8, 90, 55]]),
    }
    item = _review_item(record)
    assert item["review_boxes"] == boxes
    assert item["boxes"] == boxes
    assert item["status"] == status


def test_items_offer_suggestions_until_the_user_saves(label_server):
    dataset, record, request = label_server
    record["suggestion"] = _suggestion([[5, 8, 90, 55]])
    dataset.manifest_path.write_text(json.dumps({"version": 1, "records": [record]}))

    status, content = request("GET", "/api/items")
    assert status == 200
    item = json.loads(content)[0]
    assert item["review_boxes"] == [[5, 8, 90, 55]]
    assert item["boxes"] == []
    assert item["status"] == "unreviewed"
    assert dataset.summary()["reviewed"] == 0

    status, content = request(
        "POST",
        "/api/review",
        {"id": record["id"], "status": "positive", "boxes": item["review_boxes"]},
    )
    assert status == 200
    saved = json.loads(content)
    assert saved["status"] == "positive"
    assert saved["boxes"] == [[5, 8, 90, 55]]
    assert saved["review_boxes"] == saved["boxes"]
    assert dataset.summary()["reviewed"] == 1


def test_empty_ai_result_is_pending_until_explicit_negative_review(label_server):
    dataset, record, request = label_server
    record["suggestion"] = _suggestion([])
    dataset.manifest_path.write_text(json.dumps({"version": 1, "records": [record]}))

    status, content = request("GET", "/api/items")
    assert status == 200
    item = json.loads(content)[0]
    assert item["review_boxes"] == []
    assert item["suggestion"]["boxes"] == []
    assert item["status"] == "unreviewed"
    review = {"id": record["id"], "status": "positive", "boxes": []}
    assert request("POST", "/api/review", review)[0] == 400
    assert dataset.get_record(record["id"])["status"] == "unreviewed"
    review["status"] = "negative"
    status, content = request("POST", "/api/review", review)
    assert status == 200
    assert json.loads(content)["status"] == "negative"
    assert dataset.get_record(record["id"])["boxes"] == []
