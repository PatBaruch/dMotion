"""Check local label writes cannot bypass authentication or dataset validation."""

import io
import json
from types import SimpleNamespace

import pytest

from dmotion.dataset import Dataset
from dmotion.labeler import _handler


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
