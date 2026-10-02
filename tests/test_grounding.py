import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from dmotion.detector import Detection
from dmotion.grounding import DEFAULT_GROUNDING_MODEL, GroundingMoneyDetector


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeFrame:
    shape = (80, 120, 3)

    def __getitem__(self, index):
        assert index == (slice(None), slice(None), slice(None, None, -1))
        return SimpleNamespace(copy=lambda: "rgb pixels")


def fake_dependencies(monkeypatch, *, boxes=(), scores=(), labels=()):
    calls = []

    class Inputs(dict):
        input_ids = "cash token ids"

        def to(self, device):
            calls.append(("inputs.to", device))
            return self

    class Processor:
        def __call__(self, **kwargs):
            calls.append(("process", kwargs))
            return Inputs(pixel_values="pixels", input_ids=self.token_ids)

        token_ids = "cash token ids"

        def post_process_grounded_object_detection(
            self, outputs, input_ids, *, threshold, text_threshold, target_sizes
        ):
            calls.append(
                ("postprocess", outputs, input_ids, threshold, text_threshold, target_sizes)
            )
            return [
                {"boxes": FakeTensor(boxes), "scores": FakeTensor(scores), "text_labels": labels}
            ]

    class Model:
        def to(self, device):
            calls.append(("model.to", device))
            return self

        def eval(self):
            calls.append(("eval",))

        def __call__(self, **inputs):
            calls.append(("predict", inputs))
            return "raw outputs"

    def processor_factory(model_id, **kwargs):
        calls.append(("load_processor", model_id, kwargs))
        return Processor()

    def model_factory(model_id, **kwargs):
        calls.append(("load_model", model_id, kwargs))
        return Model()

    def image_from_array(array):
        calls.append(("image", array))
        return "PIL image"

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(
        sys.modules,
        "PIL",
        SimpleNamespace(
            Image=SimpleNamespace(
                fromarray=image_from_array,
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=processor_factory),
            AutoModelForZeroShotObjectDetection=SimpleNamespace(from_pretrained=model_factory),
        ),
    )
    return calls


def test_predict_uses_rgb_original_dimensions_and_transformers_v4_api(tmp_path, monkeypatch):
    calls = fake_dependencies(
        monkeypatch,
        boxes=[[-2.1, 3.6, 130.2, 70.4]],
        scores=[0.6],
        labels=["dollar bills"],
    )
    detector = GroundingMoneyDetector(tmp_path, image_size=640)
    assert detector.predict(FakeFrame()) == [Detection((0, 4, 120, 70), 0.6, "dollar bills")]
    assert detector.device == "cpu"
    assert calls[0] == (
        "load_processor",
        DEFAULT_GROUNDING_MODEL,
        {
            "cache_dir": str(tmp_path / ".cache/huggingface/hub"),
            "use_fast": False,
        },
    )
    assert calls[1] == (
        "load_model",
        DEFAULT_GROUNDING_MODEL,
        {
            "cache_dir": str(tmp_path / ".cache/huggingface/hub"),
            "use_safetensors": True,
            "disable_custom_kernels": True,
        },
    )
    assert ("model.to", "cpu") in calls
    assert ("eval",) in calls
    assert ("image", "rgb pixels") in calls
    assert (
        "process",
        {
            "images": "PIL image",
            "text": [["banknotes", "dollar bills", "cash money"]],
            "return_tensors": "pt",
            "size": {"shortest_edge": 640, "longest_edge": 1066},
        },
    ) in calls
    assert calls[-1] == ("postprocess", "raw outputs", "cash token ids", 0.25, 0.2, [(80, 120)])


def test_predict_ignores_low_scores_nonfinite_and_degenerate_boxes(tmp_path, monkeypatch):
    fake_dependencies(
        monkeypatch,
        boxes=[[1, 2, 20, 30], [1, 2, 20, 30], [1, 2, 1, 30], [float("nan"), 2, 20, 30]],
        scores=[0.1, float("nan"), 0.8, 0.8],
        labels=["cash"] * 4,
    )
    assert GroundingMoneyDetector(tmp_path).predict(FakeFrame()) == []


def test_predict_with_no_detections(tmp_path, monkeypatch):
    fake_dependencies(monkeypatch)
    assert GroundingMoneyDetector(tmp_path).predict(FakeFrame()) == []


def test_low_box_threshold_cannot_return_blank_labels_that_abort_draft_saving(
    tmp_path, monkeypatch
):
    fake_dependencies(
        monkeypatch,
        boxes=[[1, 2, 20, 30], [30, 2, 50, 30], [60, 2, 80, 30]],
        scores=[0.15, 0.15, 0.4],
        labels=["", "  ", " banknotes "],
    )
    detector = GroundingMoneyDetector(tmp_path, confidence=0.1, text_threshold=0.2)
    assert detector.predict(FakeFrame()) == [Detection((60, 2, 80, 30), 0.4, "banknotes")]


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [(True, True, "cuda:0"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_device_is_resolved_for_model_and_inputs(
    tmp_path, monkeypatch, cuda_available, mps_available, expected
):
    calls = fake_dependencies(monkeypatch)
    fake_torch = sys.modules["torch"]
    fake_torch.cuda = SimpleNamespace(is_available=lambda: cuda_available)
    fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps_available))
    detector = GroundingMoneyDetector(tmp_path, device="auto")
    detector.predict(FakeFrame())
    assert detector.device == expected
    assert ("model.to", expected) in calls
    assert ("inputs.to", expected) in calls


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence": -0.1},
        {"text_threshold": 1.1},
        {"image_size": 0},
        {"prompts": ()},
    ],
)
def test_invalid_options_fail_before_imports_or_downloads(tmp_path, kwargs):
    with pytest.raises(ValueError):
        GroundingMoneyDetector(tmp_path, **kwargs)
