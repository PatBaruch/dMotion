import pytest

from dmotion.config import TriggerConfig
from dmotion.detector import Detection
from dmotion.monitor import InferenceMonitor
from dmotion.trigger import DetectionTrigger


def monitor(*, consecutive_hits=3, max_age=2.0):
    trigger = DetectionTrigger(TriggerConfig(consecutive_hits=consecutive_hits))
    return InferenceMonitor(max_result_age_seconds=max_age, trigger=trigger)


def matches():
    return [Detection((10, 20, 100, 200), 0.8, "person")]


def test_no_completed_inference_is_shown_as_warming_up():
    state = monitor()
    assert state.completed == 0
    assert state.visible(0.0) == []
    assert "warming" in state.headline(0.0, checking=True, confidence=0.25).lower()


def test_expiring_overlay_does_not_interrupt_positive_inference_confirmation():
    state = monitor()
    detections = matches()

    assert not state.complete(detections, captured_at=0.0, finished_at=1.5)
    assert state.visible(1.6) == detections
    assert state.visible(2.1) == []
    assert not state.complete(detections, captured_at=1.5, finished_at=3.0)
    assert state.visible(3.6) == []
    assert state.complete(detections, captured_at=3.0, finished_at=4.5)
    assert not state.complete(detections, captured_at=4.5, finished_at=6.0)

    assert state.completed == 4
    assert state.discarded == 0
    assert state.inference_ms == pytest.approx(1500.0)


def test_slow_result_is_reported_and_cannot_fire_or_draw_a_box():
    state = monitor(consecutive_hits=1)

    assert not state.complete(matches(), captured_at=0.0, finished_at=2.1)
    assert state.completed == 1
    assert state.discarded == 1
    assert state.inference_ms == pytest.approx(2100.0)
    assert state.visible(2.1) == []
    assert "too slow" in state.headline(2.1, checking=True, confidence=0.25).lower()


def test_result_at_maximum_age_is_still_accepted():
    state = monitor(consecutive_hits=1)
    assert state.complete(matches(), captured_at=0.0, finished_at=2.0)
    assert state.discarded == 0


def test_check_passes_after_real_match_and_remembers_success_after_removal():
    state = monitor(consecutive_hits=1)
    assert state.complete(matches(), captured_at=0.0, finished_at=0.1)
    assert "AI CHECK PASSED" in state.headline(0.1, checking=True, confidence=0.25)

    assert not state.complete([], captured_at=0.2, finished_at=0.3)
    assert state.visible(0.3) == []
    assert "AI CHECK PASSED" in state.headline(0.3, checking=True, confidence=0.25)


def test_successful_empty_inference_is_distinct_from_warming_up():
    state = monitor()
    assert not state.complete([], captured_at=0.0, finished_at=0.1)
    status = state.headline(0.1, checking=True, confidence=0.25).lower()
    assert "no match" in status
    assert "warming" not in status
    assert state.completed == 1


def test_fresh_result_clears_slow_warning_without_resetting_discard_count():
    state = monitor(consecutive_hits=1)
    assert not state.complete(matches(), captured_at=0.0, finished_at=2.1)
    assert state.complete(matches(), captured_at=2.2, finished_at=2.3)
    assert state.discarded == 1
    assert state.completed == 2
    assert "too slow" not in state.headline(2.3, checking=True, confidence=0.25).lower()
    assert "AI CHECK PASSED" in state.headline(2.3, checking=True, confidence=0.25)
