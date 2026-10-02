from dmotion.config import TriggerConfig
from dmotion.trigger import DetectionTrigger


def gate(**changes):
    return DetectionTrigger(TriggerConfig(**changes))


def test_requires_consecutive_positive_inferences():
    trigger = gate()
    assert not trigger.update(True, 0.0)
    assert not trigger.update(True, 0.1)
    assert trigger.update(True, 0.2)


def test_interrupted_hits_do_not_confirm_detection():
    trigger = gate()
    trigger.update(True, 0.0)
    trigger.update(True, 0.1)
    trigger.update(False, 0.2)
    assert not trigger.update(True, 0.3)
    assert not trigger.update(True, 0.4)
    assert trigger.update(True, 0.5)


def test_held_spread_does_not_retrigger_when_cooldown_expires():
    trigger = gate(consecutive_hits=1)
    assert trigger.update(True, 0.0)
    for timestamp in (1.0, 4.0, 10.0, 60.0):
        assert not trigger.update(True, timestamp)


def test_brief_detection_dropout_does_not_rearm_sound():
    trigger = gate(consecutive_hits=1)
    assert trigger.update(True, 0.0)
    trigger.update(False, 4.0)
    assert not trigger.update(True, 4.2)


def test_removing_and_representing_spread_rearms_sound():
    trigger = gate(consecutive_hits=1)
    assert trigger.update(True, 0.0)
    trigger.update(False, 4.0)
    trigger.update(False, 5.1)
    assert trigger.update(True, 5.2)


def test_next_positive_sample_can_complete_absence_interval():
    trigger = gate(consecutive_hits=1)
    trigger.update(True, 0.0)
    trigger.update(False, 4.0)
    assert trigger.update(True, 5.1)


def test_rearmed_sound_still_respects_cooldown():
    trigger = gate(consecutive_hits=1, reset_seconds=0.5, cooldown_seconds=3.0)
    assert trigger.update(True, 0.0)
    trigger.update(False, 0.1)
    trigger.update(False, 0.7)
    assert not trigger.update(True, 0.8)
    assert not trigger.update(True, 2.9)
    assert trigger.update(True, 3.0)
