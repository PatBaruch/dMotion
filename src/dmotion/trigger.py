"""Detection event state machine. All timestamps are monotonic seconds."""

from dataclasses import dataclass

from dmotion.config import TriggerConfig


@dataclass
class DetectionTrigger:
    settings: TriggerConfig
    active: bool = False
    hits: int = 0
    absent_since: float | None = None
    last_triggered: float = float("-inf")

    def update(self, detected: bool, now: float) -> bool:
        """Return True once per confirmed presentation, subject to cooldown."""
        if not detected:
            self.hits = 0
            if self.absent_since is None:
                self.absent_since = now
            if now - self.absent_since >= self.settings.reset_seconds:
                self.active = False
            return False

        # An observed absence also counts when the next sample is positive.
        if self.absent_since is not None:
            if now - self.absent_since >= self.settings.reset_seconds:
                self.active = False
            self.absent_since = None
        self.hits += 1
        if (
            not self.active
            and self.hits >= self.settings.consecutive_hits
            and now - self.last_triggered >= self.settings.cooldown_seconds
        ):
            self.active = True
            self.last_triggered = now
            return True
        return False
