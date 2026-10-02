"""Nonblocking alert playback, with a generated beep as the default asset."""

import io
import logging
import math
import wave
from array import array

from dmotion.config import AppConfig

logger = logging.getLogger(__name__)


def beep_wav() -> bytes:
    rate = 44100
    samples = array("h")
    for frequency in (880, 660):
        count = int(rate * 0.16)
        for i in range(count):
            envelope = min(1.0, i / 200, (count - i - 1) / 200)
            samples.append(int(12000 * envelope * math.sin(2 * math.pi * frequency * i / rate)))
    import sys

    if sys.byteorder != "little":
        samples.byteswap()
    output = io.BytesIO()
    with wave.open(output, "wb") as sound:
        sound.setnchannels(1)
        sound.setsampwidth(2)
        sound.setframerate(rate)
        sound.writeframes(samples.tobytes())
    return output.getvalue()


class AlertSound:
    def __init__(self, config: AppConfig):
        self.enabled = config.audio.enabled
        self.sound = None
        self.mixer = None
        try:
            import pygame.mixer as mixer

            mixer.init(frequency=44100, size=-16, channels=1)
            self.mixer = mixer
            path = config.resolve(config.audio.file)
            if path.is_file():
                self.sound = mixer.Sound(str(path))
            else:
                logger.info("Custom sound missing; using the built-in alert beep")
                self.sound = mixer.Sound(file=io.BytesIO(beep_wav()))
            self.sound.set_volume(config.audio.volume)
        except (ImportError, RuntimeError, OSError) as exc:
            logger.warning("Audio unavailable: %s", exc)
        except Exception as exc:
            # pygame.error is not a built-in exception; camera testing can continue.
            logger.warning("Audio unavailable: %s", exc)

    def play(self, *, force: bool = False) -> bool:
        if self.sound is not None and (self.enabled or force):
            self.sound.play()
            return True
        return False

    def close(self) -> None:
        if self.mixer is not None:
            self.mixer.quit()
