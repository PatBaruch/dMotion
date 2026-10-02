import io
import wave

from dmotion.audio import beep_wav


def test_fallback_alert_is_a_playable_short_wav():
    with wave.open(io.BytesIO(beep_wav()), "rb") as sound:
        assert sound.getnchannels() == 1
        assert sound.getsampwidth() == 2
        duration = sound.getnframes() / sound.getframerate()
        assert 0.3 < duration < 0.4
        assert any(sound.readframes(sound.getnframes()))
