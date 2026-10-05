import io
import wave

import numpy as np

from seera_tts.audio import pcm16_to_wav, silence, to_pcm16


# ------------------------------------------------------------------ to_pcm16
def test_pcm16_uses_two_bytes_per_sample():
    assert len(to_pcm16(np.zeros(100, dtype=np.float32))) == 200


def test_pcm16_values_and_byte_order():
    pcm = to_pcm16(np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32))
    samples = np.frombuffer(pcm, dtype="<i2")  # read back as little-endian int16
    assert samples.tolist() == [0, 32767, -32767, 16383]


def test_pcm16_clips_out_of_range_values():
    assert to_pcm16(np.array([2.0])) == to_pcm16(np.array([1.0]))
    assert to_pcm16(np.array([-5.0])) == to_pcm16(np.array([-1.0]))


# ------------------------------------------------------------------- silence
def test_silence_length_and_values():
    s = silence(1000, 24000)
    assert len(s) == 24000
    assert s.dtype == np.float32
    assert not s.any()  # all zeros


def test_silence_short_pause():
    assert len(silence(280, 24000)) == 6720  # the sentence pause from config


# -------------------------------------------------------------- pcm16_to_wav
def test_wav_has_riff_header():
    wav = pcm16_to_wav(to_pcm16(np.zeros(240, dtype=np.float32)), 24000)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"


def test_wav_round_trip():
    pcm = to_pcm16(np.linspace(-1, 1, 480, dtype=np.float32))
    with wave.open(io.BytesIO(pcm16_to_wav(pcm, 24000))) as f:
        assert f.getnchannels() == 1
        assert f.getsampwidth() == 2
        assert f.getframerate() == 24000
        assert f.getnframes() == 480
        assert f.readframes(480) == pcm