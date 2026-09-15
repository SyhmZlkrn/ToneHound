"""The one decoder. Everything downstream assumes 48 kHz mono float32.

Worth testing directly rather than through the server, because the failure it
prevents is silent: a file that stayed at 44.1 kHz still plays, still draws a
waveform, and produces a fingerprint shifted in frequency by 8.8% -- which looks
like a bad match, not like a bug.
"""

from __future__ import annotations

import io
import pathlib

import numpy as np
import pytest
import soundfile as sf

from tonehound import audioio
from tonehound.audioio import AudioError
from tonehound.config import SAMPLE_RATE

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _tone(seconds: float, sr: int, hz: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _wav(x: np.ndarray, sr: int, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    data = np.repeat(x[:, None], channels, axis=1) if channels > 1 else x
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# -- decode ----------------------------------------------------------------


def test_decode_resamples_and_downmixes_to_the_engine_rate() -> None:
    samples, source_sr, channels = audioio.decode(_wav(_tone(1.0, 44100), 44100, 2))
    assert (source_sr, channels) == (44100, 2)
    assert samples.dtype == np.float32 and samples.ndim == 1
    assert abs(len(samples) / SAMPLE_RATE - 1.0) < 0.01


def test_decode_reports_the_source_rate_it_resampled_away_from() -> None:
    _, source_sr, _ = audioio.decode(_wav(_tone(0.2, 22050), 22050))
    assert source_sr == 22050


def test_decode_normalises_so_a_quiet_file_ranks_like_a_loud_one() -> None:
    quiet = audioio.decode(_wav(_tone(0.3, SAMPLE_RATE) * 0.01, SAMPLE_RATE))[0]
    loud = audioio.decode(_wav(_tone(0.3, SAMPLE_RATE), SAMPLE_RATE))[0]
    assert np.abs(quiet).max() == pytest.approx(audioio.PEAK_TARGET, abs=1e-3)
    assert np.abs(loud).max() == pytest.approx(audioio.PEAK_TARGET, abs=1e-3)


def test_decode_rejects_bytes_that_are_not_audio_with_a_coded_error() -> None:
    with pytest.raises(AudioError) as exc:
        audioio.decode(b"this is not a sound file", "notes.txt")
    assert exc.value.code == "unsupported_media"


def test_decode_rejects_a_file_that_holds_no_samples() -> None:
    with pytest.raises(AudioError):
        audioio.decode(_wav(np.zeros(0, dtype=np.float32), SAMPLE_RATE))


# -- windowing -------------------------------------------------------------


def test_window_takes_the_requested_span_and_renormalises_it() -> None:
    x = np.concatenate([_tone(1.0, SAMPLE_RATE) * 0.1, _tone(1.0, SAMPLE_RATE)])
    out = audioio.window(x, seconds=1.0, start_s=0.0)
    assert abs(len(out) / SAMPLE_RATE - 1.0) < 0.01
    # The quiet half alone comes back at full scale; that is the point.
    assert np.abs(out).max() == pytest.approx(audioio.PEAK_TARGET, abs=1e-3)


def test_window_past_the_end_is_an_error_rather_than_empty_audio() -> None:
    with pytest.raises(AudioError) as exc:
        audioio.window(_tone(1.0, SAMPLE_RATE), seconds=1.0, start_s=30.0)
    assert exc.value.code == "empty_window"


def test_read_round_trips_a_file_from_disk(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "take.wav"
    sf.write(path, _tone(2.0, SAMPLE_RATE), SAMPLE_RATE)
    assert abs(len(audioio.read(path)) / SAMPLE_RATE - 2.0) < 0.01
    assert abs(len(audioio.read(path, seconds=0.5)) / SAMPLE_RATE - 0.5) < 0.01


def test_read_of_a_missing_file_names_the_file(tmp_path: pathlib.Path) -> None:
    with pytest.raises(AudioError) as exc:
        audioio.read(tmp_path / "nope.wav")
    assert exc.value.code == "not_found"


# -- resampling and level --------------------------------------------------


def test_to_rate_is_a_no_op_when_the_rates_already_agree() -> None:
    x = _tone(0.1, SAMPLE_RATE)
    assert np.allclose(audioio.to_rate(x, SAMPLE_RATE), x)


def test_to_rate_preserves_a_tone_across_a_round_trip() -> None:
    x = _tone(0.5, SAMPLE_RATE, hz=440.0)
    back = audioio.to_rate(audioio.to_rate(x, SAMPLE_RATE, 24_000), 24_000, SAMPLE_RATE)
    n = min(len(x), len(back))
    # Edges ring from the resampler; the interior is what has to survive.
    assert np.corrcoef(x[1000:n - 1000], back[1000:n - 1000])[0, 1] > 0.999


def test_normalise_leaves_silence_alone_instead_of_amplifying_it() -> None:
    silence = np.zeros(128, dtype=np.float32)
    assert np.array_equal(audioio.normalise(silence), silence)


def test_stereo_decode_preserves_polarity_and_relative_channel_level():
    left = _tone(1, 44100)
    stereo = np.column_stack((left, -.5 * left))
    buf = io.BytesIO(); sf.write(buf, stereo, 44100, format='WAV', subtype='FLOAT')
    audio, sr, channels = audioio.decode(buf.getvalue(), mono=False)
    assert sr == 44100 and channels == 2 and audio.shape == (48000, 2)
    assert np.max(np.abs(audio[:, 0])) > .9
    assert np.allclose(audio[:, 1], -.5 * audio[:, 0], atol=1e-7)


def test_ffmpeg_fallback_keeps_channels_and_original_sample_rate():
    import subprocess
    exe = audioio.ffmpeg_exe()
    if exe is None: pytest.skip('ffmpeg not installed')
    left = _tone(1, 44100, 220)
    stereo = np.column_stack((left, _tone(1, 44100, 660) * .3))
    buf=io.BytesIO();sf.write(buf,stereo,44100,format='WAV',subtype='FLOAT')
    encoded=subprocess.run([exe,'-v','error','-i','pipe:0','-c:a','aac','-f','adts','pipe:1'],input=buf.getvalue(),capture_output=True,check=True).stdout
    audio,sr,channels=audioio.decode(encoded,'reference.aac',mono=False)
    assert sr == 44100 and channels == 2 and audio.ndim == 2
    assert abs(np.corrcoef(audio.T)[0,1]) < .1
    assert np.sqrt(np.mean(audio[:,0]**2))/np.sqrt(np.mean(audio[:,1]**2)) == pytest.approx(1/.3,rel=.1)
