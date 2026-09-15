"""Decode, waveform peaks, cropping, and the one id namespace.

The peaks path is the one worth testing hard. It is the only thing the UI draws
a waveform from, so an off-by-one there is a picture that disagrees with the
audio the user then crops out of it -- and it is exactly the kind of error that
looks fine at 2000 bins over three minutes and is wrong at the ends.
"""

from __future__ import annotations

import io
import pathlib
import wave

import numpy as np
import pytest
import soundfile as sf

from tonehound.config import SAMPLE_RATE
from live import media
from live.media import MediaError, MediaStore

ROOT = pathlib.Path(__file__).resolve().parents[2]
DI_DIR = ROOT / "assets" / "user_di"
SONGS = ROOT / "assets" / "song_train"

needs_di = pytest.mark.skipif(not DI_DIR.is_dir(), reason="no DI takes present")
needs_songs = pytest.mark.skipif(not SONGS.is_dir() or not list(SONGS.glob("*.mp3")),
                                 reason="no test songs present")


def _wav_bytes(x: np.ndarray, sr: int, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    data = np.repeat(x[:, None], channels, axis=1) if channels > 1 else x
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _tone(seconds: float, sr: int, hz: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


# -- decode ----------------------------------------------------------------


def test_decode_resamples_and_downmixes_to_the_engine_rate() -> None:
    raw = _wav_bytes(_tone(1.0, 44100), 44100, channels=2)
    samples, source_sr, channels = media.decode(raw, "x.wav")
    assert (source_sr, channels) == (44100, 2)
    assert samples.dtype == np.float32 and samples.ndim == 1
    assert abs(len(samples) / SAMPLE_RATE - 1.0) < 0.01


def test_decode_peak_normalises_so_a_quiet_import_ranks_like_a_loud_one() -> None:
    quiet, loud = _tone(0.5, SAMPLE_RATE) * 0.01, _tone(0.5, SAMPLE_RATE)
    a, _, _ = media.decode(_wav_bytes(quiet, SAMPLE_RATE), "q.wav")
    b, _, _ = media.decode(_wav_bytes(loud, SAMPLE_RATE), "l.wav")
    assert np.abs(a).max() == pytest.approx(media.PEAK_TARGET, abs=1e-3)
    assert np.abs(b).max() == pytest.approx(media.PEAK_TARGET, abs=1e-3)


def test_an_undecodable_file_is_a_told_error_not_a_crash() -> None:
    with pytest.raises(MediaError) as exc:
        media.decode(b"this is not audio", "notes.txt")
    assert exc.value.code == "unsupported_media"


@needs_songs
def test_a_real_mp3_decodes_from_memory() -> None:
    """A browser hands over bytes, never a path, so this must work without a
    temporary file on the way."""
    path = sorted(SONGS.glob("*.mp3"))[0]
    samples, source_sr, channels = media.decode(path.read_bytes(), path.name)
    assert source_sr in (44100, 48000) and channels in (1, 2)
    assert len(samples) > SAMPLE_RATE * 30


# -- wav encoding ----------------------------------------------------------


def test_encoded_wav_is_readable_mono_pcm16_at_the_engine_rate() -> None:
    raw = media.encode_wav(_tone(0.25, SAMPLE_RATE))
    with wave.open(io.BytesIO(raw)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, SAMPLE_RATE)
        assert w.getnframes() == int(0.25 * SAMPLE_RATE)


def test_encoding_never_wraps_a_sample_that_exceeded_full_scale() -> None:
    """Wrapping would turn a hot render into a burst of white noise, which
    sounds like a broken profile rather than a clipped one."""
    raw = media.encode_wav(np.array([2.0, -2.0, 0.0], dtype=np.float32))
    pcm = np.frombuffer(raw[-6:], dtype="<i2")
    assert pcm[0] > 32000 and pcm[1] < -32000


# -- peaks -----------------------------------------------------------------


def test_peaks_return_exactly_the_requested_bin_count() -> None:
    env = media.envelope(_tone(3.0, SAMPLE_RATE), 777)
    assert len(env["min"]) == len(env["max"]) == len(env["rms"]) == 777


def test_peaks_bracket_the_signal_and_rms_sits_between_them() -> None:
    x = _tone(2.0, SAMPLE_RATE)
    env = media.envelope(x, 200)
    lo, hi, rms = (np.array(env[k]) for k in ("min", "max", "rms"))
    assert lo.min() >= -1.0 and hi.max() <= 1.0
    assert np.all(lo <= hi)
    assert np.all(rms >= 0.0) and np.all(rms <= np.maximum(np.abs(lo), hi) + 1e-6)
    assert hi.max() == pytest.approx(np.abs(x).max(), abs=1e-3)


def test_the_last_bin_covers_the_end_of_the_signal() -> None:
    """A reshape-based binner drops the remainder, which silently truncates the
    right-hand edge of every waveform that is not a multiple of the bin count."""
    x = np.zeros(48001, dtype=np.float32)
    x[-1] = 1.0
    assert media.envelope(x, 1000)["max"][-1] == pytest.approx(1.0)


def test_more_bins_than_samples_does_not_read_the_wrong_element() -> None:
    x = np.array([0.0, 1.0, -1.0, 0.0], dtype=np.float32)
    env = media.envelope(x, 16)
    assert len(env["max"]) == 16
    assert max(env["max"]) == pytest.approx(1.0)
    assert min(env["min"]) == pytest.approx(-1.0)


def test_peaks_of_silence_are_defined_rather_than_nan() -> None:
    env = media.envelope(np.zeros(1000, dtype=np.float32), 50)
    assert all(v == 0.0 for v in env["rms"])


# -- the store -------------------------------------------------------------


def test_items_share_one_id_namespace_and_serve_both_urls() -> None:
    store = MediaStore()
    item = store.add(_tone(1.0, SAMPLE_RATE), kind="snippet", name="a")
    assert item.audio_url == f"/audio/{item.id}.wav"
    assert item.peaks_url == f"/peaks/{item.id}"
    assert store.get(item.id) is item


def test_wav_and_peaks_are_memoised_so_a_seeking_player_is_cheap() -> None:
    store = MediaStore()
    item = store.add(_tone(1.0, SAMPLE_RATE), kind="render", name="r")
    assert item.wav() is item.wav()
    assert item.peaks(500) is item.peaks(500)


def test_an_unknown_or_wrongly_typed_id_is_not_found() -> None:
    store = MediaStore()
    item = store.add(_tone(0.5, SAMPLE_RATE), kind="snippet", name="a")
    with pytest.raises(MediaError) as exc:
        store.get("deadbeef")
    assert exc.value.code == "not_found"
    with pytest.raises(MediaError):
        store.get(item.id, "di")


def test_crop_slices_the_requested_region() -> None:
    store = MediaStore()
    snippet = store.add(_tone(30.0, SAMPLE_RATE), kind="snippet", name="song")
    crop = store.crop(snippet.id, 5.0, 15.0, 3.0, 30.0)
    assert crop.duration_s == pytest.approx(10.0, abs=0.01)
    assert crop.parent == snippet.id
    assert crop.meta["start_s"] == pytest.approx(5.0)


def test_crop_is_non_destructive_and_repeatable() -> None:
    """The UI re-crops on every handle drag, so this has to stay cheap and must
    not consume the decode it came from."""
    store = MediaStore()
    snippet = store.add(_tone(30.0, SAMPLE_RATE), kind="snippet", name="song")
    first = store.crop(snippet.id, 5.0, 15.0, 3.0, 30.0)
    second = store.crop(snippet.id, 6.0, 16.0, 3.0, 30.0)
    assert first.id != second.id
    assert store.get(snippet.id).duration_s == pytest.approx(30.0, abs=0.01)


def test_crop_bounds_are_clamped_but_the_length_rules_still_apply() -> None:
    store = MediaStore()
    snippet = store.add(_tone(20.0, SAMPLE_RATE), kind="snippet", name="song")
    # Overshooting the end by a drag is not an error; it is clamped.
    assert store.crop(snippet.id, 10.0, 999.0, 3.0, 30.0).duration_s == pytest.approx(
        10.0, abs=0.01)
    with pytest.raises(MediaError) as short:
        store.crop(snippet.id, 1.0, 2.0, 3.0, 30.0)
    assert short.value.code == "crop_too_short"
    with pytest.raises(MediaError) as long:
        store.crop(snippet.id, 0.0, 20.0, 3.0, 15.0)
    assert long.value.code == "snippet_too_long"


def test_forget_drops_the_snippet_and_every_crop_from_it() -> None:
    store = MediaStore()
    snippet = store.add(_tone(30.0, SAMPLE_RATE), kind="snippet", name="song")
    crop = store.crop(snippet.id, 5.0, 15.0, 3.0, 30.0)
    other = store.add(_tone(1.0, SAMPLE_RATE), kind="di", name="di")
    gone = store.forget(snippet.id)
    assert set(gone) == {snippet.id, crop.id}
    with pytest.raises(MediaError):
        store.get(crop.id)
    assert store.get(other.id) is other


@needs_di
def test_the_shipped_di_takes_load_and_one_is_default() -> None:
    store = MediaStore()
    takes = store.load_di_directory(DI_DIR)
    assert len(takes) >= 1
    assert store.default_di_id in {t.id for t in takes}
    # The index is rendered through the Djent take, so auditions default to it.
    assert "djent" in store.get(store.default_di_id).name.lower()
    assert all(t.duration_s > 1.0 for t in takes)
