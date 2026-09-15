"""Decoding, resampling and normalisation -- in one place.

`live/media.py` grew its own decoder first, because the server was the only
thing that had to accept whatever a user dragged onto it. The pipeline now
accepts the same range of input from a CLI and from a URL, so the decoder moved
here and `media.py` calls it. One decoder means one answer to "does this file
work", rather than a server that opens an .m4a and a tool that does not.

Two rules the whole engine depends on:

*Everything lands at 48 kHz float32.* Legacy callers request mono; reference
playback and separation preserve channels with ``mono=False``. `config.SAMPLE_RATE` is NAM's anchor
rate and the renderer's only rate. A file that stayed at 44.1 kHz would have to
be resampled by whichever caller remembered to, and the one that forgot would
produce a fingerprint shifted in frequency by 8.8%.

*ffmpeg is the fallback, not the default.* libsndfile opens wav/flac/ogg/mp3
directly and quickly. It will not open m4a, wma, or the audio track of a video,
which is exactly what a download from a link tends to be, so `.cache/bin`
carries an ffmpeg and this module shells out to it when libsndfile refuses.
"""

from __future__ import annotations

import io
import pathlib
import shutil
import subprocess

import numpy as np
import soundfile as sf
from scipy import signal

from .config import SAMPLE_RATE

PEAK_TARGET = 0.98
"""Headroom left below full scale so a PCM16 round trip cannot clip."""


class AudioError(Exception):
    """A decode problem worth showing the user, with a protocol error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def ffmpeg_exe() -> str | None:
    """ffmpeg from `.cache/bin` if it is there, else whatever is on PATH."""
    bundled = pathlib.Path(__file__).resolve().parents[2] / ".cache" / "bin" / "ffmpeg.exe"
    if bundled.exists():
        return str(bundled)
    return shutil.which("ffmpeg")


def _ffmpeg_decode(raw: bytes) -> tuple[np.ndarray, int, int]:
    exe = ffmpeg_exe()
    if exe is None:
        raise FileNotFoundError("ffmpeg")
    proc = subprocess.run(
        [exe, "-v", "error", "-i", "pipe:0", "-f", "wav",
         "-acodec", "pcm_f32le", "pipe:1"],
        input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0 or not proc.stdout:
        raise AudioError("unsupported_media",
                         "could not decode this file: "
                         + (proc.stderr.decode("utf-8", "replace").strip()[-300:]
                            or "ffmpeg produced no output"))
    x, sr = sf.read(io.BytesIO(proc.stdout), always_2d=True, dtype="float32")
    return x, sr, x.shape[1]


def decode(raw: bytes, filename: str = "", *, mono: bool = True) -> tuple[np.ndarray, int, int]:
    """Decode bytes to float32 at 48 kHz; ``mono=False`` returns frames × channels.

    Returns ``(samples, source_sample_rate, source_channels)``. The two source
    values are reported rather than discarded because the UI shows them, and
    because "this was already mono at 48 kHz" is worth being able to say.
    """
    try:
        data, source_sr = sf.read(io.BytesIO(raw), always_2d=True, dtype="float32")
        channels = data.shape[1]
    except Exception as exc:
        try:
            data, source_sr, channels = _ffmpeg_decode(raw)
        except FileNotFoundError:
            raise AudioError("unsupported_media",
                             f"could not decode {filename or 'the file'}: {exc}") from exc

    if data.size == 0:
        raise AudioError("unsupported_media",
                         f"{filename or 'the file'} decoded to zero samples")

    samples = data.mean(axis=1) if mono else data
    return normalise(to_rate(samples, int(source_sr))), int(source_sr), int(channels)


def read(path: str | pathlib.Path, *, seconds: float | None = None,
         start_s: float = 0.0, mono: bool = True) -> np.ndarray:
    """Decode a file from disk, optionally taking a window out of it.

    The window is applied *after* decoding rather than by seeking, because the
    fallback path is a pipe through ffmpeg that cannot seek, and one code path
    that is slightly wasteful beats two that disagree about where second 30 is.
    """
    path = pathlib.Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AudioError("not_found", f"could not read {path}: {exc}") from exc
    x, _, _ = decode(raw, path.name, mono=mono)
    return window(x, seconds=seconds, start_s=start_s)


def window(x: np.ndarray, *, seconds: float | None = None,
           start_s: float = 0.0) -> np.ndarray:
    """``x`` restricted to ``[start_s, start_s + seconds)``, renormalised."""
    lo = max(0, int(start_s * SAMPLE_RATE))
    hi = len(x) if seconds is None else lo + int(seconds * SAMPLE_RATE)
    out = x[lo:hi]
    if out.size == 0:
        raise AudioError("empty_window",
                         f"the window at {start_s:.1f}s is past the end of "
                         f"{len(x) / SAMPLE_RATE:.1f}s of audio")
    return normalise(out)


def to_rate(x: np.ndarray, source_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Resample, or return the input untouched when the rates already agree."""
    if source_sr == target_sr:
        return np.asarray(x, dtype=np.float64)
    return signal.resample_poly(np.asarray(x, dtype=np.float64), target_sr, source_sr)


def normalise(x: np.ndarray, target: float = PEAK_TARGET) -> np.ndarray:
    """Scale to a fixed peak. Silence is returned as-is rather than amplified."""
    peak = float(np.abs(x).max()) if x.size else 0.0
    out = x / peak * target if peak > 0 else x
    return np.ascontiguousarray(out, dtype=np.float32)


def write(path: str | pathlib.Path, x: np.ndarray, *,
          sr: int = SAMPLE_RATE, peak: float = 0.95) -> None:
    """Write a mono signal, peak-normalised, format inferred from the suffix."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scale = float(np.abs(x).max()) + 1e-12
    sf.write(str(path), np.asarray(x, dtype=np.float32) / scale * peak, sr)
