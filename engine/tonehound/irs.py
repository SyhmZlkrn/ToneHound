"""Cab impulse responses.

An IR is a linear time-invariant filter, so unlike a NAM profile its entire
character is its frequency response -- no probe rendering needed, just an FFT.
That is what makes the joint (profile, IR) search affordable.

Whether an IR is wanted at all is decided by the profile's ``gear_type``: a
capture that already includes a cab must *not* have one convolved on top.
Stacking two cabs is the classic beginner mistake and it ruins a match no
matter how good the amp selection was.
"""

import numpy as np
import soundfile as sf
from scipy import signal

from .config import SAMPLE_RATE

CAB_INCLUDED_GEAR_TYPES = frozenset({"amp_cab", "amp_pedal_cab", "studio"})
"""``gear_type`` values whose capture already contains a cab."""

CAB_MISSING_GEAR_TYPES = frozenset({"amp", "preamp", "pedal_amp"})
"""``gear_type`` values that need an IR to sound like a rig."""


def needs_ir(gear_type: str | None) -> bool:
    """Whether an IR slot should be engaged for a profile.

    Unknown or missing types are treated as cab-inclusive: bypassing an IR that
    was wanted is a mild error, while doubling a cab is a severe one.
    """
    if gear_type is None:
        return False
    return gear_type.strip().lower() in CAB_MISSING_GEAR_TYPES


def load_ir(path: str, sr: int = SAMPLE_RATE, max_ms: float = 500.0) -> np.ndarray:
    """Load an IR, downmix to mono, resample if needed, and trim."""
    h, file_sr = sf.read(path, always_2d=True)
    h = h.mean(axis=1)
    if file_sr != sr:
        h = signal.resample_poly(h, sr, file_sr)
    h = h[: int(max_ms * 1e-3 * sr)]
    peak = np.abs(h).max()
    return h / peak if peak > 0 else h


def apply_ir(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolve, keeping the input length."""
    return signal.fftconvolve(x, h)[: len(x)]


def synthetic_cab_ir(sr: int = SAMPLE_RATE, dur: float = 0.08, seed: int = 0,
                     resonances=((110.0, 6.0, 2.5), (400.0, 4.0, 1.6), (2200.0, 5.0, 1.8)),
                     band=(75.0, 5200.0)) -> np.ndarray:
    """A guitar-cab-shaped IR for development and tests.

    Not a model of any real cabinet -- just the broad strokes that matter for
    validating the maths: a band-limited decaying burst with a few resonant
    peaks and a steep top-end roll-off.
    """
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    t = np.arange(n) / sr

    h = rng.standard_normal(n) * np.exp(-t / 0.012)
    sos = signal.butter(4, band, btype="band", fs=sr, output="sos")
    h = signal.sosfilt(sos, h)

    for f0, q, gain in resonances:
        b, a = signal.iirpeak(f0, q, fs=sr)
        h = h + (gain - 1.0) * signal.lfilter(b, a, h)

    peak = np.abs(h).max()
    return h / peak if peak > 0 else h
