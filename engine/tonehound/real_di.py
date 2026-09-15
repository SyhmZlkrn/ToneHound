"""Use real recorded DI takes in place of the synthesised probe.

``probe.py`` synthesises its DI with Karplus-Strong, which is deterministic and
dependency-free but lacks pick noise, vibrato, string squeak and real playing
dynamics. That was fine while the only question was profile-vs-profile
distance. It stopped being fine once separation entered the picture: the UVR
models are trained on real music, so a synthetic probe may sit outside their
training distribution, and every separation number measured so far carries that
doubt.

The structure is kept identical to the synthetic probe -- one phrase repeated at
each input level -- so a real DI is a drop-in replacement and every existing
tool works unchanged.
"""

import pathlib

import numpy as np
import soundfile as sf
from scipy import signal

from .config import SAMPLE_RATE
from .probe import PROBE_LEVELS_DBFS

GAP_S = 0.25


def load_di(path: str | pathlib.Path, sr: int = SAMPLE_RATE,
            max_seconds: float | None = 30.0) -> np.ndarray:
    """Load one DI take: mono, resampled, peak-normalised, optionally trimmed."""
    x, file_sr = sf.read(str(path), always_2d=True)
    x = x.mean(axis=1)
    if file_sr != sr:
        x = signal.resample_poly(x, sr, file_sr)
    if max_seconds is not None:
        x = x[: int(max_seconds * sr)]
    peak = np.abs(x).max()
    return x / peak if peak > 0 else x


def probe_from_di(di: np.ndarray, sr: int = SAMPLE_RATE,
                  levels=PROBE_LEVELS_DBFS, gap_s: float = GAP_S) -> np.ndarray:
    """Build a multi-level probe from a real DI take.

    Same shape as ``probe.build_probe``: the phrase at each input level, with a
    gap between passes so the level boundaries are unambiguous.
    """
    gap = np.zeros(int(gap_s * sr))
    return np.concatenate(
        [np.concatenate([di * 10 ** (db / 20.0), gap]) for db in levels])


def slices_from_di(di_len: int, sr: int = SAMPLE_RATE,
                   levels=PROBE_LEVELS_DBFS, gap_s: float = GAP_S) -> list[slice]:
    """Sample range of each level pass, mirroring ``probe.level_slices``."""
    stride = di_len + int(gap_s * sr)
    return [slice(i * stride, i * stride + di_len) for i in range(len(levels))]


def inspect(x: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, float]:
    """Cheap sanity checks on a take.

    The common mistake is submitting a *recorded amp* rather than a DI. An
    amped, cabinet-filtered signal is duller and far more compressed than a raw
    pickup, so centroid and crest factor separate the two well enough to warn
    on.
    """
    eps = 1e-20
    peak = float(np.abs(x).max() + eps)
    rms = float(np.sqrt(np.mean(x**2)) + eps)

    freqs, psd = signal.welch(x, fs=sr, nperseg=min(16384, len(x)))
    band = (freqs >= 80.0) & (freqs <= 10_000.0)
    total = psd[band].sum() + eps
    centroid = float((freqs[band] * psd[band]).sum() / total)

    # Energy above 5 kHz. Kept as a diagnostic, NOT as a DI/amp test: measured
    # against real takes, a clean guitar DI sits at 2e-6 to 8e-4 here because a
    # magnetic pickup rolls off steeply. A cabinet rolls off too, so this
    # separates neither -- it only flags synthetic signals, which is how the
    # Karplus-Strong probe (0.150, ~180-75000x a real DI) was caught.
    high = (freqs >= 5_000.0) & (freqs <= 10_000.0)
    high_ratio = float(psd[high].sum() / total)

    silent = float((np.abs(x) < 10 ** (-60 / 20)).mean())

    return {
        "seconds": len(x) / sr,
        "peak_dbfs": 20 * np.log10(peak),
        "crest_db": 20 * np.log10(peak / rms),
        "centroid_hz": centroid,
        "high_5k_ratio": high_ratio,
        "silent_fraction": silent,
    }


def warnings_for(stats: dict[str, float]) -> list[str]:
    """Human-readable problems with a take, empty if it looks usable."""
    out = []
    if stats["seconds"] < 15.0:
        out.append(f"only {stats['seconds']:.1f}s -- 30s+ gives the separator more to work with")
    # Crest factor is what actually separates a clean DI from an amped take:
    # real DIs measured 16-20 dB, while a distorted/compressed amp signal sits
    # far lower. The high-frequency test that used to live here was wrong -- it
    # flagged every genuine DI, because real pickups have almost nothing above
    # 5 kHz.
    if stats["crest_db"] < 10.0:
        out.append(f"crest factor {stats['crest_db']:.1f} dB is low (clean DIs measure "
                   "16-20 dB) -- is this an amped or heavily compressed recording?")
    if stats["centroid_hz"] > 1500.0:
        out.append(f"spectral centroid {stats['centroid_hz']:.0f} Hz is far above the "
                   "200-600 Hz a real guitar DI shows -- synthetic or processed?")
    if stats["peak_dbfs"] < -20.0:
        out.append(f"peak {stats['peak_dbfs']:.1f} dBFS is quiet -- fine, it gets "
                   "normalised, but check it is not clipped-then-reduced")
    if stats["silent_fraction"] > 0.5:
        out.append(f"{stats['silent_fraction']:.0%} of the take is near-silence -- "
                   "trim the gaps for a denser performance")
    return out
