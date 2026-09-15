"""Fingerprint extraction.

A ``.nam`` profile is a function, not audio, so profiles are compared by
rendering a fixed probe DI through them and fingerprinting the result. The
target song is fingerprinted the same way after stem separation, which puts
both into a single feature space.

The LTAS term is kept separate from the scalar descriptors because the joint
(profile, IR) search relies on LTAS being additive in the dB domain::

    LTAS(amp -> IR)  ~=  LTAS(amp) + ir_response(IR)

Convolution is multiplication in the spectrum, hence addition in dB. That turns
a 300x200 pair search from 60k renders into 60k vector additions. The scalar
descriptors measure saturation and dynamics, which a *linear* cab does not
alter, so they are taken from the amp render alone.

The approximation is only exact when |H|^2 is flat across a band, so its error
is measured rather than assumed -- see ``tests/test_additive_ltas.py``.
"""

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .config import InstrumentConfig

_EPS = 1e-20

# Welch segment length. At 48 kHz this gives ~2.9 Hz resolution, which keeps
# even the narrowest third-octave band near 40 Hz populated with real bins.
_NPERSEG = 16_384


def third_octave_edges(lo_hz: float, hi_hz: float, n_bands: int) -> np.ndarray:
    """Log-spaced band edges. Returns ``n_bands + 1`` values."""
    return np.geomspace(lo_hz, hi_hz, n_bands + 1)


def band_centers(edges: np.ndarray) -> np.ndarray:
    """Geometric centre of each band."""
    return np.sqrt(edges[:-1] * edges[1:])


def _band_stat(freqs: np.ndarray, values: np.ndarray, edges: np.ndarray, mode: str) -> np.ndarray:
    """Reduce a spectrum into bands.

    ``mode="sum"`` integrates (for power spectral densities, where the band
    total is the meaningful quantity). ``mode="mean"`` averages (for transfer
    magnitudes, where the band's average gain is what matters). The two must
    differ for the additive identity above to hold.
    """
    out = np.empty(len(edges) - 1)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (freqs >= lo) & (freqs < hi)
        if m.any():
            out[i] = np.trapz(values[m], freqs[m]) if m.sum() > 1 else values[m][0] * (hi - lo)
            if mode == "mean":
                out[i] /= hi - lo
        else:
            # Band narrower than the FFT resolution: fall back to the value at
            # the band centre.
            centre = np.sqrt(lo * hi)
            v = np.interp(centre, freqs, values)
            out[i] = v if mode == "mean" else v * (hi - lo)
    return out


def _welch(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(_NPERSEG, len(x))
    return signal.welch(x, fs=sr, nperseg=nperseg, noverlap=nperseg // 2, scaling="density")


def ltas_db(x: np.ndarray, sr: int, cfg: InstrumentConfig) -> np.ndarray:
    """Long-term average spectrum, per band, in dB.

    Not level-normalised -- callers normalise, because the additive identity is
    stated on raw dB and normalisation is a constant offset that cancels.
    """
    freqs, psd = _welch(x, sr)
    edges = third_octave_edges(cfg.ltas_lo_hz, cfg.ltas_hi_hz, cfg.n_bands)
    return 10.0 * np.log10(_band_stat(freqs, psd, edges, "sum") + _EPS)


def ir_response_db(h: np.ndarray, sr: int, cfg: InstrumentConfig,
                   reference: np.ndarray | None = None) -> np.ndarray:
    """Band-averaged power response of an impulse response, in dB.

    This is the vector that adds to an amp's LTAS to predict the cabbed result.

    ``reference`` supplies a signal whose spectrum weights the band average. It
    matters more than it looks: guitar is a *line* spectrum, so a band's energy
    sits on a handful of harmonics and the cab gain that actually applies is
    |H|^2 at those harmonics, not the band's flat mean. Weighting by a
    representative render roughly halves the approximation error where the cab
    response moves steeply.

    The weighting is a fixed reference, not the amp being paired, so each IR
    still reduces to one precomputed vector and the search stays additive.
    """
    n = int(2 ** np.ceil(np.log2(max(len(h), _NPERSEG))))
    H = np.fft.rfft(h, n=n)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mag2 = np.abs(H) ** 2
    edges = third_octave_edges(cfg.ltas_lo_hz, cfg.ltas_hi_hz, cfg.n_bands)

    if reference is None:
        return 10.0 * np.log10(_band_stat(freqs, mag2, edges, "mean") + _EPS)

    ref_freqs, ref_psd = _welch(np.asarray(reference, dtype=float), sr)
    w = np.interp(freqs, ref_freqs, ref_psd)
    num = _band_stat(freqs, mag2 * w, edges, "sum")
    den = _band_stat(freqs, w, edges, "sum")
    return 10.0 * np.log10(num / (den + _EPS) + _EPS)


def normalize_ltas(v: np.ndarray) -> np.ndarray:
    """Remove overall level so only spectral shape remains."""
    return v - v.mean()


# --------------------------------------------------------------------------
# Scalar descriptors. All are computed on the instrument's analysis band and
# are unaffected by a linear cab, so they come from the amp render alone.
# --------------------------------------------------------------------------


def _analysis_psd(x: np.ndarray, sr: int, cfg: InstrumentConfig):
    freqs, psd = _welch(x, sr)
    m = (freqs >= cfg.analysis_lo_hz) & (freqs <= cfg.analysis_hi_hz)
    return freqs[m], psd[m]


def spectral_shape(x: np.ndarray, sr: int, cfg: InstrumentConfig) -> dict[str, float]:
    freqs, psd = _analysis_psd(x, sr, cfg)
    total = psd.sum() + _EPS

    centroid = float((freqs * psd).sum() / total)
    spread = float(np.sqrt(((freqs - centroid) ** 2 * psd).sum() / total))

    cumulative = np.cumsum(psd) / total
    rolloff85 = float(freqs[np.searchsorted(cumulative, 0.85)])
    rolloff95 = float(freqs[np.searchsorted(cumulative, min(0.95, cumulative[-1]))])

    flatness = float(np.exp(np.log(psd + _EPS).mean()) / (psd.mean() + _EPS))

    # Least-squares tilt of the log-log spectrum: a single number for "dark vs
    # bright" that is robust to the overall level.
    slope = float(np.polyfit(np.log10(freqs + _EPS), 10 * np.log10(psd + _EPS), 1)[0])

    return {
        "centroid": centroid,
        "spread": spread,
        "rolloff85": rolloff85,
        "rolloff95": rolloff95,
        "flatness": flatness,
        "slope": slope,
    }


def saturation(x: np.ndarray, sr: int, cfg: InstrumentConfig) -> dict[str, float]:
    freqs, psd = _analysis_psd(x, sr, cfg)
    total = psd.sum() + _EPS

    fizz = (freqs >= cfg.fizz_lo_hz) & (freqs <= cfg.fizz_hi_hz)
    fizz_ratio = float(psd[fizz].sum() / total)

    peak = float(np.abs(x).max() + _EPS)
    rms = float(np.sqrt(np.mean(x**2)) + _EPS)
    crest = float(20 * np.log10(peak / rms))

    return {"fizz_ratio": fizz_ratio, "crest_db": crest}


def dynamics(x: np.ndarray, sr: int) -> dict[str, float]:
    """Envelope statistics. Separates a squashed high-gain amp from a clean."""
    # 10 ms RMS envelope.
    win = max(1, int(0.010 * sr))
    env = np.sqrt(np.convolve(x**2, np.ones(win) / win, mode="same") + _EPS)
    env_db = 20 * np.log10(env)

    # Spread of the envelope in dB: small means heavily compressed.
    #
    # Deliberately no rms-to-peak here: it is crest_db re-expressed, exactly
    # (crest_db == -20*log10(rms_to_peak), to machine precision), so carrying
    # both counted the same measurement twice in every distance.
    return {
        "env_std_db": float(env_db.std()),
        "env_p90_p10_db": float(np.percentile(env_db, 90) - np.percentile(env_db, 10)),
        "transient_sharpness": float(np.mean(np.abs(np.diff(env_db)))),
    }


SCALAR_KEYS = (
    "centroid",
    "spread",
    "rolloff85",
    "rolloff95",
    "flatness",
    "slope",
    "fizz_ratio",
    "crest_db",
    "env_std_db",
    "env_p90_p10_db",
    "transient_sharpness",
)


@dataclass
class Fingerprint:
    """LTAS stays separate from scalars so the joint IR search can add to it."""

    ltas: np.ndarray  # (n_bands,) dB, un-normalised
    scalars: dict[str, float]
    level_ltas: np.ndarray | None = None  # (n_levels, n_bands) dB, un-normalised

    @property
    def shape_vector(self) -> np.ndarray:
        """Level-independent spectral shape."""
        return normalize_ltas(self.ltas)

    @property
    def level_delta(self) -> np.ndarray:
        """Per-band brightness change from the quietest to the loudest pass.

        This is *how the amp breaks up* -- the quantity the three-level probe
        exists to expose, and which a single Welch average over the whole probe
        destroys by folding the levels together.

        It is also cab-invariant by construction: a cab adds the same
        ir_response to every level, so the difference cancels. That means it can
        be carried through the joint (profile, IR) search from the amp render
        alone, exactly like the scalar descriptors.
        """
        if self.level_ltas is None:
            return np.zeros(0)
        return self.level_ltas[-1] - self.level_ltas[0]

    @property
    def scalar_vector(self) -> np.ndarray:
        return np.array([self.scalars[k] for k in SCALAR_KEYS], dtype=float)

    def with_ir(self, ir_db: np.ndarray) -> "Fingerprint":
        """Predict this amp's fingerprint through a cab, without rendering.

        Scalars are carried over unchanged: a linear cab adds no distortion and
        does not meaningfully alter the envelope statistics.
        """
        return Fingerprint(
            ltas=self.ltas + ir_db,
            scalars=dict(self.scalars),
            # Broadcasts across levels; level_delta is unchanged, as it must be.
            level_ltas=None if self.level_ltas is None else self.level_ltas + ir_db,
        )


def fingerprint(x: np.ndarray, sr: int, cfg: InstrumentConfig,
                levels: "list[slice] | None" = None) -> Fingerprint:
    """Fingerprint a rendered signal.

    Pass ``levels`` (from ``probe.level_slices()``) when ``x`` is a multi-level
    probe render, so the per-level spectra are kept instead of being averaged
    into one. Without it the level sweep is wasted.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.mean(axis=1)
    scalars: dict[str, float] = {}
    scalars.update(spectral_shape(x, sr, cfg))
    scalars.update(saturation(x, sr, cfg))
    scalars.update(dynamics(x, sr))

    level_ltas = None
    if levels:
        level_ltas = np.stack([ltas_db(x[s], sr, cfg) for s in levels])

    return Fingerprint(ltas=ltas_db(x, sr, cfg), scalars=scalars, level_ltas=level_ltas)
