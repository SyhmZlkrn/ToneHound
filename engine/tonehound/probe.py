"""Reference probe DI.

Every profile in the index is fingerprinted by rendering this exact signal
through it, so the probe must be fixed and must exercise the amp's
nonlinearity: palm mutes for transients, chords for intermodulation, sustained
notes for compression behaviour, and a chromatic run for frequency coverage.

Crucially the whole phrase repeats at three input levels. Amps are nonlinear,
so drive level *is* the timbre -- a single-level probe only samples one slice of
a model's behaviour, and the slope of brightness against input level is one of
the most discriminative features available.

This module synthesises a probe with Karplus-Strong plucked-string synthesis.
That is a *stopgap*: a real recorded DI is planned (see the plan's open items)
because synthesis lacks the pick noise and playing dynamics of real material.
It is deterministic and dependency-free, which is what Phase 0 needs.
"""

import numpy as np

from .config import SAMPLE_RATE

PROBE_LEVELS_DBFS = (-36.0, -24.0, -12.0)
"""Nonlinearity is sampled at these three input levels.

Measured on the dev corpus, the original -18/-12/-6 sweep sat almost entirely
above the amps' compression knee: brightness moved only ~0.88 dB rms across it,
so the level-delta features had little to read. Reaching further down spans the
clean-to-saturated transition, which is where the interesting behaviour is.
"""


def note_hz(midi: int) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _envelope(n: int, sr: int, attack_s: float = 0.003, release_s: float = 0.015) -> np.ndarray:
    """Short ramps so concatenated notes do not click."""
    env = np.ones(n)
    a = min(int(attack_s * sr), n // 2)
    r = min(int(release_s * sr), n // 2)
    if a:
        env[:a] = np.linspace(0.0, 1.0, a)
    if r:
        env[-r:] = np.linspace(1.0, 0.0, r)
    return env


def pluck(freq: float, dur: float, sr: int = SAMPLE_RATE, decay: float = 0.998,
          damping: float = 0.5) -> np.ndarray:
    """Karplus-Strong plucked string.

    Processed a period at a time, which is the same recurrence as the
    sample-wise form but vectorised. ``decay`` sets sustain (low values give a
    palm mute); ``damping`` sets how fast the highs roll off.
    """
    period = max(2, int(round(sr / freq)))
    rng = np.random.default_rng(int(freq * 1000) % (2**31))

    buf = rng.standard_normal(period)
    buf = np.convolve(buf, np.ones(3) / 3.0, mode="same")  # tame the excitation

    n = int(dur * sr)
    blocks, produced = [], 0
    while produced < n:
        blocks.append(buf.copy())
        produced += period
        buf = decay * (damping * buf + (1.0 - damping) * np.roll(buf, 1))

    y = np.concatenate(blocks)[:n]
    return y * _envelope(n, sr)


def _chord(freqs, dur, sr, **kw) -> np.ndarray:
    return sum(pluck(f, dur, sr, **kw) for f in freqs) / len(freqs)


def _silence(dur: float, sr: int) -> np.ndarray:
    return np.zeros(int(dur * sr))


def build_phrase(sr: int = SAMPLE_RATE) -> np.ndarray:
    """One pass of the probe material, normalised to unit peak."""
    parts: list[np.ndarray] = []

    # 1. Palm-muted low-E chugs: transient response and low-end behaviour.
    for _ in range(8):
        parts.append(pluck(note_hz(40), 0.16, sr, decay=0.965, damping=0.62))
        parts.append(_silence(0.05, sr))

    # 2. Open power chords: intermodulation and sag under a heavy load.
    for root in (40, 43, 45):  # E2, G2, A2
        parts.append(_chord([note_hz(root), note_hz(root + 7)], 0.9, sr, decay=0.9985))
        parts.append(_silence(0.08, sr))

    # 3. Single-note lead, mid-neck: midrange voicing.
    for m in (57, 60, 64, 62, 59):
        parts.append(pluck(note_hz(m), 0.35, sr, decay=0.9988, damping=0.45))

    # 4. Clean arpeggio: headroom and the onset of compression.
    for m in (40, 47, 52, 55, 59, 64):
        parts.append(pluck(note_hz(m), 0.28, sr, decay=0.9990, damping=0.40))

    # 5. Chromatic run: full-range coverage.
    for m in range(40, 64, 2):
        parts.append(pluck(note_hz(m), 0.12, sr, decay=0.9985))

    y = np.concatenate(parts)
    return y / (np.abs(y).max() + 1e-12)


def build_probe(sr: int = SAMPLE_RATE, levels=PROBE_LEVELS_DBFS) -> np.ndarray:
    """The full probe: the phrase repeated at each input level."""
    phrase = build_phrase(sr)
    gap = _silence(0.25, sr)
    return np.concatenate([np.concatenate([phrase * 10 ** (db / 20.0), gap]) for db in levels])


def level_slices(sr: int = SAMPLE_RATE, levels=PROBE_LEVELS_DBFS) -> list[slice]:
    """Sample ranges of each level pass, for per-level feature extraction."""
    phrase_n = len(build_phrase(sr))
    gap_n = int(0.25 * sr)
    stride = phrase_n + gap_n
    return [slice(i * stride, i * stride + phrase_n) for i in range(len(levels))]
