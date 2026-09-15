"""Pitch detection good enough to actually tune a guitar with.

Autocorrelation was tried first and rejected. On a low E (82.4 Hz) and a low B
(61.7 Hz baritone) it reports the octave above, and it does so with confidence
0.77-0.86 -- so a confidence gate does not catch it, and the player is told
they are in tune while they are an octave out. YIN's cumulative-mean
normalisation exists precisely to kill that failure: dividing by the running
mean of the difference function penalises the first (sub-harmonic) dip far less
than the later ones, so the true period wins instead of its multiples.

Three choices below are worth their comments.

*Full sample rate, no decimation.* Decimating to 12 kHz would make the search
4x cheaper, but a tuner lives or dies on the last few cents and the parabolic
interpolation is only as good as the sample spacing under it. At 48 kHz one
evaluation costs about 0.6 ms, and it runs at 20 Hz on a worker thread -- about
1% of one core. There is nothing to buy.

*A short median on top of plain YIN.* Raw YIN is accurate but its last digit
moves frame to frame, and a needle that will not settle is a needle nobody
trusts. A five-frame median (250 ms at the 20 Hz telemetry rate) holds it
still. The history is thrown away the moment the reading moves more than a
quarter tone, so a new note is instant and only a *held* note is smoothed --
lag on the attack would be worse than the jitter it fixes.

*Silence is reported as silence.* Below about -45 dBFS, and below the
confidence floor, ``detected`` is false rather than a stale reading. Holding
the last value is the UI's job (it has the 500 ms hold), not the detector's --
if the detector lies, nothing above it can tell.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

# Sharps only, never flats: a guitarist reads "F#", and showing "Gb" for the
# same fret in a different key would be a bug report.
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# The floor has to clear a flat 7-string drop A (55 Hz) with room to spare --
# a tuner that cannot see the string you are trying to tune up to pitch is
# useless exactly when it is needed. The ceiling is above E5; nothing higher on
# a guitar sends a fundamental the pickup actually captures.
MIN_HZ = 50.0
MAX_HZ = 900.0

WINDOW_S = 0.085
"""85 ms: three periods of a 55 Hz fundamental, and short enough that the
reading tracks a re-picked note rather than averaging over two of them."""

THRESHOLD = 0.15
"""YIN's absolute threshold on the normalised difference. Cheriet's paper uses
0.10-0.15; a distorted guitar signal is noisier than the speech YIN was tuned
on, so the looser end keeps low notes from dropping out mid-decay."""

MIN_CONFIDENCE = 0.55
MIN_LEVEL_DB = -45.0


@dataclass(frozen=True)
class Pitch:
    """One reading. ``detected`` false means the other fields are meaningless."""

    detected: bool
    frequency_hz: float | None
    note: str | None
    octave: int | None
    cents: float | None
    confidence: float
    reference_hz: float


NOT_DETECTED = Pitch(False, None, None, None, None, 0.0, 440.0)


def note_for(frequency_hz: float, reference_hz: float = 440.0) -> tuple[str, int, float]:
    """Nearest note name, octave and signed cents offset.

    Scientific pitch notation, so a guitar's low E is ``E2`` and A440 is ``A4``.
    """
    midi = 69.0 + 12.0 * math.log2(frequency_hz / reference_hz)
    nearest = int(round(midi))
    cents = (midi - nearest) * 100.0
    return NOTE_NAMES[nearest % 12], nearest // 12 - 1, cents


def _difference(x: np.ndarray, tau_max: int) -> np.ndarray:
    """YIN's squared-difference function d(tau), computed through one FFT.

    d(tau) = sum_j (x[j] - x[j+tau])^2 over a window of ``w`` samples expands to
    power(0) + power(tau) - 2 * autocorr(tau), so the O(w * tau_max) double loop
    becomes two cumulative sums and one correlation. On an 85 ms window that is
    the difference between 4 ms and 0.2 ms.
    """
    w = len(x) - tau_max
    # power[k] is the energy in x[:k], so the energy of any window is one
    # subtraction and every lag's term comes out of the same array.
    power = np.concatenate([[0.0], np.cumsum(x * x)])

    n = 1 << int(np.ceil(np.log2(len(x) + w)))
    # Zero-padded to n >= len(x) + w so the circular correlation cannot wrap.
    corr = np.fft.irfft(np.fft.rfft(x, n) * np.conj(np.fft.rfft(x[:w], n)), n)[:tau_max + 1]

    lags = np.arange(tau_max + 1)
    return (power[w] - power[0]) + (power[lags + w] - power[lags]) - 2.0 * corr


def _cumulative_mean(d: np.ndarray) -> np.ndarray:
    """d'(tau): each dip judged against the average dip so far.

    This one line is the whole reason YIN does not octave-fail where plain
    autocorrelation does. A sub-harmonic at 2*tau produces a dip of similar
    depth to the true period's, but by then the running mean has risen, so the
    true (earlier) period scores lower.
    """
    out = np.empty_like(d)
    out[0] = 1.0
    running = np.cumsum(d[1:])
    taus = np.arange(1, len(d))
    out[1:] = d[1:] * taus / np.maximum(running, 1e-20)
    return out


def _parabolic(d: np.ndarray, tau: int) -> float:
    """Sub-sample the minimum. Without this the resolution at 330 Hz is 22 cents."""
    if tau <= 0 or tau >= len(d) - 1:
        return float(tau)
    a, b, c = d[tau - 1], d[tau], d[tau + 1]
    denom = 2.0 * (a - 2.0 * b + c)
    if abs(denom) < 1e-20:
        return float(tau)
    return float(tau) + float((a - c) / denom)


def estimate(x: np.ndarray, sample_rate: int,
             min_hz: float = MIN_HZ, max_hz: float = MAX_HZ,
             threshold: float = THRESHOLD) -> tuple[float | None, float]:
    """Raw YIN over one window. Returns ``(frequency or None, confidence)``."""
    tau_min = max(2, int(sample_rate / max_hz))
    tau_max = int(sample_rate / min_hz) + 1
    if len(x) < 2 * tau_max:
        return None, 0.0

    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()  # a DC offset shifts every lag's difference equally
    d = _cumulative_mean(_difference(x, tau_max))

    window = d[tau_min:tau_max + 1]
    below = np.flatnonzero(window < threshold)
    if below.size:
        # First dip under the threshold, walked down to its local minimum: YIN
        # takes the earliest acceptable period, not the globally deepest, which
        # is what stops it locking onto an octave-down multiple.
        tau = tau_min + int(below[0])
        while tau + 1 <= tau_max and d[tau + 1] < d[tau]:
            tau += 1
    else:
        tau = tau_min + int(np.argmin(window))

    value = float(d[tau])
    confidence = float(np.clip(1.0 - value, 0.0, 1.0))
    period = _parabolic(d, tau)
    if period <= 0:
        return None, 0.0
    frequency = sample_rate / period
    if not (min_hz <= frequency <= max_hz):
        return None, confidence
    return frequency, confidence


class Tuner:
    """Stateful wrapper: level gating, octave guard, and hold-steady smoothing."""

    def __init__(self, sample_rate: int, reference_hz: float = 440.0,
                 smoothing: int = 5) -> None:
        self.sample_rate = sample_rate
        self.reference_hz = reference_hz
        self._recent: deque[float] = deque(maxlen=smoothing)

    def reset(self) -> None:
        self._recent.clear()

    def __call__(self, window: np.ndarray) -> Pitch:
        if window.size == 0:
            self.reset()
            return self._silence(0.0)

        rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))
        level_db = 20.0 * math.log10(rms + 1e-12)
        if level_db < MIN_LEVEL_DB:
            self.reset()
            return self._silence(0.0)

        frequency, confidence = estimate(window, self.sample_rate)
        if frequency is None or confidence < MIN_CONFIDENCE:
            self.reset()
            return self._silence(confidence)

        frequency = self._smooth(frequency)
        note, octave, cents = note_for(frequency, self.reference_hz)
        return Pitch(True, frequency, note, octave, cents, confidence, self.reference_hz)

    def _silence(self, confidence: float) -> Pitch:
        return Pitch(False, None, None, None, None, confidence, self.reference_hz)

    def _smooth(self, frequency: float) -> float:
        """Median of recent frames while the note holds; instant on a change.

        Anything more than a quarter tone away is a new note, not jitter, so
        the history is thrown out rather than averaged across the change.
        """
        if self._recent:
            reference = float(np.median(self._recent))
            if abs(1200.0 * math.log2(frequency / reference)) > 50.0:
                self._recent.clear()
        self._recent.append(frequency)
        return float(np.median(self._recent))
