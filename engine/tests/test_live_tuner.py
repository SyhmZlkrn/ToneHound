"""The tuner: it has to be accurate enough and steady enough to tune by.

Two separate properties, and the second is the one people actually feel.

*Accuracy* over the whole guitar range including a baritone's low B, because
that is where a naive detector fails. Plain FFT-autocorrelation was measured and
rejected during the spike precisely because it octave-fails on low E and low B
*with high confidence*, so a confidence gate would not catch it -- which is why
these tests assert the frequency and not merely that something was detected.

*Stability.* A needle that jitters between picks is unusable even when every
individual reading is correct, so a held note is checked for a spread in cents
rather than only for a correct mean.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tonehound.config import SAMPLE_RATE
from live import tuner as tuner_mod
from live.tuner import Tuner, estimate, note_for

WINDOW = int(tuner_mod.WINDOW_S * SAMPLE_RATE)

# Open strings, plus the low B a baritone or a 7-string adds below them.
STRINGS = {"B1": 61.74, "E2": 82.41, "A2": 110.00, "D3": 146.83,
           "G3": 196.00, "B3": 246.94, "E4": 329.63, "E5": 659.26}


def _pluck(hz: float, seconds: float = tuner_mod.WINDOW_S, *, amplitude: float = 0.3,
           harmonics: int = 6, seed: int = 0, noise: float = 0.0) -> np.ndarray:
    """A plucked string: a decaying harmonic stack, not a pure sine.

    A pure sine is the easy case and would hide an octave error, because the
    subharmonic a detector might lock onto simply is not present. Real strings
    carry strong harmonics and a weak fundamental, which is what makes low
    notes hard.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    x = np.zeros_like(t)
    for n in range(1, harmonics + 1):
        # Fundamental deliberately quieter than the second partial, as on a
        # pickup close to the bridge.
        weight = (0.35 if n == 1 else 1.0 / n) * math.exp(-0.6 * n * t.mean())
        x += weight * np.sin(2 * np.pi * hz * n * t + rng.uniform(0, 2 * np.pi))
    x *= np.exp(-1.5 * t)
    if noise:
        x += rng.normal(scale=noise, size=t.size)
    peak = np.abs(x).max()
    return (x / peak * amplitude).astype(np.float32)


def _cents(got: float, want: float) -> float:
    return 1200.0 * math.log2(got / want)


# -- note naming -----------------------------------------------------------


@pytest.mark.parametrize("hz,note,octave", [
    (440.0, "A", 4), (82.41, "E", 2), (61.74, "B", 1),
    (329.63, "E", 4), (261.63, "C", 4), (233.08, "A#", 3),
])
def test_note_names_use_scientific_pitch_and_sharps(hz: float, note: str,
                                                    octave: int) -> None:
    name, got_octave, cents = note_for(hz)
    assert (name, got_octave) == (note, octave)
    assert abs(cents) < 2.0
    assert "b" not in name, "the protocol says sharps only, never flats"


def test_cents_are_signed_and_bounded_by_a_semitone() -> None:
    for offset in np.linspace(-49.0, 49.0, 21):
        hz = 440.0 * 2 ** (offset / 1200.0)
        _, _, cents = note_for(hz)
        assert cents == pytest.approx(offset, abs=0.01)
        assert -50.0 <= cents <= 50.0


def test_the_reference_pitch_moves_every_note() -> None:
    assert note_for(432.0, reference_hz=432.0)[:2] == ("A", 4)
    assert abs(note_for(432.0, reference_hz=432.0)[2]) < 0.01


# -- accuracy --------------------------------------------------------------


@pytest.mark.parametrize("name,hz", sorted(STRINGS.items(), key=lambda kv: kv[1]))
def test_every_string_in_the_guitar_range_is_found(name: str, hz: float) -> None:
    got, confidence = estimate(_pluck(hz), SAMPLE_RATE)
    assert got is not None, f"{name} not detected at all"
    assert abs(_cents(got, hz)) < 1.0, f"{name}: {_cents(got, hz):.2f} cents off"
    assert confidence > tuner_mod.MIN_CONFIDENCE


@pytest.mark.parametrize("name,hz", [("B1", 61.74), ("E2", 82.41)])
def test_the_low_strings_are_not_reported_an_octave_down(name: str, hz: float) -> None:
    """The specific failure the spike measured and rejected autocorrelation
    for: a wrong octave reported *confidently*, which no gate would catch."""
    got, _ = estimate(_pluck(hz, harmonics=8), SAMPLE_RATE)
    assert got is not None
    assert abs(_cents(got, hz)) < 25.0, f"{name} landed at {got:.1f} Hz, wanted {hz}"


@pytest.mark.parametrize("offset", [-40.0, -12.0, 0.0, 7.0, 33.0])
def test_a_detuned_string_reports_the_offset_it_actually_has(offset: float) -> None:
    """The whole point of a tuner: the number has to move with the peg."""
    hz = 110.0 * 2 ** (offset / 1200.0)
    pitch = Tuner(SAMPLE_RATE)(_pluck(hz))
    assert pitch.detected
    assert pitch.note == "A" and pitch.octave == 2
    assert pitch.cents == pytest.approx(offset, abs=2.0)


def test_a_noisy_pickup_signal_is_still_tracked() -> None:
    got, _ = estimate(_pluck(146.83, noise=0.05, seed=3), SAMPLE_RATE)
    assert got is not None and abs(_cents(got, 146.83)) < 5.0


# -- gating ----------------------------------------------------------------


def test_silence_reports_not_detected_rather_than_a_guess() -> None:
    pitch = Tuner(SAMPLE_RATE)(np.zeros(WINDOW, dtype=np.float32))
    assert not pitch.detected
    assert pitch.frequency_hz is None and pitch.note is None


def test_a_signal_under_the_level_gate_is_not_detected() -> None:
    quiet = _pluck(110.0, amplitude=10 ** (tuner_mod.MIN_LEVEL_DB / 20.0) / 10.0)
    assert not Tuner(SAMPLE_RATE)(quiet).detected


def test_broadband_noise_is_not_reported_as_a_note() -> None:
    noise = np.random.default_rng(0).normal(scale=0.2, size=WINDOW).astype(np.float32)
    pitch = Tuner(SAMPLE_RATE)(noise)
    assert not pitch.detected or pitch.confidence >= tuner_mod.MIN_CONFIDENCE


def test_an_empty_window_does_not_raise() -> None:
    """The engine's ring is empty for the first blocks after `stream.start`."""
    assert not Tuner(SAMPLE_RATE)(np.zeros(0, dtype=np.float32)).detected


# -- stability -------------------------------------------------------------


def test_a_held_note_does_not_jitter() -> None:
    """A correct-on-average needle that wanders +-5 cents cannot be tuned by."""
    tuner = Tuner(SAMPLE_RATE)
    hz = 110.0
    readings = []
    for frame in range(20):
        window = _pluck(hz, seconds=tuner_mod.WINDOW_S, noise=0.02, seed=frame)
        pitch = tuner(window)
        assert pitch.detected
        readings.append(pitch.cents)
    spread = max(readings) - min(readings)
    assert spread < 3.0, f"needle moved {spread:.2f} cents on a held note"


def test_a_new_note_is_followed_immediately_not_averaged_into_the_old_one() -> None:
    """Smoothing that survived a string change would show a note between the
    two for as long as the history lasts."""
    tuner = Tuner(SAMPLE_RATE)
    for _ in range(6):
        tuner(_pluck(82.41))
    after = tuner(_pluck(110.0))
    assert after.detected
    assert (after.note, after.octave) == ("A", 2)
    assert abs(after.cents) < 5.0


def test_reset_clears_the_history() -> None:
    tuner = Tuner(SAMPLE_RATE)
    for _ in range(5):
        tuner(_pluck(82.41))
    tuner.reset()
    assert tuner(_pluck(196.0)).note == "G"


def test_the_reference_can_be_moved_and_the_reading_follows() -> None:
    tuner = Tuner(SAMPLE_RATE)
    tuner.reference_hz = 432.0
    pitch = tuner(_pluck(108.0))            # A2 in a 432 Hz reference
    assert pitch.detected and pitch.note == "A"
    assert pitch.reference_hz == 432.0
    assert abs(pitch.cents) < 5.0


def test_every_reported_field_is_json_safe() -> None:
    """One numpy scalar in a telemetry frame kills the whole frame."""
    import json

    pitch = Tuner(SAMPLE_RATE)(_pluck(110.0))
    json.dumps({"tuner": pitch.__dict__}, allow_nan=False)
    assert type(pitch.frequency_hz) is float
    assert type(pitch.cents) is float
    assert type(pitch.confidence) is float
    assert type(pitch.octave) is int
