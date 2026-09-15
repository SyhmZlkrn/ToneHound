"""Stem separation, and above all what happens when there is no guitar in it.

The interesting logic is not the model, it is the fallback. htdemucs_6s returns
an empty guitar stem on plenty of real material, and the old behaviour -- rank
the raw mix and say nothing -- hands the matcher drums and vocals and presents
the result as a guitar tone. The chain here tries `other` first, and whichever
stem it ended up with is carried on the result so the caller cannot forget to
mention it.

The model itself is exercised by one integration test, skipped when the weights
are not present. Everything else drives a stub separator, because "which stem
did we pick and what did we say about it" needs no neural network.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from tonehound import separate
from tonehound.config import SAMPLE_RATE
from tonehound.separate import Separated, SeparationError, isolate

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / ".cache" / "uvr_models"

needs_model = pytest.mark.skipif(
    not (MODEL_DIR / separate.GUITAR_MODEL_FILE).exists(),
    reason="htdemucs_6s not downloaded")


def _tone(seconds: float, hz: float = 220.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


class StubSeparator:
    """Returns whatever stems the test asked it to."""

    def __init__(self, stems: dict[str, np.ndarray]) -> None:
        self.stems = stems
        self.calls: list[str] = []

    def split(self, x: np.ndarray, sr: int = SAMPLE_RATE,
              tag: str = "mix") -> dict[str, np.ndarray]:
        self.calls.append(tag)
        return dict(self.stems)


# -- the fallback chain ----------------------------------------------------


def test_a_real_guitar_stem_is_used_and_reported_as_such() -> None:
    stub = StubSeparator({"guitar": _tone(2.0, 330.0), "other": _tone(2.0, 110.0)})
    result = isolate(_tone(2.0), separator=stub)
    assert result.stem == "guitar" and result.ok
    assert result.note == "isolated guitar"


def test_a_missing_guitar_stem_falls_back_to_other_not_to_the_mix() -> None:
    stub = StubSeparator({"vocals": _tone(2.0), "other": _tone(2.0, 110.0)})
    result = isolate(_tone(2.0), separator=stub)
    assert result.stem == "other" and not result.ok
    assert "other" in result.note and "may not be only guitar" in result.note


def test_a_silent_guitar_stem_counts_as_no_guitar_stem() -> None:
    stub = StubSeparator({"guitar": np.zeros(SAMPLE_RATE, dtype=np.float32),
                          "other": _tone(2.0, 110.0)})
    assert isolate(_tone(2.0), separator=stub).stem == "other"


def test_a_guitar_stem_that_is_only_rounding_noise_counts_as_absent() -> None:
    stub = StubSeparator({"guitar": _tone(2.0, amp=1e-6),
                          "other": _tone(2.0, 110.0)})
    assert isolate(_tone(2.0), separator=stub).stem == "other"


def test_with_no_usable_stem_the_mix_is_returned_and_flagged_loudly() -> None:
    stub = StubSeparator({"vocals": _tone(2.0), "drums": _tone(2.0)})
    mix = _tone(2.0, 82.0)
    result = isolate(mix, separator=stub)
    assert result.stem == "mix" and not result.ok
    assert "raw mix" in result.note
    assert np.allclose(result.samples, mix, atol=1e-6)


def test_asking_for_a_different_stem_changes_what_counts_as_success() -> None:
    stub = StubSeparator({"bass": _tone(2.0, 55.0), "other": _tone(2.0)})
    result = isolate(_tone(2.0), separator=stub, want="bass", fallbacks=("other",))
    assert result.stem == "bass" and result.ok and result.requested == "bass"


def test_stems_are_dropped_unless_the_caller_asked_to_keep_them() -> None:
    stems = {"guitar": _tone(2.0), "drums": _tone(2.0), "other": _tone(2.0)}
    assert isolate(_tone(2.0), separator=StubSeparator(stems)).stems == {}
    kept = isolate(_tone(2.0), separator=StubSeparator(stems), keep_stems=True)
    assert set(kept.stems) == set(stems)


def test_opposite_polarity_channels_reach_demucs_without_cancellation() -> None:
    tone=_tone(2.0);stereo=np.column_stack((tone,-tone))
    class CheckingSeparator:
        def split(self,x,sr,tag):
            assert x.shape == stereo.shape and np.allclose(x,stereo)
            return {'guitar':x}
    result=isolate(stereo,separator=CheckingSeparator(),preserve_stereo=True)
    assert result.stem == 'guitar' and result.samples.shape == stereo.shape
    assert np.max(np.abs(result.samples)) > .4


def test_isolate_without_a_separator_or_a_model_dir_is_an_error() -> None:
    with pytest.raises(SeparationError) as exc:
        isolate(_tone(2.0))
    assert exc.value.code == "no_separator"


def test_the_result_carries_float32_audio_for_the_media_store() -> None:
    result = isolate(_tone(2.0), separator=StubSeparator({"guitar": _tone(2.0)}))
    assert result.samples.dtype == np.float32


# -- the note the user is shown --------------------------------------------


def test_every_outcome_has_a_sentence_and_they_differ() -> None:
    notes = {Separated(np.zeros(4), stem=s).note for s in ("guitar", "other", "mix")}
    assert len(notes) == 3
    assert all(n and not n.endswith(".") for n in notes)


# -- the real model --------------------------------------------------------


@needs_model
def test_htdemucs_6s_returns_named_stems_for_real_audio() -> None:
    """The one test that proves the backend is wired to real weights."""
    mix = (_tone(4.0, 110.0) + _tone(4.0, 330.0) * 0.5).astype(np.float32)
    result = isolate(mix, separator=separate.Separator(MODEL_DIR),
                     keep_stems=True, tag="test")
    assert set(result.stems) <= set(separate.STEMS)
    assert result.stems, "htdemucs_6s produced no stems at all"
    assert len(result.samples) == pytest.approx(len(mix), rel=0.02)
    assert result.stem in (*separate.STEMS, "mix")
