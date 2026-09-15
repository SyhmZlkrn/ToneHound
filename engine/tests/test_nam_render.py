"""Guard the ``.nam`` reader against the thing it exists to absorb: schema drift.

The reader is hand-rolled because neural-amp-modeler's own loader refuses the
legacy captures in ``assets/dev_profiles``. Hand-rolled means the weight layout
is an assumption, and a wrong layout does not crash -- it renders plausible,
wrong audio and silently poisons the index.

These are the cheap, always-available checks, and their limits are known: exact
weight consumption catches a truncated layout but *not* a permuted one (the
per-layer blocks sum to the same total in any order -- verified by feeding this
suite a file with layer1x1 and the mixer swapped, which passed every test here),
and the behavioural checks below only bound what an amplifier may do. The layout
itself is pinned in ``test_nam_render_parity.py`` against the installed package.
What is checked here:

* exact weight consumption and clean refusal of the schemas this reader will
  not guess at;
* renders that are finite, bounded, length-correct and reproducible;
* an actual nonlinearity, measured as harmonic content that grows with input
  level -- something no linear filter can do, however wrong its coefficients;
* different profiles producing different audio, which catches a reader that
  accidentally ignores most of the file.

Everything profile-driven skips cleanly when the dev profiles are not checked
out.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from tonehound.config import SAMPLE_RATE
from tonehound.nam_render import (
    DEFAULT_SAMPLE_RATE,
    UnsupportedArchitecture,
    WeightCountMismatch,
    load,
)

PROFILE_DIR = Path(__file__).resolve().parents[2] / "assets" / "dev_profiles"
PROFILES = sorted(PROFILE_DIR.glob("*.nam")) if PROFILE_DIR.is_dir() else []

needs_profiles = pytest.mark.skipif(not PROFILES, reason="assets/dev_profiles is absent")
each_profile = pytest.mark.parametrize("path", PROFILES, ids=[p.stem[:40] for p in PROFILES])

# Longer than the 4093-sample receptive field of the legacy captures, so an
# unpadded render still has samples left over.
_N = 6000


@lru_cache(maxsize=None)
def _model(path: Path):
    return load(path)


def _probe(n: int = _N, amplitude: float = 0.5, seed: int = 0) -> np.ndarray:
    """Deterministic broadband stimulus: noise shaped by a decaying envelope."""
    rng = np.random.default_rng(seed)
    env = np.exp(-3.0 * np.arange(n) / n)
    return amplitude * env * rng.standard_normal(n)


def _harmonic_ratio(y: np.ndarray, f0: float, sr: int = SAMPLE_RATE) -> float:
    """Energy in harmonics 2..6 relative to the fundamental."""
    window = np.hanning(len(y))
    power = np.abs(np.fft.rfft(y * window)) ** 2
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)

    def band(centre: float) -> float:
        return power[(freqs > centre - 30.0) & (freqs < centre + 30.0)].sum()

    return sum(band(f0 * k) for k in range(2, 7)) / max(band(f0), 1e-30)


@needs_profiles
@each_profile
def test_every_profile_loads_and_consumes_its_weights(path: Path) -> None:
    model = _model(path)
    with open(path, "r", encoding="utf-8") as fp:
        available = len(json.load(fp)["weights"])
    assert model.weight_count == available
    assert model.architecture
    assert model.receptive_field >= 1


@needs_profiles
@each_profile
def test_render_is_finite_bounded_and_length_correct(path: Path) -> None:
    model = _model(path)
    x = _probe()

    y = model.render(x)
    assert y.shape == x.shape
    assert np.isfinite(y).all()
    # A capture is a guitar amp, not an oscillator: 0.5 in cannot become 10 out.
    assert np.abs(y).max() < 10.0
    assert np.abs(y).max() > 1e-4  # and it is not silence either

    unpadded = model.render(x, pad_start=False)
    assert unpadded.shape[0] == len(x) - model.receptive_field + 1
    # The padded render is the unpadded one preceded by the warm-up region.
    assert np.allclose(y[model.receptive_field - 1 :], unpadded, atol=1e-12)


@needs_profiles
@each_profile
def test_render_is_deterministic(path: Path) -> None:
    model = _model(path)
    x = _probe(n=3000)
    assert np.array_equal(model.render(x), model.render(x))
    assert np.array_equal(load(path).render(x), model.render(x))


@needs_profiles
@each_profile
def test_sine_produces_harmonics(path: Path) -> None:
    """A pure sine must come out with harmonics on it.

    Two shapes of evidence, because the corpus contains both kinds of capture: a
    high-gain amp is already saturated at -26 dBFS, so its harmonic ratio is
    large but barely moves with level, while a clean Fender capture is nearly
    linear until driven, so its ratio is small but climbs by orders of magnitude.
    A linear filter -- which is what a mis-read weight layout most easily
    degenerates into -- can produce neither.
    """
    model = _model(path)
    t = np.arange(_N) / SAMPLE_RATE
    sine = np.sin(2.0 * np.pi * 110.0 * t)

    hot = _harmonic_ratio(model.render(0.9 * sine), 110.0)
    quiet = _harmonic_ratio(model.render(0.05 * sine), 110.0)
    assert hot > 1e-4, "no harmonic content at all"
    assert hot > 2e-3 or hot > 20.0 * quiet, f"looks linear: {hot=}, {quiet=}"


@needs_profiles
def test_different_profiles_render_differently() -> None:
    assert len(PROFILES) >= 2
    x = _probe()
    renders = [_model(p).render(x) for p in PROFILES[:4]]
    for i, a in enumerate(renders):
        for b in renders[i + 1 :]:
            difference = np.sqrt(np.mean((a - b) ** 2))
            reference = max(np.sqrt(np.mean(a**2)), np.sqrt(np.mean(b**2)))
            assert difference > 0.01 * reference


@needs_profiles
def test_legacy_captures_carry_no_metadata_and_default_their_rate() -> None:
    legacy = [p for p in PROFILES if not p.name.startswith("_")]
    assert legacy, "expected the legacy dev profiles"
    for path in legacy:
        model = _model(path)
        assert model.architecture == "WaveNet"
        assert model.version == "0.5.0"
        assert model.metadata == {}
        assert model.sample_rate == DEFAULT_SAMPLE_RATE


def test_unknown_architecture_raises() -> None:
    with pytest.raises(UnsupportedArchitecture) as excinfo:
        load({"version": "0.7.0", "architecture": "A2", "config": {}, "weights": []})
    assert excinfo.value.architecture == "A2"
    assert "A2" in str(excinfo.value)


@needs_profiles
def test_trailing_junk_weights_raise() -> None:
    with open(PROFILES[0], "r", encoding="utf-8") as fp:
        data = json.load(fp)
    consumed = len(data["weights"])
    data["weights"] = data["weights"] + [0.0] * 5
    with pytest.raises(WeightCountMismatch) as excinfo:
        load(data)
    assert excinfo.value.consumed == consumed
    assert excinfo.value.available == consumed + 5


@needs_profiles
def test_truncated_weights_raise() -> None:
    with open(PROFILES[0], "r", encoding="utf-8") as fp:
        data = json.load(fp)
    data["weights"] = data["weights"][:-20]
    with pytest.raises(WeightCountMismatch):
        load(data)


@needs_profiles
def test_head_scale_disagreement_is_reported_not_refused() -> None:
    """The trailing weight wins, and the disagreement is surfaced.

    An export hook that compensates output level rewrites ``weights[-1]`` without
    touching the config copy, so the two legitimately differ on files the plugin
    plays fine. The reference reader adopts the weight silently; this one adopts
    it and says so, so the indexer can log it instead of dropping the capture.
    """
    with open(PROFILES[0], "r", encoding="utf-8") as fp:
        data = json.load(fp)
    weight_scale = data["weights"][-1]
    data["config"]["head_scale"] = 0.5

    model = load(data)
    assert model.head_scale_config_mismatch == 0.5
    assert np.allclose(model.render(_probe(n=5000)),
                       _model(PROFILES[0]).render(_probe(n=5000)), atol=1e-12)
    assert model._net.head_scale == weight_scale


@needs_profiles
def test_float32_rounding_of_a_large_head_scale_is_not_a_mismatch() -> None:
    """The tolerance has to be relative: the weight is a float32 round of the config.

    At head_scale ~1234 the float32 ulp alone exceeds 1e-5, so an absolute
    tolerance would refuse a file the reference implementation renders happily.
    """
    with open(PROFILES[0], "r", encoding="utf-8") as fp:
        data = json.load(fp)
    data["config"]["head_scale"] = 1234.5678
    data["weights"][-1] = float(np.float32(1234.5678))

    model = load(data)
    assert model.head_scale_config_mismatch is None


@needs_profiles
def test_condition_dsp_is_refused_rather_than_ignored() -> None:
    """Its weights live in a nested blob, so the outer count check cannot catch it."""
    with open(PROFILES[0], "r", encoding="utf-8") as fp:
        data = json.load(fp)
    data["config"]["condition_dsp"] = {"architecture": "WaveNet", "config": {}, "weights": []}
    with pytest.raises(UnsupportedArchitecture, match="condition_dsp"):
        load(data)


@needs_profiles
def test_active_film_is_refused() -> None:
    with open(PROFILES[0], "r", encoding="utf-8") as fp:
        data = json.load(fp)
    data["config"]["layers"][0]["conv_pre_film"] = {"active": True, "shift": True, "groups": 1}
    with pytest.raises(UnsupportedArchitecture, match="FiLM"):
        load(data)
