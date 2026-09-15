"""The real-DI probe must have the level structure its slices claim.

``probe_from_di`` lays the take out at three input levels and ``slices_from_di``
says where each one starts. Nothing checks that they agree, and a disagreement
is silent: the fingerprint would read a -12 dBFS pass as though it were the
-36 dBFS one, and the level-delta features -- the whole reason the probe is
multi-level -- would measure the wrong thing while still producing plausible
numbers. The two functions take the DI length by different routes (one from the
array, one from an int the caller passes), so this is exactly the kind of pair
that drifts apart.

Also covers the render-cache probe stamp, for the same reason: a synthetic-probe
cache reused by a real-DI run fails silently and wrongly.
"""

import json
import pathlib
import sys

import numpy as np
import pytest

from tonehound import real_di
from tonehound.config import SAMPLE_RATE
from tonehound.probe import PROBE_LEVELS_DBFS

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

import probe_source  # noqa: E402

DI_DIR = pathlib.Path(__file__).resolve().parents[2] / "assets" / "user_di"

# The claim is exact arithmetic (a scalar gain applied to one buffer), so the
# tolerance only has to cover float64 round-off. A "fraction of a dB" bar would
# pass while hiding a real off-by-one in the slice arithmetic.
MAX_LEVEL_ERR_DB = 1e-6


def _rms_db(x: np.ndarray) -> float:
    return 20.0 * np.log10(np.sqrt(np.mean(x**2)) + 1e-30)


def _synthetic_take(seconds: float = 2.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """A DI-shaped stand-in: peak-normalised, non-constant, deterministic.

    Runs even where the recorded takes are absent, so the structural claim is
    never silently unverified.
    """
    t = np.arange(int(seconds * sr)) / sr
    x = np.sin(2 * np.pi * 82.4 * t) * (0.3 + 0.7 * np.sin(2 * np.pi * 1.7 * t) ** 2)
    return x / np.abs(x).max()


def _takes() -> list[tuple[str, np.ndarray]]:
    out = [("synthetic stand-in", _synthetic_take())]
    for path in sorted(DI_DIR.glob("*.wav")):
        out.append((path.stem, real_di.load_di(path, SAMPLE_RATE, max_seconds=4.0)))
    return out


TAKES = _takes()
TAKE_IDS = [n for n, _ in TAKES]


@pytest.mark.parametrize("name,di", TAKES, ids=TAKE_IDS)
def test_each_slice_lands_on_its_declared_level(name, di):
    probe = real_di.probe_from_di(di, SAMPLE_RATE)
    slices = real_di.slices_from_di(len(di), SAMPLE_RATE)
    assert len(slices) == len(PROBE_LEVELS_DBFS)

    base = _rms_db(di)
    for sl, db in zip(slices, PROBE_LEVELS_DBFS):
        seg = probe[sl]
        assert len(seg) == len(di), f"{name}: slice length {len(seg)} != take length {len(di)}"
        err = abs((_rms_db(seg) - base) - db)
        assert err < MAX_LEVEL_ERR_DB, f"{name}: pass at {db} dBFS is off by {err:.6f} dB"
        # Level alone would pass if the slice were shifted onto an identically
        # scaled copy; identity to the gained take pins the offset too.
        assert np.allclose(seg, di * 10 ** (db / 20.0), rtol=0, atol=1e-12)


@pytest.mark.parametrize("name,di", TAKES, ids=TAKE_IDS)
def test_gaps_between_passes_are_silent(name, di):
    probe = real_di.probe_from_di(di, SAMPLE_RATE)
    slices = real_di.slices_from_di(len(di), SAMPLE_RATE)
    gap_n = int(real_di.GAP_S * SAMPLE_RATE)

    assert len(probe) == len(PROBE_LEVELS_DBFS) * (len(di) + gap_n)
    for sl in slices:
        gap = probe[sl.stop:sl.stop + gap_n]
        assert len(gap) == gap_n
        assert not gap.any(), f"{name}: gap after a pass is not silent"


def test_probe_levels_are_ordered_and_distinct():
    """A slice mapping is only meaningful if the passes differ in level."""
    di = _synthetic_take(0.5)
    probe = real_di.probe_from_di(di, SAMPLE_RATE)
    rms = [_rms_db(probe[sl]) for sl in real_di.slices_from_di(len(di), SAMPLE_RATE)]
    assert rms == sorted(rms)
    assert min(np.diff(rms)) > 6.0


# --------------------------------------------------------------------------
# Cache binding
# --------------------------------------------------------------------------


def _fake_source(label, audio, synthetic):
    return probe_source.ProbeSource(label, audio, [slice(0, len(audio))], (0.0,),
                                    SAMPLE_RATE, synthetic)


def test_probe_digest_tracks_the_audio():
    a = _fake_source("a", np.arange(64, dtype=float), True)
    b = _fake_source("b", np.arange(64, dtype=float), False)
    c = _fake_source("c", np.arange(64, dtype=float) + 1.0, True)
    assert a.digest == b.digest      # identity is the signal, not the label
    assert a.digest != c.digest


def test_cache_stamped_by_one_probe_is_refused_by_another(tmp_path):
    synth = _fake_source("synthetic", np.arange(64, dtype=float), True)
    di = _fake_source("real DI", np.arange(64, dtype=float) + 1.0, False)

    probe_source.check_cache(tmp_path, synth, write=True)
    assert json.loads((tmp_path / probe_source.STAMP_NAME).read_text())["digest"] == synth.digest

    probe_source.check_cache(tmp_path, synth, write=False)   # same probe: fine
    with pytest.raises(SystemExit):
        probe_source.check_cache(tmp_path, di, write=False)


def test_unstamped_cache_is_synthetic_only(tmp_path):
    """An unstamped cache predates stamping, so it can only be synthetic."""
    np.save(tmp_path / "profile.npy", np.zeros(8))
    probe_source.check_cache(tmp_path, _fake_source("synthetic", np.zeros(4), True),
                             write=False)
    with pytest.raises(SystemExit):
        probe_source.check_cache(tmp_path, _fake_source("real DI", np.ones(4), False),
                                 write=False)


def test_read_only_tools_never_stamp(tmp_path):
    np.save(tmp_path / "profile.npy", np.zeros(8))
    probe_source.check_cache(tmp_path, _fake_source("synthetic", np.zeros(4), True),
                             write=False)
    assert not (tmp_path / probe_source.STAMP_NAME).exists()
