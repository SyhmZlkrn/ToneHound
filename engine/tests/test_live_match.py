"""The shared matching core: whitening, ranking, and the index cache.

Two things are being protected here.

*One implementation.* `tonehound.match` exists so the live server and
`tools/try_match.py` rank with the same arithmetic. If they drifted, the app
and the tool would disagree about the same song and there would be no way to
tell which was right, so the aliases the tools import are checked against the
functions they now alias.

*A cache that cannot outlive its renders.* The index costs ~45 s to build, so it
is cached; a cache that survived a re-render would silently rank against
fingerprints of audio that no longer exists, which is far worse than a slow
start.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

from tonehound import match
from tonehound.config import GUITAR, SAMPLE_RATE

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))


def _render(seed: int, seconds: float = 2.0) -> np.ndarray:
    """A signal with its own spectral tilt, so profiles are distinguishable."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    x = rng.normal(size=n)
    # A one-pole tilt whose coefficient varies with the seed gives each fake
    # "profile" a different LTAS, which is what the fingerprint keys on.
    a = 0.2 + 0.7 * ((seed * 37) % 11) / 11.0
    out = np.empty(n)
    acc = 0.0
    for i in range(0, n, 4096):                     # cheap block-wise IIR
        chunk = x[i:i + 4096]
        filtered = np.empty_like(chunk)
        for j, v in enumerate(chunk):
            acc = a * acc + (1.0 - a) * v
            filtered[j] = v - acc
        out[i:i + 4096] = filtered
    return np.tanh(out * 3.0) * 0.5


@pytest.fixture(scope="module")
def render_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    directory = tmp_path_factory.mktemp("renders")
    for seed in range(5):
        np.save(directory / f"fake_profile_{seed}.npy", _render(seed))
    return directory


# -- whitening and distance ------------------------------------------------


def test_whitening_standardises_against_the_corpus_not_the_query() -> None:
    shape = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    scalars = np.array([[0.0], [1.0], [2.0]])
    ref = (match.zscore_reference(shape), match.zscore_reference(scalars))
    whitened = match.whiten(shape, scalars, ref)
    # Each half is scaled by sqrt(weight / n_columns), so the block means are
    # zero and the halves carry the intended share of the vector's energy.
    assert whitened.shape == (3, 3)
    assert np.allclose(whitened.mean(axis=0), 0.0, atol=1e-12)
    shape_energy = float((whitened[:, :2] ** 2).sum())
    scalar_energy = float((whitened[:, 2:] ** 2).sum())
    assert shape_energy == pytest.approx(scalar_energy, rel=1e-9)


def test_a_constant_feature_contributes_nothing_instead_of_infinity() -> None:
    shape = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    scalars = np.array([[1.0], [2.0], [3.0]])
    ref = (match.zscore_reference(shape), match.zscore_reference(scalars))
    whitened = match.whiten(shape, scalars, ref)
    assert np.isfinite(whitened).all()
    assert np.allclose(whitened[:, 1], 0.0)


def test_cosine_distance_is_zero_against_itself_and_two_for_the_opposite() -> None:
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    d = match.cosine_distance(a, a)
    # Not exactly zero: the norms carry a 1e-12 guard against a zero vector, so
    # a unit vector's self-distance lands at ~2e-12 rather than 0.
    assert np.allclose(np.diag(d), 0.0, atol=1e-9)
    assert match.cosine_distance(a[:1], -a[:1])[0, 0] == pytest.approx(2.0)


def test_the_tools_still_share_this_one_implementation() -> None:
    """Two copies of a weak ranking would be two different weak rankings."""
    import stem_test

    assert stem_test.zscore_ref is match.zscore_reference
    assert stem_test.vectors is match.whiten
    assert stem_test.cosine is match.cosine_distance
    assert stem_test.self_retrieval is match.self_retrieval


# -- the index -------------------------------------------------------------


def test_the_index_fingerprints_every_render(render_dir: pathlib.Path) -> None:
    index = match.ProfileIndex.build(render_dir)
    assert len(index) == 5
    assert index.names == [f"fake_profile_{i}" for i in range(5)]
    assert index.shape.shape == (5, GUITAR.n_bands)
    assert np.isfinite(index.matrix).all()


def test_a_render_retrieves_its_own_entry_first(render_dir: pathlib.Path) -> None:
    """The floor the whole matcher stands on: with no degradation at all, a
    profile must rank itself first. If this fails nothing downstream can work."""
    index = match.ProfileIndex.build(render_dir)
    for i, name in enumerate(index.names):
        distances = index.rank(np.load(render_dir / f"{name}.npy"))
        assert int(np.argmin(distances)) == i, name
        assert distances[i] == pytest.approx(0.0, abs=1e-9)


def test_ranking_returns_one_distance_per_profile(render_dir: pathlib.Path) -> None:
    index = match.ProfileIndex.build(render_dir)
    distances = index.rank(_render(99))
    assert distances.shape == (5,)
    assert np.isfinite(distances).all()
    assert (distances >= 0.0).all()


def test_progress_is_reported_for_every_entry(render_dir: pathlib.Path) -> None:
    """A 45-second first build with no progress looks like a hang."""
    seen: list[tuple[int, int]] = []
    match.ProfileIndex.build(render_dir, progress=lambda d, t: seen.append((d, t)))
    assert seen == [(i, 5) for i in range(1, 6)]


def test_an_empty_render_directory_says_so(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        match.ProfileIndex.build(tmp_path)


# -- the cache -------------------------------------------------------------


def test_the_cache_round_trips_and_is_reused(render_dir: pathlib.Path,
                                             tmp_path: pathlib.Path) -> None:
    cache = tmp_path / "index.npz"
    built = match.ProfileIndex.load_or_build(render_dir, cache)
    assert cache.exists()

    calls: list[int] = []
    reloaded = match.ProfileIndex.load_or_build(
        render_dir, cache, progress=lambda d, t: calls.append(d))
    assert not calls, "a warm cache must not re-fingerprint"
    assert reloaded.names == built.names
    assert np.allclose(reloaded.matrix, built.matrix)


def test_changing_the_renders_invalidates_the_cache(render_dir: pathlib.Path,
                                                    tmp_path: pathlib.Path) -> None:
    """A cache that outlived its renders would rank against audio that no
    longer exists, and nothing downstream could detect it."""
    work = tmp_path / "renders"
    work.mkdir()
    for seed in range(3):
        np.save(work / f"p{seed}.npy", _render(seed))
    cache = tmp_path / "index.npz"
    first = match.ProfileIndex.load_or_build(work, cache)

    np.save(work / "p3.npy", _render(7))
    second = match.ProfileIndex.load_or_build(work, cache)
    assert len(second) == len(first) + 1
    assert "p3" in second.names


def test_a_corrupt_cache_rebuilds_rather_than_raising(render_dir: pathlib.Path,
                                                      tmp_path: pathlib.Path) -> None:
    cache = tmp_path / "index.npz"
    cache.write_bytes(b"not an npz at all")
    index = match.ProfileIndex.load_or_build(render_dir, cache)
    assert len(index) == 5


# -- the honesty text ------------------------------------------------------


def test_the_caveat_states_the_measured_accuracy_and_the_confound() -> None:
    """§5 of the protocol: this string is the mitigation for a weak matcher and
    it is the easiest thing to quietly soften while tidying up."""
    text = match.CAVEAT.lower()
    assert "18%" in text
    assert "capturer" in text
    assert "trust your ears" in text
