"""The library index: centring, ranking, and a cache that stays honest.

Centring is tested for what it actually does, which is narrower than it first
looked. Raw MERT vectors sit in a narrow cone -- different amps score 0.78
against each other, 0.94 once separation is in the chain -- so raw similarities
are useless as something to show a user. Subtracting the index mean fixes *that*.
It does not improve retrieval: `tools/gate_mert.py` scores both and they come out
level on clean queries, with raw slightly ahead once separation is involved. So
the tests here pin the spread it produces and the fact that it can be switched
off, and deliberately do not assert that it ranks better.

The embedder is replaced with a deterministic stand-in almost everywhere: these
tests are about bookkeeping -- which vector belongs to which file, what
invalidates a cache, what happens when a profile will not render -- and none of
that needs a 1.3 GB model to be exercised.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from tonehound import tone_index
from tonehound.config import SAMPLE_RATE
from tonehound.tone_index import IndexError_, ToneEntry, ToneIndex, scan

DIM = 32


class FakeEmbedder:
    """Maps a signal to a vector by its mean and variance, deterministically."""

    class cfg:  # noqa: N801  (mirrors MertEmbedder.cfg.key())
        @staticmethod
        def key() -> str:
            return "fake"

    dim = DIM

    def embed(self, x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        rng = np.random.default_rng(abs(int(np.sum(np.abs(x)) * 1e4)) % (2**31))
        v = rng.standard_normal(DIM)
        return (v / np.linalg.norm(v)).astype(np.float32)


def _nam_bytes(gain: float) -> bytes:
    """A minimal linear `.nam`: one weight, so it renders as a gain."""
    return json.dumps({
        "version": "0.5.2", "architecture": "Linear",
        "config": {"receptive_field": 1, "input_channels": 1, "bias": False},
        "weights": [gain], "metadata": {},
    }).encode()


@pytest.fixture()
def library(tmp_path: pathlib.Path) -> pathlib.Path:
    directory = tmp_path / "profiles"
    directory.mkdir()
    for i, gain in enumerate((0.2, 0.5, 0.9, 1.4), 1):
        (directory / f"amp {i}.nam").write_bytes(_nam_bytes(gain))
    (directory / "t3k-4211-9987 Fetched Amp.nam").write_bytes(_nam_bytes(0.7))
    return directory


@pytest.fixture()
def di() -> np.ndarray:
    t = np.arange(int(3.0 * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.4 * np.sin(2 * np.pi * 110.0 * t)).astype(np.float32)


def _vectors(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def _entries(n: int) -> list[ToneEntry]:
    return [ToneEntry(key=f"k{i}", name=f"amp {i}", path=pathlib.Path(f"{i}.nam"))
            for i in range(n)]


# -- scanning --------------------------------------------------------------


def test_scanning_finds_every_profile_and_reads_its_content_hash(
        library: pathlib.Path) -> None:
    entries = scan(library)
    assert len(entries) == 5
    assert len({e.key for e in entries}) == 5
    assert all(len(e.key) == 16 for e in entries)


def test_a_fetched_profile_keeps_its_tone_id_and_a_local_one_has_none(
        library: pathlib.Path) -> None:
    by_name = {e.name: e for e in scan(library)}
    fetched = by_name["t3k-4211-9987 Fetched Amp"]
    assert (fetched.source, fetched.tone_id, fetched.model_id) == ("tone3000", 4211, 9987)
    assert fetched.tone3000_url == "https://www.tone3000.com/tones/4211"
    assert by_name["amp 1"].source == "local"
    assert by_name["amp 1"].tone_id is None


def test_editing_a_profile_gives_it_a_different_key(library: pathlib.Path) -> None:
    before = {e.name: e.key for e in scan(library)}
    (library / "amp 1.nam").write_bytes(_nam_bytes(0.25))
    assert scan(library)[0].key != before["amp 1"]


# -- centring and ranking --------------------------------------------------


def test_centring_pulls_an_anisotropic_cone_apart() -> None:
    """Vectors crowded into a cone must come apart once the mean is removed.

    This is about the *scores*, not the order -- see the module docstring.
    """
    cone = _vectors(24) * 0.1 + np.full(DIM, 1.0, dtype=np.float32)
    cone /= np.linalg.norm(cone, axis=1, keepdims=True)
    index = ToneIndex(entries=_entries(24), vectors=cone)

    raw = cone @ cone.T
    centred = index.matrix @ index.matrix.T
    off = ~np.eye(24, dtype=bool)
    assert raw[off].mean() > 0.9              # the cone: everything looks alike
    assert centred[off].std() > raw[off].std() * 3


def test_a_profile_is_its_own_best_match() -> None:
    vectors = _vectors(12)
    index = ToneIndex(entries=_entries(12), vectors=vectors)
    for i in range(12):
        assert index.search(vectors[i], top=1)[0].entry.name == f"amp {i}"


def test_search_returns_the_requested_count_ranked_best_first() -> None:
    index = ToneIndex(entries=_entries(20), vectors=_vectors(20))
    matches = index.search(_vectors(1, seed=7)[0], top=5)
    assert [m.rank for m in matches] == [1, 2, 3, 4, 5]
    sims = [m.similarity for m in matches]
    assert sims == sorted(sims, reverse=True)


def test_distance_is_the_complement_of_similarity() -> None:
    index = ToneIndex(entries=_entries(6), vectors=_vectors(6))
    match = index.search(_vectors(1, seed=3)[0], top=1)[0]
    assert match.distance == pytest.approx(1.0 - match.similarity)


def test_the_percentile_says_how_much_of_the_index_a_match_beat() -> None:
    index = ToneIndex(entries=_entries(20), vectors=_vectors(20))
    matches = index.search(_vectors(1, seed=11)[0], top=20)
    assert matches[0].percentile > matches[-1].percentile
    assert 0.0 < matches[-1].percentile < matches[0].percentile <= 1.0


def test_asking_for_more_than_the_index_holds_returns_what_there_is() -> None:
    index = ToneIndex(entries=_entries(3), vectors=_vectors(3))
    assert len(index.search(_vectors(1, seed=1)[0], top=50)) == 3


def test_entries_and_vectors_that_disagree_are_refused_at_construction() -> None:
    with pytest.raises(IndexError_) as exc:
        ToneIndex(entries=_entries(3), vectors=_vectors(5))
    assert exc.value.code == "shape_mismatch"


# -- building --------------------------------------------------------------


def test_building_renders_and_embeds_every_profile(library: pathlib.Path,
                                                   di: np.ndarray,
                                                   tmp_path: pathlib.Path) -> None:
    index = ToneIndex.build(scan(library), di, embedder=FakeEmbedder(),
                            seconds=2.0, work_dir=tmp_path)
    assert len(index) == 5
    assert index.vectors.shape == (5, DIM)
    assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1.0, atol=1e-5)


def test_building_an_empty_library_says_so(di: np.ndarray) -> None:
    with pytest.raises(IndexError_) as exc:
        ToneIndex.build([], di, embedder=FakeEmbedder())
    assert exc.value.code == "empty"


def test_a_profile_that_cannot_be_embedded_is_dropped_not_zero_filled(
        library: pathlib.Path, di: np.ndarray, tmp_path: pathlib.Path) -> None:
    """A zero row would drag the centre, and the centre is shared by every rank."""
    class Picky(FakeEmbedder):
        def __init__(self) -> None:
            self.seen = 0

        def embed(self, x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
            self.seen += 1
            if self.seen == 2:
                raise RuntimeError("this render is unembeddable")
            return super().embed(x, sr)

    entries = scan(library)
    index = ToneIndex.build(entries, di, embedder=Picky(), seconds=2.0,
                            work_dir=tmp_path)
    assert len(index) == len(entries) - 1
    assert np.isfinite(index.vectors).all()
    assert entries[1].key not in {e.key for e in index.entries}


def test_a_render_pool_produces_the_same_index_as_plain_map(
        library: pathlib.Path, di: np.ndarray, tmp_path: pathlib.Path) -> None:
    entries = scan(library)
    a = ToneIndex.build(entries, di, embedder=FakeEmbedder(), seconds=2.0,
                        work_dir=tmp_path)
    b = ToneIndex.build(entries, di, embedder=FakeEmbedder(), seconds=2.0,
                        work_dir=tmp_path, render_map=map)
    assert np.allclose(a.vectors, b.vectors)


def test_progress_is_reported_once_per_profile(library: pathlib.Path,
                                               di: np.ndarray,
                                               tmp_path: pathlib.Path) -> None:
    seen: list[tuple[int, int, str]] = []
    ToneIndex.build(scan(library), di, embedder=FakeEmbedder(), seconds=2.0,
                    work_dir=tmp_path, progress=lambda *a: seen.append(a))
    assert [s[0] for s in seen] == [1, 2, 3, 4, 5]
    assert all(s[1] == 5 for s in seen)


def test_the_temporary_di_is_cleaned_up_after_a_build(
        library: pathlib.Path, di: np.ndarray, tmp_path: pathlib.Path) -> None:
    ToneIndex.build(scan(library), di, embedder=FakeEmbedder(), seconds=2.0,
                    work_dir=tmp_path)
    assert list(tmp_path.glob("_index_di_*.npy")) == []


# -- the cache -------------------------------------------------------------


def test_a_second_build_reuses_the_cache_and_embeds_nothing(
        library: pathlib.Path, di: np.ndarray, tmp_path: pathlib.Path) -> None:
    cache = tmp_path / "index.npz"
    first = ToneIndex.load_or_build(scan(library), di, cache,
                                    embedder=FakeEmbedder(), seconds=2.0,
                                    work_dir=tmp_path)

    class Explodes(FakeEmbedder):
        def embed(self, x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
            raise AssertionError("a cached profile must not be embedded again")

    second = ToneIndex.load_or_build(scan(library), di, cache,
                                     embedder=Explodes(), seconds=2.0,
                                     work_dir=tmp_path)
    assert np.allclose(first.vectors, second.vectors)
    assert [e.key for e in first.entries] == [e.key for e in second.entries]


def test_adding_a_profile_embeds_only_the_new_one(
        library: pathlib.Path, di: np.ndarray, tmp_path: pathlib.Path) -> None:
    cache = tmp_path / "index.npz"
    ToneIndex.load_or_build(scan(library), di, cache, embedder=FakeEmbedder(),
                            seconds=2.0, work_dir=tmp_path)
    (library / "amp 9.nam").write_bytes(_nam_bytes(1.9))

    embedded: list[int] = []

    class Counting(FakeEmbedder):
        def embed(self, x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
            embedded.append(1)
            return super().embed(x, sr)

    index = ToneIndex.load_or_build(scan(library), di, cache,
                                    embedder=Counting(), seconds=2.0,
                                    work_dir=tmp_path)
    assert len(embedded) == 1
    assert len(index) == 6


def test_a_different_di_invalidates_every_vector(
        library: pathlib.Path, di: np.ndarray, tmp_path: pathlib.Path) -> None:
    cache = tmp_path / "index.npz"
    ToneIndex.load_or_build(scan(library), di, cache, embedder=FakeEmbedder(),
                            seconds=2.0, work_dir=tmp_path)
    other = (di * 0.5 + 0.01).astype(np.float32)

    embedded: list[int] = []

    class Counting(FakeEmbedder):
        def embed(self, x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
            embedded.append(1)
            return super().embed(x, sr)

    ToneIndex.load_or_build(scan(library), other, cache, embedder=Counting(),
                            seconds=2.0, work_dir=tmp_path)
    assert len(embedded) == 5


def test_a_corrupt_cache_is_a_rebuild_rather_than_a_crash(
        library: pathlib.Path, di: np.ndarray, tmp_path: pathlib.Path) -> None:
    cache = tmp_path / "index.npz"
    cache.write_bytes(b"not an npz at all")
    index = ToneIndex.load_or_build(scan(library), di, cache,
                                    embedder=FakeEmbedder(), seconds=2.0,
                                    work_dir=tmp_path)
    assert len(index) == 5


def test_removing_a_profile_removes_it_from_the_index(
        library: pathlib.Path, di: np.ndarray, tmp_path: pathlib.Path) -> None:
    cache = tmp_path / "index.npz"
    ToneIndex.load_or_build(scan(library), di, cache, embedder=FakeEmbedder(),
                            seconds=2.0, work_dir=tmp_path)
    (library / "amp 1.nam").unlink()
    index = ToneIndex.load_or_build(scan(library), di, cache,
                                    embedder=FakeEmbedder(), seconds=2.0,
                                    work_dir=tmp_path)
    assert len(index) == 4
    assert "amp 1" not in {e.name for e in index.entries}


def test_a_cache_from_another_embedding_config_is_not_reused(
        library: pathlib.Path, di: np.ndarray, tmp_path: pathlib.Path) -> None:
    cache = tmp_path / "index.npz"
    ToneIndex.load_or_build(scan(library), di, cache, embedder=FakeEmbedder(),
                            seconds=2.0, work_dir=tmp_path)
    assert tone_index._read_cache(cache, "a different key", "x") == {}


def test_a_match_serialises_with_everything_the_ui_needs(
        library: pathlib.Path, di: np.ndarray, tmp_path: pathlib.Path) -> None:
    index = ToneIndex.build(scan(library), di, embedder=FakeEmbedder(),
                            seconds=2.0, work_dir=tmp_path)
    payload = index.search(index.vectors[0], top=1)[0].to_json()
    assert {"rank", "similarity", "distance", "percentile", "name", "tone_id",
            "url", "source"} <= set(payload)
    json.dumps(payload)   # must survive a round trip to the browser


def test_centring_can_be_turned_off_and_then_ranks_by_raw_cosine() -> None:
    """Measurement, not belief, decides this one -- so it has to be switchable."""
    vectors = _vectors(12)
    query = _vectors(1, seed=5)[0]

    centred = ToneIndex(entries=_entries(12), vectors=vectors)
    raw = ToneIndex(entries=_entries(12), vectors=vectors, centred=False)

    assert not np.allclose(centred.center, 0.0)
    assert np.allclose(raw.center, 0.0)
    assert np.allclose(raw.similarities(query),
                       _unit_rows(vectors) @ (query / np.linalg.norm(query)),
                       atol=1e-5)


def test_centring_spreads_scores_out_which_is_what_it_is_for() -> None:
    cone = _vectors(24) * 0.1 + np.full(DIM, 1.0, dtype=np.float32)
    cone /= np.linalg.norm(cone, axis=1, keepdims=True)
    entries = _entries(24)
    query = cone[3]

    raw = ToneIndex(entries=entries, vectors=cone, centred=False).similarities(query)
    centred = ToneIndex(entries=entries, vectors=cone).similarities(query)
    assert raw.std() < 0.05          # every candidate looks like a 0.99 match
    assert centred.std() > raw.std() * 3


def _unit_rows(m: np.ndarray) -> np.ndarray:
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)


def test_device_switch_reuses_existing_float64_index_cache(
        library, di, tmp_path, monkeypatch):
    cache = tmp_path / "index.npz"
    first = ToneIndex.load_or_build(scan(library), di, cache,
                                    embedder=FakeEmbedder(), seconds=2.0,
                                    work_dir=tmp_path)

    def must_not_render(*args, **kwargs):
        raise AssertionError("changing float64 render device must not rebuild cached audio")

    monkeypatch.setattr(tone_index, "_render_entry", must_not_render)
    second = ToneIndex.load_or_build(scan(library), di, cache, render_device="cuda",
                                     embedder=FakeEmbedder(), seconds=2.0,
                                     work_dir=tmp_path)
    np.testing.assert_array_equal(first.vectors, second.vectors)


def test_index_gpu_option_reaches_renderer_and_recovers_without_cuda(
        library, di, tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    expected = ToneIndex.build(scan(library), di, embedder=FakeEmbedder(),
                               seconds=2.0, work_dir=tmp_path)
    with pytest.warns(RuntimeWarning, match="unavailable"):
        actual = ToneIndex.build(scan(library), di, embedder=FakeEmbedder(),
                                 seconds=2.0, work_dir=tmp_path, render_device="cuda")
    np.testing.assert_array_equal(actual.vectors, expected.vectors)
