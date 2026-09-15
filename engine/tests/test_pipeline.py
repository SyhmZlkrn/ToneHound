"""The five steps wired together, and what the result promises to say.

The expensive parts are stubbed. What is being tested is the wiring and the
reporting: that the stages run in order, that each one's cost is recorded, that
a separation fallback survives all the way into the JSON the UI reads, and that
the index cache key actually separates two libraries. Those are the things that
break quietly -- a pipeline that silently ranks the raw mix still returns five
confident candidates.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import soundfile as sf

from tonehound import pipeline
from tonehound.config import SAMPLE_RATE
from tonehound.embed import EmbedConfig
from tonehound.pipeline import CAVEAT, Pipeline, PipelineConfig
from tonehound.separate import Separated
from tonehound.tone_index import ToneEntry, ToneIndex

DIM = 16


def _tone(seconds: float, hz: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


class StubSeparator:
    def __init__(self, stem: str = "guitar") -> None:
        self.stem = stem

    def split(self, x, sr=SAMPLE_RATE, tag="mix"):
        return {self.stem: np.asarray(x, dtype=np.float32)} if self.stem else {}


class StubEmbedder:
    class cfg:
        @staticmethod
        def key() -> str:
            return "stub"

    dim = DIM

    def embed(self, x, sr=SAMPLE_RATE):
        v = np.zeros(DIM, dtype=np.float32)
        v[0] = 1.0
        return v


def _index(n: int = 6) -> ToneIndex:
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((n, DIM)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    entries = [ToneEntry(key=f"k{i}", name=f"amp {i}",
                         path=pathlib.Path(f"amp {i}.nam"),
                         tone_id=100 + i if i % 2 else None)
               for i in range(n)]
    return ToneIndex(entries=entries, vectors=vectors)


@pytest.fixture()
def song(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "A Song.wav"
    sf.write(path, _tone(8.0), SAMPLE_RATE)
    return path


@pytest.fixture()
def pipe(tmp_path: pathlib.Path) -> Pipeline:
    cfg = PipelineConfig(cache_dir=tmp_path, seconds=4.0, top=3)
    p = Pipeline(cfg)
    p._separator = StubSeparator()
    p._embedder = StubEmbedder()
    p._index = _index()
    p._di = _tone(3.0, 110.0)
    return p


# -- configuration ---------------------------------------------------------


def test_two_profile_libraries_do_not_share_an_index_cache(
        tmp_path: pathlib.Path) -> None:
    a = PipelineConfig(cache_dir=tmp_path, profile_dir=tmp_path / "dev")
    b = PipelineConfig(cache_dir=tmp_path, profile_dir=tmp_path / "tone3000")
    assert a.index_cache != b.index_cache


def test_changing_the_embedding_changes_the_index_cache(
        tmp_path: pathlib.Path) -> None:
    a = PipelineConfig(cache_dir=tmp_path)
    b = PipelineConfig(cache_dir=tmp_path, embed=EmbedConfig(layers=(-1,)))
    assert a.index_cache != b.index_cache


def test_changing_float64_render_device_keeps_the_index_cache(tmp_path):
    a = PipelineConfig(cache_dir=tmp_path)
    b = PipelineConfig(cache_dir=tmp_path, render_device="cuda")
    assert a.index_cache == b.index_cache


def test_pipeline_passes_render_device_to_index_builder(pipe, monkeypatch):
    pipe._index = None
    pipe.cfg.render_device = "cuda"
    seen = []

    def build(*args, **kwargs):
        seen.append(kwargs["render_device"])
        return _index()

    monkeypatch.setattr(ToneIndex, "load_or_build", build)
    assert len(pipe.index(entries=_index().entries)) == 6
    assert seen == ["cuda"]


# -- a run -----------------------------------------------------------------


def test_a_run_returns_the_requested_number_of_ranked_candidates(
        pipe: Pipeline, song: pathlib.Path) -> None:
    result = pipe.match(song)
    assert [m.rank for m in result.matches] == [1, 2, 3]
    assert result.index_size == 6 and result.embed_dim == DIM


def test_every_stage_reports_what_it_cost(pipe: Pipeline,
                                          song: pathlib.Path) -> None:
    timings = pipe.match(song).timings
    assert set(timings) == {"index", "ingest", "separate", "embed"}
    assert all(v >= 0.0 for v in timings.values())


def test_progress_names_the_stages_in_order(pipe: Pipeline,
                                            song: pathlib.Path) -> None:
    stages: list[str] = []
    pipe.match(song, progress=lambda stage, f, m: stages.append(stage))
    first = [s for i, s in enumerate(stages) if i == 0 or stages[i - 1] != s]
    assert first == ["indexing", "ingesting", "separating", "embedding",
                     "ranking", "done"]


def test_the_window_is_the_one_that_was_asked_for(pipe: Pipeline,
                                                  song: pathlib.Path) -> None:
    result = pipe.match(song)
    assert abs(result.source.duration_s - 4.0) < 0.05
    assert abs(result.source.full_duration_s - 8.0) < 0.05


def test_top_can_be_overridden_per_call(pipe: Pipeline,
                                        song: pathlib.Path) -> None:
    assert len(pipe.match(song, top=5).matches) == 5


def test_stereo_survives_import_separation_and_embedding(pipe, tmp_path):
    left = _tone(8.0)
    stereo = np.column_stack((left, -left * 0.4))
    path = tmp_path / 'wide-guitars.wav'
    sf.write(path, stereo, SAMPLE_RATE, subtype='FLOAT')
    seen = {}

    class StereoSeparator(StubSeparator):
        def split(self, x, sr=SAMPLE_RATE, tag='mix'):
            seen['separation'] = x.copy()
            return {'guitar': x * 0.7}

    class StereoEmbedder(StubEmbedder):
        def embed(self, x, sr=SAMPLE_RATE):
            seen['embedding'] = x.copy()
            return super().embed(x, sr)

    pipe._separator = StereoSeparator()
    pipe._embedder = StereoEmbedder()
    result = pipe.match(path)
    assert result.source.samples.shape == (4 * SAMPLE_RATE, 2)
    assert np.allclose(seen['separation'][:, 1], seen['separation'][:, 0] * -0.4)
    assert np.allclose(seen['embedding'], seen['separation'] * 0.7)
    assert result.stem.samples.shape == (4 * SAMPLE_RATE, 2)


# -- what the result says --------------------------------------------------


def test_a_clean_separation_is_reported_as_such(pipe: Pipeline,
                                                song: pathlib.Path) -> None:
    result = pipe.match(song)
    assert result.separation_ok and result.stem.stem == "guitar"


def test_a_fallback_stem_reaches_the_json_the_ui_reads(
        pipe: Pipeline, song: pathlib.Path) -> None:
    pipe._separator = StubSeparator("other")
    payload = pipe.match(song).to_json()
    assert payload["stem"]["used"] == "other"
    assert payload["stem"]["ok"] is False
    assert "may not be only guitar" in payload["stem"]["note"]


def test_no_stem_at_all_does_not_rank_the_raw_mix(
        pipe: Pipeline, song: pathlib.Path) -> None:
    from tonehound.separate import SeparationError
    pipe._separator = StubSeparator("")
    with pytest.raises(SeparationError,match='No usable guitar stem') as failure:
        pipe.match(song)
    assert failure.value.code == 'guitar_missing'


def test_the_result_serialises_for_the_browser(pipe: Pipeline,
                                               song: pathlib.Path) -> None:
    payload = pipe.match(song).to_json()
    json.dumps(payload)
    assert payload["caveat"] == CAVEAT
    assert payload["source"]["origin"] == "file"
    assert {"rank", "similarity", "name", "url"} <= set(payload["matches"][0])


def test_a_tone3000_candidate_carries_a_link_and_a_local_one_does_not(
        pipe: Pipeline, song: pathlib.Path) -> None:
    urls = {m["name"]: m["url"] for m in pipe.match(song, top=6).to_json()["matches"]}
    assert any(u and u.startswith("https://www.tone3000.com/tones/")
               for u in urls.values())
    assert any(u is None for u in urls.values())


def test_the_caveat_says_what_the_domain_gap_is() -> None:
    """The one sentence every surface quotes; it must keep saying this."""
    assert "mix" in CAVEAT and "DI" in CAVEAT
    assert "ears" in CAVEAT


# -- auditioning -----------------------------------------------------------


def test_auditioning_renders_the_users_di_through_the_matched_profile(
        pipe: Pipeline, song: pathlib.Path, tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    nam = tmp_path / "amp 0.nam"
    nam.write_bytes(json.dumps({
        "version": "0.5.2", "architecture": "Linear",
        "config": {"receptive_field": 1, "input_channels": 1, "bias": False},
        "weights": [0.5], "metadata": {}}).encode())

    match = pipe.match(song).matches[0]
    match.entry.path = nam
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    pipe.cfg.render_device = "cuda"
    with pytest.warns(RuntimeWarning, match="unavailable"):
        rendered = pipe.audition(match, seconds=1.0)
    assert rendered.dtype == np.float32
    assert abs(len(rendered) / SAMPLE_RATE - 1.0) < 0.05
    assert np.abs(rendered).max() == pytest.approx(0.95, abs=1e-3)


# -- a library that is not there -------------------------------------------


def test_an_empty_profile_directory_is_a_clear_error(
        tmp_path: pathlib.Path) -> None:
    empty = tmp_path / "none"
    empty.mkdir()
    p = Pipeline(PipelineConfig(cache_dir=tmp_path, profile_dir=empty))
    p._embedder = StubEmbedder()
    with pytest.raises(FileNotFoundError, match="no .nam profiles"):
        p.index()


def test_match_song_is_the_same_pipeline_behind_a_function(
        monkeypatch: pytest.MonkeyPatch, song: pathlib.Path,
        tmp_path: pathlib.Path) -> None:
    built: list[Pipeline] = []
    original = Pipeline.__init__

    def spy(self, cfg=None):
        original(self, cfg)
        self._separator = StubSeparator()
        self._embedder = StubEmbedder()
        self._index = _index()
        self._di = _tone(3.0)
        built.append(self)

    monkeypatch.setattr(Pipeline, "__init__", spy)
    result = pipeline.match_song(song, PipelineConfig(cache_dir=tmp_path,
                                                      seconds=4.0, top=2))
    assert len(built) == 1 and len(result.matches) == 2
