"""Song in, Tone IDs out.

The five steps in one place, so the CLI, the server and the tests all run the
same pipeline rather than three arrangements of the same parts::

    ingest      a file or a link                     ~0 s / ~20 s for a link
    separate    htdemucs_6s -> the guitar stem        ~1.0x realtime
    embed       MERT hidden states -> a unit vector   ~1.1 s per 5 s window
    search      centred cosine over the index         instant
    audition    the user's DI through the top matches ~0.17x realtime each

Only the index build is expensive and it is cached, so the second run of a
session is separation plus a second.

**What this ranking is and is not.** Both sides of the comparison are MERT
embeddings, but they are not the same kind of recording: the query is a guitar
separated out of a finished, mastered record, and the index is a clean DI
through a capture. That domain gap is the honest limit of the method, and it is
the reason `CAVEAT` is quoted on every surface instead of a bare number. Nothing
here hides which stem the ranking actually saw: an `other` fallback is labelled,
and a missing usable stem stops ranking rather than comparing the raw mix.
"""

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from . import ingest
from .audioio import read as read_audio
from .config import SAMPLE_RATE
from .embed import EmbedConfig, MertEmbedder
from .ingest import DEFAULT_SECONDS, Source
from .separate import Separated, Separator, SeparationError, isolate
from .tone_index import DEFAULT_INDEX_SECONDS, Match, ToneIndex, scan

CAVEAT = (
    "Treat this as a shortlist. A high score is similarity within the available "
    "library, not proof of the equipment used on a record. The query is guitar "
    "separated from a finished mix; capture references come from DI performances. "
    "Playing, cabinet/microphone choices, mastering and separation artefacts all "
    "affect the order. Multi-DI comparison training improves controlled retrieval, "
    "but those tests do not establish amp-identification accuracy for commercial "
    "songs. Listen to the isolated guitar, audition the candidates, and trust "
    "your ears. If separation produced no usable guitar, choose a clearer passage."
)
"""The honesty text, owned here so every surface quotes it identically.

The historical 34-capture results remain in the workflow guide. Current,
versioned 51-capture results and their limits are in MULTI_DI_TRAINING.md and
retrieval_evaluation.json. No controlled percentage is presented as the chance
that an unlabelled song used the selected amp.
Re-measure these numbers rather than editing them.
"""

ROOT = pathlib.Path(__file__).resolve().parents[2]


@dataclass
class PipelineConfig:
    """Everything a run needs that is not the song itself."""

    profile_dir: pathlib.Path = ROOT / "assets" / "dev_profiles"
    cache_dir: pathlib.Path = ROOT / ".cache"
    di_path: pathlib.Path = ROOT / "assets" / "user_di" / "Djent DI.wav"

    seconds: float = DEFAULT_SECONDS
    start_s: float = 0.0
    index_seconds: float = DEFAULT_INDEX_SECONDS
    top: int = 5

    embed: EmbedConfig = field(default_factory=EmbedConfig)

    render_device: str = "cpu"
    persistent_catalogue: bool = False
    refresh_descriptors: bool = False
    learned_retrieval: bool = True
    """Opt-in auto/cuda float64 rendering for indexing and auditions, with CPU fallback."""

    separator_autocast: bool = False
    """fp16 autocast for htdemucs. Only does anything on a CUDA device."""

    separator_shifts: int | None = None
    """demucs's quality/time dial; `None` keeps the library's default of 2."""

    @property
    def model_dir(self) -> pathlib.Path:
        return self.cache_dir / "uvr_models"

    @property
    def download_dir(self) -> pathlib.Path:
        return self.cache_dir / "downloads"

    @property
    def index_cache(self) -> pathlib.Path:
        """One cache file per (profile directory, embedding config).

        Keyed by name so switching between the dev corpus and a TONE3000 library
        does not make the two rebuild each other on every switch.
        """
        return (self.cache_dir /
                f"tone_index_{self.profile_dir.name}_{self.embed.key()}.npz")


@dataclass
class MatchResult:
    """What one run produced, including how long each step took."""

    source: Source
    stem: Separated
    matches: list[Match]
    index_size: int
    embed_dim: int
    query: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    timings: dict[str, float] = field(default_factory=dict)
    caveat: str = CAVEAT
    retrieval: dict = field(default_factory=dict)

    @property
    def separation_ok(self) -> bool:
        return self.stem.ok

    def to_json(self) -> dict[str, Any]:
        return {
            "source": {"name": self.source.name, "origin": self.source.origin,
                       "url": self.source.url,
                       "duration_s": round(self.source.duration_s, 2),
                       "full_duration_s": round(self.source.full_duration_s, 2)},
            "stem": {"used": self.stem.stem, "ok": self.stem.ok,
                     "note": self.stem.note},
            "matches": [m.to_json() for m in self.matches],
            "index_size": self.index_size, "embed_dim": self.embed_dim,
            "timings": {k: round(v, 2) for k, v in self.timings.items()},
            "caveat": self.caveat,
            "retrieval": self.retrieval,
        }


Progress = Callable[[str, float, str], None]
"""``(stage, fraction, message)``. Fraction is -1 when the total is unknown."""


class Pipeline:
    """A loaded separator, embedder and index. Build once, match repeatedly."""

    def __init__(self, cfg: PipelineConfig | None = None) -> None:
        self.cfg = cfg or PipelineConfig()
        self._separator: Separator | None = None
        self._embedder: MertEmbedder | None = None
        self._index: ToneIndex | None = None
        self._di: np.ndarray | None = None

    # -- lazily built parts ----------------------------------------------

    @property
    def separator(self) -> Separator:
        if self._separator is None:
            self._separator = Separator(self.cfg.model_dir,
                                        autocast=self.cfg.separator_autocast,
                                        shifts=self.cfg.separator_shifts)
        return self._separator

    @property
    def devices(self) -> dict[str, str]:
        """Where each expensive step will run, for the CLI to print.

        Rendering remains opt-in and may still fall back per capture (LSTM or
        CUDA runtime failure); the other two stages select their own devices.
        """
        from .nam_render import render_device

        return {"separate": self.separator.device,
                "embed": self.embedder.device,
                "render": render_device(self.cfg.render_device)}

    @property
    def embedder(self) -> MertEmbedder:
        if self._embedder is None:
            from .lora import create_embedder
            self._embedder = create_embedder(self.cfg.embed)
        return self._embedder

    @property
    def di(self) -> np.ndarray:
        if self._di is None:
            self._di = read_audio(self.cfg.di_path,
                                  seconds=self.cfg.index_seconds)
        return self._di

    def index(self, *, progress: Progress | None = None,
              entries: Sequence[Any] | None = None) -> ToneIndex:
        """The profile library, built on first use and cached to disk."""
        if self._index is None:
            if getattr(self.embedder, 'trained', False):
                if progress:
                    progress('indexing', 0., 'Loading the LoRA matching model')
                # Surface load errors before per-capture rendering can skip them.
                self.embedder.model
            if self.cfg.persistent_catalogue and entries is None:
                from .catalogue import Catalogue
                self._index = Catalogue(self.cfg.cache_dir.parent).index(self.cfg, self.embedder, self.di, progress)
                self._index = self._retrieval_index(self._index, progress)
                return self._index
            found = list(entries) if entries is not None else scan(self.cfg.profile_dir)
            if not found:
                raise FileNotFoundError(
                    f"no .nam profiles in {self.cfg.profile_dir}")

            def step(done: int, total: int, name: str) -> None:
                if progress:
                    progress("indexing", done / total,
                             f"indexing {done}/{total}: {name}")

            self._index = ToneIndex.load_or_build(
                found, self.di, self.cfg.index_cache,
                embedder=self.embedder, seconds=self.cfg.index_seconds,
                render_device=self.cfg.render_device,
                progress=step, work_dir=self.cfg.cache_dir)
            self._index = self._retrieval_index(self._index, progress)
        return self._index

    def _retrieval_index(self, index, progress):
        if getattr(self.embedder, 'trained', False):
            # The contrastive head was trained/evaluated with ordinary cosine.
            # Never apply the unrelated 330M projection to its 128D output.
            index = ToneIndex(index.entries, index.vectors, index.embed_key, index.di_key, centred=False)
            index.learning = dict(self.embedder.retrieval, profiles=len(index))
        elif self.cfg.learned_retrieval:
            from .tone_learning import apply_active
            index = apply_active(index, self.cfg.cache_dir/'tone_learning/active.npz', progress)
        return index

    # -- the run ---------------------------------------------------------

    def match(self, spec: str | pathlib.Path, *,
              progress: Progress | None = None,
              top: int | None = None) -> MatchResult:
        """Run the whole pipeline on one song."""
        timings: dict[str, float] = {}

        def stage(name: str, fraction: float, message: str) -> None:
            if progress:
                progress(name, fraction, message)

        # The index is built before anything else so a first run reports its
        # cost up front rather than appearing to hang after separation.
        stage("indexing", 0.0, "preparing the profile index")
        started = time.monotonic()
        index = self.index(progress=progress)
        timings["index"] = time.monotonic() - started

        stage("ingesting", 0.05, f"loading {spec}")
        started = time.monotonic()
        source = ingest.load(spec, cache_dir=self.cfg.download_dir,
                             seconds=self.cfg.seconds, start_s=self.cfg.start_s,
                             mono=False,
                             progress=lambda m, f: stage("ingesting", 0.05, m))
        timings["ingest"] = time.monotonic() - started

        stage("separating", 0.15, "isolating the guitar")
        started = time.monotonic()
        stem = isolate(source.samples, SAMPLE_RATE, separator=self.separator,
                       tag="query", preserve_stereo=True)
        timings["separate"] = time.monotonic() - started
        if stem.stem not in ('guitar','other'):
            raise SeparationError('guitar_missing',
                'No usable guitar stem was recovered. Select a passage with a clearer guitar or try another recording.')

        stage("embedding", 0.75, "measuring the tone")
        started = time.monotonic()
        query = self.embedder.embed(stem.samples, SAMPLE_RATE)
        timings["embed"] = time.monotonic() - started

        stage("ranking", 0.95, f"ranking {len(index)} profiles")
        matches = index.search(query, top or self.cfg.top)

        stage("done", 1.0, f"{len(matches)} candidates")
        return MatchResult(source=source, stem=stem, matches=matches,
                           index_size=len(index), embed_dim=index.dim,
                           query=query, timings=timings, retrieval=getattr(index, 'learning', {}))

    def audition(self, match: Match, di: np.ndarray | None = None,
                 seconds: float = 8.0) -> np.ndarray:
        """The user's DI through one matched profile, so they can hear it."""
        from . import nam_render

        take = (self.di if di is None else di)[:int(seconds * SAMPLE_RATE)]
        path = match.entry.path
        if self.cfg.persistent_catalogue and match.entry.model_id is not None:
            from .catalogue import Catalogue
            path = Catalogue(self.cfg.cache_dir.parent).ensure(path.name)
        model = nam_render.load(path)
        out = model.render(np.asarray(take, dtype=np.float64), device=self.cfg.render_device)
        peak = float(np.abs(out).max())
        return (out / peak * 0.95 if peak > 0 else out).astype(np.float32)


def match_song(spec: str | pathlib.Path, cfg: PipelineConfig | None = None,
               *, progress: Progress | None = None) -> MatchResult:
    """One-shot convenience wrapper. Prefer `Pipeline` for repeated matches --
    it keeps the separator, the 1.3 GB embedder and the index loaded."""
    return Pipeline(cfg).match(spec, progress=progress)
