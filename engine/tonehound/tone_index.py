"""The searchable library: every `.nam` profile as one MERT vector.

A `.nam` file is a function, not audio, so it cannot be embedded directly. Each
profile is rendered with one fixed DI take and the render is embedded; the query
side is the guitar separated out of a song, embedded the same way. That puts a
capture and a record in one space, which is the only reason cosine between them
means anything at all.

**Vectors are centred against the index -- for readable scores, not for
accuracy.** Transformer embeddings occupy a narrow cone rather than the whole
sphere: measured on this checkpoint, raw cosine between renders of *different*
amps sits at 0.78 (0.94 once separation is in the chain) with almost no spread,
so a raw similarity is unusable as something to show a user -- every candidate
looks like a 0.9 match. Subtracting the library mean moves that to about -0.02
with a standard deviation of 0.6, which makes the numbers mean something.

It was tempting to claim this improves the ranking as well. It does not, and
`tools/gate_mert.py` prints both so the claim cannot quietly drift: on clean
unseen performances centred and raw retrieve *identically*, and with separation
in the chain raw is slightly ahead. The differences are a handful of profiles
out of 34 -- inside the noise of a corpus this size -- so `centred` is an option
rather than a verdict, defaulting to on because a score a user can read is worth
something and the retrieval cost is not established.

The centre is a property of the *library*, computed once and applied to the
query too, exactly as `match.py` whitens against the corpus rather than against
the query.

**The cache is incremental and content-addressed.** Rendering and embedding one
profile costs several seconds, and a library fetched from TONE3000 grows every
time it is searched. Each entry is keyed by the SHA-1 of its own file, so adding
fifty profiles embeds fifty profiles rather than rebuilding hundreds. The DI and
the embedding configuration key the cache as a whole, because changing either
invalidates every vector in it.
"""

from __future__ import annotations

import hashlib
import pathlib
import uuid
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .config import SAMPLE_RATE
from .embed import EmbedConfig, MertEmbedder
from .tone3000 import parse_model_reference

CACHE_VERSION = 1
DEFAULT_INDEX_SECONDS = 15.0
"""How much DI each profile is rendered through.

Long enough that the embedder gets three independent windows to average, short
enough that a first build of a few hundred profiles is minutes rather than an
afternoon.
"""

_EPS = 1e-12


class IndexError_(Exception):
    """A build that could not proceed. Named to avoid shadowing the builtin."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class ToneEntry:
    """One profile in the library, and the way back to where it came from."""

    key: str                       # SHA-1 of the file's bytes
    name: str
    path: pathlib.Path
    source: str = "local"          # "local" | "tone3000"
    tone_id: int | None = None
    model_id: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def tone3000_url(self) -> str | None:
        return (f"https://www.tone3000.com/tones/{self.tone_id}"
                if self.tone_id is not None else None)

    def to_json(self) -> dict[str, Any]:
        return {"key": self.key, "name": self.name, "path": str(self.path),
                "source": self.source, "tone_id": self.tone_id,
                "model_id": self.model_id, "url": self.tone3000_url,
                "meta": self.meta}


@dataclass
class Match:
    """One ranked candidate."""

    rank: int
    entry: ToneEntry
    similarity: float              # cosine in [-1, 1]; centred by default
    percentile: float              # share of the index this one beats

    @property
    def distance(self) -> float:
        """``1 - similarity``, for callers that prefer lower-is-closer."""
        return 1.0 - self.similarity

    def to_json(self) -> dict[str, Any]:
        return {"rank": self.rank, "similarity": self.similarity,
                "distance": self.distance, "percentile": self.percentile,
                **self.entry.to_json()}


def file_key(path: pathlib.Path) -> str:
    """SHA-1 of a profile's bytes. A `.nam` is JSON of a few megabytes at most,
    so hashing the content is affordable and means an edited capture becomes a
    different entry instead of silently changing under a cached vector."""
    return hashlib.sha1(path.read_bytes()).hexdigest()[:16]


def scan(directory: str | pathlib.Path, source: str = "local") -> list[ToneEntry]:
    """Every `.nam` under a directory, as index entries.

    Files downloaded by `tone3000.download_model` carry their tone and model ids
    in the file name, so a mixed directory of local captures and fetched ones
    still knows which is which.
    """
    directory = pathlib.Path(directory)
    entries = []
    for path in sorted(directory.glob("*.nam")):
        tone_id, model_id = parse_model_reference(path)
        entries.append(ToneEntry(
            key=file_key(path), name=path.stem, path=path,
            source="tone3000" if tone_id is not None else source,
            tone_id=tone_id, model_id=model_id))
    return entries


# --------------------------------------------------------------------------
# rendering, in a shape a worker process can import
# --------------------------------------------------------------------------


def _render_entry(args: tuple[str, str, int], *, device: str = "cpu") -> np.ndarray:
    """One profile rendered through the DI on disk.

    Module level and tuple-argued so `ProcessPoolExecutor.map` can pickle it;
    the DI travels as a path rather than an array because otherwise every task
    would carry a copy of the same 15 seconds of audio.
    """
    from . import nam_render

    nam_path, di_path, samples = args
    di = np.load(di_path)[:samples]
    model = nam_render.load(nam_path)
    out = model.render(np.asarray(di, dtype=np.float64), device=device)
    peak = float(np.abs(out).max())
    return (out / peak * 0.95 if peak > 0 else out).astype(np.float32)


# --------------------------------------------------------------------------
# the index
# --------------------------------------------------------------------------


@dataclass
class ToneIndex:
    """Profiles plus their vectors, and the centre those vectors define."""

    entries: list[ToneEntry]
    vectors: np.ndarray            # (n, dim) unit rows, un-centred
    embed_key: str = ""
    di_key: str = ""
    centred: bool = True
    """Subtract the library mean before comparing. See the module docstring:
    this buys readable scores, not better retrieval."""

    def __post_init__(self) -> None:
        self.vectors = np.asarray(self.vectors, dtype=np.float32)
        if len(self.entries) != len(self.vectors):
            raise IndexError_("shape_mismatch",
                              f"{len(self.entries)} entries but "
                              f"{len(self.vectors)} vectors")
        self.center = (self.vectors.mean(axis=0) if len(self.vectors)
                       else np.zeros(self.vectors.shape[-1], dtype=np.float32))
        if not self.centred:
            self.center = np.zeros_like(self.center)
        self.matrix = _unit_rows(self.vectors - self.center)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if len(self.vectors) else 0

    def similarities(self, query: np.ndarray) -> np.ndarray:
        """Centred cosine from ``query`` to every entry, in index order."""
        if len(self) == 0:
            return np.zeros(0, dtype=np.float32)
        q = np.asarray(query, dtype=np.float32) - self.center
        q = q / (np.linalg.norm(q) + _EPS)
        return self.matrix @ q

    def search(self, query: np.ndarray, top: int = 5) -> list[Match]:
        """The ``top`` closest profiles, best first."""
        sims = self.similarities(query)
        if len(sims) == 0:
            return []
        order = np.argsort(-sims)[:max(1, top)]
        return [Match(rank=rank,
                      entry=self.entries[i],
                      similarity=float(sims[i]),
                      percentile=float((np.sum(sims < sims[i]) + 0.5) / len(sims)))
                for rank, i in enumerate(order, 1)]

    # -- construction ----------------------------------------------------

    @classmethod
    def build(cls, entries: Sequence[ToneEntry], di: np.ndarray, *,
              embedder: MertEmbedder | None = None,
              seconds: float = DEFAULT_INDEX_SECONDS,
              cached: dict[str, np.ndarray] | None = None,
              render_map: Callable[..., Iterable[Any]] | None = None,
              render_device: str = "cpu",
              progress: Callable[[int, int, str], None] | None = None,
              work_dir: pathlib.Path | None = None) -> "ToneIndex":
        """Render and embed every entry not already in ``cached``.

        ``render_map`` defaults to `map`; pass `pool.map` to spread the render
        across cores. Only the render is parallelised -- the embedder holds a
        1.3 GB model, and a copy of it per worker would cost more memory than
        the parallelism saves.

        ``render_device="auto"`` or ``"cuda"`` opts into float64 GPU rendering
        with CPU fallback. Use the default sequential map on GPU so workers do
        not compete with each other and MERT for VRAM. Both devices retain
        float64 rendering, so existing embedding caches remain valid.
        """
        if not entries:
            raise IndexError_("empty", "no profiles to index")

        embedder = embedder or MertEmbedder()
        cached = cached or {}
        di = np.asarray(di, dtype=np.float32)[:int(seconds * SAMPLE_RATE)]

        todo = [e for e in entries if e.key not in cached]
        vectors = dict(cached)
        total = len(todo)

        if todo:
            work_dir = pathlib.Path(work_dir or pathlib.Path.cwd() / ".cache")
            work_dir.mkdir(parents=True, exist_ok=True)
            # Plugin instances may index the same probe concurrently. Each job
            # owns its temporary file, including cleanup after cancellation.
            di_path = work_dir / f"_index_di_{_array_key(di)}_{uuid.uuid4().hex}.npy"
            np.save(di_path, di)
            try:
                work = [(str(e.path), str(di_path), len(di)) for e in todo]
                renders = (render_map or map)(partial(_render_entry, device=render_device), work)
                for done, (entry, rendered) in enumerate(zip(todo, renders), 1):
                    if progress:
                        progress(done, total, entry.name)
                    try:
                        vectors[entry.key] = embedder.embed(rendered, SAMPLE_RATE)
                    except Exception:
                        # A profile that renders to something unembeddable
                        # (silence, a handful of samples) is dropped from the
                        # index rather than poisoning the centre with a zero row.
                        continue
            finally:
                di_path.unlink(missing_ok=True)

        kept = [e for e in entries if e.key in vectors]
        if not kept:
            raise IndexError_("all_failed",
                              f"none of the {len(entries)} profiles could be "
                              "rendered and embedded")
        return cls(entries=kept,
                   vectors=np.stack([vectors[e.key] for e in kept]),
                   embed_key=embedder.cfg.key(), di_key=_array_key(di))

    # -- cache -----------------------------------------------------------

    @classmethod
    def load_or_build(cls, entries: Sequence[ToneEntry], di: np.ndarray,
                      cache_path: str | pathlib.Path, *,
                      embedder: MertEmbedder | None = None,
                      **kwargs: Any) -> "ToneIndex":
        """Build, reusing every vector on disk that is still valid.

        A changed DI or embedding configuration invalidates the whole file; a
        changed, added or removed profile invalidates only itself.
        """
        embedder = embedder or MertEmbedder()
        cache_path = pathlib.Path(cache_path)
        seconds = kwargs.get("seconds", DEFAULT_INDEX_SECONDS)
        di_key = _array_key(np.asarray(di, dtype=np.float32)[:int(seconds * SAMPLE_RATE)])
        embed_key = embedder.cfg.key()

        cached = _read_cache(cache_path, embed_key, di_key)
        index = cls.build(entries, di, embedder=embedder, cached=cached, **kwargs)
        if len(cached) != len(index) or set(cached) != {e.key for e in index.entries}:
            _write_cache(cache_path, index)
        return index


def _array_key(x: np.ndarray) -> str:
    """A short hash of an array's bytes, for cache keys."""
    return hashlib.sha1(np.ascontiguousarray(x, dtype=np.float32).tobytes()).hexdigest()[:16]


def _unit_rows(m: np.ndarray) -> np.ndarray:
    return (m / (np.linalg.norm(m, axis=1, keepdims=True) + _EPS)).astype(np.float32)


def _read_cache(path: pathlib.Path, embed_key: str,
                di_key: str) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=False) as data:
            if (str(data["embed_key"]) != embed_key or str(data["di_key"]) != di_key
                    or int(data["version"]) != CACHE_VERSION):
                return {}
            return {str(k): v for k, v in zip(data["keys"], data["vectors"])}
    except Exception:
        return {}  # a corrupt or older cache is a rebuild, never a crash


def _write_cache(path: pathlib.Path, index: ToneIndex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp.npz')
    np.savez(tmp, version=CACHE_VERSION, embed_key=index.embed_key,
             di_key=index.di_key,
             keys=np.array([e.key for e in index.entries]),
             vectors=index.vectors)
    tmp.replace(path)
