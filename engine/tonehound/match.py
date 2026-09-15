"""Rank the profile library against a piece of recorded guitar.

This is the matcher `engine/tools/try_match.py` grew, lifted out of that script
so the live server and the CLI run *the same* code. Two implementations of a
ranking that is already weak would be two different weak rankings, and the
difference would show up as "the app disagrees with the tool" rather than as a
bug anyone could find.

The pipeline is three steps and each one is honest about what it costs:

    isolate_guitar()   UVR htdemucs_6s, ~1.0x realtime, the whole wall clock
    fingerprint()      features.py, ~0.2 s for a 15 s query
    ProfileIndex.rank  cosine over a whitened vector, instant

**How good is it.** Measured with the query taken from a *different*
performance than the one in the index, top-1 is about 0.176 on the dev corpus,
and every same-amp pair in that corpus shares a capturer, so some of that 0.176
may be the ranking sorting by who made the capture rather than by how the amp
sounds. `CAVEAT` below is the sentence the product shows, and it lives here --
next to the code it describes -- so the server and the CLI cannot drift into
saying different things about the same numbers.

**Whitening is against the index, not the query.** `zscore_reference` is
computed once over the corpus and reused for every query. Standardising a
single query against itself would be meaningless, and standardising the corpus
per query would make distances incomparable between runs.

**The index is expensive and therefore cached.** Fingerprinting 34 profile
renders takes ~45 s single-threaded, which is a first launch that looks hung.
`ProfileIndex.load_or_build` keys a `.npz` on the identity of the render files
themselves, so a stale cache cannot silently outlive the renders it describes,
and `map_fn` lets a caller with a process pool fan the first build out across
cores.
"""

from __future__ import annotations

import hashlib
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .config import GUITAR, SAMPLE_RATE, InstrumentConfig
from .features import fingerprint

CAVEAT = (
    "Ranking is a rough draft. Measured against unseen performances it puts the "
    "right amp first about 18% of the time, and the profiles it learned from "
    "share capturers, so it may be sorting by who made the capture as much as "
    "by how the amp sounds. Listen to the isolated guitar above, then listen to "
    "the candidates. Trust your ears over the order."
)
"""The honesty text, owned here so every surface quotes it identically."""

LTAS_WEIGHT = 0.5
"""Share of the vector's energy given to spectral shape over the scalars."""

_CACHE_VERSION = 2
"""Bump when anything that changes a stored fingerprint changes."""


# --------------------------------------------------------------------------
# whitening and distance
# --------------------------------------------------------------------------


def zscore_reference(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and standard deviation, with zero-variance columns held
    at 1.0 so a constant feature contributes nothing instead of infinity."""
    mean, std = block.mean(axis=0), block.std(axis=0)
    return mean, np.where(std > 1e-12, std, 1.0)


Reference = tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def whiten(shape: np.ndarray, scalars: np.ndarray, ref: Reference,
           ltas_weight: float = LTAS_WEIGHT) -> np.ndarray:
    """Standardise against the *corpus* statistics and weight the two halves.

    Each half is scaled by ``sqrt(weight / n_columns)`` so that the 48-band
    spectral shape and the handful of scalars contribute the intended share of
    the cosine regardless of how many columns each happens to have.
    """
    (shape_mean, shape_std), (scal_mean, scal_std) = ref
    z_shape = (shape - shape_mean) / shape_std
    z_scal = (scalars - scal_mean) / scal_std
    a = z_shape * np.sqrt(ltas_weight / z_shape.shape[1])
    b = z_scal * np.sqrt((1.0 - ltas_weight) / z_scal.shape[1])
    return np.hstack([a, b])


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise ``1 - cos`` between rows of ``a`` and rows of ``b``."""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return 1.0 - a @ b.T


def self_retrieval(query: np.ndarray, index: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Rank of each row's own index entry when queried by its degraded self."""
    d = cosine_distance(query, index)
    order = np.argsort(d, axis=1)
    ranks = np.array([int(np.where(order[i] == i)[0][0]) + 1 for i in range(len(d))])
    return ranks, float((ranks == 1).mean()), float((ranks <= 5).mean())


# --------------------------------------------------------------------------
# the index
# --------------------------------------------------------------------------


def _fingerprint_render(args: tuple[str, int, str]) -> tuple[np.ndarray, np.ndarray]:
    """One index entry. Module level and tuple-argued so it can be pickled to a
    worker process; ``ProcessPoolExecutor.map`` cannot carry a closure."""
    path, sr, _cfg_name = args
    fp = fingerprint(np.load(path), sr, GUITAR)
    return fp.shape_vector, fp.scalar_vector


def _identity(paths: Sequence[pathlib.Path]) -> str:
    """A key that changes whenever the renders do.

    Content-hashing would be honest but the renders are 1.2 GB; name, size and
    modification time change on any regeneration, and the alternative failure
    (a stale cache surviving a re-render) is the one that matters.
    """
    h = hashlib.sha1(f"v{_CACHE_VERSION}".encode())
    for p in paths:
        st = p.stat()
        h.update(f"{p.name}\0{st.st_size}\0{st.st_mtime_ns}\0".encode())
    return h.hexdigest()


@dataclass
class ProfileIndex:
    """Fingerprints of every profile render, plus the whitening they define."""

    names: list[str]
    shape: np.ndarray     # (n_profiles, n_bands)
    scalars: np.ndarray   # (n_profiles, n_scalars)
    di_name: str

    def __post_init__(self) -> None:
        self.reference: Reference = (zscore_reference(self.shape),
                                     zscore_reference(self.scalars))
        self.matrix = whiten(self.shape, self.scalars, self.reference)

    def __len__(self) -> int:
        return len(self.names)

    def rank(self, x: np.ndarray, sr: int = SAMPLE_RATE,
             cfg: InstrumentConfig = GUITAR) -> np.ndarray:
        """Distance from ``x`` to every profile, in index order (lower is closer)."""
        fp = fingerprint(x, sr, cfg)
        return self.rank_fingerprint(fp.shape_vector, fp.scalar_vector)

    def rank_fingerprint(self, shape: np.ndarray, scalars: np.ndarray) -> np.ndarray:
        q = whiten(shape[None, :], scalars[None, :], self.reference)
        return cosine_distance(q, self.matrix)[0]

    # -- construction ----------------------------------------------------

    @classmethod
    def build(cls, render_dir: pathlib.Path, *, sr: int = SAMPLE_RATE,
              cfg: InstrumentConfig = GUITAR,
              map_fn: Callable[..., Iterable[Any]] | None = None,
              progress: Callable[[int, int], None] | None = None) -> "ProfileIndex":
        """Fingerprint every ``.npy`` render in ``render_dir``.

        ``map_fn`` defaults to ``map``; pass ``pool.map`` to spread the ~1.3 s
        per profile across cores. ``progress(done, total)`` is called as results
        arrive, which is the only way the caller can show anything during a
        45-second first run.
        """
        paths = sorted(render_dir.glob("*.npy"))
        if not paths:
            raise FileNotFoundError(f"no profile renders in {render_dir}")

        work = [(str(p), sr, cfg.name) for p in paths]
        results = (map_fn or map)(_fingerprint_render, work)

        shapes, scalars = [], []
        for i, (shape_vec, scalar_vec) in enumerate(results, 1):
            shapes.append(shape_vec)
            scalars.append(scalar_vec)
            if progress:
                progress(i, len(paths))

        return cls(names=[p.stem for p in paths],
                   shape=np.array(shapes), scalars=np.array(scalars),
                   di_name=render_dir.name)

    @classmethod
    def load_or_build(cls, render_dir: pathlib.Path, cache_path: pathlib.Path,
                      **kwargs: Any) -> "ProfileIndex":
        """Return a cached index if it still describes these renders, else build.

        The cache is keyed on the renders' identity, so deleting or
        regenerating one invalidates it rather than producing a matcher that
        silently ranks against files that no longer exist.
        """
        paths = sorted(render_dir.glob("*.npy"))
        if not paths:
            raise FileNotFoundError(f"no profile renders in {render_dir}")
        want = _identity(paths)

        if cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as data:
                    if str(data["identity"]) == want:
                        return cls(names=[str(n) for n in data["names"]],
                                   shape=data["shape"], scalars=data["scalars"],
                                   di_name=str(data["di_name"]))
            except Exception:
                pass  # a corrupt or older cache is a rebuild, never a crash

        index = cls.build(render_dir, **kwargs)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp.npz")
        np.savez(tmp, identity=want, names=np.array(index.names),
                 shape=index.shape, scalars=index.scalars,
                 di_name=index.di_name)
        tmp.replace(cache_path)
        return index


# --------------------------------------------------------------------------
# separation
# --------------------------------------------------------------------------

GUITAR_MODEL = "htdemucs_6s.yaml"
"""The 6-stem Demucs is the only model here that emits a guitar stem at all.
UVR's stronger models are vocal/instrumental splitters with nothing to say
about a guitar-and-drums mix."""


def make_separator(model_dir: pathlib.Path, work_dir: pathlib.Path) -> Any:
    """Load UVR. Imported lazily: `audio_separator` pulls in onnxruntime and
    costs seconds, which nothing that only ranks should have to pay."""
    from audio_separator.separator import Separator

    sep = Separator(log_level=40, output_dir=str(work_dir),
                    model_file_dir=str(model_dir), use_soundfile=True)
    sep.load_model(model_filename=GUITAR_MODEL)
    return sep


def isolate_guitar(mix: np.ndarray, sr: int, separator: Any,
                   work_dir: pathlib.Path, tag: str = "mix") -> np.ndarray | None:
    """Return the guitar stem, or ``None`` if the separator produced none.

    ``None`` is a real outcome, not an error: on some material htdemucs_6s
    returns nothing for guitar, and the caller has to say so out loud rather
    than pass silence downstream as if it were a stem.
    """
    import soundfile as sf
    from scipy import signal

    work_dir.mkdir(parents=True, exist_ok=True)
    src = work_dir / f"{tag}.wav"
    peak = float(np.abs(mix).max()) + 1e-12
    sf.write(src, mix / peak * 0.98, sr)

    outputs = separator.separate(str(src))
    guitar = [p for p in outputs if "guitar" in str(p).lower()]
    try:
        if not guitar:
            return None
        path = pathlib.Path(guitar[0])
        if not path.is_absolute():
            path = work_dir / path
        y, file_sr = sf.read(path, always_2d=True)
        y = y.mean(axis=1)
        if file_sr != sr:
            y = signal.resample_poly(y, sr, file_sr)
        return y * peak
    finally:
        # htdemucs_6s writes all six stems per call and nothing deletes them.
        # A long session would otherwise leave gigabytes behind, and a full
        # disk shows up as a silently truncated write, not as an error.
        for p in [src, *(pathlib.Path(o) if pathlib.Path(o).is_absolute()
                         else work_dir / o for o in outputs)]:
            p.unlink(missing_ok=True)
