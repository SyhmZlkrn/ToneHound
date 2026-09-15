"""The expensive half of the server, in a shape a worker process can import.

Everything here is a plain module-level function taking and returning arrays,
because `ProcessPoolExecutor` on Windows pickles by qualified name -- a closure
or a bound method cannot cross into a worker. Keeping them in their own module
(rather than in `server.py`) also means a worker imports numpy, torch and UVR
without importing the server, its sockets or its media store.

The costs these hide, measured on a 15 s crop:

    separate       14.67 s   ~1.0x realtime, all cores. Dominates everything.
    fingerprint     0.19 s
    render          1.33 s per 8 s of DI

The separator is a module-level singleton because loading htdemucs_6s takes
seconds and a match is a repeated action. That is also why separation gets a
one-worker pool of its own in the server: with several workers the model would
be loaded again on whichever worker happened to pick up the next job, and the
user would pay for it every time.
"""

from __future__ import annotations

import pathlib
import tempfile
from typing import Any

import numpy as np

from tonehound import match, nam_render
from tonehound.config import GUITAR, SAMPLE_RATE

_separator: Any = None
_work_dir: pathlib.Path | None = None


def init_worker() -> None:
    """Silence a worker's stdout chatter.

    UVR draws tqdm bars on stderr and several of its dependencies log at INFO.
    In a worker that output has nowhere sensible to go: it interleaves with the
    server's own log and tells the user nothing they are not already being sent
    as `match.progress`.
    """
    import logging
    import os

    os.environ.setdefault("TQDM_DISABLE", "1")
    logging.getLogger().setLevel(logging.ERROR)


def _ensure_separator(model_dir: str) -> tuple[Any, pathlib.Path]:
    global _separator, _work_dir
    if _separator is None:
        _work_dir = pathlib.Path(tempfile.mkdtemp(prefix="tonehound_sep_"))
        _separator = match.make_separator(pathlib.Path(model_dir), _work_dir)
    assert _work_dir is not None
    return _separator, _work_dir


def separate_job(samples: np.ndarray, model_dir: str) -> np.ndarray | None:
    """Isolate the guitar. ``None`` means the separator produced no guitar stem
    -- a real outcome the caller must report, not silently paper over."""
    separator, work_dir = _ensure_separator(model_dir)
    return match.isolate_guitar(np.asarray(samples, dtype=np.float64),
                                SAMPLE_RATE, separator, work_dir, "query")


def fingerprint_job(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shape and scalar halves of one query fingerprint."""
    fp = match.fingerprint(np.asarray(samples, dtype=np.float64),
                           SAMPLE_RATE, GUITAR)
    return fp.shape_vector, fp.scalar_vector


def render_job(nam_path: str, di: np.ndarray) -> np.ndarray:
    """A DI take through one profile, offline.

    float64 and the whole-signal `render()` on purpose: this is not the live
    path, and the offline renderer is the one holding 1.1e-06 parity with
    neural-amp-modeler. Peak-normalised on the way out so a quiet profile and a
    loud one can be compared without reaching for the volume control between
    them -- which would defeat the comparison.
    """
    model = nam_render.load(nam_path)
    out = model.render(np.asarray(di, dtype=np.float64))
    peak = float(np.abs(out).max())
    return (out / peak * 0.95 if peak > 0 else out).astype(np.float32)


def build_index_job(render_dir: str, cache_path: str) -> dict[str, Any]:
    """Whole index build inside one worker, for the no-pool fallback path."""
    index = match.ProfileIndex.load_or_build(pathlib.Path(render_dir),
                                             pathlib.Path(cache_path))
    return {"names": index.names, "shape": index.shape,
            "scalars": index.scalars, "di_name": index.di_name}
