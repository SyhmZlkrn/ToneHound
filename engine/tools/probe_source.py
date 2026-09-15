"""One probe identity, shared by every measurement tool.

The synthetic Karplus-Strong probe and a real recorded DI are not variations on
each other -- measured against four real DI takes the synthetic one carries 180x
to 75000x too much energy above 5 kHz -- so every number a tool prints belongs
to exactly one of them and is meaningless without saying which. Hence the
banner on every run.

The render cache is the dangerous part. Nothing about a ``.npy`` of a render
says what was pushed through the profile to make it, so a cache filled by the
synthetic probe and read back by a real-DI run yields plausible, entirely wrong
results with no error anywhere. The cache therefore carries a stamp of the probe
that filled it, and a mismatch aborts rather than warns. An *unstamped* cache is
legacy, i.e. synthetic by construction, which is why a real-DI run refuses one
outright instead of assuming.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tonehound import real_di
from tonehound.config import SAMPLE_RATE
from tonehound.probe import PROBE_LEVELS_DBFS, build_probe, level_slices

STAMP_NAME = "_probe.json"


@dataclass
class ProbeSource:
    """The probe a run is using, plus everything needed to talk about it."""

    label: str
    audio: np.ndarray
    slices: list[slice]
    levels: tuple[float, ...]
    sample_rate: int
    synthetic: bool
    _digest: str | None = field(default=None, repr=False)

    @property
    def seconds(self) -> float:
        return len(self.audio) / self.sample_rate

    @property
    def digest(self) -> str:
        """Content hash of the rendered signal -- the cache key that matters."""
        if self._digest is None:
            buf = np.ascontiguousarray(self.audio, dtype=np.float64).tobytes()
            self._digest = hashlib.sha256(buf).hexdigest()[:16]
        return self._digest

    def banner(self) -> str:
        levels = ", ".join(f"{d:g}" for d in self.levels)
        pass_s = (self.slices[0].stop - self.slices[0].start) / self.sample_rate
        return (f"probe        : {self.label}\n"
                f"               {len(self.audio)} samples = {self.seconds:.2f} s total, "
                f"{len(self.slices)} passes of {pass_s:.2f} s at {levels} dBFS\n"
                f"               peak {np.abs(self.audio).max():.4f}, id {self.digest}")

    def stamp(self) -> dict:
        return {"digest": self.digest, "label": self.label, "synthetic": self.synthetic,
                "samples": len(self.audio), "sample_rate": self.sample_rate,
                "levels": list(self.levels)}


def synthetic_source(sr: int = SAMPLE_RATE) -> ProbeSource:
    return ProbeSource("synthetic Karplus-Strong (tonehound.probe.build_probe)",
                       build_probe(sr), level_slices(sr), tuple(PROBE_LEVELS_DBFS),
                       sr, synthetic=True)


def di_source(path: str | Path, seconds: float = 30.0,
              sr: int = SAMPLE_RATE) -> ProbeSource:
    path = Path(path)
    di = real_di.load_di(path, sr, max_seconds=seconds)
    return ProbeSource(f"real DI {path.name} (take trimmed to {len(di) / sr:.2f} s)",
                       real_di.probe_from_di(di, sr), real_di.slices_from_di(len(di), sr),
                       tuple(PROBE_LEVELS_DBFS), sr, synthetic=False)


def add_probe_args(ap: argparse.ArgumentParser, default_seconds: float = 30.0) -> None:
    ap.add_argument("--di", type=Path, default=None,
                    help="render a real recorded DI instead of the synthetic probe")
    ap.add_argument("--di-seconds", type=float, default=default_seconds,
                    help="trim the DI take to this many seconds (--di only)")


def probe_from_args(args: argparse.Namespace, sr: int = SAMPLE_RATE) -> ProbeSource:
    di = getattr(args, "di", None)
    if di is None:
        return synthetic_source(sr)
    return di_source(di, getattr(args, "di_seconds", 30.0), sr)


# --------------------------------------------------------------------------
# Cache stamping
# --------------------------------------------------------------------------


def read_stamp(cache: Path) -> dict | None:
    path = Path(cache) / STAMP_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _abort(lines: list[str]) -> None:
    sys.stdout.flush()   # keep the banner above the refusal when stdout is piped
    bar = "!" * 78
    print(f"\n{bar}", file=sys.stderr)
    for line in lines:
        print(line, file=sys.stderr)
    print(f"{bar}\n", file=sys.stderr)
    raise SystemExit(2)


def check_cache(cache: Path | None, src: ProbeSource, write: bool) -> None:
    """Bind a render cache to one probe, fatally.

    ``write`` marks the tool that fills the cache (gate_b); it stamps an
    unstamped directory. Read-only tools never stamp, so they cannot launder a
    cache they did not create into looking verified.
    """
    if cache is None:
        return
    cache = Path(cache)
    if write:
        cache.mkdir(parents=True, exist_ok=True)

    stamp = read_stamp(cache)
    if stamp is not None:
        if stamp.get("digest") != src.digest:
            _abort([
                f"REFUSING TO USE THE RENDER CACHE {cache}",
                "",
                f"  cache was filled by : {stamp.get('label', '?')}  "
                f"(id {stamp.get('digest', '?')}, {stamp.get('samples', '?')} samples)",
                f"  this run asks for   : {src.label}  "
                f"(id {src.digest}, {len(src.audio)} samples)",
                "",
                "Reusing it would produce plausible numbers for the wrong probe.",
                "Use a separate --cache directory per probe.",
            ])
        return

    populated = any(cache.glob("*.npy")) if cache.exists() else False
    if populated and not src.synthetic:
        _abort([
            f"REFUSING TO USE THE RENDER CACHE {cache}",
            "",
            "  It holds renders but carries no probe stamp, so it predates probe",
            "  tracking and was filled by the synthetic probe. This run asks for",
            f"  {src.label}.",
            "",
            "Use a separate --cache directory for the real-DI renders.",
        ])
    if populated:
        print(f"cache        : {cache} is unstamped (legacy) -- treating its renders as "
              f"synthetic-probe renders")
    if write:
        (cache / STAMP_NAME).write_text(json.dumps(src.stamp(), indent=2))
