"""The profile library: what is on disk, which of it will load, and why not.

`profiles.list` in §4.5 of the protocol is emphatic that an unloadable `.nam`
stays in the list, greyed, with its reason. That is the whole reason this is a
catalogue rather than a `glob`: the alternative -- silently listing only what
worked -- makes a bad file look like a missing file, and the user goes looking
for it in Explorer instead of reading the sentence that would have told them
`nam_render` does not implement architecture A2.

`profile_id` is the first 16 hex of the SHA-1 of the file bytes. Stable across
restarts, so the UI can remember a selection; content-addressed, so a profile
that was edited becomes a different profile rather than quietly changing under
a saved reference. Two identical files with different names collide onto one
id, which is correct -- they are the same capture.

Probing a file means parsing it, and parsing 35 profiles costs about a second
of pure JSON. That is small, but it happens before the first frame of UI and
the answer never changes for a given file, so it is cached in a small JSON
sidecar keyed by the content hash. A cache miss is a re-probe, never an error.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

from tonehound import nam_render
from tonehound.config import SAMPLE_RATE

CACHE_VERSION = 1


@dataclass
class ProfileEntry:
    """One `.nam` on disk, in the shape §4.5 asks for."""

    profile_id: str
    name: str
    path: pathlib.Path
    architecture: str = ""
    sample_rate: int = SAMPLE_RATE
    receptive_field: int | None = None
    weight_count: int | None = None
    loadable: bool = False
    unsupported_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rate_mismatch(self) -> bool:
        return self.sample_rate != SAMPLE_RATE

    def to_json(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "architecture": self.architecture,
            "sample_rate": self.sample_rate,
            "receptive_field": self.receptive_field,
            "weight_count": self.weight_count,
            "loadable": self.loadable,
            "unsupported_reason": self.unsupported_reason,
            "rate_mismatch": self.rate_mismatch,
            "metadata": self.metadata,
        }


def profile_id_for(raw: bytes) -> str:
    return hashlib.sha1(raw).hexdigest()[:16]


def _probe(raw: bytes, path: pathlib.Path) -> dict[str, Any]:
    """Load far enough to describe the file, and turn every failure into text.

    `UnsupportedArchitecture` and a corrupt file are different sentences to the
    user but the same outcome here: an entry that lists but will not load.
    """
    try:
        model = nam_render.load(json.loads(raw.decode("utf-8")))
    except nam_render.UnsupportedArchitecture as exc:
        arch = getattr(exc, "architecture", "") or ""
        return {"architecture": str(arch), "loadable": False,
                "unsupported_reason": str(exc)}
    except Exception as exc:
        return {"architecture": "", "loadable": False,
                "unsupported_reason": f"{type(exc).__name__}: {exc}"[:400]}
    return {
        "architecture": model.architecture,
        "sample_rate": model.sample_rate,
        "receptive_field": model.receptive_field,
        "weight_count": model.weight_count,
        "loadable": True,
        "unsupported_reason": None,
        "metadata": model.metadata if isinstance(model.metadata, dict) else {},
    }


class ProfileCatalogue:
    """Every `.nam` under a directory, probed once and remembered."""

    def __init__(self, directory: pathlib.Path,
                 cache_path: pathlib.Path | None = None) -> None:
        self.directory = pathlib.Path(directory)
        self.cache_path = cache_path
        self.entries: dict[str, ProfileEntry] = {}
        self.by_name: dict[str, ProfileEntry] = {}

    def scan(self) -> list[ProfileEntry]:
        cache = self._read_cache()
        entries: dict[str, ProfileEntry] = {}
        dirty = False

        for path in sorted(self.directory.glob("*.nam")):
            raw = path.read_bytes()
            pid = profile_id_for(raw)
            probed = cache.get(pid)
            if probed is None:
                probed = _probe(raw, path)
                cache[pid] = probed
                dirty = True
            entries[pid] = ProfileEntry(
                profile_id=pid, name=path.stem, path=path,
                architecture=probed.get("architecture", ""),
                sample_rate=int(probed.get("sample_rate", SAMPLE_RATE)),
                receptive_field=probed.get("receptive_field"),
                weight_count=probed.get("weight_count"),
                loadable=bool(probed.get("loadable")),
                unsupported_reason=probed.get("unsupported_reason"),
                metadata=probed.get("metadata") or {})

        self.entries = entries
        self.by_name = {e.name: e for e in entries.values()}
        if dirty:
            self._write_cache(cache)
        return list(entries.values())

    # -- queries ---------------------------------------------------------

    def get(self, profile_id: str) -> ProfileEntry | None:
        return self.entries.get(profile_id)

    def for_name(self, name: str) -> ProfileEntry | None:
        """A render in the index is named after its profile's file stem, which
        is the only link between the ranking and something loadable."""
        return self.by_name.get(name)

    def listing(self, query: str = "") -> dict[str, Any]:
        entries = list(self.entries.values())
        matched = ([e for e in entries if query.lower() in e.name.lower()]
                   if query else entries)
        matched.sort(key=lambda e: e.name.lower())
        return {
            "profiles": [e.to_json() for e in matched],
            "total": len(entries),
            "unsupported": sum(1 for e in entries if not e.loadable),
        }

    # -- cache -----------------------------------------------------------

    def _read_cache(self) -> dict[str, Any]:
        if self.cache_path is None or not self.cache_path.exists():
            return {}
        try:
            data = json.loads(self.cache_path.read_text("utf-8"))
            if data.get("version") != CACHE_VERSION:
                return {}
            return data.get("profiles", {})
        except Exception:
            return {}

    def _write_cache(self, profiles: dict[str, Any]) -> None:
        if self.cache_path is None:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"version": CACHE_VERSION,
                                       "profiles": profiles}), "utf-8")
            tmp.replace(self.cache_path)
        except OSError:
            pass  # a read-only cache directory slows startup; it must not stop it
