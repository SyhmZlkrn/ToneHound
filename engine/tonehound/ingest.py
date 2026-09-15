"""Turn "what the user gave us" into audio, whether that was a file or a link.

Step 0 of the pipeline. A user has a song in mind; they either have it on disk
or they have a URL for it. Both have to arrive at the same place: 48 kHz mono
float32, plus enough provenance to put a title on the result page.

Links go through **yt-dlp**, which is what StemRoller spawns for the same job.
It is used as a library when it imports and as a subprocess when it does not,
because a pip install and a standalone `yt-dlp.exe` are both normal ways to
have it and neither should be the one that works.

*The download is cached by URL.* Re-running a match on the same link during
tuning is the common case, and a cache turns a 20-second fetch into a file
read. The key is a hash of the URL, so two different links never collide and
the same link never re-downloads.

*Only the analysed window is kept in memory, but the whole file is fetched.*
yt-dlp can download a byte range, but the range has to land on keyframes and a
song is a few megabytes; paying for the whole file buys a cache entry that any
later window can be cut from.

Nothing here uploads or publishes: a URL is fetched, decoded, and analysed
locally.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np

from .audioio import AudioError, ffmpeg_exe, read, window
from .config import SAMPLE_RATE

Progress = Callable[[str, float], None]
"""``(message, fraction)``. Fraction is -1 when the total is unknown."""

DEFAULT_SECONDS = 20.0
"""How much of a song the matcher listens to by default.

Separation runs at about realtime, so this is most of the wall clock. Twenty
seconds is long enough for a fingerprint to settle over a full phrase and short
enough that a mismatch is found in under a minute.
"""


@dataclass
class Source:
    """One piece of audio the pipeline was pointed at, and where it came from."""

    samples: np.ndarray            # float32 mono or frames × channels @ SAMPLE_RATE
    name: str
    origin: str                    # "file" | "url"
    url: str | None = None
    path: pathlib.Path | None = None
    full_duration_s: float = 0.0   # before windowing
    start_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return len(self.samples) / SAMPLE_RATE


class IngestError(Exception):
    """A fetch that failed in a way the user can act on."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def is_url(spec: str) -> bool:
    """True for something that should be fetched rather than opened.

    Deliberately narrow: only http and https. A Windows path like
    ``C:\\music\\song.mp3`` parses with a scheme of ``c``, so testing for "has a
    scheme" would send every absolute path on this platform to yt-dlp.
    """
    try:
        parsed = urlparse(spec.strip())
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


# --------------------------------------------------------------------------
# fetching a link
# --------------------------------------------------------------------------


def _cache_key(url: str) -> str:
    return hashlib.sha1(url.strip().encode("utf-8")).hexdigest()[:16]


def _cached(cache_dir: pathlib.Path, key: str) -> tuple[pathlib.Path, dict] | None:
    """A previous download of this URL, if both halves of it survived."""
    sidecar = cache_dir / f"{key}.json"
    if not sidecar.exists():
        return None
    try:
        meta = json.loads(sidecar.read_text("utf-8"))
        path = pathlib.Path(meta["path"])
    except Exception:
        return None
    return (path, meta) if path.exists() else None


def _ydl_options(cache_dir: pathlib.Path, key: str) -> dict[str, Any]:
    ffmpeg = ffmpeg_exe()
    opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(cache_dir / f"{key}.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        # A playlist URL that slipped past `noplaylist` would otherwise fetch
        # every track before anyone noticed.
        "playlist_items": "1",
    }
    if ffmpeg:
        opts["ffmpeg_location"] = str(pathlib.Path(ffmpeg).parent)
    return opts


def _download_with_library(url: str, cache_dir: pathlib.Path, key: str,
                           progress: Progress | None) -> tuple[pathlib.Path, dict]:
    import yt_dlp

    opts = _ydl_options(cache_dir, key)
    if progress:
        def hook(status: dict) -> None:
            if status.get("status") != "downloading":
                return
            total = status.get("total_bytes") or status.get("total_bytes_estimate")
            done = status.get("downloaded_bytes") or 0
            progress("downloading", done / total if total else -1.0)
        opts["progress_hooks"] = [hook]

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    if info.get("_type") == "playlist":
        entries = info.get("entries") or []
        if not entries:
            raise IngestError("download_failed", f"{url} contained nothing playable")
        info = entries[0]
    path = pathlib.Path(ydl.prepare_filename(info))
    if not path.exists():  # a postprocessor changed the extension under us
        matches = sorted(cache_dir.glob(f"{key}.*"))
        matches = [p for p in matches if p.suffix != ".json"]
        if not matches:
            raise IngestError("download_failed", f"yt-dlp wrote no file for {url}")
        path = matches[0]
    return path, info


def _download_with_cli(url: str, cache_dir: pathlib.Path, key: str,
                       progress: Progress | None) -> tuple[pathlib.Path, dict]:
    """StemRoller's approach: spawn the yt-dlp executable.

    Used when the module will not import. The metadata comes from a separate
    ``--dump-json`` call rather than being parsed out of the progress output,
    because scraping a progress bar for a title is how a rename breaks a build.
    """
    exe = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
    if exe is None:
        raise IngestError(
            "yt_dlp_missing",
            "fetching a link needs yt-dlp; install it with `pip install yt-dlp`")

    if progress:
        progress("downloading", -1.0)
    args = [exe, "--no-playlist", "--playlist-items", "1",
            "-f", "bestaudio/best", "--no-progress", "--quiet",
            "-o", str(cache_dir / f"{key}.%(ext)s")]
    ffmpeg = ffmpeg_exe()
    if ffmpeg:
        args += ["--ffmpeg-location", str(pathlib.Path(ffmpeg).parent)]
    proc = subprocess.run(args + [url], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise IngestError("download_failed",
                          f"yt-dlp failed: {proc.stderr.strip()[-300:] or 'no output'}")

    written = [p for p in sorted(cache_dir.glob(f"{key}.*")) if p.suffix != ".json"]
    if not written:
        raise IngestError("download_failed", f"yt-dlp wrote no file for {url}")

    info: dict[str, Any] = {}
    dump = subprocess.run([exe, "--no-playlist", "--dump-json", "--quiet", url],
                          capture_output=True, text=True, check=False)
    if dump.returncode == 0 and dump.stdout.strip():
        try:
            info = json.loads(dump.stdout.splitlines()[0])
        except json.JSONDecodeError:
            pass
    return written[0], info


def fetch(url: str, cache_dir: pathlib.Path,
          progress: Progress | None = None) -> tuple[pathlib.Path, dict[str, Any]]:
    """Download ``url`` to ``cache_dir``. Returns the file and its metadata."""
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(url)

    hit = _cached(cache_dir, key)
    if hit is not None:
        if progress:
            progress("cached", 1.0)
        return hit

    try:
        path, info = _download_with_library(url, cache_dir, key, progress)
    except ImportError:
        path, info = _download_with_cli(url, cache_dir, key, progress)

    meta = {
        "path": str(path),
        "url": info.get("webpage_url") or url,
        "title": info.get("title") or path.stem,
        "uploader": info.get("uploader") or info.get("channel") or "",
        "duration_s": float(info.get("duration") or 0.0),
        "extractor": info.get("extractor_key") or info.get("extractor") or "",
        "video_id": info.get("id") or "",
    }
    try:
        (cache_dir / f"{key}.json").write_text(json.dumps(meta), "utf-8")
    except OSError:
        pass  # an unwritable cache costs a re-download, not a failure
    return path, meta


# --------------------------------------------------------------------------
# the one entry point
# --------------------------------------------------------------------------


def load(spec: str | pathlib.Path, *, cache_dir: pathlib.Path,
         seconds: float | None = DEFAULT_SECONDS, start_s: float = 0.0,
         progress: Progress | None = None, mono: bool = True) -> Source:
    """Resolve a path or a URL to a `Source`.

    ``seconds=None`` keeps the whole thing, which is what a caller that intends
    to let the user crop interactively wants; every other caller should pass a
    window, because separation cost is linear in it.
    """
    spec = str(spec).strip()

    if is_url(spec):
        path, meta = fetch(spec, pathlib.Path(cache_dir), progress)
        origin, url = "url", meta.get("url", spec)
        name = meta.get("title") or path.stem
    else:
        path = pathlib.Path(spec)
        if not path.exists():
            raise IngestError("not_found", f"no such file: {path}")
        origin, url, meta = "file", None, {}
        name = path.stem

    if progress:
        progress("decoding", -1.0)
    try:
        full = read(path, mono=mono)
    except AudioError as exc:
        raise IngestError(exc.code, exc.message) from exc

    full_duration = len(full) / SAMPLE_RATE
    samples = full if seconds is None and start_s == 0.0 else window(
        full, seconds=seconds, start_s=start_s)

    return Source(samples=samples, name=name, origin=origin, url=url, path=path,
                  full_duration_s=full_duration, start_s=start_s, meta=meta)
