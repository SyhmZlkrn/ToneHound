"""Everything the browser can point an `<audio>` element or a waveform at.

One store, one id namespace. §3 of the protocol says `snippet_id`, `crop_id`,
`di_id` and `render_id` are all opaque hex that `GET /audio/{id}.wav` accepts,
and the cheapest way to keep that promise is to actually have one dictionary --
a store per kind would eventually disagree with itself about which prefix means
what.

Three decisions worth explaining.

*Everything is decoded to 48 kHz mono float32 on the way in.* The engine has
exactly one sample rate, and a snippet that stayed at 44.1 kHz would have to be
resampled somewhere later, by whichever caller remembered. Mono because the
matcher analyses mono and a stereo waveform display would be showing something
the matcher never sees.

*The analysed audio is peak-normalised, and the stored audio is too.* This
mirrors `tools/try_match.py`, which normalises the clip it analyses. It matters
because the fingerprint's scalar half is not fully level-invariant, so a quiet
import and a loud one would rank differently -- a difference in the recording,
not in the amp.

*Crops are derived, not destructive.* `snippet.crop` is expected to fire
repeatedly while the user drags a handle, so a crop is a view recorded as a
sample range plus its own id, and re-cropping costs a slice. The full decode
stays until `snippet.forget`.

WAV bodies and peak arrays are both memoised, because the browser refetches
them: `<audio>` alone will issue a `Range` request per seek, and re-encoding a
15-second render for each one would be pure waste.
"""

from __future__ import annotations

import io
import pathlib
import wave
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tonehound import audioio
from tonehound.audioio import AudioError
from tonehound.config import SAMPLE_RATE

from .wsproto import token

PEAK_TARGET = audioio.PEAK_TARGET
"""Headroom left below full scale so a PCM16 round trip cannot clip."""

DEFAULT_BINS = 2000
MAX_BINS = 20000


class MediaError(Exception):
    """Something the user should be told about, with a protocol error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class Item:
    """One piece of audio the UI can play, draw, or match against."""

    id: str
    kind: str                     # "snippet" | "crop" | "di" | "stem" | "render"
    name: str
    samples: np.ndarray           # float32, mono, SAMPLE_RATE
    source_sample_rate: int = SAMPLE_RATE
    source_channels: int = 1
    parent: str | None = None     # snippet a crop came from
    meta: dict[str, Any] = field(default_factory=dict)
    _wav: bytes | None = field(default=None, repr=False)
    _peaks: dict[int, dict[str, Any]] = field(default_factory=dict, repr=False)

    @property
    def duration_s(self) -> float:
        return len(self.samples) / SAMPLE_RATE

    @property
    def audio_url(self) -> str:
        return f"/audio/{self.id}.wav"

    @property
    def peaks_url(self) -> str:
        return f"/peaks/{self.id}"

    def wav(self) -> bytes:
        if self._wav is None:
            self._wav = encode_wav(self.samples)
        return self._wav

    def peaks(self, bins: int) -> dict[str, Any]:
        bins = max(1, min(int(bins), MAX_BINS))
        cached = self._peaks.get(bins)
        if cached is None:
            cached = envelope(self.samples, bins)
            cached["id"] = self.id
            cached["duration_s"] = self.duration_s
            self._peaks[bins] = cached
        return cached


# --------------------------------------------------------------------------
# encode / decode
# --------------------------------------------------------------------------


def encode_wav(x: np.ndarray) -> bytes:
    """Mono 48 kHz PCM16. Written with `wave` rather than `soundfile` so the
    bytes come back without a temporary file on the way."""
    peak = float(np.abs(x).max()) if x.size else 0.0
    scaled = x if peak <= 1.0 else x / peak
    pcm = np.clip(np.rint(scaled * 32767.0), -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def decode(raw: bytes, filename: str = "") -> tuple[np.ndarray, int, int]:
    """Decode to mono float32 at 48 kHz. Returns ``(samples, source_sr, channels)``.

    The decoding itself lives in `tonehound.audioio`, because the CLI and the URL
    ingest path have to accept exactly the same range of files as this server
    does; only the error type differs, and it differs because the protocol
    speaks in codes.
    """
    try:
        return audioio.decode(raw, filename)
    except AudioError as exc:
        raise MediaError(exc.code, exc.message) from exc


def normalise(x: np.ndarray, target: float = PEAK_TARGET) -> np.ndarray:
    return audioio.normalise(x, target)


def envelope(x: np.ndarray, bins: int) -> dict[str, Any]:
    """min/max/rms per display bin, in one pass over the samples.

    `reduceat` rather than a reshape because the sample count is never a
    multiple of the bin count, and padding it would put a false zero-crossing
    in the last bin of every waveform.
    """
    n = len(x)
    if n == 0:
        zeros = [0.0] * bins
        return {"bins": bins, "min": zeros, "max": list(zeros), "rms": list(zeros)}

    edges = np.linspace(0, n, bins + 1).astype(np.int64)
    starts = np.minimum(edges[:-1], n - 1)
    # An empty bin (more bins than samples) would make reduceat read the wrong
    # element, so every bin is forced to span at least one sample.
    lo = np.minimum.reduceat(x, starts)
    hi = np.maximum.reduceat(x, starts)
    energy = np.add.reduceat(np.square(x.astype(np.float64)), starts)
    counts = np.maximum(np.diff(edges), 1)
    rms = np.sqrt(energy / counts)
    return {"bins": bins,
            "min": np.round(lo, 5).tolist(),
            "max": np.round(hi, 5).tolist(),
            "rms": np.round(rms, 5).tolist()}


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------


class MediaStore:
    """The one id namespace. Not thread-safe by design: it is touched only from
    the server's event loop, and adding a lock would invite calling it from a
    worker, where a 30-second decode would then block every other client."""

    def __init__(self) -> None:
        self.items: dict[str, Item] = {}
        self.default_di_id: str | None = None

    # -- lookup ----------------------------------------------------------

    def get(self, item_id: str, kind: str | None = None) -> Item:
        item = self.items.get(item_id)
        if item is None or (kind is not None and item.kind != kind):
            raise MediaError("not_found", f"no {kind or 'item'} with id {item_id!r}")
        return item

    def of_kind(self, kind: str) -> list[Item]:
        return [i for i in self.items.values() if i.kind == kind]

    # -- adding ----------------------------------------------------------

    def add(self, samples: np.ndarray, *, kind: str, name: str,
            source_sample_rate: int = SAMPLE_RATE, source_channels: int = 1,
            parent: str | None = None, meta: dict[str, Any] | None = None) -> Item:
        item = Item(id=token(), kind=kind, name=name,
                    samples=np.ascontiguousarray(samples, dtype=np.float32),
                    source_sample_rate=source_sample_rate,
                    source_channels=source_channels, parent=parent,
                    meta=meta or {})
        self.items[item.id] = item
        return item

    def load_di_directory(self, directory: pathlib.Path,
                          prefer: str = "Djent") -> list[Item]:
        """Register the DI takes that shipped with the app.

        ``prefer`` picks the default because the profile index is rendered
        through that take; auditioning through a different DI than the index was
        built from is allowed but is no longer the comparison the ranking made.
        """
        added: list[Item] = []
        for path in sorted(directory.glob("*.wav")):
            try:
                samples, source_sr, channels = decode(path.read_bytes(), path.name)
            except MediaError:
                continue
            added.append(self.add(samples, kind="di", name=path.stem,
                                  source_sample_rate=source_sr,
                                  source_channels=channels,
                                  meta={"path": str(path)}))
        if self.default_di_id is None and added:
            chosen = next((i for i in added if prefer.lower() in i.name.lower()),
                          added[0])
            self.default_di_id = chosen.id
        return added

    def crop(self, snippet_id: str, start_s: float, end_s: float,
             min_crop_s: float, max_crop_s: float) -> Item:
        """A new item over ``[start_s, end_s)`` of a snippet.

        Bounds are clamped rather than rejected -- a drag can overshoot the end
        of the waveform by a pixel and that is not an error -- but a crop that
        is still too short or too long after clamping is refused, because
        separation cost and fingerprint stability both depend on the length.
        """
        snippet = self.get(snippet_id, "snippet")
        duration = snippet.duration_s
        start = float(np.clip(start_s, 0.0, duration))
        end = float(np.clip(end_s, 0.0, duration))
        if end <= start:
            raise MediaError("crop_too_short", "the crop end must follow its start")
        if end - start < min_crop_s:
            raise MediaError(
                "crop_too_short",
                f"{end - start:.1f}s is under the {min_crop_s:.0f}s minimum; "
                "the fingerprint needs a few seconds of playing to settle")
        if end - start > max_crop_s:
            raise MediaError(
                "snippet_too_long",
                f"{end - start:.1f}s is over the {max_crop_s:.0f}s maximum; "
                "separation runs at about realtime, so a longer crop is a "
                "longer wait for the same answer")

        lo, hi = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
        item = self.add(normalise(snippet.samples[lo:hi]), kind="crop",
                        name=f"{snippet.name} [{start:.1f}-{end:.1f}s]",
                        parent=snippet_id,
                        meta={"start_s": start, "end_s": end})
        return item

    def forget(self, snippet_id: str) -> list[str]:
        """Drop a snippet and every crop derived from it. Returns the ids gone."""
        self.get(snippet_id, "snippet")
        gone = [snippet_id] + [i.id for i in self.items.values()
                               if i.parent == snippet_id]
        for item_id in gone:
            self.items.pop(item_id, None)
        return gone
