"""File or link, both arriving as the same thing.

The tests that matter here are the two that are easy to get wrong and hard to
notice: telling a Windows path from a URL (``C:\\song.mp3`` parses with a scheme
of ``c``, so a naive check hands every absolute path on this platform to
yt-dlp), and the download cache actually being used, because a link that
re-downloads on every run turns a tuning loop into a rate limit.

Nothing here reaches the network. The yt-dlp call is the seam, and it is
replaced.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import soundfile as sf

from tonehound import ingest
from tonehound.config import SAMPLE_RATE
from tonehound.ingest import IngestError, Source


def _tone(seconds: float, sr: int = SAMPLE_RATE, hz: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


@pytest.fixture()
def song(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "song.wav"
    sf.write(path, _tone(6.0), SAMPLE_RATE)
    return path


# -- telling the two kinds of input apart ----------------------------------


@pytest.mark.parametrize("spec", [
    "https://www.youtube.com/watch?v=abc",
    "http://example.com/track.mp3",
    "https://soundcloud.com/x/y",
])
def test_urls_are_recognised(spec: str) -> None:
    assert ingest.is_url(spec)


@pytest.mark.parametrize("spec", [
    r"C:\Users\me\Music\song.mp3",     # parses with scheme "c"
    r"D:/songs/take.wav",
    "/home/me/song.flac",
    "assets/song_train/x.mp3",
    "song.mp3",
    "ftp://host/file.mp3",             # a scheme, but not one to fetch
    "",
])
def test_paths_and_other_schemes_are_not_urls(spec: str) -> None:
    assert not ingest.is_url(spec)


# -- files -----------------------------------------------------------------


def test_a_local_file_loads_as_a_windowed_source(song: pathlib.Path) -> None:
    src = ingest.load(song, cache_dir=song.parent, seconds=2.0)
    assert isinstance(src, Source)
    assert src.origin == "file" and src.url is None
    assert src.name == "song"
    assert abs(src.duration_s - 2.0) < 0.05
    assert abs(src.full_duration_s - 6.0) < 0.05


def test_seconds_none_keeps_the_whole_file(song: pathlib.Path) -> None:
    src = ingest.load(song, cache_dir=song.parent, seconds=None)
    assert abs(src.duration_s - 6.0) < 0.05


def test_start_offset_moves_the_window(song: pathlib.Path) -> None:
    src = ingest.load(song, cache_dir=song.parent, seconds=2.0, start_s=3.0)
    assert src.start_s == 3.0
    assert abs(src.duration_s - 2.0) < 0.05


def test_a_missing_file_is_a_coded_error_not_a_traceback(
        tmp_path: pathlib.Path) -> None:
    with pytest.raises(IngestError) as exc:
        ingest.load(tmp_path / "absent.mp3", cache_dir=tmp_path)
    assert exc.value.code == "not_found"


# -- links -----------------------------------------------------------------


def test_a_cached_download_is_reused_instead_of_refetched(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
        song: pathlib.Path) -> None:
    url = "https://example.com/watch?v=cached"
    key = ingest._cache_key(url)
    (tmp_path / f"{key}.json").write_text(json.dumps({
        "path": str(song), "url": url, "title": "Cached Song",
        "uploader": "someone", "duration_s": 6.0}), "utf-8")

    def explode(*_a: object, **_k: object) -> None:
        raise AssertionError("a cached URL must not be downloaded again")

    monkeypatch.setattr(ingest, "_download_with_library", explode)
    monkeypatch.setattr(ingest, "_download_with_cli", explode)

    src = ingest.load(url, cache_dir=tmp_path, seconds=1.0)
    assert src.origin == "url" and src.url == url
    assert src.name == "Cached Song"


def test_the_cache_key_is_stable_and_url_specific() -> None:
    a = ingest._cache_key("https://example.com/a")
    assert a == ingest._cache_key("https://example.com/a  ")
    assert a != ingest._cache_key("https://example.com/b")


def test_a_cache_entry_whose_file_vanished_is_ignored(
        tmp_path: pathlib.Path) -> None:
    key = ingest._cache_key("https://example.com/gone")
    (tmp_path / f"{key}.json").write_text(
        json.dumps({"path": str(tmp_path / "missing.wav")}), "utf-8")
    assert ingest._cached(tmp_path, key) is None


def test_fetch_writes_a_sidecar_naming_the_downloaded_file(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
        song: pathlib.Path) -> None:
    def fake(url: str, cache_dir: pathlib.Path, key: str, progress: object):
        return song, {"title": "Fake Track", "uploader": "band",
                      "duration": 6, "webpage_url": url, "id": "xyz"}

    monkeypatch.setattr(ingest, "_download_with_library", fake)
    path, meta = ingest.fetch("https://example.com/new", tmp_path)
    assert path == song and meta["title"] == "Fake Track"

    sidecar = tmp_path / f"{ingest._cache_key('https://example.com/new')}.json"
    assert json.loads(sidecar.read_text("utf-8"))["title"] == "Fake Track"


def test_a_link_becomes_a_source_carrying_its_title_and_url(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
        song: pathlib.Path) -> None:
    monkeypatch.setattr(ingest, "_download_with_library",
                        lambda url, d, k, p: (song, {"title": "Some Song",
                                                     "webpage_url": url}))
    src = ingest.load("https://example.com/x", cache_dir=tmp_path, seconds=1.0)
    assert (src.origin, src.name, src.url) == ("url", "Some Song",
                                               "https://example.com/x")


def test_without_yt_dlp_the_error_says_how_to_install_it(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    def no_module(*_a: object, **_k: object) -> None:
        raise ImportError("no yt_dlp")

    monkeypatch.setattr(ingest, "_download_with_library", no_module)
    monkeypatch.setattr(ingest.shutil, "which", lambda _n: None)
    with pytest.raises(IngestError) as exc:
        ingest.fetch("https://example.com/x", tmp_path)
    assert exc.value.code == "yt_dlp_missing"
    assert "pip install yt-dlp" in exc.value.message
