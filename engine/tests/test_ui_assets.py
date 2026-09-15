"""Guard the two properties of the live UI that are easy to break silently.

There is no JavaScript test runner in this project and adding one would drag in
a toolchain the plugin build cannot use. But two classes of regression here are
worth catching in CI anyway, because both look *better* when broken and would
sail through a visual review:

1. **The honesty requirements of docs/ui_protocol.md §5.** A cleaner-looking
   results panel is exactly what you get by deleting the caveat, burying the
   isolated-stem player under the ranking, or dropping the raw distance in
   favour of a confident-looking order. The matcher is a rough draft (~0.176
   top-1 against unseen performances, with a shared-capturer confound), so
   those elements are the product, not decoration.

2. **Dependency freedom.** The UI is destined for a JUCE WebView with no
   bundler and no network. One `<script src="https://cdn...">` works fine on a
   dev machine and produces a blank panel in the plugin.

These are structural assertions -- element ids, source order, absence of remote
URLs -- not snapshots of wording, so ordinary editing does not trip them.
"""

from __future__ import annotations

import pathlib
import re

import pytest

UI = pathlib.Path(__file__).resolve().parents[1] / "live" / "ui"
FILES = {
    "index.html": UI / "index.html",
    "app.css": UI / "css" / "app.css",
    "client.js": UI / "js" / "client.js",
    "mock.js": UI / "js" / "mock.js",
    "app.js": UI / "js" / "app.js",
    "waveform.js": UI / "js" / "waveform.js",
    "meters.js": UI / "js" / "meters.js",
    "wav.js": UI / "js" / "wav.js",
}


def read(name: str) -> str:
    return FILES[name].read_text(encoding="utf-8")


@pytest.mark.parametrize("name", sorted(FILES))
def test_file_present(name: str) -> None:
    assert FILES[name].is_file(), f"{name} is missing from {UI}"


def test_index_references_every_asset() -> None:
    """A file nobody loads is a file nobody notices has rotted."""
    html = read("index.html")
    for src in ("css/app.css", "js/wav.js", "js/client.js", "js/mock.js",
                "js/waveform.js", "js/meters.js", "js/app.js"):
        assert src in html, f"index.html does not load {src}"


def test_no_remote_resources() -> None:
    """No CDN, no web fonts: the JUCE WebView has no network."""
    remote = re.compile(r"""(?:src|href)\s*=\s*["']https?://""", re.I)
    for name in ("index.html", "app.css"):
        hits = remote.findall(read(name))
        assert not hits, f"{name} loads a remote resource; the plugin WebView cannot"
    # url(https://...) in CSS would smuggle a web font in the same way.
    assert "url(http" not in read("app.css").replace(" ", "")


def test_stem_player_precedes_the_ranking() -> None:
    """§5.1: the isolated stem is the user's own check on the separator.

    Source order is the check, because it is what decides which one a person
    sees first on a screen that scrolls.
    """
    html = read("index.html")
    stem = html.index('id="stem-audio"')
    ranking = html.index('id="candidates"')
    assert stem < ranking, "the isolated-stem player must come before the ranking"


def test_separation_failure_is_loud() -> None:
    html, js = read("index.html"), read("app.js")
    assert 'id="stem-flag"' in html
    assert "separation_ok" in js, "app.js never reads separation_ok"
    assert "SEPARATION FAILED" in js, "a failed separation must say so in words"


def test_caveat_is_rendered_verbatim() -> None:
    """§5.3: server-owned text, in the flow, not behind a tooltip."""
    html, js = read("index.html"), read("app.js")
    assert 'id="caveat"' in html
    assert 'class="caveat"' in html
    assert re.search(r"""\$\("#caveat"\)\.textContent\s*=\s*r\.caveat""", js), \
        "the caveat must be printed from the server's own string"
    # title/aria-label would mean "hidden behind an icon".
    assert not re.search(r'id="caveat"[^>]*(title=|aria-label=)', html)


def test_every_candidate_shows_its_distance_in_context() -> None:
    """§5.2: the raw number, placed against the spread of the whole index."""
    js = read("app.js")
    assert "c.distance.toFixed" in js, "candidates must show the raw distance"
    assert "distance_range" in js, "distance must be placed against the index range"
    assert "dist-note" in js, "the UI must say whether the top-N are bunched"


def test_rank_one_is_judged_against_rank_two() -> None:
    """§5.2, the half that is easy to get subtly wrong.

    Reading only the rank-1..rank-N span and concluding "rank 1 is measurably
    closer" is false whenever rank 2 sits on top of rank 1, which happens
    constantly with a matcher that is right ~18% of the time. The note must
    look at the gap between the top two, and must be able to say they are tied.
    """
    js = read("app.js")
    assert re.search(r"cands\[1\]\.distance\s*-\s*cands\[0\]\.distance", js), \
        "the distance note must measure rank 1 against rank 2, not just the top-N span"
    assert "effectively tied" in js, \
        "the UI must be able to say the top two are indistinguishable"


def test_space_does_not_steal_focus_from_buttons() -> None:
    """The page is full of buttons (Play live, per-candidate transports) and a
    global Space handler that preventDefaults unconditionally makes every one
    of them unreachable from the keyboard."""
    js = read("app.js")
    space = js[js.index('e.code === "Space"'):]
    assert '"BUTTON"' in space[:600], "Space must yield to a focused button"


def test_driver_supplied_names_are_escaped() -> None:
    """Device names come from the OS. `<option>` labels are the only markup
    this UI builds by concatenation, so they are the only place a stray angle
    bracket can eat the rest of a list."""
    js = read("app.js")
    assert re.search(r"const esc\s*=", js), "app.js has no escaping helper"
    for field in ("a.name", "dev.name", "t.name"):
        assert f"esc({field})" in js, f"{field} is interpolated into markup unescaped"


def test_latency_is_labelled_as_driver_reported() -> None:
    """Nothing in this system measures a true acoustic round trip."""
    html = read("index.html").lower()
    assert "reported by driver" in html or "reported by the driver" in html


def test_perf_headroom_is_surfaced() -> None:
    """The user must be able to see dropouts rather than wonder."""
    html, js = read("index.html"), read("app.js")
    assert 'id="perf-headroom"' in html and 'id="perf-xruns"' in html
    assert "headroom_x" in js and "xruns" in js


def test_mock_mode_is_opt_in() -> None:
    """A page that faked telemetry by default would be indistinguishable from a
    broken engine."""
    js = read("app.js")
    assert 'qs.has("mock")' in js
    assert 'id="mock-banner"' in read("index.html"), "mock mode must announce itself"
