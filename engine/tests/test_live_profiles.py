"""The profile catalogue.

The rule this file exists to hold: an unloadable `.nam` stays in the listing
with its reason. Hiding it makes a bad file look like a missing file, and the
user goes hunting in Explorer instead of reading the sentence that would have
told them what is wrong with it.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from tonehound.config import SAMPLE_RATE
from live.profiles import ProfileCatalogue, profile_id_for

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROFILE_DIR = ROOT / "assets" / "dev_profiles"

needs_profiles = pytest.mark.skipif(
    not PROFILE_DIR.is_dir() or not list(PROFILE_DIR.glob("*.nam")),
    reason="no dev profiles present")


@pytest.fixture(scope="module")
def catalogue() -> ProfileCatalogue:
    cat = ProfileCatalogue(PROFILE_DIR)          # no cache: probe for real
    cat.scan()
    return cat


# -- ids -------------------------------------------------------------------


def test_the_id_is_content_addressed_and_stable() -> None:
    assert profile_id_for(b"abc") == profile_id_for(b"abc")
    assert profile_id_for(b"abc") != profile_id_for(b"abd")
    assert len(profile_id_for(b"abc")) == 16
    assert all(c in "0123456789abcdef" for c in profile_id_for(b"abc"))


@needs_profiles
def test_a_rescan_gives_every_profile_the_same_id(catalogue: ProfileCatalogue) -> None:
    """The UI remembers a selection across restarts, so an id that moved would
    silently load a different amp than the one the user picked."""
    again = ProfileCatalogue(PROFILE_DIR)
    again.scan()
    assert set(again.entries) == set(catalogue.entries)


# -- scanning --------------------------------------------------------------


@needs_profiles
def test_every_file_on_disk_appears(catalogue: ProfileCatalogue) -> None:
    assert len(catalogue.entries) == len(list(PROFILE_DIR.glob("*.nam")))


@needs_profiles
def test_loadable_entries_carry_the_facts_the_ui_shows(
        catalogue: ProfileCatalogue) -> None:
    loadable = [e for e in catalogue.entries.values() if e.loadable]
    assert loadable
    for entry in loadable:
        assert entry.architecture in ("WaveNet", "LSTM", "Linear")
        assert entry.receptive_field and entry.receptive_field >= 1
        assert entry.weight_count and entry.weight_count > 0
        assert entry.unsupported_reason is None
        assert isinstance(entry.metadata, dict)


@needs_profiles
def test_rate_mismatch_is_reported_rather_than_resampled_away(
        catalogue: ProfileCatalogue) -> None:
    for entry in catalogue.entries.values():
        assert entry.rate_mismatch == (entry.sample_rate != SAMPLE_RATE)


def test_an_unsupported_architecture_stays_listed_with_its_reason(
        tmp_path: pathlib.Path) -> None:
    (tmp_path / "future.nam").write_text(json.dumps({
        "architecture": "A2", "version": "0.5.4", "config": {}, "weights": [],
        "sample_rate": 48000}))
    entries = ProfileCatalogue(tmp_path).scan()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.loadable is False
    assert entry.name == "future"
    assert "A2" in (entry.unsupported_reason or "")
    assert entry.to_json()["unsupported_reason"]


def test_a_corrupt_file_is_listed_too_rather_than_vanishing(
        tmp_path: pathlib.Path) -> None:
    (tmp_path / "truncated.nam").write_text('{"architecture": "WaveNet"')
    entry = ProfileCatalogue(tmp_path).scan()[0]
    assert entry.loadable is False
    assert entry.unsupported_reason


def test_the_listing_counts_totals_before_filtering(tmp_path: pathlib.Path) -> None:
    (tmp_path / "good.nam").write_text(json.dumps({
        "architecture": "Linear", "version": "0.5.4",
        "config": {"receptive_field": 2, "bias": False},
        "weights": [0.5, 0.5], "sample_rate": 48000}))
    (tmp_path / "bad.nam").write_text(json.dumps({
        "architecture": "A2", "version": "0.5.4", "config": {}, "weights": []}))
    catalogue = ProfileCatalogue(tmp_path)
    catalogue.scan()

    everything = catalogue.listing()
    assert everything["total"] == 2 and everything["unsupported"] == 1
    assert len(everything["profiles"]) == 2

    filtered = catalogue.listing("goo")
    assert len(filtered["profiles"]) == 1
    assert filtered["total"] == 2, "total counts the library, not the filter"


@needs_profiles
def test_filtering_is_case_insensitive(catalogue: ProfileCatalogue) -> None:
    name = next(iter(catalogue.entries.values())).name
    token = name[:5]
    assert catalogue.listing(token.lower())["profiles"]
    assert (len(catalogue.listing(token.lower())["profiles"])
            == len(catalogue.listing(token.upper())["profiles"]))


@needs_profiles
def test_a_render_name_resolves_to_something_loadable(
        catalogue: ProfileCatalogue) -> None:
    """The ranking only knows a render's file stem, so this lookup is the only
    link between a candidate and a profile the user can actually load."""
    renders = ROOT / ".cache" / "renders_djent"
    if not renders.is_dir():
        pytest.skip("no render index present")
    stems = [p.stem for p in renders.glob("*.npy")]
    assert stems
    missing = [s for s in stems if catalogue.for_name(s) is None]
    assert not missing, f"ranked renders with no profile behind them: {missing[:3]}"


# -- caching ---------------------------------------------------------------


def test_the_probe_cache_is_written_and_reused(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "profiles"
    source.mkdir()
    (source / "lin.nam").write_text(json.dumps({
        "architecture": "Linear", "version": "0.5.4",
        "config": {"receptive_field": 2, "bias": False},
        "weights": [0.5, 0.5], "sample_rate": 48000}))
    cache = tmp_path / "probe.json"

    first = ProfileCatalogue(source, cache)
    first.scan()
    assert cache.exists()

    payload = json.loads(cache.read_text())
    assert payload["version"] >= 1 and payload["profiles"]

    second = ProfileCatalogue(source, cache)
    second.scan()
    assert [e.to_json() for e in second.entries.values()] == \
           [e.to_json() for e in first.entries.values()]


def test_a_corrupt_cache_reprobes_instead_of_failing(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "profiles"
    source.mkdir()
    (source / "lin.nam").write_text(json.dumps({
        "architecture": "Linear", "version": "0.5.4",
        "config": {"receptive_field": 2, "bias": False},
        "weights": [0.5, 0.5], "sample_rate": 48000}))
    cache = tmp_path / "probe.json"
    cache.write_text("{not json")
    entries = ProfileCatalogue(source, cache).scan()
    assert entries[0].loadable
