"""The TONE3000 client: PKCE, tokens, query shapes, and cached downloads.

Nothing here reaches TONE3000. The seams are `requests.get`, `requests.post`
and the browser callback, and all three are replaced -- a test suite that needed
an API key and a network would be a test suite nobody runs.

What is worth pinning down is the stuff a wrong guess makes silently wrong: the
search query's separators (gears join on underscores, creators on commas,
because a username may contain an underscore), the state check that stops a
callback from another sign-in being accepted, and the file naming, which is the
only thread connecting "the ranking chose this file" back to a Tone ID.
"""

from __future__ import annotations

import json
import pathlib
import time
import urllib.parse

import pytest

from tonehound import tone3000
from tonehound.tone3000 import (Credentials, Model, Tone, Tone3000Client,
                             Tone3000Error, parse_model_reference)

KEY = "t3k_pub_testkey"


@pytest.fixture()
def api(tmp_path: pathlib.Path) -> Tone3000Client:
    return Tone3000Client(KEY, cache_dir=tmp_path)


@pytest.fixture()
def connected(api: Tone3000Client) -> Tone3000Client:
    api._creds = Credentials("access-1", "refresh-1", time.time() + 3600)
    return api


class FakeResponse:
    def __init__(self, payload: object = None, status: int = 200,
                 content: bytes = b"") -> None:
        self._payload, self.status_code, self._content = payload, status, content
        self.text = json.dumps(payload) if payload is not None else ""

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def json(self) -> object:
        return self._payload

    def iter_content(self, chunk_size: int = 0):
        yield self._content


# -- credentials -----------------------------------------------------------


def test_a_key_from_the_environment_is_found(monkeypatch: pytest.MonkeyPatch,
                                             tmp_path: pathlib.Path) -> None:
    monkeypatch.setenv(tone3000.KEY_ENV, "t3k_pub_from_env")
    assert Tone3000Client(cache_dir=tmp_path).key == "t3k_pub_from_env"


def test_an_explicit_key_wins_over_the_environment(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    monkeypatch.setenv(tone3000.KEY_ENV, "t3k_pub_env")
    assert Tone3000Client("t3k_pub_arg", cache_dir=tmp_path).key == "t3k_pub_arg"


def test_a_key_file_is_the_last_resort(monkeypatch: pytest.MonkeyPatch,
                                       tmp_path: pathlib.Path) -> None:
    monkeypatch.delenv(tone3000.KEY_ENV, raising=False)
    key_dir = tmp_path / "tone3000"
    key_dir.mkdir(parents=True)
    (key_dir / "publishable_key.txt").write_text(" t3k_pub_file \n", "utf-8")
    assert Tone3000Client(cache_dir=tmp_path).key == "t3k_pub_file"


def test_with_no_key_anywhere_the_client_reports_itself_unconfigured(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    monkeypatch.delenv(tone3000.KEY_ENV, raising=False)
    assert not Tone3000Client(cache_dir=tmp_path).configured


def test_not_being_connected_is_a_state_not_a_crash(api: Tone3000Client) -> None:
    assert not api.connected
    with pytest.raises(Tone3000Error) as exc:
        api._access_token()
    assert exc.value.code == "not_connected"


def test_a_token_is_stale_a_minute_before_it_expires() -> None:
    assert Credentials("a", "r", time.time() + 30).stale
    assert not Credentials("a", "r", time.time() + 3600).stale


def test_tokens_survive_a_restart_and_a_corrupt_file_does_not(
        api: Tone3000Client, tmp_path: pathlib.Path) -> None:
    api._write_tokens(Credentials("access-9", "refresh-9", time.time() + 900))
    assert Tone3000Client(KEY, cache_dir=tmp_path).connected

    api.token_path.write_text("{ not json", "utf-8")
    assert not Tone3000Client(KEY, cache_dir=tmp_path).connected


def test_disconnecting_removes_the_stored_tokens(connected: Tone3000Client) -> None:
    connected._write_tokens(connected._creds)
    connected.disconnect()
    assert not connected.connected and not connected.token_path.exists()


# -- the authorize URL -----------------------------------------------------


def test_the_authorize_url_carries_pkce_and_the_client_id(
        api: Tone3000Client) -> None:
    verifier, challenge, state = tone3000._pkce()
    url = api.authorize_url((verifier, challenge, state))
    query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert url.startswith("https://www.tone3000.com/api/v1/oauth/authorize?")
    assert query["client_id"] == [KEY]
    assert query["response_type"] == ["code"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["code_challenge"] == [challenge] and query["state"] == [state]
    assert query["redirect_uri"] == [tone3000.DEFAULT_REDIRECT_URI]


def test_extra_parameters_scope_the_catalogue(api: Tone3000Client) -> None:
    url = api.authorize_url(tone3000._pkce(), prompt="select_tone", gears="amp")
    query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert query["prompt"] == ["select_tone"] and query["gears"] == ["amp"]


def test_pkce_is_fresh_every_time_and_the_challenge_hashes_the_verifier() -> None:
    import base64
    import hashlib

    verifier, challenge, state = tone3000._pkce()
    expected = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    assert challenge == expected
    assert (verifier, state) != tone3000._pkce()[::2]


def test_connecting_without_a_key_explains_where_to_get_one(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    monkeypatch.delenv(tone3000.KEY_ENV, raising=False)
    with pytest.raises(Tone3000Error) as exc:
        Tone3000Client(cache_dir=tmp_path).connect(open_browser=False)
    assert exc.value.code == "no_key" and "API Keys" in exc.value.message


def test_a_callback_from_a_different_sign_in_is_rejected(
        monkeypatch: pytest.MonkeyPatch, api: Tone3000Client) -> None:
    monkeypatch.setattr(tone3000, "_await_callback",
                        lambda uri, t: {"code": "c", "state": "not-ours"})
    monkeypatch.setattr(tone3000.webbrowser, "open", lambda _u: True)
    with pytest.raises(Tone3000Error) as exc:
        api.connect(open_browser=False)
    assert exc.value.code == "state_mismatch"


def test_a_refused_sign_in_reports_what_tone3000_said(
        monkeypatch: pytest.MonkeyPatch, api: Tone3000Client) -> None:
    monkeypatch.setattr(tone3000, "_await_callback",
                        lambda uri, t: {"error": "access_denied"})
    with pytest.raises(Tone3000Error) as exc:
        api.connect(open_browser=False)
    assert exc.value.code == "denied" and "access_denied" in exc.value.message


def test_a_completed_sign_in_stores_the_tokens(monkeypatch: pytest.MonkeyPatch,
                                               api: Tone3000Client) -> None:
    captured: dict[str, str] = {}

    def fake_callback(uri: str, timeout: float) -> dict[str, str]:
        return {"code": "auth-code", "state": captured["state"]}

    def fake_authorize(pkce, **extra):
        captured["state"] = pkce[2]
        captured["verifier"] = pkce[0]
        return "https://example.invalid/authorize"

    monkeypatch.setattr(api, "authorize_url", fake_authorize)
    monkeypatch.setattr(tone3000, "_await_callback", fake_callback)
    monkeypatch.setattr(api, "_post_token",
                        lambda body: {"access_token": "AT", "refresh_token": "RT",
                                      "expires_in": 3600} if body else {})
    api.connect(open_browser=False)
    assert api.connected and api._creds.access_token == "AT"
    assert json.loads(api.token_path.read_text("utf-8"))["refresh_token"] == "RT"


# -- requests --------------------------------------------------------------


def test_a_request_carries_the_bearer_token(monkeypatch: pytest.MonkeyPatch,
                                            connected: Tone3000Client) -> None:
    seen: dict[str, object] = {}

    def fake_get(url, params=None, headers=None, timeout=0, stream=False):
        seen.update(url=url, headers=headers, params=params)
        return FakeResponse({"ok": True})

    monkeypatch.setattr("requests.get", fake_get)
    assert connected.request("/api/v1/user") == {"ok": True}
    assert seen["headers"] == {"Authorization": "Bearer access-1"}
    assert seen["url"] == "https://www.tone3000.com/api/v1/user"


def test_an_expired_token_is_refreshed_and_the_request_retried(
        monkeypatch: pytest.MonkeyPatch, connected: Tone3000Client) -> None:
    calls: list[str] = []

    def fake_get(url, params=None, headers=None, timeout=0, stream=False):
        calls.append(headers["Authorization"])
        return (FakeResponse(status=401) if len(calls) == 1
                else FakeResponse({"ok": True}))

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr(connected, "_post_token",
                        lambda body: {"access_token": "AT2", "refresh_token": "RT2",
                                      "expires_in": 3600})
    assert connected.request("/api/v1/user") == {"ok": True}
    assert calls == ["Bearer access-1", "Bearer AT2"]


def test_a_rate_limit_is_named_rather_than_reported_as_a_generic_failure(
        monkeypatch: pytest.MonkeyPatch, connected: Tone3000Client) -> None:
    monkeypatch.setattr("requests.get",
                        lambda *a, **k: FakeResponse(status=429))
    with pytest.raises(Tone3000Error) as exc:
        connected.request("/api/v1/user")
    assert exc.value.code == "rate_limited"


def test_external_model_host_never_receives_account_bearer(monkeypatch, connected):
    seen = {}
    def fake_get(url, **kwargs):
        seen.update(kwargs)
        return FakeResponse(content=b'x')
    monkeypatch.setattr('requests.get', fake_get)
    connected.request('https://storage.example.invalid/model.nam', stream=True)
    assert seen['headers'] == {}


def test_http_failure_does_not_echo_signed_url_or_response_secret(monkeypatch, connected):
    monkeypatch.setattr('requests.get', lambda *a, **kw: FakeResponse({'secret': 'private'}, status=403))
    with pytest.raises(Tone3000Error) as exc:
        connected.request('https://storage.example.invalid/file.nam?token=private', stream=True)
    assert exc.value.code == 'forbidden'
    assert 'private' not in str(exc.value)


def test_network_failure_does_not_echo_signed_download_url(monkeypatch, connected):
    import requests
    def fail(*args, **kwargs):
        raise requests.ConnectionError('https://storage.example.invalid/file?token=private')
    monkeypatch.setattr('requests.get', fail)
    with pytest.raises(Tone3000Error) as exc:
        connected.request('https://storage.example.invalid/file?token=private')
    assert exc.value.code == 'network_error' and 'private' not in str(exc.value)


# -- search ----------------------------------------------------------------


def test_search_uses_the_separators_the_api_expects(
        monkeypatch: pytest.MonkeyPatch, connected: Tone3000Client) -> None:
    seen: dict[str, object] = {}

    def fake_get(url, params=None, headers=None, timeout=0, stream=False):
        seen.update(params=params)
        return FakeResponse({"data": [], "total_pages": 1})

    monkeypatch.setattr("requests.get", fake_get)
    connected.search_tones("plexi", gears=("amp", "amp-cab"),
                           tags=("crunch", "vintage"), creators=("a_b", "c"),
                           architecture=2)
    params = seen["params"]
    assert params["gears"] == "amp_amp-cab"
    assert params["tags"] == "crunch_vintage"
    assert params["creators"] == "a_b,c"          # commas: usernames hold "_"
    assert params["query"] == "plexi" and params["architecture"] == 2
    assert params["format"] == "nam"


def test_paging_walks_until_the_limit_is_reached(
        monkeypatch: pytest.MonkeyPatch, connected: Tone3000Client) -> None:
    def page(index: int) -> dict:
        base = index * 2
        return {"data": [{"id": base + 1, "title": f"t{base + 1}"},
                         {"id": base + 2, "title": f"t{base + 2}"}],
                "total_pages": 5}

    monkeypatch.setattr("requests.get",
                        lambda url, params=None, **k: FakeResponse(
                            page(int(params["page"]) - 1)))
    tones = list(connected.iter_tones(limit=5))
    assert [t.id for t in tones] == [1, 2, 3, 4, 5]


def test_paging_stops_at_the_last_page_even_below_the_limit(
        monkeypatch: pytest.MonkeyPatch, connected: Tone3000Client) -> None:
    monkeypatch.setattr("requests.get",
                        lambda *a, **k: FakeResponse(
                            {"data": [{"id": 1, "title": "only"}],
                             "total_pages": 1}))
    assert len(list(connected.iter_tones(limit=100))) == 1


def test_a_tone_parses_the_fields_the_ui_shows() -> None:
    tone = Tone.from_json({
        "id": 42, "title": "Blue Amp", "gear": "amp", "format": "nam",
        "user": {"username": "someone"}, "makes": [{"name": "Marshall"}],
        "tags": [{"name": "crunch"}], "downloads_count": 91,
        "url": "https://www.tone3000.com/tones/42"})
    assert (tone.id, tone.title, tone.creator) == (42, "Blue Amp", "someone")
    assert tone.makes == ["Marshall"] and tone.tags == ["crunch"]
    assert tone.downloads == 91


# -- downloads -------------------------------------------------------------


def _model() -> Model:
    return Model.from_json({
        "id": 9987, "tone_id": 4211, "name": "Ceriatone: King/Kong Ch2",
        "model_url": "https://www.tone3000.com/api/v1/models/9987/file.nam",
        "size": "standard", "architecture_version": "2"})


def test_a_download_names_the_file_after_its_tone_and_model_ids(
        monkeypatch: pytest.MonkeyPatch, connected: Tone3000Client,
        tmp_path: pathlib.Path) -> None:
    monkeypatch.setattr("requests.get",
                        lambda *a, **k: FakeResponse(content=b'{"weights":[]}'))
    path = connected.download_model(_model(), tmp_path)
    assert path.name.startswith("t3k-4211-9987 ")
    assert path.suffix == ".nam"
    assert path.read_bytes() == b'{"weights":[]}'
    assert "/" not in path.stem and ":" not in path.stem   # safe on Windows


def test_a_second_download_of_the_same_model_is_served_from_disk(
        monkeypatch: pytest.MonkeyPatch, connected: Tone3000Client,
        tmp_path: pathlib.Path) -> None:
    calls: list[int] = []

    def fake_get(*a, **k):
        calls.append(1)
        return FakeResponse(content=b"x")

    monkeypatch.setattr("requests.get", fake_get)
    connected.download_model(_model(), tmp_path)
    connected.download_model(_model(), tmp_path)
    assert len(calls) == 1


def test_no_partial_file_is_left_where_a_profile_should_be(
        monkeypatch: pytest.MonkeyPatch, connected: Tone3000Client,
        tmp_path: pathlib.Path) -> None:
    monkeypatch.setattr("requests.get",
                        lambda *a, **k: FakeResponse(content=b"x"))
    connected.download_model(_model(), tmp_path)
    assert list(tmp_path.glob("*.part")) == []


def test_download_size_limit_cleans_partial_file(monkeypatch, connected, tmp_path):
    monkeypatch.setattr('requests.get', lambda *a, **kw: FakeResponse(content=b'12345'))
    with pytest.raises(Tone3000Error) as exc:
        connected.download_model(_model(), tmp_path, max_bytes=4)
    assert exc.value.code == 'model_too_large'
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name,expected", [
    ("t3k-4211-9987 Ceriatone King Kong", (4211, 9987)),
    ("t3k-1-2 x", (1, 2)),
    ("Helga B 6505+ OD808", (None, None)),
    ("t3k-notanumber-9 x", (None, None)),
    ("t3k-", (None, None)),
])
def test_a_file_name_leads_back_to_its_tone_id(name: str,
                                               expected: tuple) -> None:
    assert parse_model_reference(f"{name}.nam") == expected
