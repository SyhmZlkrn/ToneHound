"""TONE3000 API client: search the catalogue, fetch `.nam` files by Tone ID.

The last step of the pipeline. The matcher ranks *embeddings*; what the user
wants back is a tone they can actually load, and TONE3000 is where those live.
The library is far larger than anything that fits in `assets/dev_profiles`, so
the index is built from tones fetched here and the ranking's output is a Tone ID.

**Authentication is the user's, performed by the user.** TONE3000 uses OAuth 2.0
with PKCE. `connect()` opens the user's own browser at TONE3000's authorize
page, and they sign in and grant access there; nothing in this module types a
password or clicks a consent button, and there is no path that authenticates
without the browser round trip. The publishable key (`t3k_pub_...`) is read from
the environment or from a file the user writes -- never hard-coded, because a key
in a repository need not bind every fork to the developer's integration. A
publishable key is a public OAuth app identifier; account access/refresh tokens
and server secret keys must never be committed or distributed.

**Not being connected is a normal state.** `Tone3000Client.connected` is false
until someone runs the connect command, and the pipeline falls back to the local
profile directory. An analysis tool that refuses to start because a web service
is unconfigured would be worse than one with a smaller library.

**Downloads are content-addressed and cached.** A `.nam` fetched once is kept
under `<cache>/tone3000/` keyed by model id, because the index has to render
every profile it ranks and re-downloading a library on each run would dominate
the cost of using it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import pathlib
import re
import secrets
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Iterator, Sequence

T3K_API = "https://www.tone3000.com"
API = "/api/v1"

DEFAULT_REDIRECT_URI = "http://localhost:3001"
"""Localhost origins are accepted without registration; anything else has to be
added to the key in TONE3000 Settings -> API Keys."""

RATE_LIMIT_PER_MIN = 100
_MIN_INTERVAL = 60.0 / RATE_LIMIT_PER_MIN

KEY_ENV = "TONE3000_PUBLISHABLE_KEY"
REDIRECT_ENV = "TONE3000_REDIRECT_URI"


class Tone3000Error(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------


@dataclass
class Credentials:
    access_token: str
    refresh_token: str
    expires_at: float          # unix seconds

    @property
    def stale(self) -> bool:
        """True inside the last minute of validity, so a long call cannot
        expire mid-flight."""
        return time.time() > self.expires_at - 60.0

    @classmethod
    def from_token_response(cls, data: dict[str, Any]) -> "Credentials":
        return cls(access_token=data["access_token"],
                   refresh_token=data.get("refresh_token", ""),
                   expires_at=time.time() + float(data.get("expires_in", 3600)))


def publishable_key(explicit: str | None = None,
                    key_file: pathlib.Path | None = None) -> str | None:
    """The `t3k_pub_...` key, from the argument, the environment, or a file."""
    if explicit:
        return explicit.strip()
    env = os.environ.get(KEY_ENV, "").strip()
    if env:
        return env
    if key_file and key_file.exists():
        try:
            return key_file.read_text("utf-8").strip() or None
        except OSError:
            return None
    return None


# --------------------------------------------------------------------------
# PKCE and the loopback listener
# --------------------------------------------------------------------------


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _pkce() -> tuple[str, str, str]:
    """``(verifier, challenge, state)`` for one authorization attempt."""
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge, _b64url(secrets.token_bytes(16))


_DONE_PAGE = b"""<!doctype html><meta charset="utf-8"><title>ToneHound</title>
<body style="font:16px system-ui;background:#14161a;color:#e6e8ec;padding:48px">
<h1 style="font-size:20px">TONE3000 connected</h1>
<p>You can close this tab and go back to ToneHound.</p>"""

_FAIL_PAGE = b"""<!doctype html><meta charset="utf-8"><title>ToneHound</title>
<body style="font:16px system-ui;background:#14161a;color:#e6e8ec;padding:48px">
<h1 style="font-size:20px">Sign-in did not complete</h1>
<p>Go back to ToneHound for the reason.</p>"""


class _CallbackHandler(BaseHTTPRequestHandler):
    result: dict[str, str] = {}

    def do_GET(self) -> None:  # noqa: N802  (stdlib naming)
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        got = {k: v[0] for k, v in query.items()}
        if "code" in got or "error" in got:
            type(self).result.update(got)
        ok = "code" in got
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_DONE_PAGE if ok else _FAIL_PAGE)

    def log_message(self, *args: Any) -> None:
        """Silence. The listener lives for one request and its access log would
        land in the middle of the CLI's own output."""


def _await_callback(redirect_uri: str, timeout: float) -> dict[str, str]:
    """Listen on the redirect URI's own host and port until the browser lands.

    The bind address comes from the URI rather than being hard-coded, so a LAN
    redirect (a headless device showing a QR code) listens where it said it
    would. `localhost` binds to 127.0.0.1; a browser that resolved it to ::1
    first retries on the v4 address.
    """
    parsed = urllib.parse.urlparse(redirect_uri)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    host = parsed.hostname or "127.0.0.1"
    bind = "127.0.0.1" if host == "localhost" else host
    handler = type("_Once", (_CallbackHandler,), {"result": {}})

    server = HTTPServer((bind, port), handler)
    server.timeout = 1.0
    thread = threading.Thread(target=_serve_until, args=(server, handler, timeout),
                              daemon=True)
    thread.start()
    thread.join(timeout + 2.0)
    server.server_close()
    return dict(handler.result)


def _serve_until(server: HTTPServer, handler: type, timeout: float) -> None:
    deadline = time.time() + timeout
    while not handler.result and time.time() < deadline:
        server.handle_request()


# --------------------------------------------------------------------------
# the client
# --------------------------------------------------------------------------


@dataclass
class Tone:
    """A catalogue entry. One tone has one or more downloadable models."""

    id: int
    title: str
    gear: str = ""
    format: str = ""
    creator: str = ""
    makes: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    url: str = ""
    downloads: int = 0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def description(self) -> str:
        # Descriptions are creator-supplied text, never executable UI markup.
        import html
        return html.unescape(re.sub(r"<[^>]+>", " ", str(self.raw.get("description") or ""))).strip()[:12000]

    @property
    def gear_models(self) -> list[str]:
        # Only explicit catalogue fields qualify as make/model metadata. A
        # creative capture title is not proof of the physical amplifier used.
        values = self.raw.get("gear_models") or self.raw.get("gear_model") or []
        if isinstance(values, (str, dict)):
            values = [values]
        return [str(v.get("name", "") if isinstance(v, dict) else v) for v in values
                if isinstance(v, (str, dict))]

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Tone":
        return cls(
            id=int(data["id"]), title=str(data.get("title", "")),
            gear=str(data.get("gear", "")), format=str(data.get("format", "")),
            creator=str((data.get("user") or {}).get("username", "")),
            makes=[str(m.get("name", "") if isinstance(m, dict) else m) for m in (data.get("makes") or [])],
            tags=[str(t.get("name", "") if isinstance(t, dict) else t) for t in (data.get("tags") or [])],
            url=str(data.get("url", "")),
            downloads=int(data.get("downloads_count") or 0),
            raw=data)


@dataclass
class Model:
    """One downloadable file belonging to a tone."""

    id: int
    tone_id: int
    name: str
    model_url: str
    size: str = ""
    architecture: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Model":
        return cls(id=int(data["id"]), tone_id=int(data.get("tone_id") or 0),
                   name=str(data.get("name", "")),
                   model_url=str(data.get("model_url", "")),
                   size=str(data.get("size", "")),
                   architecture=str(data.get("architecture_version", "")),
                   raw=data)


class Tone3000Client:
    """Authenticated access to the TONE3000 catalogue."""

    def __init__(self, key: str | None = None, *,
                 cache_dir: str | pathlib.Path = ".cache",
                 redirect_uri: str | None = None,
                 api: str = T3K_API) -> None:
        self.cache_dir = pathlib.Path(cache_dir) / "tone3000"
        self.api = api.rstrip("/")
        self.key = publishable_key(key, self.cache_dir / "publishable_key.txt")
        self.redirect_uri = (redirect_uri or os.environ.get(REDIRECT_ENV)
                             or DEFAULT_REDIRECT_URI)
        self.token_path = self.cache_dir / "tokens.json"
        self._creds: Credentials | None = self._read_tokens()
        self._last_request = 0.0

    # -- credentials -----------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._creds is not None

    @property
    def configured(self) -> bool:
        return bool(self.key)

    def _read_tokens(self) -> Credentials | None:
        if not self.token_path.exists():
            return None
        try:
            return Credentials(**json.loads(self.token_path.read_text("utf-8")))
        except Exception:
            return None  # a corrupt token file is a reconnect, never a crash

    def _write_tokens(self, creds: Credentials) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.token_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(creds)), "utf-8")
        tmp.replace(self.token_path)
        try:
            os.chmod(self.token_path, 0o600)
        except OSError:
            pass  # best effort; Windows ACLs are not POSIX modes

    def disconnect(self) -> None:
        self._creds = None
        self.token_path.unlink(missing_ok=True)

    # -- the OAuth round trip --------------------------------------------

    def authorize_url(self, verifier_state: tuple[str, str, str],
                      **extra: str) -> str:
        _, challenge, state = verifier_state
        params = {
            "client_id": self.key or "", "redirect_uri": self.redirect_uri,
            "response_type": "code", "code_challenge": challenge,
            "code_challenge_method": "S256", "state": state, **extra,
        }
        return f"{self.api}{API}/oauth/authorize?" + urllib.parse.urlencode(params)

    def connect(self, *, open_browser: bool = True, timeout: float = 300.0,
                print_url: bool = True) -> None:
        """Run the standard flow: the user signs in and grants access.

        Blocks until the browser comes back or ``timeout`` passes. Call this
        from an explicit "connect" command -- never implicitly as part of a
        match, because it takes over the user's browser.
        """
        if not self.key:
            raise Tone3000Error(
                "no_key",
                f"set {KEY_ENV} to your t3k_pub_ key (TONE3000 -> Settings -> "
                f"API Keys), or write it to {self.cache_dir / 'publishable_key.txt'}")

        pkce = _pkce()
        url = self.authorize_url(pkce)
        if open_browser:
            webbrowser.open(url)
        if print_url:
            print(f"Opening TONE3000 to sign in.\nIf nothing opened, visit:\n  {url}\n")

        got = _await_callback(self.redirect_uri, timeout)
        if not got:
            raise Tone3000Error("timeout",
                                f"no callback within {timeout:.0f}s -- was the "
                                f"redirect URI {self.redirect_uri} registered?")
        if "error" in got:
            raise Tone3000Error("denied", f"TONE3000 returned: {got['error']}")
        if got.get("state") != pkce[2]:
            raise Tone3000Error("state_mismatch",
                                "the callback's state did not match; the sign-in "
                                "was not the one this process started")

        data = self._post_token({
            "grant_type": "authorization_code", "code": got["code"],
            "code_verifier": pkce[0], "redirect_uri": self.redirect_uri,
            "client_id": self.key,
        })
        self._creds = Credentials.from_token_response(data)
        self._write_tokens(self._creds)

    def _post_token(self, body: dict[str, str]) -> dict[str, Any]:
        import requests

        try:
            response = requests.post(f"{self.api}{API}/oauth/token", data=body,
                                     timeout=30)
        except requests.RequestException:
            raise Tone3000Error("network_error", "Could not reach TONE3000. Check your connection and try again.") from None
        if not response.ok:
            raise Tone3000Error("token_exchange_failed",
                                f"TONE3000 token endpoint returned "
                                f"HTTP {response.status_code}. Reconnect to TONE3000.")
        return response.json()

    def _access_token(self) -> str:
        if self._creds is None:
            raise Tone3000Error("not_connected",
                                "not connected to TONE3000; run the connect step first")
        if self._creds.stale and self._creds.refresh_token:
            data = self._post_token({"grant_type": "refresh_token",
                                     "refresh_token": self._creds.refresh_token,
                                     "client_id": self.key or ""})
            self._creds = Credentials.from_token_response(data)
            self._write_tokens(self._creds)
        return self._creds.access_token

    # -- requests --------------------------------------------------------

    def _throttle(self) -> None:
        """Stay under 100 requests a minute without having to think about it."""
        wait = _MIN_INTERVAL - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def request(self, path: str, params: dict[str, Any] | None = None,
                *, stream: bool = False) -> Any:
        import requests

        self._throttle()
        url = path if path.startswith("http") else f"{self.api}{path}"
        # Signed model storage URLs can be on another host. Never forward the
        # account bearer token to a third-party CDN/storage origin.
        parsed, origin = urllib.parse.urlparse(url), urllib.parse.urlparse(self.api)
        authenticated = (parsed.scheme, parsed.netloc) == (origin.scheme, origin.netloc)
        if parsed.scheme != "https" and parsed.hostname not in ("localhost", "127.0.0.1"):
            raise Tone3000Error("invalid_url", "TONE3000 downloads must use HTTPS.")
        headers = {"Authorization": f"Bearer {self._access_token()}"} if authenticated else {}
        try:
            response = requests.get(url, params=params, headers=headers,
                                    timeout=60, stream=stream)
        except requests.RequestException:
            raise Tone3000Error("network_error", "Could not reach TONE3000 or its model storage. Try again when connected.") from None

        if response.status_code == 401 and authenticated:
            # Force a refresh and retry once: the access token can expire in the
            # gap between the staleness check and the request landing.
            if self._creds is not None:
                self._creds.expires_at = 0.0
            headers = {"Authorization": f"Bearer {self._access_token()}"}
            try:
                response = requests.get(url, params=params, headers=headers,
                                        timeout=60, stream=stream)
            except requests.RequestException:
                raise Tone3000Error("network_error", "Could not reach TONE3000. Try again when connected.") from None
        if response.status_code == 429:
            raise Tone3000Error("rate_limited",
                                "TONE3000 rate limit reached. Wait a minute before trying again.")
        if response.status_code == 403:
            raise Tone3000Error("forbidden", "TONE3000 denied access. Reconnect and check the API key/account permissions.")
        if not response.ok:
            raise Tone3000Error("http_error",
                                f"TONE3000 returned HTTP {response.status_code}.")
        return response if stream else response.json()

    # -- catalogue -------------------------------------------------------

    def search_tones(self, query: str = "", *, page: int = 1, page_size: int = 25,
                     sort: str = "trending", gears: Sequence[str] = ("amp", "amp-cab"),
                     fmt: str = "nam", tags: Sequence[str] = (),
                     makes: Sequence[str] = (), creators: Sequence[str] = (),
                     architecture: int | None = None) -> dict[str, Any]:
        """One page of `/tones/search`.

        The separators are the API's, not a choice: `gears`, `tags` and `makes`
        join on underscores, `creators` on commas, because a username may
        contain an underscore.
        """
        params: dict[str, Any] = {"page": page, "page_size": page_size, "sort": sort}
        if query:
            params["query"] = query
        if gears:
            params["gears"] = "_".join(gears)
        if fmt:
            params["format"] = fmt
        if tags:
            params["tags"] = "_".join(tags)
        if makes:
            params["makes"] = "_".join(makes)
        if creators:
            params["creators"] = ",".join(creators)
        if architecture is not None:
            params["architecture"] = architecture
        return self.request(f"{API}/tones/search", params)

    def iter_tones(self, limit: int = 100, **kwargs: Any) -> Iterator[Tone]:
        """Walk search results across pages, stopping at ``limit``."""
        seen, page = 0, 1
        while seen < limit:
            payload = self.search_tones(page=page, **kwargs)
            rows = payload.get("data") or []
            if not rows:
                return
            for row in rows:
                yield Tone.from_json(row)
                seen += 1
                if seen >= limit:
                    return
            if page >= int(payload.get("total_pages") or 1):
                return
            page += 1

    def get_tone(self, tone_id: int | str) -> Tone:
        return Tone.from_json(self.request(f"{API}/tones/{tone_id}"))

    def list_models(self, tone_id: int | str, *, architecture: int | None = None,
                    page: int = 1, page_size: int = 50) -> list[Model]:
        params: dict[str, Any] = {"tone_id": tone_id, "page": page,
                                  "page_size": page_size}
        if architecture is not None:
            params["architecture"] = architecture
        payload = self.request(f"{API}/models", params)
        return [Model.from_json(m) for m in (payload.get("data") or [])]

    def get_user(self) -> dict[str, Any]:
        return self.request(f"{API}/user")

    # -- downloads -------------------------------------------------------

    def download_model(self, model: Model, dest_dir: pathlib.Path | None = None,
                       *, refresh: bool = False, max_bytes: int = 32 * 1024 * 1024) -> pathlib.Path:
        """Fetch a model's file. Cached by model id, so a rebuild is free.

        The file name carries the tone id and model id as well as the name, so
        two tones that both called a capture "Crunch" do not overwrite each
        other and so a file on disk can always be traced back to a Tone ID.
        """
        import requests

        dest_dir = pathlib.Path(dest_dir or self.cache_dir / "profiles")
        dest_dir.mkdir(parents=True, exist_ok=True)

        # A signed storage URL need not have an extension. This API client is
        # downloading NAM captures, whose on-disk extension is always .nam.
        suffix = ".nam"
        stem = "".join(c if c.isalnum() or c in " -_" else "_"
                       for c in (model.name or "model")).strip()[:80]
        path = dest_dir / f"t3k-{model.tone_id}-{model.id} {stem}{suffix}"
        if path.exists() and not refresh:
            return path

        response = self.request(model.model_url, stream=True)
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.part")
        try:
            total = 0
            with open(tmp, "wb") as fh:
                for block in response.iter_content(chunk_size=1 << 16):
                    total += len(block)
                    if total > max_bytes:
                        raise Tone3000Error("model_too_large", f"Model exceeds the {max_bytes // (1024 * 1024)} MB download limit.")
                    fh.write(block)
            tmp.replace(path)
        except requests.RequestException:
            raise Tone3000Error("network_error", "Model download was interrupted. Try this capture again.") from None
        finally:
            tmp.unlink(missing_ok=True)
            close = getattr(response, "close", None)
            if close:
                close()
        return path


def parse_model_reference(path: str | pathlib.Path) -> tuple[int | None, int | None]:
    """Recover ``(tone_id, model_id)`` from a downloaded file name.

    The index stores profiles as files, so this is the link back from "the
    ranking picked this file" to "quote this Tone ID at the user".
    """
    stem = pathlib.Path(path).stem
    if not stem.startswith("t3k-"):
        return None, None
    parts = stem[4:].split(" ", 1)[0].split("-")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return None, None
