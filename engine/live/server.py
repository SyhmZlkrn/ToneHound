"""The control process: HTTP, WebSocket, and every job that is not real time.

`docs/ui_protocol.md` is the contract; this file is the implementation of it.
The shape of the process comes from one measurement -- a match run is ~15 s of
all-core CPU and the index costs ~45 s to fingerprint -- so the rule here is
that **nothing slow happens on the event loop**. Decoding, separating,
fingerprinting and rendering all go to a process pool, and the loop does
nothing but move JSON and bytes. If the socket ever goes quiet during a match,
something has been called directly that should have been submitted.

Three things are worth knowing before reading further.

*Two pools, not one.* Separation gets a pool of exactly one worker so that
htdemucs_6s stays loaded between matches; it takes seconds to load and a match
is something people do repeatedly. Everything else gets a small general pool,
which is also what fans the first index build out across cores -- ~45 s
single-threaded is a first launch that looks hung.

*Cancellation is cooperative, and says so.* A `ProcessPoolExecutor` future that
has already started cannot be cancelled, and separation is the 15-second stage.
So `match.cancel` is acknowledged at once and the job stops at its next stage
boundary. Claiming otherwise would mean showing a stopped spinner over a
machine still running all-core.

*Missing audio hardware is not a fatal error.* If the audio process will not
start -- no interface, no PortAudio, a driver that took it down -- the server
still serves the UI, imports snippets and runs matches, and reports the fault
in the frames that carry device state. Refusing to boot would make an offline
matcher impossible on a machine that simply has no interface plugged in.

Honesty rules from §5 of the protocol are enforced on this side too, since a
server that omitted the fields could not be fixed by any UI: `caveat` is sent
with `hello` and again with every `match.result`, `distance` and
`distance_range` are always populated, `separation_ok` is reported truthfully
rather than hidden behind a fallback, and every latency figure is passed
through with the warning that the driver reported it rather than anything
having measured it.
"""

from __future__ import annotations

import asyncio
import concurrent.futures as cf
import contextlib
import json
import mimetypes
import os
import pathlib
import time
import traceback
from typing import Any, Awaitable, Callable

import numpy as np

from tonehound import match as matching
from tonehound.config import SAMPLE_RATE

from . import jobs, wsproto
from .audio_proc import AudioProcess, AudioProcessError
from .media import DEFAULT_BINS, MediaError, MediaStore
from .media import decode as media_decode
from .profiles import ProfileCatalogue
from .wsproto import Request, Response, WebSocket, token

ROOT = pathlib.Path(__file__).resolve().parents[2]
UI_DIR = pathlib.Path(__file__).resolve().parent / "ui"

PROTOCOL = 1
SERVER_VERSION = "0.1.0"
DEFAULT_PORT = 8730
PING_INTERVAL_S = 15.0

MAX_SNIPPET_S = 600.0
"""Ten minutes. Long enough for any song, and it bounds what one import costs:
a full decode is held in memory at 48 kHz mono float32 (~115 MB at the cap)
until `snippet.forget`."""

MIN_CROP_S = 3.0
MAX_CROP_S = 30.0
"""Separation runs at about realtime, so this cap is the difference between a
30-second wait and a two-minute one for an answer of the same quality."""

DEFAULT_AUDITION_S = 8.0
ALLOWED_BLOCK_SIZES = (256, 512, 1024, 2048)
DEFAULT_BLOCK_SIZE = 512

SEPARATION_XRT = 1.0
"""Measured: htdemucs_6s takes about one second per second of audio."""
RENDER_S_PER_DI_S = 1.33 / 8.0

_DROPPABLE = ("telemetry", "perf")
"""Frames the protocol allows the server to drop for a slow client. Everything
else is a state change a second tab needs, so it is delivered or the client is
disconnected -- silently skipping one would desynchronise it forever."""


def _jsonable(value: Any) -> Any:
    """Last-resort coercion for values `json.dumps` will not take.

    Numpy scalars and arrays are the realistic case: they arrive from the
    meters, the fingerprint and the ranking, and every one of those is coerced
    at its source. This is here because the failure mode without it is bad out
    of proportion to the mistake -- a single `np.float64` raises inside the
    broadcast, which drops the whole frame and leaves the UI showing a stale
    meter with nothing logged that points at why.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return str(value)


def dumps(frame: dict[str, Any]) -> str:
    return json.dumps(frame, default=_jsonable, allow_nan=False)


class CommandError(Exception):
    """A protocol-level failure with a code from §2 of the doc."""

    def __init__(self, code: str, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail


# --------------------------------------------------------------------------
# clients
# --------------------------------------------------------------------------


class Client:
    """One WebSocket, with a bounded outbound queue.

    The queue exists because telemetry is 20 Hz and a browser tab that has been
    backgrounded, or a laptop that just woke, can stop reading. Unbounded
    buffering there turns into memory growth that shows up minutes later as
    something unrelated; a bounded queue turns it into a dropped meter frame,
    which is exactly what §4.4 says to do.
    """

    QUEUE = 64

    def __init__(self, socket: WebSocket) -> None:
        self.socket = socket
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=self.QUEUE)
        self.dropped = 0
        self.alive = True

    def send(self, frame: dict[str, Any]) -> None:
        """Queue a frame. Never blocks, never raises."""
        if not self.alive:
            return
        try:
            text = dumps(frame)
        except (TypeError, ValueError) as exc:
            # `allow_nan=False` means a NaN distance raises here rather than
            # emitting bare `NaN`, which `JSON.parse` rejects and which would
            # break the client's whole frame loop. Say what happened instead of
            # dropping it: a silent gap is the one outcome with no way back.
            text = dumps({"type": "error", "id": frame.get("id"),
                          "code": "internal",
                          "message": f"a {frame.get('type')!r} frame could not "
                                     f"be encoded: {exc}",
                          "detail": {}})
        try:
            self.queue.put_nowait(text)
        except asyncio.QueueFull:
            if frame.get("type") in _DROPPABLE:
                self.dropped += 1
            else:
                # 64 frames behind on a loopback socket is not slowness, it is
                # a client that has stopped reading. Dropping a state change
                # would leave it wrong forever, so end it instead.
                self.alive = False

    async def writer(self) -> None:
        try:
            while self.alive:
                text = await self.queue.get()
                await self.socket.send_text(text)
        except (ConnectionError, OSError, asyncio.CancelledError):
            pass
        finally:
            self.alive = False


# --------------------------------------------------------------------------
# server
# --------------------------------------------------------------------------


class ToneHoundServer:
    def __init__(self, *, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 profile_dir: pathlib.Path | None = None,
                 render_dir: pathlib.Path | None = None,
                 di_dir: pathlib.Path | None = None,
                 cache_dir: pathlib.Path | None = None,
                 enable_audio: bool = True) -> None:
        self.host, self.port = host, port
        self.profile_dir = profile_dir or ROOT / "assets" / "dev_profiles"
        self.render_dir = render_dir or ROOT / ".cache" / "renders_djent"
        self.di_dir = di_dir or ROOT / "assets" / "user_di"
        self.cache_dir = cache_dir or ROOT / ".cache"
        self.enable_audio = enable_audio

        self.media = MediaStore()
        self.catalogue = ProfileCatalogue(self.profile_dir,
                                          self.cache_dir / "profile_probe.json")
        self.index: matching.ProfileIndex | None = None
        self.index_state: dict[str, Any] = {
            "type": "index.status", "ready": False, "profiles": 0, "total": 0,
            "progress": 0.0, "eta_s": None, "di_name": self.render_dir.name,
        }

        self.clients: set[Client] = set()
        self.audio = AudioProcess(self._on_engine_event, self._on_engine_death)
        self.audio_fault: str | None = None
        self._stream_settings: dict[str, Any] = {}
        self._stream_status: dict[str, Any] = _idle_status()
        self._loaded_profile: dict[str, Any] | None = None
        self._devices: dict[str, Any] = _empty_devices()

        workers = max(1, min(4, (os.cpu_count() or 2) - 1))
        self.pool = cf.ProcessPoolExecutor(max_workers=workers,
                                           initializer=jobs.init_worker)
        self.sep_pool = cf.ProcessPoolExecutor(max_workers=1,
                                               initializer=jobs.init_worker)

        self.job: dict[str, Any] | None = None      # the one running match
        self._tasks: set[asyncio.Task] = set()

    # -- lifecycle -------------------------------------------------------

    async def start(self) -> asyncio.base_events.Server:
        self.catalogue.scan()
        self.media.load_di_directory(self.di_dir)

        if self.enable_audio:
            await self._start_audio()
        self._spawn(self._build_index())

        # A closure rather than a partial, so the port is read when a
        # connection arrives rather than when the listener is built. With
        # `port=0` the real port is not known until after the bind, and origin
        # checking against the port we *asked* for would then reject every
        # browser that connected to the port we actually got.
        async def connection(reader: asyncio.StreamReader,
                             writer: asyncio.StreamWriter) -> None:
            await wsproto.serve_connection(
                reader, writer, handler=self._http, upgrader=self._websocket,
                host=self.host, port=self.port)

        server = await asyncio.start_server(connection, self.host, self.port)
        self.port = int(server.sockets[0].getsockname()[1])
        return server

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        with contextlib.suppress(Exception):
            await self.audio.close()
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.sep_pool.shutdown(wait=False, cancel_futures=True)

    def _spawn(self, coro: Awaitable[Any]) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _start_audio(self) -> None:
        """Bring the audio process up, and survive it not coming up."""
        try:
            await self.audio.ensure_started()
            self._devices = await self.audio.call("devices_list", refresh=False)
            self._stream_status = await self.audio.call("status")
            self._stream_settings = await self.audio.call("settings")
            self.audio_fault = None
        except Exception as exc:
            self.audio_fault = (f"the audio engine could not start: {exc}. "
                                "Importing and matching still work; live play "
                                "does not.")
            self._devices = _empty_devices()
            self._stream_status = _idle_status()
            self._stream_settings = _default_settings()

    # -- broadcast -------------------------------------------------------

    def broadcast(self, frame: dict[str, Any]) -> None:
        frame.setdefault("id", None)
        for client in list(self.clients):
            client.send(frame)
            if not client.alive:
                self.clients.discard(client)

    def _on_engine_event(self, event: dict[str, Any]) -> None:
        """Anything the audio process pushed without being asked."""
        kind = event.get("type")
        if kind == "stream.status":
            self._stream_status = event
        elif kind == "engine.fault":
            self.broadcast({"type": "error", "code": "internal",
                            "message": event.get("message", "audio engine fault"),
                            "detail": {}})
            return
        self.broadcast(dict(event))

    def _on_engine_death(self, message: str) -> None:
        self.audio_fault = message
        self._stream_status = _idle_status()
        self.broadcast({"type": "error", "code": "device_busy",
                        "message": message, "detail": {}})
        self.broadcast(dict(self._stream_status))

    # -- the profile index ----------------------------------------------

    async def _build_index(self) -> None:
        """Fingerprint the profile renders, cached, off the loop.

        Fanned across the general pool because ~45 s single-threaded is a first
        launch the user would reasonably assume had hung; a warm cache makes it
        instant, so this cost is paid once per set of renders.
        """
        loop = asyncio.get_running_loop()
        cache = self.cache_dir / f"index_{self.render_dir.name}.npz"
        started = time.monotonic()

        def progress(done: int, total: int) -> None:
            elapsed = time.monotonic() - started
            eta = (elapsed / done) * (total - done) if done else None
            loop.call_soon_threadsafe(self._index_progress, done, total, eta)

        def build() -> matching.ProfileIndex:
            return matching.ProfileIndex.load_or_build(
                self.render_dir, cache, map_fn=self.pool.map, progress=progress)

        try:
            self.index = await loop.run_in_executor(None, build)
        except Exception as exc:
            self.index_state.update(ready=False, eta_s=None)
            self.broadcast({"type": "error", "code": "index_not_ready",
                            "message": f"the profile index could not be built: {exc}",
                            "detail": {}})
            self.broadcast(dict(self.index_state))
            return

        n = len(self.index)
        self.index_state.update(ready=True, profiles=n, total=n, progress=1.0,
                                eta_s=None, di_name=self.index.di_name)
        self.broadcast(dict(self.index_state))

    def _index_progress(self, done: int, total: int, eta: float | None) -> None:
        self.index_state.update(profiles=done, total=total,
                                progress=done / total if total else 0.0,
                                eta_s=eta)
        self.broadcast(dict(self.index_state))

    # ==================================================================
    # HTTP
    # ==================================================================

    async def _http(self, request: Request) -> Response:
        try:
            return await self._route(request)
        except MediaError as exc:
            return _json_response(400, {"code": exc.code, "message": exc.message})
        except CommandError as exc:
            return _json_response(400, {"code": exc.code, "message": exc.message})
        except Exception as exc:
            traceback.print_exc()
            return _json_response(500, {"code": "internal", "message": str(exc)})

    async def _route(self, request: Request) -> Response:
        path = request.path
        if request.method not in ("GET", "HEAD", "POST"):
            return Response(405, b"method not allowed")

        if request.method == "POST" and path == "/upload":
            return await self._upload(request)

        if path.startswith("/audio/"):
            item = self.media.get(path[len("/audio/"):].removesuffix(".wav"))
            return Response(200, item.wav(), "audio/wav")

        if path.startswith("/peaks/"):
            item = self.media.get(path[len("/peaks/"):])
            bins = int(request.query.get("bins") or DEFAULT_BINS)
            return _json_response(200, item.peaks(bins))

        return self._static(path)

    def _static(self, path: str) -> Response:
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        target = (UI_DIR / rel).resolve()
        try:
            target.relative_to(UI_DIR.resolve())
        except ValueError:
            return Response(403, b"outside the UI directory")
        if not target.is_file():
            return Response(404, b"not found")
        kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if kind.startswith("text/") or kind in ("application/javascript",):
            kind += "; charset=utf-8"
        return Response(200, target.read_bytes(), kind)

    async def _upload(self, request: Request) -> Response:
        """`POST /upload` -- a browser cannot hand the server a file path.

        Decoding happens in the pool: a 5-minute MP3 is a few hundred
        milliseconds of libsndfile plus a resample, and doing it on the loop
        would stall every other client's telemetry for that long.
        """
        fields = wsproto.parse_multipart(request.body, request.header("content-type"))
        file_field = fields.get("file")
        if not isinstance(file_field, tuple):
            return _json_response(400, {"code": "bad_request",
                                        "message": "expected a 'file' part"})
        filename, raw = file_field
        kind = str(fields.get("kind") or "snippet")
        loop = asyncio.get_running_loop()

        try:
            # Decoding a 5-minute MP3 is ~1 s of libsndfile and resampling, so
            # it goes to a thread; the store itself is only ever mutated on the
            # loop, which is what lets it stay lock-free.
            samples, source_sr, channels = await loop.run_in_executor(
                None, media_decode, raw, filename)
        except MediaError as exc:
            return _json_response(400, {"code": exc.code, "message": exc.message})

        if kind == "di":
            item = self.media.add(samples, kind="di",
                                  name=pathlib.Path(filename).stem or "DI",
                                  source_sample_rate=source_sr,
                                  source_channels=channels)
            if self.media.default_di_id is None:
                self.media.default_di_id = item.id
            self.broadcast(self._di_listing())
            return _json_response(200, {"di_id": item.id})

        seconds = len(samples) / SAMPLE_RATE
        if seconds > MAX_SNIPPET_S:
            return _json_response(413, {
                "code": "snippet_too_long",
                "message": f"{seconds / 60:.1f} minutes is longer than the "
                           f"{MAX_SNIPPET_S / 60:.0f}-minute import limit -- "
                           "trim it before importing"})
        item = self.media.add(samples, kind="snippet", name=filename or "snippet",
                              source_sample_rate=source_sr,
                              source_channels=channels)

        self.broadcast({
            "type": "snippet.ready", "snippet_id": item.id, "filename": item.name,
            "duration_s": item.duration_s,
            "source_sample_rate": item.source_sample_rate,
            "source_channels": item.source_channels,
            "audio_url": item.audio_url, "peaks_url": item.peaks_url,
        })
        return _json_response(200, {"snippet_id": item.id})

    # ==================================================================
    # WebSocket
    # ==================================================================

    async def _websocket(self, request: Request, socket: WebSocket) -> None:
        client = Client(socket)
        self.clients.add(client)
        writer = self._spawn(client.writer())
        pinger = self._spawn(self._ping(socket))
        try:
            while True:
                text = await socket.recv_text()
                if text is None:
                    break
                self._spawn(self._handle(client, text))
        except wsproto.ProtocolError as exc:
            with contextlib.suppress(Exception):
                await socket.close(1002)
            del exc
        except (ConnectionError, OSError):
            pass
        finally:
            client.alive = False
            self.clients.discard(client)
            writer.cancel()
            pinger.cancel()
            with contextlib.suppress(Exception):
                await socket.close()

    async def _ping(self, socket: WebSocket) -> None:
        while True:
            await asyncio.sleep(PING_INTERVAL_S)
            try:
                await socket.ping()
            except (ConnectionError, OSError):
                return

    async def _handle(self, client: Client, text: str) -> None:
        """One command in, exactly one terminal reply out."""
        try:
            frame = json.loads(text)
            if not isinstance(frame, dict):
                raise ValueError("frame is not an object")
        except Exception:
            client.send({"type": "error", "id": None, "code": "bad_request",
                         "message": "frame was not a JSON object", "detail": {}})
            return

        command_id = frame.get("id")
        kind = str(frame.get("type", ""))
        handler = _COMMANDS.get(kind)
        if handler is None:
            client.send({"type": "error", "id": command_id, "code": "unknown_type",
                         "message": f"the server does not implement {kind!r}",
                         "detail": {}})
            return

        try:
            reply = await handler(self, client, frame)
            if reply is not None:
                reply["id"] = command_id
                client.send(reply)
        except CommandError as exc:
            client.send({"type": "error", "id": command_id, "code": exc.code,
                         "message": exc.message, "detail": exc.detail})
        except MediaError as exc:
            client.send({"type": "error", "id": command_id, "code": exc.code,
                         "message": exc.message, "detail": {}})
        except AudioProcessError as exc:
            client.send({"type": "error", "id": command_id, "code": exc.code,
                         "message": exc.message, "detail": {}})
        except Exception as exc:
            trace_id = token(4)
            print(f"[{trace_id}] unhandled error in {kind}:")
            traceback.print_exc()
            client.send({"type": "error", "id": command_id, "code": "internal",
                         "message": f"{type(exc).__name__}: {exc}",
                         "detail": {"traceback_id": trace_id}})

    # -- helpers ---------------------------------------------------------

    async def _audio_call(self, op: str, **kwargs: Any) -> Any:
        """Call the audio process, respawning it if it died since last time."""
        if not self.enable_audio:
            raise CommandError("device_open_failed",
                               "this server was started with --no-audio; "
                               "importing and matching work, live play does not")
        if not self.audio.alive:
            await self.audio.ensure_started()
            self.audio_fault = None
        return await self.audio.call(op, **kwargs)

    def _di_listing(self) -> dict[str, Any]:
        takes = sorted(self.media.of_kind("di"), key=lambda i: i.name.lower())
        return {"type": "di.list",
                "takes": [{"di_id": i.id, "name": i.name,
                           "duration_s": i.duration_s,
                           "audio_url": i.audio_url, "peaks_url": i.peaks_url}
                          for i in takes],
                "default_di_id": self.media.default_di_id}

    def _require_index(self) -> matching.ProfileIndex:
        if self.index is None:
            raise CommandError(
                "index_not_ready",
                "the profile index is still being fingerprinted "
                f"({self.index_state['profiles']}/{self.index_state['total']})")
        return self.index

    # ==================================================================
    # commands
    # ==================================================================

    async def cmd_hello(self, client: Client, frame: dict[str, Any]) -> dict[str, Any]:
        if int(frame.get("protocol", PROTOCOL)) != PROTOCOL:
            raise CommandError("bad_request",
                               f"this server speaks protocol {PROTOCOL}, "
                               f"the client asked for {frame.get('protocol')}")
        if self.audio_fault:
            client.send({"type": "error", "id": None, "code": "device_open_failed",
                         "message": self.audio_fault, "detail": {}})
        return {
            "type": "hello", "protocol": PROTOCOL, "server_version": SERVER_VERSION,
            "sample_rate": SAMPLE_RATE,
            "default_block_size": DEFAULT_BLOCK_SIZE,
            "allowed_block_sizes": list(ALLOWED_BLOCK_SIZES),
            "max_snippet_s": MAX_SNIPPET_S,
            "min_crop_s": MIN_CROP_S, "max_crop_s": MAX_CROP_S,
            "stream": dict(self._stream_status),
            "settings": dict(self._stream_settings),
            "devices": dict(self._devices),
            "profile": dict(self._loaded_profile) if self._loaded_profile else None,
            "index": dict(self.index_state),
            "caveat": matching.CAVEAT,
        }

    async def cmd_devices_list(self, client: Client,
                               frame: dict[str, Any]) -> dict[str, Any]:
        self._devices = await self._audio_call(
            "devices_list", refresh=bool(frame.get("refresh", False)))
        return {"type": "devices.list", **self._devices}

    async def cmd_devices_select(self, client: Client,
                                 frame: dict[str, Any]) -> dict[str, Any]:
        try:
            selection = {
                "input_device": int(frame["input_device"]),
                "input_channel": int(frame.get("input_channel", 0)),
                "output_device": int(frame["output_device"]),
                "output_channels": [int(c) for c in frame.get("output_channels", [0, 1])],
                "block_size": int(frame.get("block_size", DEFAULT_BLOCK_SIZE)),
                "latency_ms": (None if frame.get("latency_ms") is None
                               else float(frame["latency_ms"])),
                "exclusive": bool(frame.get("exclusive", False)),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise CommandError("bad_request", f"bad device selection: {exc}") from exc

        result = await self._audio_call("configure", **selection)
        self._stream_status = await self._audio_call("status")
        self.broadcast(dict(self._stream_status))
        return {"type": "devices.selected", **result}

    async def cmd_stream_start(self, client: Client,
                               frame: dict[str, Any]) -> dict[str, Any]:
        self._stream_status = await self._audio_call("start")
        self.broadcast(dict(self._stream_status))
        return dict(self._stream_status)

    async def cmd_stream_stop(self, client: Client,
                              frame: dict[str, Any]) -> dict[str, Any]:
        self._stream_status = await self._audio_call("stop")
        self.broadcast(dict(self._stream_status))
        return dict(self._stream_status)

    async def cmd_stream_set(self, client: Client,
                             frame: dict[str, Any]) -> dict[str, Any]:
        allowed = ("input_gain_db", "output_gain_db", "bypass", "monitor",
                   "tuner_enabled", "tuner_reference_hz")
        changes = {k: frame[k] for k in allowed if k in frame}
        if "monitor" in changes and changes["monitor"] not in ("amp", "dry", "mute"):
            raise CommandError("bad_request",
                               'monitor must be "amp", "dry" or "mute", '
                               f'not {changes["monitor"]!r}')
        if not changes:
            return dict(self._stream_settings)
        self._stream_settings = await self._audio_call("apply_settings", **changes)
        self.broadcast(dict(self._stream_settings))
        return dict(self._stream_settings)

    async def cmd_profiles_list(self, client: Client,
                                frame: dict[str, Any]) -> dict[str, Any]:
        return {"type": "profiles.list",
                **self.catalogue.listing(str(frame.get("query") or ""))}

    async def cmd_profile_load(self, client: Client,
                               frame: dict[str, Any]) -> dict[str, Any]:
        profile_id = str(frame.get("profile_id", ""))
        entry = self.catalogue.get(profile_id)
        if entry is None:
            raise CommandError("not_found", f"no profile with id {profile_id!r}",
                               profile_id=profile_id)
        if not entry.loadable:
            raise CommandError("profile_unsupported",
                               entry.unsupported_reason or "this file will not load",
                               profile_id=profile_id)

        command_id = frame.get("id")

        def on_progress(update: dict[str, Any]) -> None:
            client.send({"type": "profile.loading", "id": command_id,
                         "profile_id": profile_id,
                         "stage": update.get("stage", "reading")})

        info = await self._audio_call_progress(
            "load_profile", on_progress, path=str(entry.path), profile_id=profile_id)
        info = {"type": "profile.loaded", **info}
        self._loaded_profile = {k: v for k, v in info.items() if k != "type"}
        # Other tabs need to see the change; the sender gets it as its reply.
        for other in self.clients:
            if other is not client:
                other.send({**info, "id": None})
        return info

    async def _audio_call_progress(self, op: str,
                                   on_progress: Callable[[dict[str, Any]], None],
                                   **kwargs: Any) -> Any:
        if not self.enable_audio:
            raise CommandError("device_open_failed",
                               "this server was started with --no-audio; "
                               "profiles cannot be loaded for live play")
        if not self.audio.alive:
            await self.audio.ensure_started()
        return await self.audio.call(op, on_progress=on_progress, **kwargs)

    async def cmd_index_status(self, client: Client,
                               frame: dict[str, Any]) -> dict[str, Any]:
        return dict(self.index_state)

    # -- snippets --------------------------------------------------------

    async def cmd_snippet_crop(self, client: Client,
                               frame: dict[str, Any]) -> dict[str, Any]:
        try:
            snippet_id = str(frame["snippet_id"])
            start_s, end_s = float(frame["start_s"]), float(frame["end_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CommandError("bad_request",
                               f"crop needs snippet_id, start_s and end_s: {exc}") from exc
        item = self.media.crop(snippet_id, start_s, end_s, MIN_CROP_S, MAX_CROP_S)
        return {"type": "snippet.cropped", "snippet_id": snippet_id,
                "crop_id": item.id, "start_s": item.meta["start_s"],
                "end_s": item.meta["end_s"], "duration_s": item.duration_s,
                "audio_url": item.audio_url, "peaks_url": item.peaks_url}

    async def cmd_snippet_forget(self, client: Client,
                                 frame: dict[str, Any]) -> dict[str, Any]:
        snippet_id = str(frame.get("snippet_id", ""))
        self.media.forget(snippet_id)
        return {"type": "snippet.forgotten", "snippet_id": snippet_id}

    async def cmd_di_list(self, client: Client,
                          frame: dict[str, Any]) -> dict[str, Any]:
        return self._di_listing()

    # -- match -----------------------------------------------------------

    async def cmd_match_run(self, client: Client,
                            frame: dict[str, Any]) -> dict[str, Any] | None:
        if self.job is not None:
            raise CommandError("busy", "a match is already running",
                               job_id=self.job["job_id"])
        index = self._require_index()
        crop = self.media.get(str(frame.get("crop_id", "")))
        if crop.kind not in ("crop", "snippet"):
            raise CommandError("not_found", "match.run needs a crop_id")

        di_id = frame.get("di_id") or self.media.default_di_id
        if di_id is None:
            raise CommandError("not_found", "no DI take is available to audition with")
        di = self.media.get(str(di_id), "di")

        top = max(1, min(int(frame.get("top", 5)), 10))
        render_auditions = bool(frame.get("render_auditions", True))
        job = {"job_id": token(), "cancelled": False, "command_id": frame.get("id"),
               "client": client}
        self.job = job
        self._spawn(self._run_match(job, crop, di, top, render_auditions, index))
        return None            # the terminal reply comes from the job

    async def _run_match(self, job: dict[str, Any], crop: Any, di: Any, top: int,
                         render_auditions: bool,
                         index: matching.ProfileIndex) -> None:
        client: Client = job["client"]
        command_id = job["command_id"]
        loop = asyncio.get_running_loop()

        def progress(stage: str, value: float, message: str,
                     eta: float | None = None) -> None:
            client.send({"type": "match.progress", "id": None,
                         "job_id": job["job_id"], "stage": stage,
                         "progress": round(value, 3), "message": message,
                         "eta_s": eta})

        def check_cancelled() -> None:
            if job["cancelled"]:
                raise CommandError("job_cancelled", "the match was cancelled",
                                   job_id=job["job_id"])

        try:
            seconds = crop.duration_s

            # --- separate -------------------------------------------------
            estimate = seconds * SEPARATION_XRT
            progress("separating", 0.02, "isolating the guitar", estimate)
            started = time.monotonic()
            future = loop.run_in_executor(
                self.sep_pool, jobs.separate_job, crop.samples,
                str(self.cache_dir / "uvr_models"))
            ticker = self._spawn(self._separation_ticker(job, progress, started,
                                                         estimate, future))
            try:
                stem_samples = await future
            finally:
                ticker.cancel()
            check_cancelled()

            separation_ok = stem_samples is not None
            if not separation_ok:
                # A real outcome, not an error: htdemucs_6s returns nothing for
                # guitar on some material. Matching the raw mix is the fallback,
                # and the reply says so rather than presenting it as a stem.
                stem_samples = crop.samples
            stem = self.media.add(np.asarray(stem_samples, dtype=np.float32),
                                  kind="stem",
                                  name=f"{crop.name} - isolated guitar",
                                  parent=crop.parent or crop.id)

            # --- fingerprint and rank -------------------------------------
            progress("fingerprinting", 0.60, "measuring the tone")
            shape, scalars = await loop.run_in_executor(
                self.pool, jobs.fingerprint_job, stem.samples)
            check_cancelled()

            progress("ranking", 0.65, f"ranking {len(index)} profiles")
            distances = index.rank_fingerprint(shape, scalars)
            order = np.argsort(distances)[:top]

            candidates = []
            for rank, i in enumerate(order, 1):
                name = index.names[i]
                entry = self.catalogue.for_name(name)
                distance = float(distances[i])
                candidates.append({
                    "rank": rank,
                    "profile_id": entry.profile_id if entry else "",
                    "name": name,
                    "distance": distance,
                    # "top X% of the index": the share that is at least as close.
                    "percentile": float((np.sum(distances < distance) + 0.5)
                                        / len(distances)),
                    "audition_url": None,
                    "metadata": entry.metadata if entry else {},
                })

            # --- render auditions -----------------------------------------
            if render_auditions:
                di_audio = di.samples[:int(DEFAULT_AUDITION_S * SAMPLE_RATE)]
                per = DEFAULT_AUDITION_S * RENDER_S_PER_DI_S
                for n, candidate in enumerate(candidates):
                    check_cancelled()
                    entry = self.catalogue.get(candidate["profile_id"])
                    if entry is None or not entry.loadable:
                        continue
                    progress("rendering", 0.65 + 0.35 * n / len(candidates),
                             f"rendering {n + 1}/{len(candidates)}: {candidate['name']}",
                             per * (len(candidates) - n))
                    try:
                        rendered = await loop.run_in_executor(
                            self.pool, jobs.render_job, str(entry.path), di_audio)
                    except Exception as exc:
                        candidate["metadata"] = {**candidate["metadata"],
                                                 "render_error": str(exc)}
                        continue
                    item = self.media.add(rendered, kind="render",
                                          name=f"{candidate['name']} through {di.name}",
                                          meta={"profile_id": candidate["profile_id"],
                                                "di_id": di.id})
                    candidate["audition_url"] = item.audio_url

            finite = distances[np.isfinite(distances)]
            client.send({
                "type": "match.result", "id": command_id, "job_id": job["job_id"],
                "crop_id": crop.id,
                "stem_audio_url": stem.audio_url, "stem_peaks_url": stem.peaks_url,
                "separation_ok": separation_ok,
                "di_id": di.id, "di_audio_url": di.audio_url,
                "index_size": len(index), "query_seconds": seconds,
                "distance_range": {"min": float(finite.min()),
                                   "max": float(finite.max()),
                                   "median": float(np.median(finite))},
                "candidates": candidates,
                "caveat": matching.CAVEAT,
            })
        except CommandError as exc:
            client.send({"type": "error", "id": command_id, "code": exc.code,
                         "message": exc.message, "detail": exc.detail})
        except Exception as exc:
            traceback.print_exc()
            code = "separation_failed" if "separat" in str(exc).lower() else "internal"
            client.send({"type": "error", "id": command_id, "code": code,
                         "message": f"{type(exc).__name__}: {exc}", "detail": {}})
        finally:
            if self.job is job:
                self.job = None

    async def _separation_ticker(self, job: dict[str, Any],
                                 progress: Callable[..., None], started: float,
                                 estimate: float, future: Any) -> None:
        """Progress for the one stage that cannot report its own.

        UVR gives no callback, so this interpolates from the measured ~1.0x
        realtime rate and never passes 0.55, which is where separation ends.
        The bar is an estimate; `eta_s` is what it is estimating from.
        """
        while not future.done():
            await asyncio.sleep(0.5)
            elapsed = time.monotonic() - started
            fraction = min(elapsed / estimate, 1.0) if estimate > 0 else 0.0
            progress("separating", 0.02 + 0.53 * fraction,
                     "isolating the guitar", max(estimate - elapsed, 0.0))

    async def cmd_match_cancel(self, client: Client,
                               frame: dict[str, Any]) -> dict[str, Any]:
        job_id = str(frame.get("job_id", ""))
        if self.job is None or self.job["job_id"] != job_id:
            raise CommandError("not_found", f"no running match with id {job_id!r}")
        self.job["cancelled"] = True
        return {"type": "match.cancelled", "job_id": job_id}

    # -- audition --------------------------------------------------------

    async def cmd_audition_render(self, client: Client,
                                  frame: dict[str, Any]) -> dict[str, Any]:
        profile_id = str(frame.get("profile_id", ""))
        entry = self.catalogue.get(profile_id)
        if entry is None:
            raise CommandError("not_found", f"no profile with id {profile_id!r}")
        if not entry.loadable:
            raise CommandError("profile_unsupported",
                               entry.unsupported_reason or "this file will not load")
        di_id = frame.get("di_id") or self.media.default_di_id
        if di_id is None:
            raise CommandError("not_found", "no DI take is available")
        di = self.media.get(str(di_id), "di")
        seconds = float(frame.get("seconds") or DEFAULT_AUDITION_S)
        seconds = max(0.5, min(seconds, di.duration_s))

        command_id = frame.get("id")
        client.send({"type": "audition.progress", "id": command_id, "progress": 0.1})
        loop = asyncio.get_running_loop()
        rendered = await loop.run_in_executor(
            self.pool, jobs.render_job, str(entry.path),
            di.samples[:int(seconds * SAMPLE_RATE)])
        item = self.media.add(rendered, kind="render",
                              name=f"{entry.name} through {di.name}",
                              meta={"profile_id": profile_id, "di_id": di.id})
        return {"type": "audition.ready", "render_id": item.id,
                "profile_id": profile_id, "di_id": di.id,
                "audio_url": item.audio_url, "peaks_url": item.peaks_url,
                "seconds": seconds}


_COMMANDS: dict[str, Callable[..., Awaitable[Any]]] = {
    "hello": ToneHoundServer.cmd_hello,
    "devices.list": ToneHoundServer.cmd_devices_list,
    "devices.select": ToneHoundServer.cmd_devices_select,
    "stream.start": ToneHoundServer.cmd_stream_start,
    "stream.stop": ToneHoundServer.cmd_stream_stop,
    "stream.set": ToneHoundServer.cmd_stream_set,
    "profiles.list": ToneHoundServer.cmd_profiles_list,
    "profile.load": ToneHoundServer.cmd_profile_load,
    "index.status": ToneHoundServer.cmd_index_status,
    "snippet.crop": ToneHoundServer.cmd_snippet_crop,
    "snippet.forget": ToneHoundServer.cmd_snippet_forget,
    "di.list": ToneHoundServer.cmd_di_list,
    "match.run": ToneHoundServer.cmd_match_run,
    "match.cancel": ToneHoundServer.cmd_match_cancel,
    "audition.render": ToneHoundServer.cmd_audition_render,
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


_HTTP_STATUS = {"not_found": 404, "unsupported_media": 415,
                "snippet_too_long": 413, "internal": 500}
"""HTTP has its own vocabulary for these and a browser acts on it: a 404 for a
forgotten snippet lets `<audio>` give up cleanly, where a 400 reads as "the
request was malformed" and invites a retry that cannot succeed."""


def _json_response(status: int, payload: dict[str, Any]) -> Response:
    code = payload.get("code")
    if code:
        status = _HTTP_STATUS.get(code, status)
    return Response(status, dumps(payload).encode("utf-8"),
                    "application/json; charset=utf-8")


def _idle_status() -> dict[str, Any]:
    return {"type": "stream.status", "running": False, "sample_rate": SAMPLE_RATE,
            "block_size": DEFAULT_BLOCK_SIZE,
            "block_budget_ms": DEFAULT_BLOCK_SIZE / SAMPLE_RATE * 1000.0,
            "reported_latency_ms": None, "estimated_total_ms": None,
            "xruns": 0, "last_xrun_at_s": None, "uptime_s": 0.0,
            "device_names": None}


def _default_settings() -> dict[str, Any]:
    return {"type": "stream.settings", "input_gain_db": 0.0, "output_gain_db": 0.0,
            "bypass": False, "monitor": "amp", "tuner_enabled": True,
            "tuner_reference_hz": 440.0}


def _empty_devices() -> dict[str, Any]:
    return {"hostapis": [], "inputs": [], "outputs": [], "default_input": None,
            "default_output": None, "recommended": None}


async def serve(**kwargs: Any) -> None:
    server = ToneHoundServer(**kwargs)
    listener = await server.start()
    host, port = server.host, server.port
    print(f"ToneHound at http://{host}:{port}/")
    print(f"  profiles {len(server.catalogue.entries)} from {server.profile_dir}")
    print(f"  index    {server.render_dir} (fingerprinting in the background)")
    if server.audio_fault:
        print(f"  audio    UNAVAILABLE: {server.audio_fault}")
    try:
        async with listener:
            await listener.serve_forever()
    finally:
        await server.close()


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="ToneHound live server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-audio", action="store_true",
                    help="skip the audio process (matching and import still work)")
    ap.add_argument("--render-dir", type=pathlib.Path, default=None)
    ap.add_argument("--open", action="store_true", help="open a browser")
    args = ap.parse_args(argv)

    if args.open:
        import threading
        import webbrowser
        threading.Timer(1.0, webbrowser.open,
                        (f"http://{args.host}:{args.port}/",)).start()
    try:
        asyncio.run(serve(host=args.host, port=args.port,
                          enable_audio=not args.no_audio,
                          render_dir=args.render_dir))
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
