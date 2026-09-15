"""The audio engine in its own OS process, and the proxy that drives it.

This split is a measurement, not a preference. One match run is 14.7 s of
all-core CPU for separation alone and the profile index costs 45 s to
fingerprint at startup; an audio callback sharing that interpreter would drop
out continuously, whatever the GIL did. The GIL argument holds too -- the
callback survived a deliberately GIL-hungry Python thread at block 512 with
zero xruns, but only because TorchScript releases the GIL for the whole forward
pass, and one 24.6 ms outlier still appeared against a 10.67 ms budget. So the
audio path gets an interpreter where nothing else ever runs, and the server
talks to it down a pipe.

**The wire.** A `multiprocessing.Connection` carrying pickled dicts. Commands
are `{"rid": n, "op": ..., ...}` and get exactly one `{"rid": n, "ok": ...}`
back; anything with no `rid` is an unsolicited event (telemetry, perf, a status
change); a `{"rid": n, "progress": ...}` is a non-terminal update for a command
still running. That mirrors the WebSocket protocol's own envelope on purpose --
`profile.loading` then `profile.loaded` is the same shape at both hops, so the
server forwards rather than translates.

**Reading it without blocking the loop.** On Windows a `Connection` is a named
pipe handle that asyncio cannot select on, so a plain thread does the blocking
`recv()` and hands each message to the loop with `call_soon_threadsafe`. Sends
happen only on the loop thread, so the connection has one writer and one
reader and needs no lock. In the child the reverse is true -- the command loop
and the telemetry thread both send -- so there the sends take a lock, which is
safe because neither of those is the audio thread.

**Death is expected.** PortAudio can take a process down (a WDM-KS device
yanked mid-stream is the realistic case). The proxy notices EOF, fails every
pending call rather than hanging the UI on a promise that will never settle,
and respawns on the next command with the settings and profile replayed. Device
selection is deliberately *not* replayed: reopening a device that just killed
the process, without the user asking, is how a crash loop starts.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import multiprocessing.connection as mpc
import threading
import time
from typing import Any, Callable

TELEMETRY_HZ = 20.0
PERF_HZ = 1.0
CALL_TIMEOUT_S = 45.0
"""Generous: `devices.list(refresh=True)` re-enumerates PortAudio and a device
open retries once with a 0.25 s pause. A call that outlives this has hung, and
an error the UI can show beats a spinner that never stops."""


# --------------------------------------------------------------------------
# child process
# --------------------------------------------------------------------------


class _Child:
    """The command dispatcher inside the audio process."""

    def __init__(self, conn: mpc.Connection) -> None:
        self.conn = conn
        self.send_lock = threading.Lock()
        self.running = True

        from .audio_engine import AudioEngine  # imported here: this is the only
        self.engine = AudioEngine()            # process that may touch PortAudio

    # -- wire ------------------------------------------------------------

    def send(self, message: dict[str, Any]) -> None:
        try:
            with self.send_lock:
                self.conn.send(message)
        except (BrokenPipeError, OSError):
            self.running = False

    # -- operations ------------------------------------------------------

    def op_devices_list(self, refresh: bool = False) -> dict[str, Any]:
        from . import devices
        # The recommender verifies a pair by opening it, and a WDM-KS open
        # takes the device exclusively -- which would interrupt the stream this
        # call is meant to be helping configure. So while audio is running the
        # recommendation goes back to being a name heuristic, and says so.
        return devices.device_list(bool(refresh), probe=not self.engine.running)

    def op_configure(self, **kwargs: Any) -> dict[str, Any]:
        return self.engine.configure(**kwargs)

    def op_start(self) -> dict[str, Any]:
        self.engine.start()
        return self.engine.status()

    def op_stop(self) -> dict[str, Any]:
        self.engine.stop()
        return self.engine.status()

    def op_status(self) -> dict[str, Any]:
        return self.engine.status()

    def op_settings(self) -> dict[str, Any]:
        return self.engine.settings()

    def op_apply_settings(self, **kwargs: Any) -> dict[str, Any]:
        return self.engine.apply_settings(**kwargs)

    def op_load_profile(self, rid: int, path: str, profile_id: str) -> dict[str, Any]:
        def progress(stage: str) -> None:
            self.send({"rid": rid, "progress": {"stage": stage}})
        return self.engine.load_profile(path, profile_id, progress)

    def op_close(self) -> dict[str, Any]:
        self.engine.close()
        self.running = False
        return {}

    # -- loops -----------------------------------------------------------

    def telemetry_loop(self) -> None:
        """Meters, tuner and perf, plus the status edge the protocol says must
        be broadcast on any xrun. Never touches the audio callback's data other
        than through the engine's own destructive-read accessors."""
        # Deadlines advance by a fixed period rather than from "now", because
        # Windows sleeps long: asking for 5 ms yields 10-15 ms, and rescheduling
        # from the wake-up time turned a nominal 20 Hz into a measured 16 Hz.
        # Advancing the deadline lets a late frame be followed by an immediate
        # one, so the average rate is the one the protocol promises.
        period_tel = 1.0 / TELEMETRY_HZ
        period_perf = 1.0 / PERF_HZ
        next_tel = time.monotonic()
        next_perf = next_tel
        last_xruns = -1
        last_running = None

        while self.running:
            now = time.monotonic()
            try:
                if self.engine.running and now >= next_tel:
                    self.send({"event": self.engine.telemetry()})
                    next_tel = max(next_tel + period_tel, now - period_tel)
                if now >= next_perf:
                    if self.engine.running:
                        self.send({"event": self.engine.perf()})
                    next_perf = max(next_perf + period_perf, now - period_perf)
                if self.engine.xruns != last_xruns or self.engine.running != last_running:
                    last_xruns = self.engine.xruns
                    last_running = self.engine.running
                    self.send({"event": self.engine.status()})
            except Exception as exc:  # telemetry must never take the process down
                self.send({"event": {"type": "engine.fault", "message": str(exc)}})
            time.sleep(0.002)

    def command_loop(self) -> None:
        while self.running:
            try:
                message = self.conn.recv()
            except (EOFError, OSError):
                break
            rid = message.get("rid")
            op = message.get("op", "")
            args = {k: v for k, v in message.items() if k not in ("rid", "op")}
            handler = getattr(self, f"op_{op.replace('.', '_')}", None)
            if handler is None:
                self.send({"rid": rid, "ok": False, "code": "unknown_type",
                           "message": f"audio process has no operation {op!r}"})
                continue
            try:
                if op == "load_profile":
                    result = handler(rid=rid, **args)
                else:
                    result = handler(**args)
                self.send({"rid": rid, "ok": True, "result": result})
            except Exception as exc:
                self.send({"rid": rid, "ok": False,
                           "code": getattr(exc, "code", "internal"),
                           "message": str(exc),
                           "error_type": type(exc).__name__})
        self.running = False


def child_main(conn: mpc.Connection) -> None:
    """Entry point of the audio process. Nothing else may run here."""
    child = _Child(conn)
    telemetry = threading.Thread(target=child.telemetry_loop, daemon=True,
                                 name="tonehound-telemetry")
    telemetry.start()
    try:
        child.command_loop()
    finally:
        child.running = False
        try:
            child.engine.close()
        except Exception:
            pass
        conn.close()


# --------------------------------------------------------------------------
# parent proxy
# --------------------------------------------------------------------------


class AudioProcessError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class AudioProcess:
    """Async handle on the audio process, from the server's event loop."""

    def __init__(self, on_event: Callable[[dict[str, Any]], None],
                 on_death: Callable[[str], None] | None = None) -> None:
        self.on_event = on_event
        self.on_death = on_death
        self._proc: mp.process.BaseProcess | None = None
        self._conn: mpc.Connection | None = None
        self._reader: threading.Thread | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._progress: dict[int, Callable[[dict[str, Any]], None]] = {}
        self._rid = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._generation = 0
        self._closing = False

        # Replayed after a respawn. Device selection is not: see the module
        # docstring -- reopening the device that just killed the process is a
        # decision for the user, not for a recovery path.
        self._settings: dict[str, Any] = {}
        self._profile: tuple[str, str] | None = None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    # -- lifecycle -------------------------------------------------------

    async def ensure_started(self) -> None:
        if self.alive:
            return
        self._loop = asyncio.get_running_loop()
        self._closing = False
        self._teardown()
        # "spawn" is the only start method on Windows and the right one
        # anywhere here: a forked copy of a process that has already imported
        # torch and opened sockets is a much worse starting state than a fresh
        # interpreter that imports what the audio path needs and nothing else.
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        proc = ctx.Process(target=child_main, args=(child_conn,),
                           name="tonehound-audio", daemon=True)
        proc.start()
        child_conn.close()

        self._proc, self._conn = proc, parent_conn
        self._generation += 1
        self._reader = threading.Thread(target=self._read_loop,
                                        args=(parent_conn, self._generation),
                                        daemon=True, name="tonehound-audio-reader")
        self._reader.start()

        if self._settings:
            await self.call("apply_settings", **self._settings)
        if self._profile is not None:
            path, profile_id = self._profile
            await self.call("load_profile", path=path, profile_id=profile_id)

    def _teardown(self) -> None:
        # Invalidate the current reader thread's generation *first*: a clean
        # shutdown closes the pipe, and without this the EOF that follows would
        # be reported to the UI as "the audio process stopped unexpectedly".
        self._generation += 1
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
        self._conn = None
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
        self._proc = None

    async def close(self) -> None:
        # Set before the request goes out: the child answers, exits and drops
        # the pipe, and the EOF that follows would otherwise reach the UI as
        # "the audio process stopped unexpectedly" on an ordinary quit.
        self._closing = True
        if self.alive:
            try:
                await asyncio.wait_for(self.call("close"), timeout=3.0)
            except Exception:
                pass
        self._teardown()

    # -- wire ------------------------------------------------------------

    def _read_loop(self, conn: mpc.Connection, generation: int) -> None:
        """Blocking reads on a thread; every message is handed to the loop."""
        loop = self._loop
        assert loop is not None
        while True:
            try:
                message = conn.recv()
            except (EOFError, OSError):
                break
            try:
                loop.call_soon_threadsafe(self._dispatch, message)
            except RuntimeError:
                return  # loop is shutting down
        try:
            loop.call_soon_threadsafe(self._on_eof, generation)
        except RuntimeError:
            pass

    def _dispatch(self, message: dict[str, Any]) -> None:
        if "event" in message:
            self.on_event(message["event"])
            return
        rid = message.get("rid")
        if "progress" in message:
            callback = self._progress.get(rid)
            if callback:
                callback(message["progress"])
            return
        future = self._pending.pop(rid, None)
        self._progress.pop(rid, None)
        if future is None or future.done():
            return
        if message.get("ok"):
            future.set_result(message.get("result"))
        else:
            future.set_exception(AudioProcessError(
                message.get("code", "internal"),
                message.get("message", "the audio process reported an error")))

    def _on_eof(self, generation: int) -> None:
        if generation != self._generation or self._closing:
            return
        exit_code = self._proc.exitcode if self._proc is not None else None
        for future in self._pending.values():
            if not future.done():
                future.set_exception(AudioProcessError(
                    "internal", "the audio process stopped before replying"))
        self._pending.clear()
        self._progress.clear()
        self._teardown()
        if self.on_death is not None:
            self.on_death(
                f"the audio process stopped unexpectedly (exit code {exit_code}). "
                "Re-select the device and start the stream again.")

    async def call(self, op: str, *, timeout: float = CALL_TIMEOUT_S,
                   on_progress: Callable[[dict[str, Any]], None] | None = None,
                   **args: Any) -> Any:
        """Run one operation in the audio process and wait for its reply."""
        if not self.alive:
            raise AudioProcessError("internal", "the audio process is not running")
        loop = asyncio.get_running_loop()
        self._rid += 1
        rid = self._rid
        future: asyncio.Future = loop.create_future()
        self._pending[rid] = future
        if on_progress is not None:
            self._progress[rid] = on_progress
        try:
            assert self._conn is not None
            self._conn.send({"rid": rid, "op": op, **args})
        except (BrokenPipeError, OSError) as exc:
            self._pending.pop(rid, None)
            self._progress.pop(rid, None)
            raise AudioProcessError("internal",
                                    f"could not reach the audio process: {exc}") from exc

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            self._pending.pop(rid, None)
            self._progress.pop(rid, None)
            raise AudioProcessError(
                "internal", f"the audio process did not answer {op!r} "
                            f"within {timeout:.0f}s") from exc
        finally:
            self._progress.pop(rid, None)

        # Remember what is cheap and safe to restore after a respawn.
        if op == "apply_settings":
            self._settings.update(args)
        elif op == "load_profile":
            self._profile = (args["path"], args["profile_id"])
        return result
