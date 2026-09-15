"""The protocol, over a real socket.

Everything here goes through `asyncio.start_server`, a real TCP connection and
a real RFC 6455 handshake, because the two bugs this level catches are exactly
the ones a handler-level test cannot see: a handshake a browser would reject,
and a reply that is well-formed as a dict but not encodable as JSON.

Audio is disabled. The audio process needs a sound card, and the parts of the
protocol that do not touch one -- the whole import, crop and ranking half, plus
every error path -- are the parts a machine without an interface still has to
get right. Live-audio behaviour is covered against real hardware separately;
what is asserted here is that the commands are refused with a code the UI can
show, rather than hanging.

The one deliberately slow thing (UVR separation, ~1x realtime) is not run: it
is a 15-second wait for a code path already exercised end to end by hand. What
is tested is everything around it, including that `match.run` refuses correctly
before it gets there.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import pathlib
import struct
import urllib.error
import urllib.request

import numpy as np
import pytest

from live import wsproto
from live.server import ToneHoundServer

# -- a minimal client, masked and framed the way a browser does it ----------


class Client:
    def __init__(self, reader: asyncio.StreamReader,
                 writer: asyncio.StreamWriter) -> None:
        self.reader, self.writer = reader, writer
        self.seq = 0
        self.events: list[dict] = []

    @classmethod
    async def connect(cls, port: int, origin: str | None = None) -> "Client":
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        key = base64.b64encode(os.urandom(16)).decode()
        head = (f"GET /ws HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n")
        if origin is not None:
            head += f"Origin: {origin}\r\n"
        writer.write((head + "\r\n").encode())
        await writer.drain()
        response = await reader.readuntil(b"\r\n\r\n")
        if b"101" not in response:
            raise AssertionError(response.decode("latin-1", "replace"))
        # A browser verifies this; a hand-written client that skips it will
        # happily talk to a server no browser can connect to.
        import hashlib
        want = base64.b64encode(
            hashlib.sha1(key.encode() + wsproto._GUID).digest()).decode()
        assert f"Sec-WebSocket-Accept: {want}".encode() in response
        return cls(reader, writer)

    async def send(self, frame: dict) -> None:
        payload = json.dumps(frame).encode()
        mask = os.urandom(4)
        n = len(payload)
        if n < 126:
            head = struct.pack(">BB", 0x81, 0x80 | n)
        elif n < 65536:
            head = struct.pack(">BBH", 0x81, 0x80 | 126, n)
        else:
            head = struct.pack(">BBQ", 0x81, 0x80 | 127, n)
        self.writer.write(head + mask
                          + bytes(b ^ mask[i & 3] for i, b in enumerate(payload)))
        await self.writer.drain()

    async def recv(self) -> dict:
        head = await self.reader.readexactly(2)
        opcode, n = head[0] & 0x0F, head[1] & 0x7F
        if n == 126:
            n = struct.unpack(">H", await self.reader.readexactly(2))[0]
        elif n == 127:
            n = struct.unpack(">Q", await self.reader.readexactly(8))[0]
        data = await self.reader.readexactly(n) if n else b""
        if opcode == wsproto.OP_PING:
            return await self.recv()
        return json.loads(data)

    async def call(self, kind: str, **fields) -> dict:
        self.seq += 1
        command_id = f"t{self.seq}"
        await self.send({"type": kind, "id": command_id, **fields})
        while True:
            frame = await asyncio.wait_for(self.recv(), timeout=30)
            if frame.get("id") == command_id and not frame["type"].endswith(".progress"):
                return frame
            self.events.append(frame)

    async def drain_for(self, kind: str, timeout: float = 5.0) -> dict:
        for frame in list(self.events):
            if frame["type"] == kind:
                self.events.remove(frame)
                return frame
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            frame = await asyncio.wait_for(self.recv(), timeout=timeout)
            if frame["type"] == kind:
                return frame
            self.events.append(frame)
        raise AssertionError(f"no {kind} arrived")

    def close(self) -> None:
        self.writer.close()

    async def wait_for_index(self) -> dict:
        for _ in range(60):
            status = await self.call("index.status")
            if status["ready"]:
                return status
            await asyncio.sleep(0.25)
        raise AssertionError(f"test index never became ready: {status}")


def _upload(port: int, name: str, blob: bytes, kind: str | None = None) -> dict:
    boundary = "----tonehound-test"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{name}\"\r\n\r\n").encode() + blob + b"\r\n"
    if kind:
        body += (f"--{boundary}\r\nContent-Disposition: form-data; "
                 f"name=\"kind\"\r\n\r\n{kind}\r\n").encode()
    body += f"--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/upload", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def _get(port: int, path: str, headers: dict | None = None) -> tuple[int, dict, bytes]:
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                     headers=headers or {})
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, dict(response.headers), response.read()


def _wav(seconds: float, sr: int = 48000) -> bytes:
    import io
    import wave

    t = np.arange(int(seconds * sr)) / sr
    pcm = (np.sin(2 * np.pi * 220.0 * t) * 20000).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())
    return buf.getvalue()


# -- harness ---------------------------------------------------------------


class Harness:
    """Runs the server on its own loop in this process, on an ephemeral port."""

    def __init__(self, tmp_path: pathlib.Path) -> None:
        self.loop = asyncio.new_event_loop()
        renders = tmp_path / "renders_test"
        renders.mkdir()
        rng = np.random.default_rng(0)
        for i in range(3):
            np.save(renders / f"fake_{i}.npy",
                    np.tanh(rng.normal(size=48000) * (1.0 + i)) * 0.4)
        # Protocol checks must run in a clean checkout without private captures
        # or recordings. Include both a loadable model and a corrupt file.
        profiles = tmp_path / "profiles"
        profiles.mkdir()
        (profiles / "Ceriatone fixture.nam").write_text(json.dumps({
            "architecture": "Linear", "version": "0.5.0",
            "config": {"receptive_field": 1, "bias": False},
            "weights": [0.5], "sample_rate": 48000,
        }), encoding="utf-8")
        (profiles / "Broken fixture.nam").write_text("broken model", encoding="utf-8")
        di = tmp_path / "di"
        di.mkdir()
        (di / "Protocol test DI.wav").write_bytes(_wav(12))
        self.server = ToneHoundServer(host="127.0.0.1", port=0, enable_audio=False,
                                   profile_dir=profiles, di_dir=di,
                                   render_dir=renders, cache_dir=tmp_path / "cache")
        self.listener = self.run(self.server.start())
        self.port = self.listener.sockets[0].getsockname()[1]

    def run(self, coro):
        return self.loop.run_until_complete(coro)

    def request(self, fn, *args):
        """Serve HTTP while the blocking client runs on a worker thread.

        Calling urllib directly between run() calls deadlocks: the same thread
        that waits for the response is the only one that can run the server.
        """
        return self.run(asyncio.to_thread(fn, self.port, *args))

    def close(self) -> None:
        self.listener.close()
        self.run(self.server.close())
        self.run(self.loop.shutdown_default_executor())
        self.loop.close()


@pytest.fixture(scope="module")
def live(tmp_path_factory: pytest.TempPathFactory):
    harness = Harness(tmp_path_factory.mktemp("server"))
    try:
        yield harness
    finally:
        harness.close()


# -- session ---------------------------------------------------------------


def test_hello_carries_a_complete_first_frame(live: Harness) -> None:
    async def scenario():
        client = await Client.connect(live.port, f"http://127.0.0.1:{live.port}")
        try:
            return await client.call("hello", protocol=1)
        finally:
            client.close()

    hello = live.run(scenario())
    for field in ("protocol", "server_version", "sample_rate", "default_block_size",
                  "allowed_block_sizes", "max_snippet_s", "min_crop_s", "max_crop_s",
                  "stream", "settings", "devices", "profile", "index", "caveat"):
        assert field in hello, field
    assert hello["protocol"] == 1
    assert hello["sample_rate"] == 48000
    assert hello["default_block_size"] == 512
    assert 128 not in hello["allowed_block_sizes"]
    assert hello["stream"]["running"] is False
    assert set(hello["devices"]) >= {"hostapis", "inputs", "outputs", "recommended"}
    assert hello["caveat"].startswith("Ranking is a rough draft")


def test_a_protocol_mismatch_is_refused_not_guessed_at(live: Harness) -> None:
    async def scenario():
        client = await Client.connect(live.port)
        try:
            return await client.call("hello", protocol=99)
        finally:
            client.close()

    frame = live.run(scenario())
    assert frame["type"] == "error" and frame["code"] == "bad_request"


def test_an_unknown_command_gets_one_terminal_error(live: Harness) -> None:
    async def scenario():
        client = await Client.connect(live.port)
        try:
            await client.call("hello", protocol=1)
            return await client.call("nonsense.command")
        finally:
            client.close()

    frame = live.run(scenario())
    assert frame["type"] == "error" and frame["code"] == "unknown_type"


def test_a_non_json_frame_does_not_kill_the_connection(live: Harness) -> None:
    async def scenario():
        client = await Client.connect(live.port)
        try:
            await client.send({"type": "hello", "id": "x", "protocol": 1})
            await client.recv()
            client.writer.write(b"\x81\x83\x00\x00\x00\x00abc")   # masked "abc"
            await client.writer.drain()
            first = await asyncio.wait_for(client.recv(), timeout=5)
            after = await client.call("profiles.list")
            return first, after
        finally:
            client.close()

    first, after = live.run(scenario())
    assert first["type"] == "error" and first["code"] == "bad_request"
    assert after["type"] == "profiles.list"


def test_an_unknown_origin_is_refused(live: Harness) -> None:
    """A page on any site can reach loopback from the user's own browser."""
    async def scenario():
        reader, writer = await asyncio.open_connection("127.0.0.1", live.port)
        writer.write((f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{live.port}\r\n"
                      "Origin: https://evil.example\r\n\r\n").encode())
        await writer.drain()
        head = await reader.readuntil(b"\r\n\r\n")
        writer.close()
        return head

    assert b"403" in live.run(scenario())


# -- library ---------------------------------------------------------------


def test_profiles_list_keeps_unloadable_files_visible(live: Harness) -> None:
    async def scenario():
        client = await Client.connect(live.port)
        try:
            await client.call("hello", protocol=1)
            everything = await client.call("profiles.list")
            filtered = await client.call("profiles.list", query="ceriatone")
            return everything, filtered
        finally:
            client.close()

    everything, filtered = live.run(scenario())
    assert everything["total"] == len(everything["profiles"])
    assert everything["total"] > 0
    assert any(p["loadable"] for p in everything["profiles"])
    assert any(not p["loadable"] and p["unsupported_reason"] for p in everything["profiles"])
    for profile in everything["profiles"]:
        assert set(profile) >= {"profile_id", "name", "architecture", "sample_rate",
                                "receptive_field", "weight_count", "loadable",
                                "unsupported_reason", "rate_mismatch", "metadata"}
    assert filtered["total"] == everything["total"], "total is the library, not the filter"
    assert len(filtered["profiles"]) < len(everything["profiles"])
    assert len(filtered["profiles"]) == 1


def test_di_list_offers_a_default_take(live: Harness) -> None:
    async def scenario():
        client = await Client.connect(live.port)
        try:
            await client.call("hello", protocol=1)
            return await client.call("di.list")
        finally:
            client.close()

    listing = live.run(scenario())
    assert listing["takes"]
    assert listing["default_di_id"] in {t["di_id"] for t in listing["takes"]}
    for take in listing["takes"]:
        assert take["audio_url"].startswith("/audio/")
        assert take["duration_s"] > 0


# -- import, crop, serve ---------------------------------------------------


def test_upload_crop_and_serve_round_trip(live: Harness) -> None:
    async def scenario():
        client = await Client.connect(live.port)
        try:
            await client.call("hello", protocol=1)
            uploaded = await asyncio.get_running_loop().run_in_executor(
                None, _upload, live.port, "tone.wav", _wav(40.0))
            ready = await client.drain_for("snippet.ready")
            cropped = await client.call("snippet.crop",
                                        snippet_id=uploaded["snippet_id"],
                                        start_s=5.0, end_s=17.0)
            forgotten = await client.call("snippet.forget",
                                          snippet_id=uploaded["snippet_id"])
            return uploaded, ready, cropped, forgotten
        finally:
            client.close()

    uploaded, ready, cropped, forgotten = live.run(scenario())
    assert ready["snippet_id"] == uploaded["snippet_id"]
    assert ready["duration_s"] == pytest.approx(40.0, abs=0.1)
    assert ready["source_sample_rate"] == 48000 and ready["source_channels"] == 1

    assert cropped["crop_id"] != uploaded["snippet_id"]
    assert cropped["duration_s"] == pytest.approx(12.0, abs=0.05)
    assert forgotten["type"] == "snippet.forgotten"

    # The crop is derived from the snippet, so forgetting one drops both.
    with pytest.raises(urllib.error.HTTPError) as gone:
        live.request(_get, f"/peaks/{cropped['crop_id']}")
    assert gone.value.code == 404


def test_peaks_and_audio_are_served_over_http(live: Harness) -> None:
    async def scenario():
        client = await Client.connect(live.port)
        try:
            await client.call("hello", protocol=1)
            uploaded = await asyncio.get_running_loop().run_in_executor(
                None, _upload, live.port, "tone.wav", _wav(20.0))
            await client.drain_for("snippet.ready")
            return uploaded["snippet_id"]
        finally:
            client.close()

    snippet_id = live.run(scenario())

    status, _, body = live.request(_get, f"/peaks/{snippet_id}?bins=640")
    peaks = json.loads(body)
    assert status == 200
    assert peaks["bins"] == 640
    assert len(peaks["min"]) == len(peaks["max"]) == len(peaks["rms"]) == 640
    assert peaks["duration_s"] == pytest.approx(20.0, abs=0.1)

    status, headers, body = live.request(_get, f"/audio/{snippet_id}.wav")
    assert status == 200
    assert headers["Content-Type"] == "audio/wav"
    assert body[:4] == b"RIFF"
    assert headers.get("Accept-Ranges") == "bytes"

    # `<audio>` seeks with Range; a 200 here means every seek refetches.
    status, headers, chunk = live.request(_get, f"/audio/{snippet_id}.wav",
                                  {"Range": "bytes=100-199"})
    assert status == 206 and len(chunk) == 100
    assert headers["Content-Range"].startswith("bytes 100-199/")


def test_a_di_upload_joins_the_di_list(live: Harness) -> None:
    async def scenario():
        client = await Client.connect(live.port)
        try:
            await client.call("hello", protocol=1)
            before = await client.call("di.list")
            await asyncio.get_running_loop().run_in_executor(
                None, _upload, live.port, "mytake.wav", _wav(6.0), "di")
            after = await client.call("di.list")
            return before, after
        finally:
            client.close()

    before, after = live.run(scenario())
    assert len(after["takes"]) == len(before["takes"]) + 1
    assert "mytake" in {t["name"] for t in after["takes"]}
    assert after["default_di_id"] == before["default_di_id"], \
        "importing a take must not silently move the default the index was built on"


def test_an_undecodable_upload_is_a_told_error(live: Harness) -> None:
    with pytest.raises(urllib.error.HTTPError) as failed:
        live.request(_upload, "notes.txt", b"this is not audio at all")
    assert failed.value.code == 415
    assert json.loads(failed.value.read())["code"] == "unsupported_media"


def test_crop_bounds_are_enforced_with_the_documented_codes(live: Harness) -> None:
    async def scenario():
        client = await Client.connect(live.port)
        try:
            hello = await client.call("hello", protocol=1)
            uploaded = await asyncio.get_running_loop().run_in_executor(
                None, _upload, live.port, "tone.wav", _wav(60.0))
            await client.drain_for("snippet.ready")
            snippet_id = uploaded["snippet_id"]
            short = await client.call("snippet.crop", snippet_id=snippet_id,
                                      start_s=0.0, end_s=hello["min_crop_s"] / 2)
            long = await client.call("snippet.crop", snippet_id=snippet_id,
                                     start_s=0.0, end_s=hello["max_crop_s"] + 10)
            missing = await client.call("snippet.crop", snippet_id="0" * 16,
                                        start_s=0.0, end_s=10.0)
            malformed = await client.call("snippet.crop", snippet_id=snippet_id)
            return short, long, missing, malformed
        finally:
            client.close()

    short, long, missing, malformed = live.run(scenario())
    assert short["code"] == "crop_too_short"
    assert long["code"] == "snippet_too_long"
    assert missing["code"] == "not_found"
    assert malformed["code"] == "bad_request"


# -- match refusals --------------------------------------------------------


def test_match_refuses_a_crop_it_does_not_have(live: Harness) -> None:
    async def scenario():
        client = await Client.connect(live.port)
        try:
            await client.call("hello", protocol=1)
            await client.wait_for_index()
            return await client.call("match.run", crop_id="0" * 16, top=3)
        finally:
            client.close()

    assert live.run(scenario())["code"] == "not_found"


def test_audition_refuses_an_unknown_profile(live: Harness) -> None:
    async def scenario():
        client = await Client.connect(live.port)
        try:
            await client.call("hello", protocol=1)
            return await client.call("audition.render", profile_id="0" * 16)
        finally:
            client.close()

    assert live.run(scenario())["code"] == "not_found"


def test_cancelling_an_unknown_job_is_not_found(live: Harness) -> None:
    async def scenario():
        client = await Client.connect(live.port)
        try:
            await client.call("hello", protocol=1)
            return await client.call("match.cancel", job_id="nope")
        finally:
            client.close()

    assert live.run(scenario())["code"] == "not_found"


# -- audio absence ---------------------------------------------------------


def test_audio_commands_are_refused_with_a_code_not_a_hang(live: Harness) -> None:
    """A machine with no interface must still import and match; the live half
    has to say why it cannot run rather than leaving a spinner up."""
    async def scenario():
        client = await Client.connect(live.port)
        try:
            await client.call("hello", protocol=1)
            return [await client.call(kind) for kind in
                    ("devices.list", "stream.start", "stream.stop")]
        finally:
            client.close()

    for frame in live.run(scenario()):
        assert frame["type"] == "error"
        assert frame["code"] == "device_open_failed"
        assert "no-audio" in frame["message"]


# -- broadcast -------------------------------------------------------------


def test_a_second_tab_sees_state_changes(live: Harness) -> None:
    """Every state change is broadcast so two open tabs cannot disagree."""
    async def scenario():
        first = await Client.connect(live.port)
        second = await Client.connect(live.port)
        try:
            await first.call("hello", protocol=1)
            await second.call("hello", protocol=1)
            await asyncio.get_running_loop().run_in_executor(
                None, _upload, live.port, "shared.wav", _wav(8.0))
            return (await first.drain_for("snippet.ready"),
                    await second.drain_for("snippet.ready"))
        finally:
            first.close()
            second.close()

    a, b = live.run(scenario())
    assert a["snippet_id"] == b["snippet_id"]
    assert a["id"] is None, "a broadcast is an event and carries no command id"


def test_the_index_becomes_ready_and_is_reported(live: Harness) -> None:
    async def scenario():
        client = await Client.connect(live.port)
        try:
            return await client.wait_for_index()
        finally:
            client.close()

    status = live.run(scenario())
    assert status["ready"] is True
    assert status["profiles"] == status["total"] == 3
    assert status["progress"] == 1.0
    assert status["di_name"] == "renders_test"
