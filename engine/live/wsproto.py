"""The HTTP/1.1 and RFC 6455 subset this app actually speaks.

There is no WebSocket library in this environment. `websockets` is not
installed and `aiohttp` is only here as a transitive dependency of
`audio-separator`, so building the control channel on it would mean the app
breaks the day that package changes its requirements -- a failure with no
relationship to anything we did. What the protocol needs is small and closed:
one origin, one loopback client class, text frames only, `GET`/`HEAD` plus one
`POST`. That is a few hundred lines of well-specified behaviour, so it is
written out rather than depended upon.

What is deliberately *not* implemented, because nothing here sends it: chunked
request bodies (browsers use `Content-Length` for `FormData`), chunked
responses (every body is in memory and its length is known), compression
extensions (`Sec-WebSocket-Extensions` is simply not negotiated), and binary
WebSocket frames (§1 of the protocol: bulk data goes over HTTP).

What *is* implemented, because leaving it out would look like a bug:

*Range requests.* `<audio>` seeks with `Range: bytes=...`. Without a 206 the
browser has to refetch a whole WAV to move the playhead, and a 15 s render is
1.4 MB. It is a few lines here and a visibly different UI.

*Fragmented and masked frames.* Every browser frame is masked, and a large one
may arrive in fragments. Both are required by the RFC even though the messages
this app sends are small.

*Origin checking.* The server binds loopback only, but a page on any origin can
still reach `127.0.0.1` from the user's own browser. So the handshake rejects
an `Origin` the server did not serve, and that check lives at the transport
edge where it cannot be forgotten by a handler.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import re
import struct
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable

_GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
"""The RFC 6455 handshake GUID. It is checked against the specification's
own worked example in the tests, because a wrong constant here fails in the
one place a hand-written test client will not notice: the browser validates
`Sec-WebSocket-Accept` and refuses the connection, while a client that only
looks for "101" connects happily."""

MAX_BODY = 256 * 1024 * 1024
"""Upload ceiling. A 10-minute WAV is ~100 MB; past this the client is either
confused or hostile, and either way the answer is 413 rather than an OOM."""

MAX_FRAME = 4 * 1024 * 1024
"""No legitimate control frame is close to this. It exists so a malformed
length header cannot make the server allocate an arbitrary buffer."""

STATUS_TEXT = {
    200: "OK", 204: "No Content", 206: "Partial Content", 304: "Not Modified",
    400: "Bad Request", 403: "Forbidden", 404: "Not Found",
    405: "Method Not Allowed", 413: "Payload Too Large",
    415: "Unsupported Media Type", 416: "Range Not Satisfiable",
    500: "Internal Server Error",
}


class ProtocolError(Exception):
    """A request or frame the peer should not have sent."""


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


@dataclass
class Request:
    method: str
    path: str
    query: dict[str, str]
    headers: dict[str, str]
    body: bytes = b""

    def header(self, name: str, default: str = "") -> str:
        return self.headers.get(name.lower(), default)


@dataclass
class Response:
    status: int = 200
    body: bytes = b""
    content_type: str = "text/plain; charset=utf-8"
    headers: dict[str, str] = field(default_factory=dict)


async def read_request(reader: asyncio.StreamReader) -> Request | None:
    """Read one request. ``None`` on a clean connection close."""
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, ConnectionResetError):
        return None
    except asyncio.LimitOverrunError as exc:
        raise ProtocolError("request head too large") from exc

    lines = head.decode("latin-1").split("\r\n")
    try:
        method, target, _version = lines[0].split(" ", 2)
    except ValueError as exc:
        raise ProtocolError(f"malformed request line: {lines[0]!r}") from exc

    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            continue
        name, _, value = line.partition(":")
        # Later duplicates win; no header this app reads is legitimately repeated.
        headers[name.strip().lower()] = value.strip()

    if "chunked" in headers.get("transfer-encoding", "").lower():
        raise ProtocolError("chunked request bodies are not accepted")

    length = int(headers.get("content-length", "0") or 0)
    if length > MAX_BODY:
        raise ProtocolError(f"body of {length} bytes exceeds the {MAX_BODY} limit")
    body = await reader.readexactly(length) if length else b""

    parsed = urllib.parse.urlsplit(target)
    query = {k: v[-1] for k, v in urllib.parse.parse_qs(parsed.query).items()}
    return Request(method=method.upper(),
                   path=urllib.parse.unquote(parsed.path),
                   query=query, headers=headers, body=body)


def _header_block(status: int, headers: Iterable[tuple[str, str]]) -> bytes:
    text = f"HTTP/1.1 {status} {STATUS_TEXT.get(status, 'OK')}\r\n"
    text += "".join(f"{k}: {v}\r\n" for k, v in headers)
    return (text + "\r\n").encode("latin-1")


async def write_response(writer: asyncio.StreamWriter, request: Request | None,
                         response: Response) -> None:
    """Send one response, honouring ``Range`` and ``HEAD``.

    Range is here rather than in a handler because every audio URL needs it and
    a handler that forgot would produce a player that cannot seek -- a bug that
    reads as sluggishness rather than as a missing feature.
    """
    body = response.body
    status = response.status
    extra = dict(response.headers)

    if status == 200 and request is not None and body:
        extra.setdefault("Accept-Ranges", "bytes")
        rng = _parse_range(request.header("range"), len(body))
        if rng == "unsatisfiable":
            status, body = 416, b""
            extra["Content-Range"] = f"bytes */{len(response.body)}"
        elif rng is not None:
            start, end = rng
            status, body = 206, body[start:end + 1]
            extra["Content-Range"] = f"bytes {start}-{end}/{len(response.body)}"

    headers = [("Content-Type", response.content_type),
               ("Content-Length", str(len(body))),
               ("Cache-Control", "no-store"),
               ("Connection", "keep-alive"),
               *extra.items()]
    writer.write(_header_block(status, headers))
    if request is None or request.method != "HEAD":
        writer.write(body)
    await writer.drain()


_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


def _parse_range(value: str, size: int) -> tuple[int, int] | str | None:
    """``(start, end)`` inclusive, ``"unsatisfiable"``, or ``None`` for no range."""
    if not value:
        return None
    m = _RANGE.match(value.strip())
    if not m:
        return None
    first, last = m.group(1), m.group(2)
    if not first and not last:
        return None
    if not first:                                   # bytes=-N, the final N bytes
        n = min(int(last), size)
        return (size - n, size - 1) if n else "unsatisfiable"
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start > end or start >= size:
        return "unsatisfiable"
    return start, end


# --------------------------------------------------------------------------
# multipart/form-data
# --------------------------------------------------------------------------


def parse_multipart(body: bytes, content_type: str) -> dict[str, Any]:
    """Return ``{field: str}`` plus ``{field: (filename, bytes)}`` for file parts.

    Only what `POST /upload` sends: one file part and one optional text field.
    """
    match = re.search(r'boundary="?([^";]+)"?', content_type, re.I)
    if not match:
        raise ProtocolError("multipart body without a boundary")
    sep = b"--" + match.group(1).encode("latin-1")

    fields: dict[str, Any] = {}
    for chunk in body.split(sep)[1:]:
        if chunk[:2] == b"--":                       # closing boundary
            break
        chunk = chunk.lstrip(b"\r\n")
        head, _, data = chunk.partition(b"\r\n\r\n")
        if not _:
            continue
        data = data[:-2] if data.endswith(b"\r\n") else data
        disposition = ""
        for line in head.decode("latin-1", "replace").split("\r\n"):
            if line.lower().startswith("content-disposition:"):
                disposition = line
        name = re.search(r'name="([^"]*)"', disposition)
        if not name:
            continue
        filename = re.search(r'filename="([^"]*)"', disposition)
        if filename:
            fields[name.group(1)] = (filename.group(1), data)
        else:
            fields[name.group(1)] = data.decode("utf-8", "replace")
    return fields


# --------------------------------------------------------------------------
# WebSocket
# --------------------------------------------------------------------------

OP_CONT, OP_TEXT, OP_BINARY = 0x0, 0x1, 0x2
OP_CLOSE, OP_PING, OP_PONG = 0x8, 0x9, 0xA


def handshake_headers(request: Request) -> list[tuple[str, str]] | None:
    """Accept headers for a valid upgrade, or ``None`` if this is not one."""
    if request.header("upgrade").lower() != "websocket":
        return None
    key = request.header("sec-websocket-key")
    if not key or request.header("sec-websocket-version") != "13":
        raise ProtocolError("unsupported WebSocket version")
    accept = base64.b64encode(hashlib.sha1(key.encode() + _GUID).digest()).decode()
    return [("Upgrade", "websocket"), ("Connection", "Upgrade"),
            ("Sec-WebSocket-Accept", accept)]


class WebSocket:
    """One connection. Reads are single-consumer; writes are serialised.

    The write lock matters: telemetry fires at 20 Hz from a broadcast task
    while a command reply may be going out from a handler, and two coroutines
    interleaving their frames on one socket corrupts the stream in a way that
    looks like the client is buggy.
    """

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer
        self.closed = False
        self._lock = asyncio.Lock()

    async def send_text(self, text: str) -> None:
        await self._send(OP_TEXT, text.encode("utf-8"))

    async def ping(self) -> None:
        await self._send(OP_PING, b"")

    async def close(self, code: int = 1000) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            await self._send(OP_CLOSE, struct.pack(">H", code))
        except (OSError, ConnectionError):
            pass
        finally:
            self.writer.close()

    async def _send(self, opcode: int, payload: bytes) -> None:
        if self.closed and opcode != OP_CLOSE:
            return
        n = len(payload)
        if n < 126:
            head = struct.pack(">BB", 0x80 | opcode, n)
        elif n < 65536:
            head = struct.pack(">BBH", 0x80 | opcode, 126, n)
        else:
            head = struct.pack(">BBQ", 0x80 | opcode, 127, n)
        async with self._lock:
            self.writer.write(head + payload)
            await self.writer.drain()

    async def recv_text(self) -> str | None:
        """Next text message, or ``None`` once the peer has closed.

        Control frames are handled here rather than surfaced: a ping must be
        ponged within the read loop or the browser gives up on a connection the
        application layer thinks is healthy.
        """
        buffer = bytearray()
        buffer_op = 0
        while True:
            frame = await self._recv_frame()
            if frame is None:
                return None
            fin, opcode, payload = frame

            if opcode == OP_CLOSE:
                await self.close()
                return None
            if opcode == OP_PING:
                await self._send(OP_PONG, payload)
                continue
            if opcode == OP_PONG:
                continue
            if opcode == OP_BINARY:
                raise ProtocolError("binary frames are not part of this protocol")

            if opcode == OP_CONT:
                if not buffer_op:
                    raise ProtocolError("continuation frame with nothing to continue")
            else:
                if buffer_op:
                    raise ProtocolError("new message began before the last one finished")
                buffer_op = opcode
            buffer += payload
            if len(buffer) > MAX_FRAME:
                raise ProtocolError("message exceeds the frame limit")
            if fin:
                return buffer.decode("utf-8")

    async def _recv_frame(self) -> tuple[bool, int, bytes] | None:
        try:
            head = await self.reader.readexactly(2)
        except (asyncio.IncompleteReadError, ConnectionResetError, OSError):
            return None
        fin = bool(head[0] & 0x80)
        opcode = head[0] & 0x0F
        masked = bool(head[1] & 0x80)
        length = head[1] & 0x7F

        try:
            if length == 126:
                length = struct.unpack(">H", await self.reader.readexactly(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", await self.reader.readexactly(8))[0]
            if length > MAX_FRAME:
                raise ProtocolError(f"frame of {length} bytes exceeds the limit")
            mask = await self.reader.readexactly(4) if masked else b""
            payload = await self.reader.readexactly(length) if length else b""
        except (asyncio.IncompleteReadError, ConnectionResetError, OSError):
            return None

        if masked:
            payload = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
        elif opcode != OP_CLOSE:
            # Required by the RFC; an unmasked client frame means something is
            # speaking a protocol we do not know, so stop rather than guess.
            raise ProtocolError("client frames must be masked")
        return fin, opcode, payload


# --------------------------------------------------------------------------
# the connection loop
# --------------------------------------------------------------------------

Handler = Callable[[Request], Awaitable[Response]]
Upgrader = Callable[[Request, WebSocket], Awaitable[None]]


def origin_allowed(origin: str, host: str, port: int) -> bool:
    """A page on any origin can reach loopback from the user's own browser, so
    an `Origin` we did not serve is refused. An absent header is allowed: that
    is a non-browser client (curl, a test) which no web page can impersonate."""
    if not origin or origin == "null":
        return True
    try:
        parsed = urllib.parse.urlsplit(origin)
    except ValueError:
        return False
    if parsed.hostname not in ("127.0.0.1", "localhost", "::1", host):
        return False
    return (parsed.port or (443 if parsed.scheme == "https" else 80)) == port


async def serve_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                           *, handler: Handler, upgrader: Upgrader,
                           host: str, port: int) -> None:
    """Serve requests on one connection until it closes or upgrades."""
    try:
        while True:
            try:
                request = await read_request(reader)
            except ProtocolError as exc:
                await write_response(writer, None,
                                     Response(400, str(exc).encode()))
                return
            except (ConnectionResetError, OSError):
                return
            if request is None:
                return

            if not origin_allowed(request.header("origin"), host, port):
                await write_response(writer, request,
                                     Response(403, b"origin not allowed"))
                return

            try:
                accept = handshake_headers(request)
            except ProtocolError as exc:
                await write_response(writer, request, Response(400, str(exc).encode()))
                return

            if accept is not None:
                writer.write(_header_block(101, accept))
                await writer.drain()
                await upgrader(request, WebSocket(reader, writer))
                return

            response = await handler(request)
            await write_response(writer, request, response)
            if request.header("connection").lower() == "close":
                return
    finally:
        try:
            writer.close()
        except Exception:
            pass


def token(n: int = 8) -> str:
    """An opaque lowercase-hex id, as §3 of the protocol requires."""
    return os.urandom(n).hex()
