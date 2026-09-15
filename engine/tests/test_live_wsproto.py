"""The hand-written HTTP/WebSocket subset.

This transport is written out rather than depended on (see `wsproto`'s module
docstring), so it has to be tested like a dependency would have been: the
framing edge cases a browser actually produces, not just the happy path a
hand-rolled client would exercise. Masking, fragmentation, control frames
interleaved mid-message and `Range` requests are all things Chrome does and a
naive implementation gets wrong silently.
"""

from __future__ import annotations

import asyncio
import struct

import pytest

from live import wsproto
from live.wsproto import Request, Response


def _reader(payload: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(payload)
    reader.feed_eof()
    return reader


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class _Sink:
    """Stands in for a StreamWriter and keeps what was written."""

    def __init__(self) -> None:
        self.data = bytearray()

    def write(self, chunk: bytes) -> None:
        self.data += chunk

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None


# -- HTTP ------------------------------------------------------------------


def test_request_line_query_and_headers_are_parsed() -> None:
    raw = (b"GET /peaks/abc123?bins=512&x=1 HTTP/1.1\r\n"
           b"Host: 127.0.0.1:8730\r\nOrigin: http://127.0.0.1:8730\r\n\r\n")
    request = _run(wsproto.read_request(_reader(raw)))
    assert request.method == "GET"
    assert request.path == "/peaks/abc123"
    assert request.query == {"bins": "512", "x": "1"}
    assert request.header("origin") == "http://127.0.0.1:8730"


def test_percent_escapes_in_a_path_are_decoded() -> None:
    raw = b"GET /a%20b HTTP/1.1\r\nHost: x\r\n\r\n"
    assert _run(wsproto.read_request(_reader(raw))).path == "/a b"


def test_a_body_is_read_by_content_length() -> None:
    raw = b"POST /upload HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello"
    assert _run(wsproto.read_request(_reader(raw))).body == b"hello"


def test_chunked_request_bodies_are_refused_rather_than_mis_parsed() -> None:
    """Nothing here sends one, and silently treating the chunk header as data
    would corrupt an upload instead of failing it."""
    raw = b"POST /upload HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n"
    with pytest.raises(wsproto.ProtocolError):
        _run(wsproto.read_request(_reader(raw)))


def test_an_oversized_declared_body_is_refused_before_allocating() -> None:
    raw = (f"POST /upload HTTP/1.1\r\nContent-Length: {wsproto.MAX_BODY + 1}\r\n\r\n"
           ).encode()
    with pytest.raises(wsproto.ProtocolError):
        _run(wsproto.read_request(_reader(raw)))


def test_a_closed_connection_reads_as_none_not_an_error() -> None:
    assert _run(wsproto.read_request(_reader(b""))) is None


@pytest.mark.parametrize("header,size,expected", [
    ("bytes=0-9", 100, (0, 9)),
    ("bytes=10-", 100, (10, 99)),
    ("bytes=-10", 100, (90, 99)),
    ("bytes=0-999", 100, (0, 99)),      # clamped to what exists
    ("bytes=200-", 100, "unsatisfiable"),
    ("", 100, None),
    ("bytes=x-y", 100, None),           # unparseable: ignored, whole body sent
])
def test_range_headers(header: str, size: int, expected) -> None:
    assert wsproto._parse_range(header, size) == expected


def test_a_range_request_gets_206_and_only_that_slice() -> None:
    """`<audio>` seeks with Range; without this every seek refetches the file."""
    request = Request("GET", "/audio/x.wav", {}, {"range": "bytes=4-7"})
    sink = _Sink()
    _run(wsproto.write_response(sink, request, Response(200, b"0123456789")))
    head, _, body = bytes(sink.data).partition(b"\r\n\r\n")
    assert b"206 Partial Content" in head
    assert b"Content-Range: bytes 4-7/10" in head
    assert body == b"4567"


def test_head_sends_the_length_but_not_the_body() -> None:
    request = Request("HEAD", "/audio/x.wav", {}, {})
    sink = _Sink()
    _run(wsproto.write_response(sink, request, Response(200, b"0123456789")))
    head, _, body = bytes(sink.data).partition(b"\r\n\r\n")
    assert b"Content-Length: 10" in head
    assert body == b""


# -- multipart -------------------------------------------------------------


def test_multipart_yields_the_file_bytes_and_the_text_field() -> None:
    boundary = "----ez"
    body = (f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="riff.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n").encode() + b"RIFF\x00\xff" + \
           (f"\r\n--{boundary}\r\n"
            'Content-Disposition: form-data; name="kind"\r\n\r\ndi\r\n'
            f"--{boundary}--\r\n").encode()
    fields = wsproto.parse_multipart(body, f"multipart/form-data; boundary={boundary}")
    assert fields["file"] == ("riff.wav", b"RIFF\x00\xff")
    assert fields["kind"] == "di"


def test_multipart_without_a_boundary_is_an_error() -> None:
    with pytest.raises(wsproto.ProtocolError):
        wsproto.parse_multipart(b"", "multipart/form-data")


# -- WebSocket -------------------------------------------------------------


def _client_frame(payload: bytes, opcode: int = wsproto.OP_TEXT,
                  fin: bool = True, mask: bytes = b"\x01\x02\x03\x04") -> bytes:
    n = len(payload)
    first = (0x80 if fin else 0x00) | opcode
    if n < 126:
        head = struct.pack(">BB", first, 0x80 | n)
    elif n < 65536:
        head = struct.pack(">BBH", first, 0x80 | 126, n)
    else:
        head = struct.pack(">BBQ", first, 0x80 | 127, n)
    return head + mask + bytes(b ^ mask[i & 3] for i, b in enumerate(payload))


def _socket(payload: bytes) -> tuple[wsproto.WebSocket, _Sink]:
    sink = _Sink()
    return wsproto.WebSocket(_reader(payload), sink), sink


def test_handshake_accept_key_matches_the_rfc_example() -> None:
    request = Request("GET", "/ws", {}, {
        "upgrade": "websocket", "sec-websocket-version": "13",
        "sec-websocket-key": "dGhlIHNhbXBsZSBub25jZQ=="})
    headers = dict(wsproto.handshake_headers(request))
    assert headers["Sec-WebSocket-Accept"] == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_a_plain_get_is_not_an_upgrade() -> None:
    assert wsproto.handshake_headers(Request("GET", "/", {}, {})) is None


def test_a_masked_text_frame_round_trips() -> None:
    socket, _ = _socket(_client_frame(b'{"type":"hello"}'))
    assert _run(socket.recv_text()) == '{"type":"hello"}'


def test_a_fragmented_message_is_reassembled() -> None:
    payload = (_client_frame(b'{"type":', wsproto.OP_TEXT, fin=False)
               + _client_frame(b'"hello"}', wsproto.OP_CONT, fin=True))
    socket, _ = _socket(payload)
    assert _run(socket.recv_text()) == '{"type":"hello"}'


def test_a_ping_between_fragments_is_ponged_without_breaking_the_message() -> None:
    """Chrome does exactly this, and an implementation that treats a control
    frame as message data corrupts the JSON instead of answering the ping."""
    payload = (_client_frame(b"ab", wsproto.OP_TEXT, fin=False)
               + _client_frame(b"hi", wsproto.OP_PING)
               + _client_frame(b"cd", wsproto.OP_CONT, fin=True))
    socket, sink = _socket(payload)
    assert _run(socket.recv_text()) == "abcd"
    assert bytes(sink.data) == struct.pack(">BB", 0x80 | wsproto.OP_PONG, 2) + b"hi"


def test_a_close_frame_ends_the_stream() -> None:
    socket, _ = _socket(_client_frame(b"", wsproto.OP_CLOSE))
    assert _run(socket.recv_text()) is None
    assert socket.closed


def test_an_unmasked_client_frame_is_refused() -> None:
    """Required by the RFC. Something sending unmasked frames is speaking a
    protocol we do not know, so stop rather than guess at it."""
    socket, _ = _socket(struct.pack(">BB", 0x81, 2) + b"hi")
    with pytest.raises(wsproto.ProtocolError):
        _run(socket.recv_text())


def test_binary_frames_are_refused() -> None:
    socket, _ = _socket(_client_frame(b"\x00\x01", wsproto.OP_BINARY))
    with pytest.raises(wsproto.ProtocolError):
        _run(socket.recv_text())


def test_an_oversized_declared_frame_is_refused_before_allocating() -> None:
    head = struct.pack(">BBQ", 0x81, 0x80 | 127, wsproto.MAX_FRAME + 1)
    socket, _ = _socket(head + b"\x00\x00\x00\x00")
    with pytest.raises(wsproto.ProtocolError):
        _run(socket.recv_text())


@pytest.mark.parametrize("length", [0, 5, 125, 126, 200, 65535, 65536])
def test_server_frames_use_the_right_length_encoding(length: int) -> None:
    socket, sink = _socket(b"")
    _run(socket.send_text("x" * length))
    data = bytes(sink.data)
    assert data[0] == 0x80 | wsproto.OP_TEXT
    assert data[1] & 0x80 == 0                    # server frames are never masked
    header = 2 if length < 126 else (4 if length < 65536 else 10)
    assert len(data) == header + length


# -- origin ----------------------------------------------------------------


@pytest.mark.parametrize("origin,allowed", [
    ("http://127.0.0.1:8730", True),
    ("http://localhost:8730", True),
    ("", True),                                   # curl, a test: not a browser
    ("null", True),
    ("http://127.0.0.1:9999", False),             # another local server's page
    ("https://evil.example", False),
    ("http://evil.example:8730", False),
])
def test_origin_checking(origin: str, allowed: bool) -> None:
    """The server binds loopback, but any page in the user's browser can still
    reach loopback -- so the origin it did not serve is refused."""
    assert wsproto.origin_allowed(origin, "127.0.0.1", 8730) is allowed
