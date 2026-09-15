"""Static file server for developing the UI without the engine.

The real server (engine/live/server.py) serves `GET /` from this directory and
owns the WebSocket. This script exists only so the page can be opened with a
plain HTTP origin -- `file://` blocks `fetch`, blob playback and pointer capture
in ways that would send you chasing bugs that do not exist in the product.

    python engine/live/ui/devserve.py            # then open the printed URL

It serves nothing but this directory, on loopback, and understands no protocol.
Use `?mock=1` in the browser; everything dynamic comes from js/mock.js.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import pathlib
import socketserver

HERE = pathlib.Path(__file__).resolve().parent


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Same as the stdlib handler, minus a log line per asset, plus no caching.

    The stdlib handler sends ``Last-Modified`` and honours ``If-Modified-Since``,
    so a browser will happily keep running the ``app.js`` it fetched before your
    last edit. On a page with no build step and no reload banner that failure is
    invisible: the code on disk is right, the code in the tab is not, and you
    debug the wrong file. A dev server has nothing to gain from caching.
    """

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def send_header(self, keyword: str, value: str) -> None:
        # Suppress the validator itself; without it there is nothing for the
        # browser to revalidate against and every reload is a real fetch.
        if keyword.lower() == "last-modified":
            return
        super().send_header(keyword, value)

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: D102
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8731)
    args = ap.parse_args()

    handler = functools.partial(QuietHandler, directory=str(HERE))
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"UI at http://127.0.0.1:{args.port}/index.html?mock=1")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
