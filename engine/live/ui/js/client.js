/*
  WebSocket client for the protocol in docs/ui_protocol.md.

  Two jobs, and only two:

  1. Turn a command into a promise. The protocol guarantees exactly one
     terminal reply per command id (a success reply or an `error`), preceded by
     any number of non-terminal `*.progress` / `profile.loading` frames. That
     maps cleanly onto a promise plus an `onProgress` callback, and it means the
     UI never has to hand-track which reply belongs to which spinner.

  2. Re-emit every inbound frame to subscribers. Broadcasts (`stream.status`,
     `stream.settings`, `telemetry`, `index.status`, `snippet.ready`) arrive
     with no id and are the way a second tab stays in sync, so the UI listens
     for state changes rather than only reading command replies -- the same
     handler then works whether a change came from this tab or another one.

  Reconnect: the server pings every 15 s; a client that has heard nothing for
  30 s reconnects and re-runs `hello`, which carries a full snapshot, so there
  is no resync protocol to get wrong.

  The mock server in mock.js implements this same surface, so app.js never
  learns which one it is talking to.
*/
(function (global) {
  "use strict";

  const NONTERMINAL = /(\.progress$)|(^profile\.loading$)/;
  const SILENCE_MS = 30000;

  class EZClient {
    constructor(url) {
      this.url = url;
      this.ws = null;
      this.mock = false;
      this.seq = 0;
      this.pending = new Map();          // id -> {resolve, reject, onProgress}
      this.listeners = new Map();        // type ("*" for all) -> Set<fn>
      this.state = "closed";
      this._lastHeard = 0;
      this._watchdog = null;
      this._retryMs = 500;
    }

    /* --- subscription ------------------------------------------------ */

    on(type, fn) {
      if (!this.listeners.has(type)) this.listeners.set(type, new Set());
      this.listeners.get(type).add(fn);
      return () => this.listeners.get(type).delete(fn);
    }

    emit(msg) {
      const direct = this.listeners.get(msg.type);
      if (direct) direct.forEach((fn) => fn(msg));
      const all = this.listeners.get("*");
      if (all) all.forEach((fn) => fn(msg));
    }

    _setState(s, detail) {
      this.state = s;
      this.emit({ type: "conn", state: s, detail: detail || null });
    }

    /* --- connection --------------------------------------------------- */

    connect() {
      this._setState("connecting");
      let ws;
      try {
        ws = new WebSocket(this.url);
      } catch (err) {
        this._scheduleReconnect();
        return;
      }
      this.ws = ws;

      ws.onopen = () => {
        this._retryMs = 500;
        this._lastHeard = Date.now();
        this._startWatchdog();
        this._setState("open");
      };

      ws.onmessage = (ev) => {
        this._lastHeard = Date.now();
        let msg;
        try {
          msg = JSON.parse(ev.data);
        } catch (err) {
          console.warn("[ez] unparseable frame", ev.data);
          return;
        }
        this._dispatch(msg);
      };

      ws.onclose = () => {
        this._stopWatchdog();
        this._failAll("connection closed");
        this._setState("closed");
        this._scheduleReconnect();
      };

      ws.onerror = () => { /* onclose always follows; nothing useful to add */ };
    }

    _dispatch(msg) {
      const p = msg.id != null ? this.pending.get(msg.id) : null;
      if (p && NONTERMINAL.test(msg.type)) {
        if (p.onProgress) p.onProgress(msg);
      } else if (p) {
        this.pending.delete(msg.id);
        if (msg.type === "error") p.reject(Object.assign(new Error(msg.message || msg.code), { frame: msg }));
        else p.resolve(msg);
      }
      this.emit(msg);                     // events AND replies, so state handlers see both
    }

    _scheduleReconnect() {
      const wait = this._retryMs;
      this._retryMs = Math.min(this._retryMs * 2, 5000);
      setTimeout(() => this.connect(), wait);
    }

    _startWatchdog() {
      this._stopWatchdog();
      this._watchdog = setInterval(() => {
        if (Date.now() - this._lastHeard > SILENCE_MS) {
          console.warn("[ez] %d s of silence, reconnecting", SILENCE_MS / 1000);
          try { this.ws.close(); } catch (err) { /* already gone */ }
        }
      }, 5000);
    }

    _stopWatchdog() {
      if (this._watchdog) clearInterval(this._watchdog);
      this._watchdog = null;
    }

    _failAll(reason) {
      this.pending.forEach((p) => p.reject(new Error(reason)));
      this.pending.clear();
    }

    /* --- commands ------------------------------------------------------ */

    send(type, fields, onProgress) {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
        return Promise.reject(new Error("not connected"));
      }
      const id = "c" + ++this.seq;
      const frame = Object.assign({ type, id }, fields || {});
      return new Promise((resolve, reject) => {
        this.pending.set(id, { resolve, reject, onProgress });
        try {
          this.ws.send(JSON.stringify(frame));
        } catch (err) {
          this.pending.delete(id);
          reject(err);
        }
      });
    }

    /* Bulk data is HTTP, never the socket (§3): resolve server-relative URLs. */
    httpUrl(path) {
      if (!path) return path;
      if (/^(https?:|blob:|data:)/.test(path)) return path;
      return this.origin + path;
    }

    get origin() {
      const u = new URL(this.url);
      return (u.protocol === "wss:" ? "https://" : "http://") + u.host;
    }

    async upload(file, kind) {
      const form = new FormData();
      form.append("file", file);
      if (kind) form.append("kind", kind);
      const res = await fetch(this.origin + "/upload", { method: "POST", body: form });
      const body = await res.json();
      if (!res.ok) throw Object.assign(new Error(body.message || "upload failed"), { frame: body });
      return body;
    }

    async peaks(url, bins) {
      const res = await fetch(this.httpUrl(url) + "?bins=" + bins);
      if (!res.ok) throw new Error("peaks " + res.status);
      return res.json();
    }
  }

  global.EZClient = EZClient;
})(window);
