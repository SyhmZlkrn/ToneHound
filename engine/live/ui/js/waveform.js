/*
  Waveform display and crop selection.

  Draws the three arrays from `GET /peaks/{id}?bins=N` -- min, max and rms --
  rather than sample data: the server computes the envelope precisely so the UI
  never downloads a whole song, and drawing rms inside min/max is what makes a
  quiet intro visibly different from a loud chorus at a glance. That difference
  is the whole point of the panel; the user is hunting for the passage with the
  guitar in it.

  Selection is pointer-driven with draggable edges and reports twice: live
  while dragging (so the readout tracks the hand) and once on release (so the
  server only ever gets one `snippet.crop` per gesture). Cropping is cheap and
  non-destructive server-side, but it is still a round trip.
*/
(function (global) {
  "use strict";

  const CSS = getComputedStyle(document.documentElement);
  const col = (n, fallback) => (CSS.getPropertyValue(n) || fallback).trim() || fallback;

  function paint(canvas, peaks, opts) {
    const o = opts || {};
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);

    if (!peaks) return;
    const n = peaks.min.length;
    const mid = h / 2;
    const body = o.body || col("--text-far", "#616874");
    const core = o.core || col("--amber", "#e8a33d");

    // zero line
    g.strokeStyle = "rgba(255,255,255,.06)";
    g.beginPath(); g.moveTo(0, mid + 0.5); g.lineTo(w, mid + 0.5); g.stroke();

    g.fillStyle = body;
    for (let i = 0; i < n; i++) {
      const x = (i / n) * w;
      const bw = Math.max(1, w / n);
      const top = mid - peaks.max[i] * mid * 0.94;
      const bot = mid - peaks.min[i] * mid * 0.94;
      g.fillRect(x, top, bw, Math.max(1, bot - top));
    }
    g.fillStyle = core;
    for (let i = 0; i < n; i++) {
      const x = (i / n) * w;
      const bw = Math.max(1, w / n);
      const r = peaks.rms[i] * mid * 0.94;
      g.fillRect(x, mid - r, bw, Math.max(1, r * 2));
    }
  }

  class Waveform {
    /**
     * @param {HTMLCanvasElement} canvas
     * @param {HTMLElement} overlay  positioned box on top of the canvas
     * @param {HTMLElement} selEl    the selection rectangle
     * @param {HTMLElement} headEl   the playhead line
     * @param {{onChange:(s:number,e:number,final:boolean)=>void, onSeek?:(t:number)=>void}} opts
     */
    constructor(canvas, overlay, selEl, headEl, opts) {
      this.canvas = canvas;
      this.overlay = overlay;
      this.sel = selEl;
      this.head = headEl;
      this.opts = opts || {};
      this.peaks = null;
      this.duration = 0;
      this.start = null;
      this.end = null;
      this._drag = null;

      overlay.addEventListener("pointerdown", (e) => this._down(e));
      window.addEventListener("pointermove", (e) => this._move(e));
      window.addEventListener("pointerup", (e) => this._up(e));
      window.addEventListener("resize", () => this.redraw());
    }

    setPeaks(peaks, duration) {
      this.peaks = peaks;
      this.duration = duration;
      this.redraw();
    }

    clear() {
      this.peaks = null;
      this.start = this.end = null;
      this.sel.classList.remove("on");
      this.redraw();
    }

    redraw() {
      paint(this.canvas, this.peaks);
      this._layoutSel();
    }

    setSelection(start, end, silent) {
      this.start = start; this.end = end;
      this._layoutSel();
      if (!silent && this.opts.onChange) this.opts.onChange(start, end, true);
    }

    setPlayhead(t) {
      if (t == null || !this.duration) { this.head.hidden = true; return; }
      this.head.hidden = false;
      this.head.style.left = (t / this.duration) * 100 + "%";
    }

    /* --- internals ----------------------------------------------------- */

    _layoutSel() {
      if (this.start == null || !this.duration) { this.sel.classList.remove("on"); return; }
      this.sel.classList.add("on");
      const a = Math.min(this.start, this.end) / this.duration;
      const b = Math.max(this.start, this.end) / this.duration;
      this.sel.style.left = a * 100 + "%";
      this.sel.style.width = (b - a) * 100 + "%";
    }

    _time(e) {
      const r = this.overlay.getBoundingClientRect();
      const x = Math.max(0, Math.min(r.width, e.clientX - r.left));
      return (x / r.width) * this.duration;
    }

    _down(e) {
      if (!this.peaks) return;
      this.overlay.setPointerCapture(e.pointerId);
      const edge = e.target && e.target.dataset ? e.target.dataset.edge : null;
      const t = this._time(e);
      if (edge === "l") this._drag = { mode: "l" };
      else if (edge === "r") this._drag = { mode: "r" };
      else { this._drag = { mode: "new" }; this.start = t; this.end = t; }
      this._move(e);
    }

    _move(e) {
      if (!this._drag) return;
      const t = this._time(e);
      if (this._drag.mode === "l") this.start = t;
      else this.end = t;
      this._layoutSel();
      if (this.opts.onChange) {
        this.opts.onChange(Math.min(this.start, this.end), Math.max(this.start, this.end), false);
      }
    }

    _up(e) {
      if (!this._drag) return;
      this._drag = null;
      let a = Math.min(this.start, this.end), b = Math.max(this.start, this.end);
      if (b - a < 0.02) {                       // a click, not a drag: seek instead
        this.start = this.end = null;
        this._layoutSel();
        if (this.opts.onSeek) this.opts.onSeek(a);
        if (this.opts.onChange) this.opts.onChange(0, 0, true);
        return;
      }
      this.start = a; this.end = b;
      this._layoutSel();
      if (this.opts.onChange) this.opts.onChange(a, b, true);
    }
  }

  global.EZWaveform = Waveform;
  global.EZPaintPeaks = paint;
})(window);
