/*
  Meters and tuner needle: everything that has to move smoothly.

  Telemetry arrives at 20 Hz (§4.4) and may drop frames. Painting straight from
  those messages gives a meter that visibly steps 20 times a second and a needle
  that twitches, which reads as "the app is struggling" even when the audio is
  perfect. So telemetry only ever sets a *target*, and a single
  requestAnimationFrame loop eases the drawn value toward it at the display's
  own rate. One rAF loop for the whole page, not one per widget, so a hidden tab
  costs nothing and the widgets stay in step.

  The ballistics are the conventional ones for audio meters: fast attack, slow
  release, and a peak marker that holds before falling. A meter that fell as
  fast as it rose would be unreadable on a plucked note.

  The tuner holds its last reading for 500 ms before blanking, as the protocol
  advises, so the needle does not flicker to nothing between picks.
*/
(function (global) {
  "use strict";

  const DB_FLOOR = -60;
  const norm = (db) => Math.max(0, Math.min(1, (db - DB_FLOOR) / (0 - DB_FLOOR)));

  /* One clock for every animated widget, and it stops when everything has
     settled: a stopped stream should cost nothing, not 60 repaints a second
     of an unchanging meter. Widgets call wake() when new telemetry lands and
     report through needsFrame() whether they are still moving. */
  const widgets = [];
  let last = 0, running = false;

  function loop(t) {
    const dt = last ? Math.min(0.1, (t - last) / 1000) : 0.016;
    last = t;
    let more = false;
    for (let i = 0; i < widgets.length; i++) {
      widgets[i].tick(dt, t / 1000);
      if (widgets[i].needsFrame()) more = true;
    }
    if (more) requestAnimationFrame(loop);
    else { running = false; last = 0; }
  }

  function wake() {
    if (running) return;
    running = true;
    requestAnimationFrame(loop);
  }

  /** Exponential approach with a time constant, framerate independent. */
  const ease = (cur, target, tau, dt) => cur + (target - cur) * (1 - Math.exp(-dt / tau));

  class Meter {
    constructor(canvas) {
      this.canvas = canvas;
      this.rms = DB_FLOOR;
      this.peak = DB_FLOOR;
      this.targetRms = DB_FLOOR;
      this.targetPeak = DB_FLOOR;
      this.hold = DB_FLOOR;
      this.holdUntil = 0;
      this.clip = 0;
      widgets.push({ tick: (dt, now) => this._tick(dt, now), needsFrame: () => this._busy() });
      wake();                            // one frame to paint the empty meter
    }

    set(rmsDb, peakDb, clipped) {
      this.targetRms = rmsDb == null ? DB_FLOOR : rmsDb;
      this.targetPeak = peakDb == null ? DB_FLOOR : peakDb;
      if (clipped) this.clip = 1.0;
      wake();
    }

    silence() { this.targetRms = this.targetPeak = DB_FLOOR; wake(); }

    _busy() {
      return this.clip > 0 ||
        Math.abs(this.rms - this.targetRms) > 0.05 ||
        Math.abs(this.peak - this.targetPeak) > 0.05 ||
        this.hold > Math.max(this.peak, DB_FLOOR) + 0.05;
    }

    _tick(dt, now) {
      const upTau = 0.012, downTau = 0.20;
      this.rms = ease(this.rms, this.targetRms, this.targetRms > this.rms ? upTau : downTau, dt);
      this.peak = ease(this.peak, this.targetPeak, this.targetPeak > this.peak ? 0.004 : 0.35, dt);
      if (this.peak >= this.hold) { this.hold = this.peak; this.holdUntil = now + 1.1; }
      else if (now > this.holdUntil) this.hold = Math.max(this.peak, this.hold - 26 * dt);
      if (this.clip > 0) this.clip = Math.max(0, this.clip - dt / 1.2);
      this._paint();
    }

    _paint() {
      const c = this.canvas;
      const dpr = window.devicePixelRatio || 1;
      const w = c.clientWidth, h = c.clientHeight;
      if (!w || !h) return;
      if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
      const g = c.getContext("2d");
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);

      g.fillStyle = "#0a0b0d";
      g.fillRect(0, 0, w, h);

      const grad = g.createLinearGradient(0, 0, w, 0);
      grad.addColorStop(0, "#2f8f5b");
      grad.addColorStop(norm(-18), "#46cf7c");
      grad.addColorStop(norm(-6), "#e8a33d");
      grad.addColorStop(norm(-1), "#e2564a");
      grad.addColorStop(1, "#e2564a");
      g.fillStyle = grad;
      g.fillRect(0, 0, w * norm(this.rms), h);

      // peak-hold marker
      const px = w * norm(this.hold);
      if (this.hold > DB_FLOOR + 0.5) {
        g.fillStyle = "rgba(255,255,255,.85)";
        g.fillRect(Math.min(w - 2, px), 0, 2, h);
      }

      // scale marks at -40 -30 -20 -12 -6 -3 0
      g.fillStyle = "rgba(255,255,255,.13)";
      [-40, -30, -20, -12, -6, -3, 0].forEach((db) => g.fillRect(w * norm(db), 0, 1, h));

      if (this.clip > 0) {
        g.fillStyle = "rgba(226,86,74," + (0.35 * this.clip).toFixed(3) + ")";
        g.fillRect(0, 0, w, h);
      }
    }
  }

  class Tuner {
    constructor(els) {
      this.els = els;                    // {root, needle, name, oct, cents, hz, hint, strings}
      this.cents = 0;
      this.target = 0;
      this.detected = false;
      this.reading = null;
      this.seenWall = -99;               // wall clock, same base as the rAF hold test
      this._buildScale();
      this._buildStrings();
      widgets.push({ tick: (dt, now) => this._tick(dt, now), needsFrame: () => this._busy() });
      wake();
    }

    _busy() {
      // Keep animating while the needle is still moving, and for the 500 ms
      // hold after the last detection so the blank-out is not abrupt.
      return Math.abs(this.cents - (this._held ? this.target : 0)) > 0.02 ||
        performance.now() / 1000 - this.seenWall < 0.6;
    }

    _buildScale() {
      const g = this.els.ticks;
      if (!g) return;
      let html = "";
      for (let c = -50; c <= 50; c += 5) {
        const a = (c / 50) * (Math.PI / 4);             // +/- 45 degrees
        const cx = 200, cy = 140, r0 = c % 25 === 0 ? 96 : 106, r1 = 116;
        const sx = cx + Math.sin(a) * r0, sy = cy - Math.cos(a) * r0;
        const ex = cx + Math.sin(a) * r1, ey = cy - Math.cos(a) * r1;
        html += '<line class="' + (c % 25 === 0 ? "major" : "") + '" x1="' + sx.toFixed(1) + '" y1="' + sy.toFixed(1) +
                '" x2="' + ex.toFixed(1) + '" y2="' + ey.toFixed(1) + '"/>';
        if (c % 25 === 0) {
          const tx = cx + Math.sin(a) * 86, ty = cy - Math.cos(a) * 86 + 3;
          html += '<text x="' + tx.toFixed(1) + '" y="' + ty.toFixed(1) + '">' + (c > 0 ? "+" + c : c) + "</text>";
        }
      }
      g.innerHTML = html;
    }

    _buildStrings() {
      const el = this.els.strings;
      if (!el) return;
      el.innerHTML = ["E2", "A2", "D3", "G3", "B3", "E4"]
        .map((s) => '<span class="string-pip" data-string="' + s + '">' + s + "</span>").join("");
    }

    /** @param {object|null} t the protocol's Tuner object, or null when off */
    set(t) {
      if (!t) { this.reading = null; this.detected = false; return; }
      if (t.detected) {
        this.reading = t;
        this.detected = true;
        this.seenWall = performance.now() / 1000;
        this.target = Math.max(-50, Math.min(50, t.cents == null ? 0 : t.cents));
        wake();
      } else {
        this.detected = false;                       // held for 500 ms by _tick
      }
    }

    disable() { this.reading = null; this.detected = false; this.seenWall = -99; wake(); }

    _tick(dt, now) {
      const held = !!this.reading && (performance.now() / 1000 - this.seenWall < 0.5);
      this._held = held;
      this.cents = ease(this.cents, held ? this.target : 0, 0.075, dt);
      const els = this.els;
      els.needle.setAttribute("transform", "rotate(" + (this.cents / 50 * 45).toFixed(2) + " 200 140)");

      let state = "idle";
      if (held) state = Math.abs(this.target) <= 5 ? "in" : "off";
      els.root.dataset.state = state;

      if (held && this.reading) {
        const r = this.reading;
        els.name.textContent = r.note || "-";
        els.oct.textContent = r.octave == null ? "" : String(r.octave);
        els.cents.textContent = (r.cents > 0 ? "+" : "") + (r.cents == null ? "-" : r.cents.toFixed(1));
        els.hz.textContent = r.frequency_hz == null ? "-" : r.frequency_hz.toFixed(2);
        els.hint.textContent = state === "in" ? "in tune" :
          (r.cents < 0 ? "flat - tune up" : "sharp - tune down");
        const key = (r.note || "") + (r.octave == null ? "" : r.octave);
        els.strings.querySelectorAll(".string-pip").forEach((p) => {
          const on = p.dataset.string === key;
          p.classList.toggle("active", on);
          p.classList.toggle("in-tune", on && state === "in");
        });
      } else {
        els.name.textContent = "-";
        els.oct.textContent = "";
        els.cents.textContent = "-";
        els.hz.textContent = "-";
        els.hint.textContent = this._hint || "play a single string";
        els.strings.querySelectorAll(".string-pip").forEach((p) => {
          p.classList.remove("active", "in-tune");
        });
      }
    }

    setHint(text) { this._hint = text; }
  }

  global.EZMeter = Meter;
  global.EZTuner = Tuner;
  global.EZdbNorm = norm;
})(window);
