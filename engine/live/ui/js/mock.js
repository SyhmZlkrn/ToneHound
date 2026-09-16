/*
  Mock engine (`?mock=1`).

  The UI has to be buildable and demoable with no Python running, and -- more
  usefully -- with the awkward states forced on demand: an index that is not
  ready, a device that refuses to open, a separation that failed, a machine
  that has run out of headroom. Those are exactly the states the real engine
  produces rarely and at the worst moment, so they get simulated here on flags:

      ?mock=1                the happy path
      ?mock=1&index=slow     index still fingerprinting for 30 s
      ?mock=1&sepfail=1      match.result with separation_ok:false
      ?mock=1&xrun=1         perf headroom under 1.0, xruns climbing
      ?mock=1&spread=leader  one profile clearly ahead of the index
      ?mock=1&spread=flat    every profile equally far away

  It implements the same surface as EZClient (on / send / upload / peaks /
  httpUrl), so app.js cannot tell the difference and no code path exists only
  in mock mode.

  Audio is genuinely synthesised rather than left silent: a results panel whose
  players do nothing would let a broken player pass review unnoticed.
*/
(function (global) {
  "use strict";

  const SR = 48000;

  const PROFILE_NAMES = [
    "Helga B 5150 BlockLetter - Boosted", "Helga B 5150 BlockLetter - Clean",
    "Mesa Mark IIC+ - Lead", "Mesa Rectifier Orange - Rhythm",
    "Peavey 6505+ - Crunch", "Peavey 6505+ - Scooped",
    "Marshall JCM800 - Hot", "Marshall Plexi - Edge",
    "Diezel VH4 - Ch3", "Diezel Herbert - Ch2",
    "ENGL Savage - Lead", "ENGL Powerball - Rhythm",
    "Fender Twin - Clean", "Fender Deluxe - Breakup",
    "Orange Rockerverb - Dirty", "Friedman BE100 - BE",
    "Soldano SLO100 - OD", "Bogner Uberschall - Red",
    "Revv Generator 120 - Purple", "Victory Kraken - Ch2",
    "EVH 5150III 6L6 - Blue", "EVH 5150III EL34 - Red",
    "Randall Satan - Lead", "Krank Rev1 - Kranked",
    "Laney Ironheart - Lead", "PRS Archon - Ch2",
    "Vox AC30 - Top Boost", "Ampeg VT22 - Loud",
    "Splawn Quick Rod - OD1", "Jackson Ampworks - Britain",
    "Suhr PT100 - Ch2", "Wizard MC2 - Ch2",
    "Synergy SLO - Overdrive", "Cameron CCV - Ch3"
  ];

  const CAVEAT =
    "Ranking is a rough draft. Measured against unseen performances it puts the " +
    "right amp first about 18% of the time, and the profiles it learned from share " +
    "capturers, so it may be sorting by who made the capture as much as by how the " +
    "amp sounds. Listen to the isolated guitar above, then listen to the candidates. " +
    "Trust your ears over the order.";

  let idc = 0;
  const nid = (p) => p + (++idc).toString(16).padStart(4, "0") + "a3f1c07d";

  function hostapis() {
    return [
      { index: 0, name: "MME", supports_exclusive: false, honours_latency_hint: true,
        latency_note: "low latency but unstable here (17-54 xruns measured)" },
      { index: 1, name: "Windows DirectSound", supports_exclusive: false, honours_latency_hint: true,
        latency_note: "9-33 ms with an explicit hint" },
      { index: 2, name: "Windows WASAPI", supports_exclusive: true, honours_latency_hint: false,
        latency_note: "ignores the latency hint on this machine: 47-55 ms whatever you ask for" },
      { index: 3, name: "Windows WDM-KS", supports_exclusive: false, honours_latency_hint: true,
        latency_note: "lowest latency here (10-20 ms) but takes the device exclusively" }
    ];
  }

  function devices() {
    const mk = (index, name, hostapi, ch, low) => ({
      index, name, hostapi, hostapi_name: hostapis()[hostapi].name,
      max_channels: ch, default_samplerate: 48000, supports_48k: true, low_latency_ms: low
    });
    return {
      hostapis: hostapis(),
      inputs: [
        mk(1, "Komplete Audio 2", 2, 2, 46.7), mk(5, "Komplete Audio 2", 3, 2, 4.0),
        mk(9, "Microphone (Realtek)", 2, 2, 48.0)
      ],
      outputs: [
        mk(2, "Komplete Audio 2", 2, 2, 46.7), mk(6, "Komplete Audio 2", 3, 2, 4.0),
        mk(10, "Speakers (Realtek)", 2, 2, 50.0)
      ],
      default_input: 1, default_output: 2,
      recommended: {
        input: 5, output: 6, hostapi: 3, block_size: 512, latency_ms: 5.0, exclusive: false,
        reason: "WDM-KS honours the latency hint; WASAPI reports 47-55 ms regardless"
      }
    };
  }

  function profiles() {
    return PROFILE_NAMES.map((name, i) => {
      const dead = i === 12 || i === 27;             // two files this loader cannot play
      return {
        profile_id: "p" + i.toString(16).padStart(2, "0") + "9f2a1c3b7e00",
        name,
        architecture: dead ? "A2" : (i % 11 === 3 ? "LSTM" : "WaveNet"),
        sample_rate: i === 20 ? 44100 : 48000,
        receptive_field: dead ? null : 4093,
        weight_count: dead ? null : [13802, 13802, 13802, 41288, 48854][i % 5],
        loadable: !dead,
        unsupported_reason: dead
          ? "unsupported .nam model (architecture 'A2'): this loader implements Linear, LSTM, WaveNet"
          : null,
        rate_mismatch: i === 20,
        metadata: { modeled_by: ["Helga B", "Mikko", "Studio One", "T. Ang"][i % 4], gear: "amp" }
      };
    });
  }

  /* --- synthesis ------------------------------------------------------- */

  /** A palm-muted chug bed, tone shaped by `bright` and `drive`, so every
   *  candidate audition sounds like a different amp rather than the same file. */
  function synthAmp(seconds, bright, drive, seed) {
    const n = Math.floor(seconds * SR);
    const out = new Float32Array(n);
    let rnd = seed || 1;
    const rand = () => (rnd = (rnd * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff - 0.5;
    const f0 = 82.41;                                       // low E
    let lp = 0, hp = 0, prev = 0;
    const noteEvery = Math.floor(SR * 0.28);
    for (let i = 0; i < n; i++) {
      const pos = i % noteEvery;
      const env = Math.exp(-pos / (SR * 0.09));
      const t = i / SR;
      let s = 0;
      for (let h = 1; h <= 7; h++) s += Math.sin(2 * Math.PI * f0 * h * t) / h;
      s = s * env * 0.5 + rand() * env * 0.06;
      s = Math.tanh(s * drive) / Math.tanh(drive);          // preamp
      lp += (s - lp) * bright;                              // cabinet-ish LPF
      hp = 0.995 * (hp + lp - prev); prev = lp;             // and a highpass
      out[i] = hp * 0.55;
    }
    return out;
  }

  /** A crude "isolated guitar": band-limit the mix and duck the wideband parts
   *  so it audibly differs from the source, the way a real stem does. */
  function fakeStem(src) {
    const out = new Float32Array(src.length);
    let lp = 0, hp = 0, prev = 0;
    for (let i = 0; i < src.length; i++) {
      lp += (src[i] - lp) * 0.35;
      hp = 0.99 * (hp + lp - prev); prev = lp;
      out[i] = hp * 1.1;
    }
    return out;
  }

  /* --- the mock server -------------------------------------------------- */

  class MockClient {
    constructor(opts) {
      this.opts = opts || {};
      this.mock = true;
      this.listeners = new Map();
      this.state = "closed";
      this.items = new Map();            // id -> {samples, sr, url}
      this.settings = {
        input_gain_db: 0, output_gain_db: 0, bypass: false,
        monitor: "amp", tuner_enabled: true, tuner_reference_hz: 440
      };
      this.stream = {
        running: false, sample_rate: SR, block_size: 512, block_budget_ms: 10.67,
        reported_latency_ms: null, estimated_total_ms: null, xruns: 0,
        last_xrun_at_s: null, uptime_s: 0, device_names: null
      };
      this.loaded = null;
      this.index = { ready: false, profiles: 0, total: 34, progress: 0, eta_s: null, di_name: "Djent" };
      this._t0 = 0;
      this._job = null;
    }

    on(type, fn) {
      if (!this.listeners.has(type)) this.listeners.set(type, new Set());
      this.listeners.get(type).add(fn);
      return () => this.listeners.get(type).delete(fn);
    }

    emit(msg) {
      const d = this.listeners.get(msg.type);
      if (d) d.forEach((fn) => fn(msg));
      const a = this.listeners.get("*");
      if (a) a.forEach((fn) => fn(msg));
    }

    httpUrl(u) { return u; }             // mock URLs are already blob: URLs

    /* Parity with EZClient: app.js prints the origin in its "no engine" toast,
       and a mock that lacked the getter would throw instead of the real client
       being the only thing that differs. */
    get origin() { return location.origin; }

    connect() {
      this.state = "open";
      setTimeout(() => {
        this.emit({ type: "conn", state: "mock" });
        this._startIndex();
      }, 60);
    }

    _startIndex() {
      const slow = this.opts.index === "slow";
      const total = 34, per = slow ? 900 : 160;
      const tick = () => {
        this.index.profiles = Math.min(total, this.index.profiles + 2);
        this.index.progress = this.index.profiles / total;
        this.index.eta_s = ((total - this.index.profiles) / 2) * (per / 1000);
        this.index.ready = this.index.profiles >= total;
        this.emit(Object.assign({ type: "index.status", id: null }, this.index));
        if (!this.index.ready) setTimeout(tick, per);
      };
      setTimeout(tick, per);
    }

    /* --- command dispatch ------------------------------------------- */

    send(type, fields, onProgress) {
      const f = fields || {};
      const id = "m" + ++idc;
      const reply = (o) => { const m = Object.assign({ id }, o); this.emit(m); return m; };
      const fail = (code, message) => {
        const m = { type: "error", id, code, message, detail: {} };
        this.emit(m);
        return Promise.reject(Object.assign(new Error(message), { frame: m }));
      };
      const later = (ms, fn) => new Promise((res) => setTimeout(() => res(fn()), ms));

      switch (type) {
        case "hello":
          return later(40, () => reply({
            type: "hello", protocol: 1, server_version: "0.0.0-mock", sample_rate: SR,
            default_block_size: 512, allowed_block_sizes: [256, 512, 1024, 2048],
            max_snippet_s: 120, min_crop_s: 3, max_crop_s: 30,
            stream: this.stream, settings: this.settings, devices: devices(),
            profile: this.loaded, index: this.index, caveat: CAVEAT
          }));

        case "devices.list":
          return later(f.refresh ? 400 : 30, () => reply(Object.assign({ type: "devices.list" }, devices())));

        case "devices.select": {
          const api = hostapis().find((a) => {
            const d = devices().inputs.concat(devices().outputs).find((x) => x.index === f.input_device);
            return d && a.index === d.hostapi;
          }) || hostapis()[2];
          const ks = api.name.indexOf("WDM-KS") >= 0;
          const rt = ks ? 10.0 : 52.7;
          const warnings = [];
          if (!api.honours_latency_hint) warnings.push(api.name + " ignores the latency hint on this machine.");
          if (ks) warnings.push("WDM-KS takes the device exclusively; other apps will lose it.");
          this.stream.block_size = f.block_size;
          this.stream.block_budget_ms = (f.block_size / SR) * 1000;
          this.stream.reported_latency_ms = rt;
          this.stream.estimated_total_ms = rt + this.stream.block_budget_ms;
          this.stream.device_names = { input: "Komplete Audio 2", output: "Komplete Audio 2" };
          this.stream.running = false;
          return later(280, () => {
            this.emit(Object.assign({ type: "stream.status", id: null }, this.stream));
            return reply({
              type: "devices.selected",
              input_device: f.input_device, output_device: f.output_device,
              input_channel: f.input_channel, output_channels: f.output_channels,
              block_size: f.block_size, sample_rate: SR,
              reported_latency: { input_ms: rt / 2, output_ms: rt / 2, roundtrip_ms: rt },
              estimated_total_ms: this.stream.estimated_total_ms, warnings
            });
          });
        }

        case "stream.start":
          this.stream.running = true;
          this.stream.xruns = 0;
          this._t0 = performance.now();
          this._startTelemetry();
          return later(120, () => reply(Object.assign({ type: "stream.status" }, this.stream)));

        case "stream.stop":
          this.stream.running = false;
          this._stopTelemetry();
          return later(60, () => reply(Object.assign({ type: "stream.status" }, this.stream)));

        case "stream.set": {
          ["input_gain_db", "output_gain_db", "bypass", "monitor", "tuner_enabled", "tuner_reference_hz"]
            .forEach((k) => { if (f[k] !== undefined) this.settings[k] = f[k]; });
          if (f.bypass !== undefined) this.settings.monitor = f.bypass ? "dry" : "amp";
          if (f.monitor !== undefined) this.settings.bypass = f.monitor === "dry";
          return later(10, () => reply(Object.assign({ type: "stream.settings" }, this.settings)));
        }

        case "profiles.list": {
          const all = profiles();
          const q = (f.query || "").toLowerCase();
          const sel = q ? all.filter((p) => p.name.toLowerCase().indexOf(q) >= 0) : all;
          return later(40, () => reply({
            type: "profiles.list", profiles: sel, total: all.length,
            unsupported: all.filter((p) => !p.loadable).length
          }));
        }

        case "profile.load": {
          const p = profiles().find((x) => x.profile_id === f.profile_id);
          if (!p) return fail("not_found", "no such profile_id");
          if (!p.loadable) return fail("profile_unsupported", p.unsupported_reason);
          const stage = (s, ms) => later(ms, () => {
            const m = { type: "profile.loading", id, profile_id: p.profile_id, stage: s };
            if (onProgress) onProgress(m);
            this.emit(m);
          });
          return stage("reading", 40).then(() => stage("building", 30)).then(() => stage("priming", 60))
            .then(() => later(40, () => {
              this.loaded = {
                profile_id: p.profile_id, name: p.name, architecture: p.architecture,
                sample_rate: p.sample_rate, receptive_field: p.receptive_field,
                rate_mismatch: p.rate_mismatch,
                warnings: p.rate_mismatch
                  ? ["Captured at 44100 Hz and played at 48000 Hz without resampling: it will sound slightly bright and fast."]
                  : []
              };
              return reply(Object.assign({ type: "profile.loaded" }, this.loaded));
            }));
        }

        case "di.list": {
          const takes = ["Djent", "Funk", "Thall", "Baritone"].map((name, i) => {
            const id2 = "d" + i;
            if (!this.items.has(id2)) {
              const s = synthAmp(6, 0.5, 1.2, i + 7);
              this.items.set(id2, { samples: s, sr: SR, url: EZWav.wavUrl(s, SR) });
            }
            const it = this.items.get(id2);
            return { di_id: id2, name, duration_s: it.samples.length / SR, audio_url: it.url, peaks_url: "peaks:" + id2 };
          });
          return later(30, () => reply({ type: "di.list", takes, default_di_id: "d0" }));
        }

        case "snippet.crop": {
          const src = this.items.get(f.snippet_id);
          if (!src) return fail("not_found", "no such snippet_id");
          const dur = f.end_s - f.start_s;
          if (dur < 3) return fail("crop_too_short", "crop must be at least 3.0 s");
          if (dur > 30) return fail("snippet_too_long", "crop must be at most 30.0 s");
          const a = Math.floor(f.start_s * src.sr), z = Math.floor(f.end_s * src.sr);
          const cut = src.samples.slice(a, z);
          const cid = nid("k");
          this.items.set(cid, { samples: cut, sr: src.sr, url: EZWav.wavUrl(cut, src.sr) });
          return later(90, () => reply({
            type: "snippet.cropped", snippet_id: f.snippet_id, crop_id: cid,
            start_s: f.start_s, end_s: f.end_s, duration_s: dur,
            audio_url: this.items.get(cid).url, peaks_url: "peaks:" + cid
          }));
        }

        case "snippet.forget":
          this.items.delete(f.snippet_id);
          return later(20, () => reply({ type: "snippet.forgotten", snippet_id: f.snippet_id }));

        case "match.run":
          return this._match(id, f, onProgress, reply, fail);

        case "match.cancel":
          if (this._job) this._job.cancelled = true;
          return later(20, () => reply({ type: "match.cancelled" }));

        case "audition.render": {
          const rid = nid("r");
          const s = synthAmp(f.seconds || 8, 0.3 + Math.random() * 0.4, 1 + Math.random() * 6, idc);
          this.items.set(rid, { samples: s, sr: SR, url: EZWav.wavUrl(s, SR) });
          return later(700, () => reply({
            type: "audition.ready", render_id: rid, profile_id: f.profile_id,
            di_id: f.di_id || "d0", audio_url: this.items.get(rid).url,
            peaks_url: "peaks:" + rid, seconds: f.seconds || 8
          }));
        }

        default:
          return fail("unknown_type", "mock does not implement " + type);
      }
    }

    _match(id, f, onProgress, reply, fail) {
      if (!this.index.ready) return fail("index_not_ready", "profile index is still fingerprinting");
      if (this._job && !this._job.done) return fail("busy", "a match is already running");
      const crop = this.items.get(f.crop_id);
      if (!crop) return fail("not_found", "no such crop_id");

      const job_id = nid("j");
      const job = { job_id, cancelled: false, done: false };
      this._job = job;
      const top = f.top || 5;
      const cropSeconds = crop.samples.length / crop.sr;

      /* Timeline compressed ~4x against the real thing (separation is ~1.0x
         realtime), but the stage ORDER and relative weights are the real ones,
         so the progress bar is honest about which part is slow. */
      const steps = [];
      const sepSteps = 12;
      for (let i = 1; i <= sepSteps; i++) {
        steps.push({ stage: "separating", progress: 0.65 * (i / sepSteps),
          message: "separating guitar from the mix", eta_s: (cropSeconds * (1 - i / sepSteps)) });
      }
      steps.push({ stage: "fingerprinting", progress: 0.72, message: "fingerprinting the stem", eta_s: 7 });
      steps.push({ stage: "ranking", progress: 0.78, message: "ranking 34 profiles", eta_s: 6 });
      const names = profiles().filter((p) => p.loadable);
      for (let i = 0; i < top; i++) {
        steps.push({ stage: "rendering", progress: 0.78 + 0.22 * ((i + 1) / top),
          message: "rendering " + (i + 1) + "/" + top + ": " + names[i].name, eta_s: 1.3 * (top - i - 1) });
      }

      return new Promise((resolve, reject) => {
        let k = 0;
        const tick = () => {
          if (job.cancelled) {
            job.done = true;
            const m = { type: "error", id, code: "job_cancelled", message: "match cancelled", detail: { job_id } };
            this.emit(m);
            reject(Object.assign(new Error(m.message), { frame: m }));
            return;
          }
          if (k < steps.length) {
            const m = Object.assign({ type: "match.progress", id, job_id }, steps[k++]);
            if (onProgress) onProgress(m);
            this.emit(m);
            setTimeout(tick, 260);
            return;
          }
          job.done = true;

          const stemSamples = fakeStem(crop.samples);
          const sid = nid("s");
          this.items.set(sid, { samples: stemSamples, sr: crop.sr, url: EZWav.wavUrl(stemSamples, crop.sr) });
          const di = this.items.get("d0") || { url: "" };

          /* Distances that look like the real thing: a wide index spread and a
             top-5 that is NOT dramatically separated -- the honest picture, and
             the default because it is what the real matcher does.

             `?spread=` forces the other two shapes, because the results panel
             says something different about each and only one of the three turns
             up by chance. `leader` is one profile clearly ahead of the field;
             `flat` is an index where every profile is equally far away, which is
             what a failed separation tends to produce. */
          const shape = this.opts.spread;
          const dists = [];
          for (let i = 0; i < 34; i++) {
            if (shape === "leader") dists.push(i === 0 ? 0.18 : 0.55 + Math.random() * 0.18);
            else if (shape === "flat") dists.push(0.48 + Math.random() * 0.01);
            else dists.push(0.18 + Math.random() * 0.55);
          }
          dists.sort((a, b) => a - b);
          const candidates = [];
          for (let i = 0; i < top; i++) {
            const p = names[i];
            const rid = nid("r");
            const s = synthAmp(8, 0.22 + i * 0.06, 1.5 + i * 1.7, i + 3);
            this.items.set(rid, { samples: s, sr: SR, url: EZWav.wavUrl(s, SR) });
            candidates.push({
              rank: i + 1, profile_id: p.profile_id, name: p.name,
              distance: dists[i], percentile: (i + 0.5) / 34,
              audition_url: f.render_auditions === false ? null : this.items.get(rid).url,
              metadata: p.metadata
            });
          }
          const sorted = dists.slice();
          resolve(reply({
            type: "match.result", job_id, crop_id: f.crop_id,
            stem_audio_url: this.items.get(sid).url, stem_peaks_url: "peaks:" + sid,
            separation_ok: this.opts.sepfail !== "1",
            di_id: f.di_id || "d0", di_audio_url: di.url,
            index_size: 34, query_seconds: cropSeconds,
            distance_range: { min: sorted[0], max: sorted[sorted.length - 1], median: sorted[17] },
            candidates, caveat: CAVEAT
          }));
        };
        setTimeout(tick, 200);
      });
    }

    /* --- telemetry ---------------------------------------------------- */

    _startTelemetry() {
      this._stopTelemetry();
      const strings = [
        { note: "E", octave: 2, hz: 82.41 }, { note: "A", octave: 2, hz: 110.0 },
        { note: "D", octave: 3, hz: 146.83 }, { note: "G", octave: 3, hz: 196.0 },
        { note: "B", octave: 3, hz: 246.94 }, { note: "E", octave: 4, hz: 329.63 }
      ];
      let str = strings[0], cents = 14, sinceNote = 0, playing = true, level = -18;

      this._tel = setInterval(() => {
        const t = (performance.now() - this._t0) / 1000;
        sinceNote += 0.05;
        if (sinceNote > 3.2) {                      // a new note every ~3 s
          sinceNote = 0;
          str = strings[Math.floor(Math.random() * strings.length)];
          cents = (Math.random() * 2 - 1) * 32;
          playing = true;
        }
        if (sinceNote > 2.4) playing = false;       // and it dies away
        cents += (Math.random() - 0.5) * 1.2;
        cents = Math.max(-50, Math.min(50, cents - Math.sign(cents) * 0.35));

        const env = Math.exp(-sinceNote / 1.2);
        level = -46 + env * 34 + Math.random() * 2;
        const peak = level + 4 + Math.random();

        this.emit({
          type: "telemetry", id: null, t_s: t,
          input_peak_db: peak, input_rms_db: level,
          output_peak_db: peak + (this.settings.bypass ? 0 : 5) + this.settings.output_gain_db,
          output_rms_db: level + (this.settings.bypass ? 0 : 7) + this.settings.output_gain_db,
          input_clipped: peak > -0.5, output_clipped: peak + 5 > 0.0,
          tuner: this.settings.tuner_enabled ? {
            detected: playing,
            frequency_hz: playing ? str.hz * Math.pow(2, cents / 1200) : null,
            note: playing ? str.note : null,
            octave: playing ? str.octave : null,
            cents: playing ? cents : null,
            confidence: playing ? 0.85 + Math.random() * 0.1 : 0.1,
            reference_hz: this.settings.tuner_reference_hz
          } : null
        });
      }, 50);

      this._perf = setInterval(() => {
        const bad = this.opts.xrun === "1";
        const budget = this.stream.block_budget_ms;
        const p95 = bad ? budget * (0.9 + Math.random() * 0.3) : 3.3 + Math.random() * 0.9;
        if (bad && Math.random() < 0.5) {
          this.stream.xruns += 1;
          this.stream.last_xrun_at_s = (performance.now() - this._t0) / 1000;
          this.emit(Object.assign({ type: "stream.status", id: null }, this.stream));
        }
        this.stream.uptime_s = (performance.now() - this._t0) / 1000;
        this.emit({
          type: "perf", id: null,
          block_ms_mean: p95 * 0.85, block_ms_p95: p95, block_ms_max: p95 * 1.4,
          budget_ms: budget, headroom_x: budget / p95, xruns: this.stream.xruns
        });
      }, 1000);
    }

    _stopTelemetry() {
      if (this._tel) clearInterval(this._tel);
      if (this._perf) clearInterval(this._perf);
      this._tel = this._perf = null;
    }

    /* --- HTTP stand-ins ------------------------------------------------ */

    async upload(file) {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const buf = await ctx.decodeAudioData(await file.arrayBuffer());
      const n = buf.length;
      const mono = new Float32Array(n);
      for (let c = 0; c < buf.numberOfChannels; c++) {
        const d = buf.getChannelData(c);
        for (let i = 0; i < n; i++) mono[i] += d[i] / buf.numberOfChannels;
      }
      const sid = nid("s");
      this.items.set(sid, { samples: mono, sr: buf.sampleRate, url: URL.createObjectURL(file) });
      ctx.close();
      setTimeout(() => this.emit({
        type: "snippet.ready", id: null, snippet_id: sid, filename: file.name,
        duration_s: n / buf.sampleRate, source_sample_rate: buf.sampleRate,
        source_channels: buf.numberOfChannels,
        audio_url: this.items.get(sid).url, peaks_url: "peaks:" + sid
      }), 30);
      return { snippet_id: sid };
    }

    async peaks(url, bins) {
      const id = String(url).replace(/^peaks:/, "");
      const it = this.items.get(id);
      if (!it) throw new Error("no such item " + id);
      const p = EZWav.peaksOf(it.samples, bins);
      return Object.assign({ id, duration_s: it.samples.length / it.sr }, p);
    }
  }

  global.EZMockClient = MockClient;
})(window);
