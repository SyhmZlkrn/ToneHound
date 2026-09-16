/*
  ToneHound UI wiring.

  One module, because the page is one screen and splitting it across five files
  would cost more in indirection than it saves. It is organised in the order of
  docs/ui_protocol.md: session, devices, stream, telemetry, profiles, snippet,
  match, audition.

  Two rules shape almost every decision here:

  * The server owns the truth. Controls do not change their own appearance on
    click; they send a command and re-render from the broadcast that comes back
    (`stream.settings`, `stream.status`, `profile.loaded`). That is what keeps a
    second browser tab in sync, and it means the UI can never show ENGAGE lit
    while the stream is actually stopped.

  * The honest numbers stay visible. Latency is labelled as driver-reported,
    `perf.headroom_x` gets a permanent strip rather than a hidden log, every
    candidate carries its raw distance against the index range, and the server's
    caveat is printed verbatim above the ranking. Those are §5 requirements, not
    styling choices -- see engine/tests/test_ui_assets.py, which fails if they
    are quietly removed.
*/
(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);

  /* Device and host-API names come from the OS driver, and `<option>` labels
     are the one place this file builds markup by concatenation (a <select> has
     nowhere to hang a textContent-built child cheaply). A card called
     "Line 1/2 <Fast Track>" would silently eat the rest of the list. */
  const esc = (s) => String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  const qs = new URLSearchParams(location.search);
  const MOCK = qs.has("mock") && qs.get("mock") !== "0";

  const client = MOCK
    ? new EZMockClient({ index: qs.get("index"), sepfail: qs.get("sepfail"), xrun: qs.get("xrun"), spread: qs.get("spread") })
    : new EZClient(qs.get("ws") || ((location.protocol === "https:" ? "wss://" : "ws://") +
        (qs.get("server") || location.host || "127.0.0.1:8730") + "/ws"));

  const S = {
    limits: { min_crop_s: 3, max_crop_s: 30, max_snippet_s: 120, allowed_block_sizes: [256, 512, 1024, 2048] },
    devices: null,
    settings: null,
    stream: null,
    profiles: [],
    loaded: null,
    index: null,
    caveat: "",
    snippet: null,     // {snippet_id, duration_s, audio_url, filename}
    sel: null,         // {start, end}
    crop: null,        // {crop_id, start_s, end_s}
    di: [],
    match: null,
    job: null,         // {id, job_id} of a running match
    playing: false
  };

  /* ==================================================================== */
  /* toasts                                                               */
  /* ==================================================================== */

  function toast(kind, title, body, ms) {
    const el = document.createElement("div");
    el.className = "toast " + kind;
    el.innerHTML = "<b></b><span></span>";
    el.querySelector("b").textContent = title;
    el.querySelector("span").textContent = body;
    $("#toasts").appendChild(el);
    setTimeout(() => el.remove(), ms || (kind === "err" ? 9000 : 5000));
    return el;
  }

  function fail(err, what) {
    const f = err && err.frame;
    console.warn("[ez]", what, err);
    toast("err", f ? f.code : "error", (f ? f.message : String(err && err.message || err)));
  }

  /* ==================================================================== */
  /* 4.1 session                                                          */
  /* ==================================================================== */

  let closes = 0, toldAboutMock = false;
  client.on("conn", (m) => {
    const dot = $("#conn-dot"), txt = $("#conn-text");
    dot.dataset.state = m.state;
    txt.textContent = m.state === "mock" ? "mock engine" : m.state;
    if (m.state === "open" || m.state === "mock") { closes = 0; hello(); }
    if (m.state === "closed") {
      meterIn.silence(); meterOut.silence(); tuner.disable();
      setEngage(false);
      // The page reconnects for ever, quietly. Say once what is actually wrong,
      // because an amber dot that never goes green looks like a UI bug.
      if (++closes === 2 && !toldAboutMock) {
        toldAboutMock = true;
        toast("err", "no engine",
          "Nothing is answering at " + client.origin.replace(/^https?:\/\//, "") +
          ". Start the engine, or add ?mock=1 to this URL to drive the UI with fake data.", 20000);
      }
    }
  });

  async function hello() {
    try {
      const h = await client.send("hello", { protocol: 1 });
      S.limits = {
        min_crop_s: h.min_crop_s, max_crop_s: h.max_crop_s, max_snippet_s: h.max_snippet_s,
        allowed_block_sizes: h.allowed_block_sizes, default_block_size: h.default_block_size
      };
      S.caveat = h.caveat || "";
      $("#drop-limits").textContent =
        "up to " + Math.round(h.max_snippet_s) + " s of audio; crop between " +
        h.min_crop_s + " and " + h.max_crop_s + " s (separation runs at about real time)";
      $("#stat-rate").textContent = String(h.sample_rate).replace(/\B(?=(\d{3})+$)/g, " ");
      fillBlockSizes(h.allowed_block_sizes, h.default_block_size);
      applyDevices(h.devices);
      applySettings(h.settings);
      applyStream(h.stream);
      applyIndex(h.index);
      if (h.profile) applyLoaded(h.profile);
      if (S.caveat) $("#caveat").textContent = S.caveat;
      loadProfiles("");
      loadDiTakes();
    } catch (err) {
      fail(err, "hello");
    }
  }

  /* ==================================================================== */
  /* 4.2 devices                                                          */
  /* ==================================================================== */

  function fillBlockSizes(sizes, current) {
    const sel = $("#sel-block");
    sel.innerHTML = sizes.map((n) =>
      '<option value="' + n + '">' + n + " (" + (n / 48).toFixed(1) + " ms)</option>").join("");
    sel.value = String(current || 512);
    $("#stat-block-ms").textContent = (Number(sel.value) / 48).toFixed(1);
  }

  function applyDevices(d) {
    if (!d) return;
    S.devices = d;
    const apis = d.hostapis || [];
    const used = new Set(d.inputs.concat(d.outputs).map((x) => x.hostapi));
    const api = $("#sel-hostapi");
    const keep = api.value;
    api.innerHTML = apis.filter((a) => used.has(a.index)).map((a) =>
      '<option value="' + a.index + '">' + esc(a.name) + (a.honours_latency_hint ? "" : "  (ignores latency hint)") +
      "</option>").join("");
    const rec = d.recommended;
    api.value = keep && api.querySelector('option[value="' + keep + '"]') ? keep
      : String(rec ? rec.hostapi : (apis[0] ? apis[0].index : ""));
    api.onchange = () => { fillDeviceSelects(); noteHostApi(); };
    fillDeviceSelects();
    noteHostApi();
    if (rec) {
      if ($("#sel-input").querySelector('option[value="' + rec.input + '"]')) $("#sel-input").value = String(rec.input);
      if ($("#sel-output").querySelector('option[value="' + rec.output + '"]')) $("#sel-output").value = String(rec.output);
      $("#sel-block").value = String(rec.block_size);
      $("#stat-block-ms").textContent = (rec.block_size / 48).toFixed(1);
    }
    fillChannels();
  }

  function fillDeviceSelects() {
    const apiIdx = Number($("#sel-hostapi").value);
    const opt = (dev) => '<option value="' + dev.index + '">' + esc(dev.name) +
      (dev.supports_48k ? "" : "  (no 48 kHz)") + "</option>";
    const ins = S.devices.inputs.filter((x) => x.hostapi === apiIdx);
    const outs = S.devices.outputs.filter((x) => x.hostapi === apiIdx);
    $("#sel-input").innerHTML = ins.map(opt).join("") || '<option value="">none on this host API</option>';
    $("#sel-output").innerHTML = outs.map(opt).join("") || '<option value="">none on this host API</option>';
    fillChannels();
  }

  function fillChannels() {
    const dev = (S.devices.inputs || []).find((x) => String(x.index) === $("#sel-input").value);
    const n = dev ? Math.max(1, dev.max_channels) : 1;
    const cur = $("#sel-input-ch").value;
    $("#sel-input-ch").innerHTML = Array.from({ length: n }, (_, i) =>
      '<option value="' + i + '">' + (i + 1) + "</option>").join("");
    if (cur && Number(cur) < n) $("#sel-input-ch").value = cur;
  }
  $("#sel-input").addEventListener("change", fillChannels);

  function noteHostApi() {
    const a = (S.devices.hostapis || []).find((x) => x.index === Number($("#sel-hostapi").value));
    $("#sel-hostapi").title = a && a.latency_note ? a.latency_note : "";
  }

  $("#btn-rescan").onclick = async () => {
    try { applyDevices(await client.send("devices.list", { refresh: true })); toast("ok", "devices", "rescanned"); }
    catch (err) { fail(err, "devices.list"); }
  };

  $("#btn-apply-io").onclick = async () => {
    const dev = (S.devices.outputs || []).find((x) => String(x.index) === $("#sel-output").value);
    const chans = dev && dev.max_channels >= 2 ? [0, 1] : [0];
    try {
      const r = await client.send("devices.select", {
        input_device: Number($("#sel-input").value),
        input_channel: Number($("#sel-input-ch").value || 0),
        output_device: Number($("#sel-output").value),
        output_channels: chans,
        block_size: Number($("#sel-block").value),
        latency_ms: null,
        exclusive: false
      });
      showLatency(r.reported_latency ? r.reported_latency.roundtrip_ms : null, r.estimated_total_ms);
      $("#stat-block-ms").textContent = (r.block_size / 48).toFixed(1);
      (r.warnings || []).forEach((w) => toast("", "device", w, 8000));
      toast("ok", "devices", "selected - press ENGAGE to start the stream");
    } catch (err) {
      fail(err, "devices.select");
    }
  };

  function showLatency(rtMs, totalMs) {
    $("#stat-latency").textContent = totalMs == null && rtMs == null ? "-"
      : (totalMs == null ? rtMs : totalMs).toFixed(1);
  }

  /* ==================================================================== */
  /* 4.3 stream control                                                   */
  /* ==================================================================== */

  const meterIn = new EZMeter($("#meter-in"));
  const meterOut = new EZMeter($("#meter-out"));
  const tuner = new EZTuner({
    root: $("#tuner"), needle: $("#needle"), ticks: $("#dial-ticks"),
    name: $("#note-name"), oct: $("#note-oct"), cents: $("#cents-val"),
    hz: $("#tuner-hz"), hint: $("#tuner-hint"), strings: $("#tuner-strings")
  });

  function applyStream(st) {
    if (!st) return;
    S.stream = st;
    setEngage(st.running);
    $("#stat-block-ms").textContent = st.block_budget_ms ? st.block_budget_ms.toFixed(1) : "-";
    showLatency(st.reported_latency_ms, st.estimated_total_ms);
    $("#perf-xruns").textContent = st.xruns || 0;
    $("#perf-uptime").textContent = fmtClock(st.uptime_s || 0);
    if (!st.running) {
      meterIn.silence(); meterOut.silence(); tuner.disable();
      // Telemetry stops with the stream, so a CLIP lamp lit by the last frame
      // before a stop would stay lit for ever and read as a live warning.
      clearLeds();
    }
  }

  function setEngage(running) {
    const b = $("#btn-engage");
    const bypassed = S.settings && S.settings.bypass;
    b.dataset.state = running ? (bypassed ? "bypass" : "on") : "off";
    b.setAttribute("aria-pressed", running ? "true" : "false");
    b.querySelector(".engage-sub").textContent = running ? (bypassed ? "bypassed" : "live") : "stopped";
  }

  $("#btn-engage").onclick = async () => {
    const want = !(S.stream && S.stream.running);
    try { applyStream(await client.send(want ? "stream.start" : "stream.stop", {})); }
    catch (err) { fail(err, "stream"); }
  };

  function applySettings(s) {
    if (!s) return;
    S.settings = s;
    $("#in-gain").value = s.input_gain_db;
    $("#out-gain").value = s.output_gain_db;
    $("#lbl-in-gain").textContent = s.input_gain_db.toFixed(1) + " dB";
    $("#lbl-out-gain").textContent = s.output_gain_db.toFixed(1) + " dB";
    $("#btn-bypass").setAttribute("aria-pressed", s.bypass ? "true" : "false");
    document.querySelectorAll(".seg[data-monitor]").forEach((el) =>
      el.setAttribute("aria-pressed", el.dataset.monitor === s.monitor ? "true" : "false"));
    $("#tuner-on").checked = !!s.tuner_enabled;
    $("#tuner-ref").value = s.tuner_reference_hz;
    if (!s.tuner_enabled) { tuner.disable(); tuner.setHint("tuner off"); } else tuner.setHint("play a single string");
    setEngage(S.stream ? S.stream.running : false);
  }

  /* Gain sliders stream continuously; the engine ramps over one block, so the
     only thing to protect is the socket. 30 Hz, as the protocol suggests. */
  let gainTimer = null, pendingGain = {};
  function setStream(fields, immediate) {
    Object.assign(pendingGain, fields);
    if (immediate) { flushStream(); return; }
    if (gainTimer) return;
    gainTimer = setTimeout(flushStream, 33);
  }
  function flushStream() {
    clearTimeout(gainTimer); gainTimer = null;
    const f = pendingGain; pendingGain = {};
    if (!Object.keys(f).length) return;
    client.send("stream.set", f).catch((err) => fail(err, "stream.set"));
  }

  $("#in-gain").addEventListener("input", (e) => {
    $("#lbl-in-gain").textContent = Number(e.target.value).toFixed(1) + " dB";
    setStream({ input_gain_db: Number(e.target.value) });
  });
  $("#out-gain").addEventListener("input", (e) => {
    $("#lbl-out-gain").textContent = Number(e.target.value).toFixed(1) + " dB";
    setStream({ output_gain_db: Number(e.target.value) });
  });
  $("#btn-bypass").onclick = () => toggleBypass();
  function toggleBypass() {
    const now = S.settings ? !S.settings.bypass : true;
    setStream({ bypass: now }, true);
  }
  document.querySelectorAll(".seg[data-monitor]").forEach((el) => {
    el.onclick = () => setStream({ monitor: el.dataset.monitor }, true);
  });
  $("#tuner-on").addEventListener("change", (e) => setStream({ tuner_enabled: e.target.checked }, true));
  $("#tuner-ref").addEventListener("change", (e) => {
    const v = Math.max(410, Math.min(470, Number(e.target.value) || 440));
    e.target.value = v;
    setStream({ tuner_reference_hz: v }, true);
  });

  client.on("stream.status", (m) => applyStream(m));
  client.on("stream.settings", (m) => applySettings(m));

  /* ==================================================================== */
  /* 4.4 telemetry                                                        */
  /* ==================================================================== */

  client.on("telemetry", (m) => {
    meterIn.set(m.input_rms_db, m.input_peak_db, m.input_clipped);
    meterOut.set(m.output_rms_db, m.output_peak_db, m.output_clipped);
    led("#clip-in", m.input_clipped);
    led("#clip-out", m.output_clipped);
    tuner.set(m.tuner);
  });

  const ledUntil = {};
  function led(sel, on) {
    const el = $(sel);
    if (on) { ledUntil[sel] = Date.now() + 1200; el.classList.add("on"); }
    else if (ledUntil[sel] && Date.now() > ledUntil[sel]) el.classList.remove("on");
  }

  function clearLeds() {
    ["#clip-in", "#clip-out"].forEach((sel) => {
      delete ledUntil[sel];
      $(sel).classList.remove("on");
    });
  }

  client.on("perf", (m) => {
    const strip = $("#perf-strip");
    $("#perf-headroom").textContent = m.headroom_x.toFixed(2) + "x";
    $("#perf-p95").textContent = m.block_ms_p95.toFixed(2) + " ms";
    $("#perf-budget").textContent = m.budget_ms.toFixed(2) + " ms";
    $("#perf-xruns").textContent = m.xruns;
    // 1.0x is "the callback exactly fills its budget": at that point audio drops.
    $("#perf-bar-fill").style.width = Math.max(2, Math.min(100, (m.headroom_x / 4) * 100)) + "%";
    strip.classList.toggle("warn", m.headroom_x < 2.0 && m.headroom_x >= 1.3);
    strip.classList.toggle("bad", m.headroom_x < 1.3 || m.xruns > 0);
    const warn = $("#perf-warn");
    if (m.xruns > 0) {
      warn.hidden = false;
      warn.textContent = m.xruns + " dropout" + (m.xruns === 1 ? "" : "s") +
        " - raise the block size or pick a lighter profile";
    } else if (m.headroom_x < 2.0) {
      warn.hidden = false;
      warn.textContent = "close to the budget - a bigger block size buys headroom";
    } else {
      warn.hidden = true;
    }
  });

  const fmtClock = (s) => Math.floor(s / 60) + ":" + String(Math.floor(s % 60)).padStart(2, "0");

  /* ==================================================================== */
  /* 4.5 profiles                                                         */
  /* ==================================================================== */

  async function loadProfiles(query) {
    try {
      const r = await client.send("profiles.list", { query: query || "" });
      S.profiles = r.profiles;
      $("#profile-count").textContent = r.profiles.length + " of " + r.total +
        (r.unsupported ? " - " + r.unsupported + " unsupported" : "");
      renderProfiles();
    } catch (err) {
      fail(err, "profiles.list");
    }
  }

  function renderProfiles() {
    const list = $("#profile-list");
    list.innerHTML = "";
    S.profiles.forEach((p) => {
      const row = document.createElement("div");
      row.className = "prow" + (p.loadable ? "" : " dead");
      row.setAttribute("aria-current", S.loaded && S.loaded.profile_id === p.profile_id ? "true" : "false");
      const meta = p.loadable
        ? p.architecture + " - " + (p.weight_count || "?") + " weights - " + p.sample_rate + " Hz" +
          (p.rate_mismatch ? " - not 48 kHz" : "")
        : p.unsupported_reason;
      row.innerHTML = '<div><div class="prow-name"></div><div class="' +
        (p.loadable ? "prow-meta" : "prow-reason") + '"></div></div>' +
        '<div class="prow-meta">' + (p.loadable ? "load" : "skipped") + "</div>";
      row.querySelector(".prow-name").textContent = p.name;
      row.querySelector(p.loadable ? ".prow-meta" : ".prow-reason").textContent = meta;
      if (p.loadable) row.onclick = () => loadProfile(p.profile_id);
      list.appendChild(row);
    });
  }

  let searchTimer = null;
  $("#profile-search").addEventListener("input", (e) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => loadProfiles(e.target.value), 180);
  });

  async function loadProfile(profile_id) {
    const box = $("#loaded-profile");
    const stage = document.createElement("div");
    stage.className = "loading-stage";
    stage.textContent = "loading...";
    box.appendChild(stage);
    try {
      const r = await client.send("profile.load", { profile_id }, (p) => {
        stage.textContent = p.stage + "...";
      });
      applyLoaded(r);
    } catch (err) {
      stage.remove();
      fail(err, "profile.load");
    }
  }

  function applyLoaded(p) {
    S.loaded = p;
    const box = $("#loaded-profile");
    box.innerHTML = '<div class="loaded-name"></div><div class="loaded-meta"></div>';
    box.querySelector(".loaded-name").textContent = p.name;
    box.querySelector(".loaded-meta").textContent =
      p.architecture + " - receptive field " + p.receptive_field + " - captured at " + p.sample_rate + " Hz";
    (p.warnings || []).forEach((w) => {
      const d = document.createElement("div");
      d.className = "loaded-warn";
      d.textContent = w;
      box.appendChild(d);
    });
    renderProfiles();
    markLiveCandidate();
    setEngage(S.stream ? S.stream.running : false);
  }

  client.on("profile.loaded", (m) => { if (m.id == null) applyLoaded(m); });

  function applyIndex(ix) {
    if (!ix) return;
    S.index = ix;
    const box = $("#index-status");
    box.hidden = !!ix.ready;
    $("#index-text").textContent = ix.ready ? "index ready"
      : "Fingerprinting profile index " + ix.profiles + "/" + ix.total +
        (ix.eta_s != null ? " - about " + Math.ceil(ix.eta_s) + " s left" : "") +
        " (matching is unavailable until it finishes)";
    $("#index-bar").style.width = (ix.progress * 100).toFixed(1) + "%";
    updateMatchButton();
  }
  client.on("index.status", (m) => applyIndex(m));

  /* ==================================================================== */
  /* 4.6 snippet import and crop                                          */
  /* ==================================================================== */

  const wave = new EZWaveform($("#wave"), $("#wave-overlay"), $("#sel"), $("#playhead"), {
    onChange: onSelection,
    onSeek: (t) => { if (S.snippet) { audio.currentTime = t; } }
  });
  const audio = $("#snippet-audio");

  const drop = $("#drop");
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault(); drop.classList.add("over");
  }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]);
  });
  $("#btn-browse").onclick = () => $("#file-input").click();
  $("#file-input").addEventListener("change", (e) => { if (e.target.files[0]) upload(e.target.files[0]); });

  async function upload(file) {
    $("#snippet-name").textContent = "decoding " + file.name + "...";
    try {
      await client.upload(file);            // snippet.ready arrives as a broadcast
    } catch (err) {
      $("#snippet-name").textContent = "nothing loaded";
      fail(err, "upload");
    }
  }

  client.on("snippet.ready", async (m) => {
    S.snippet = m;
    S.crop = null;
    $("#snippet-name").textContent = m.filename + " - " + m.duration_s.toFixed(1) + " s - " +
      m.source_sample_rate + " Hz " + (m.source_channels === 1 ? "mono" : "stereo");
    $("#btn-forget").hidden = false;
    $("#drop").hidden = true;
    $("#wave-wrap").hidden = false;
    audio.src = client.httpUrl(m.audio_url);
    try {
      const bins = Math.max(400, Math.min(4000, Math.round($("#wave").clientWidth * 2)));
      const p = await client.peaks(m.peaks_url, bins);
      wave.setPeaks(p, m.duration_s);
    } catch (err) {
      fail(err, "peaks");
    }
    // A sensible starting crop: a chorus-length window a quarter of the way in.
    const len = Math.min(15, S.limits.max_crop_s, Math.max(S.limits.min_crop_s, m.duration_s));
    const start = Math.max(0, Math.min(m.duration_s - len, m.duration_s * 0.25));
    wave.setSelection(start, start + len);
  });

  $("#btn-forget").onclick = async () => {
    if (!S.snippet) return;
    const id = S.snippet.snippet_id;
    stopPlayback();
    S.snippet = null; S.crop = null; S.sel = null;
    wave.clear();
    $("#wave-wrap").hidden = true;
    $("#drop").hidden = false;
    $("#btn-forget").hidden = true;
    $("#snippet-name").textContent = "nothing loaded";
    updateMatchButton();
    try { await client.send("snippet.forget", { snippet_id: id }); } catch (err) { fail(err, "snippet.forget"); }
  };

  let cropTimer = null;
  function onSelection(start, end, final) {
    if (!S.snippet) return;
    const dur = end - start;
    S.sel = dur > 0 ? { start, end } : null;
    $("#sel-readout").textContent = S.sel
      ? fmtTime(start) + " - " + fmtTime(end) + "   (" + dur.toFixed(2) + " s)"
      : "no selection";
    const warn = $("#sel-warn");
    let problem = null;
    if (S.sel && dur < S.limits.min_crop_s) problem = "shorter than the " + S.limits.min_crop_s + " s minimum";
    else if (S.sel && dur > S.limits.max_crop_s) problem = "longer than the " + S.limits.max_crop_s + " s maximum";
    warn.hidden = !problem;
    warn.textContent = problem || "";
    if (problem) { S.crop = null; updateMatchButton(); return; }
    if (!final || !S.sel) { if (!S.sel) { S.crop = null; updateMatchButton(); } return; }

    clearTimeout(cropTimer);
    cropTimer = setTimeout(async () => {
      try {
        const r = await client.send("snippet.crop", {
          snippet_id: S.snippet.snippet_id, start_s: start, end_s: end
        });
        S.crop = r;
        updateMatchButton();
      } catch (err) {
        S.crop = null;
        updateMatchButton();
        fail(err, "snippet.crop");
      }
    }, 120);
  }

  const fmtTime = (t) => Math.floor(t / 60) + ":" + (t % 60).toFixed(2).padStart(5, "0");

  /* --- selection playback --------------------------------------------- */

  $("#btn-play-sel").onclick = () => togglePlayback();

  function togglePlayback() {
    if (S.playing) { stopPlayback(); return; }
    if (!S.snippet || !S.sel) return;
    audio.currentTime = S.sel.start;
    audio.play().then(() => {
      S.playing = true;
      $("#btn-play-sel").querySelector(".play-glyph").textContent = "■";
      requestAnimationFrame(followPlayhead);
    }).catch((err) => fail(err, "playback"));
  }

  function stopPlayback() {
    audio.pause();
    S.playing = false;
    $("#btn-play-sel").querySelector(".play-glyph").textContent = "▶";
    wave.setPlayhead(null);
  }

  function followPlayhead() {
    if (!S.playing) return;
    wave.setPlayhead(audio.currentTime);
    if (S.sel && audio.currentTime >= S.sel.end) { stopPlayback(); return; }
    requestAnimationFrame(followPlayhead);
  }
  audio.addEventListener("ended", stopPlayback);

  /* --- DI takes -------------------------------------------------------- */

  async function loadDiTakes() {
    try {
      const r = await client.send("di.list", {});
      S.di = r.takes || [];
      $("#sel-di").innerHTML = S.di.map((t) =>
        '<option value="' + esc(t.di_id) + '">' + esc(t.name) + " (" + t.duration_s.toFixed(0) + " s)</option>").join("");
      if (r.default_di_id) $("#sel-di").value = r.default_di_id;
    } catch (err) {
      fail(err, "di.list");
    }
  }

  $("#sel-top").innerHTML = [3, 5, 8, 10].map((n) =>
    '<option value="' + n + '"' + (n === 5 ? " selected" : "") + ">" + n + "</option>").join("");

  /* ==================================================================== */
  /* 4.7 match                                                            */
  /* ==================================================================== */

  function updateMatchButton() {
    const ready = S.index ? S.index.ready : false;
    const b = $("#btn-match");
    b.disabled = !(S.crop && ready) || !!S.job;
    b.title = !S.crop ? "select at least " + S.limits.min_crop_s + " s of the song first"
      : (!ready ? "the profile index is still fingerprinting" : "");
  }

  $("#btn-match").onclick = async () => {
    if (!S.crop) return;
    const job = $("#match-job");
    job.hidden = false;
    $("#job-bar").style.width = "0%";
    $("#job-stage").textContent = "starting";
    $("#job-msg").textContent = "";
    $("#job-eta").textContent = "";
    S.job = {};
    updateMatchButton();
    try {
      const r = await client.send("match.run", {
        crop_id: S.crop.crop_id,
        di_id: $("#sel-di").value || null,
        top: Number($("#sel-top").value),
        render_auditions: true
      }, (p) => {
        S.job.job_id = p.job_id;
        $("#job-stage").textContent = p.stage;
        $("#job-msg").textContent = p.message || "";
        $("#job-eta").textContent = p.eta_s != null ? "~" + Math.ceil(p.eta_s) + " s left" : "";
        $("#job-bar").style.width = (p.progress * 100).toFixed(1) + "%";
      });
      renderResult(r);
    } catch (err) {
      if (!(err.frame && err.frame.code === "job_cancelled")) fail(err, "match.run");
    } finally {
      S.job = null;
      job.hidden = true;
      updateMatchButton();
    }
  };

  $("#btn-cancel-match").onclick = () => {
    if (S.job && S.job.job_id) client.send("match.cancel", { job_id: S.job.job_id }).catch(() => {});
  };

  async function renderResult(r) {
    S.match = r;
    $("#results-empty").hidden = true;
    $("#results").hidden = false;

    /* §5.1 -- the stem first, and say it loudly when separation failed. */
    const stem = $("#stem");
    stem.classList.toggle("failed", r.separation_ok === false);
    $("#stem-flag").hidden = r.separation_ok !== false;
    if (r.separation_ok === false) {
      $("#stem-flag").textContent = "SEPARATION FAILED - the raw mix was matched instead";
      toast("err", "separation failed",
        "No guitar stem came back, so the whole mix was fingerprinted. Treat the ranking below as noise.");
    }
    $("#stem-audio").src = client.httpUrl(r.stem_audio_url);
    $("#di-audio").src = client.httpUrl(r.di_audio_url);

    $("#results-meta").textContent =
      r.index_size + " profiles ranked over " + r.query_seconds.toFixed(1) + " s of audio";

    /* §5.3 -- verbatim, in the flow. */
    $("#caveat").textContent = r.caveat || S.caveat;

    /* §5.2 -- distance in context: where the top-N sit in the whole index. */
    const dr = r.distance_range;
    $("#dist-scale-range").textContent =
      "min " + dr.min.toFixed(3) + "  median " + dr.median.toFixed(3) + "  max " + dr.max.toFixed(3);
    const track = $("#dist-track");
    track.querySelectorAll(".dist-pin").forEach((n) => n.remove());
    const pos = (d) => Math.max(0, Math.min(1, (d - dr.min) / Math.max(1e-9, dr.max - dr.min)));
    $("#dist-median").style.left = (pos(dr.median) * 100) + "%";
    r.candidates.forEach((c) => {
      const pin = document.createElement("div");
      pin.className = "dist-pin" + (c.rank === 1 ? " rank1" : "");
      pin.style.left = (pos(c.distance) * 100) + "%";
      pin.title = "#" + c.rank + " " + c.name + " - distance " + c.distance.toFixed(4);
      pin.innerHTML = "<span>" + c.rank + "</span>";
      track.appendChild(pin);
    });
    /* Two different questions live here, and only the second one is about
       rank 1. How tightly the whole top-N is bunched says how much the ORDER
       is worth; whether rank 1 is separated from rank 2 says whether the
       WINNER is worth anything. They come apart constantly -- a top-5 that
       spans a fifth of the index can still have rank 1 and rank 2 sitting on
       top of each other, and reading the span alone would then announce that
       rank 1 is "measurably closer" when it is a coin toss. Given a matcher
       that is right about 18% of the time, that is exactly the false
       confidence §5 exists to prevent, so the gap is stated explicitly. */
    const cands = r.candidates;
    const full = Math.max(1e-9, dr.max - dr.min);
    const span = cands.length > 1 ? cands[cands.length - 1].distance - cands[0].distance : 0;
    const gap = cands.length > 1 ? cands[1].distance - cands[0].distance : 0;
    const pct = (x) => (x * 100).toFixed(0) + "%";
    let verdict;
    if (cands.length < 2) {
      verdict = "Only one candidate was returned, so there is nothing here to compare it against.";
    } else if (gap / full < 0.05) {
      verdict = "Rank 1 and rank 2 are " + gap.toFixed(4) + " apart, " + pct(gap / full) +
        " of that spread: they are effectively tied, so which one came first is close to arbitrary.";
    } else if (span / full < 0.15) {
      verdict = "The whole top " + cands.length + " sits inside " + pct(span / full) +
        " of the index spread, so the order between them means very little.";
    } else {
      verdict = "Rank 1 is " + gap.toFixed(4) + " ahead of rank 2, " + pct(gap / full) +
        " of the index spread - a real gap, but still only a hint.";
    }
    $("#dist-note").textContent =
      "Rank 1 to rank " + cands.length + " spans " + span.toFixed(4) +
      " of an index spread of " + full.toFixed(4) + ". " + verdict;

    renderCandidates(r.candidates);
    /* Load the top candidate's peaks under the stem player so the user can see
       whether the separator returned music or mush before pressing play. */
    try {
      const p = await client.peaks(r.stem_peaks_url, 1200);
      EZPaintPeaks($("#stem-wave"), p, { body: "#3c6b4c", core: "#46cf7c" });
    } catch (err) { /* the player still works without the picture */ }
  }

  function renderCandidates(cands) {
    const ol = $("#candidates");
    ol.innerHTML = "";
    cands.forEach((c) => {
      const li = document.createElement("li");
      li.className = "cand";
      li.dataset.profileId = c.profile_id;
      li.innerHTML =
        '<div class="cand-rank"></div>' +
        '<div><div class="cand-name"></div><div class="cand-nums"></div></div>' +
        '<div class="cand-actions"><button class="btn cand-live">Play live</button></div>';
      li.querySelector(".cand-rank").textContent = c.rank;
      li.querySelector(".cand-name").textContent = c.name;
      li.querySelector(".cand-nums").innerHTML =
        'distance <span class="dist"></span> - top <span class="pct"></span>% of the index';
      li.querySelector(".dist").textContent = c.distance.toFixed(4);
      li.querySelector(".pct").textContent = (c.percentile * 100).toFixed(0);
      if (c.audition_url) {
        const a = document.createElement("audio");
        a.controls = true;
        a.preload = "none";
        a.src = client.httpUrl(c.audition_url);
        li.querySelector(".cand-actions").insertBefore(a, li.querySelector(".cand-live"));
      }
      li.querySelector(".cand-live").onclick = () => loadProfile(c.profile_id);
      ol.appendChild(li);
    });
    markLiveCandidate();
  }

  function markLiveCandidate() {
    document.querySelectorAll(".cand").forEach((li) => {
      const on = S.loaded && li.dataset.profileId === S.loaded.profile_id;
      li.setAttribute("aria-current", on ? "true" : "false");
      const b = li.querySelector(".cand-live");
      if (b) {
        b.setAttribute("aria-pressed", on ? "true" : "false");
        b.textContent = on ? "Playing live" : "Play live";
      }
    });
  }

  /* ==================================================================== */
  /* spontaneous faults (e.g. the device disappearing mid-stream)          */
  /* ==================================================================== */

  client.on("error", (m) => {
    if (m.id == null) fail({ frame: m }, "engine");
  });

  /* ==================================================================== */
  /* keyboard                                                             */
  /* ==================================================================== */

  document.addEventListener("keydown", (e) => {
    const t = e.target;
    const tag = t && t.tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || (t && t.isContentEditable)) return;
    if (e.key === "b" || e.key === "B") { e.preventDefault(); toggleBypass(); }
    else if (e.code === "Space") {
      // Space is the browser's own "activate the focused control". Stealing it
      // from a focused button or transport would break keyboard operation of
      // every Play live button on the page, so the shortcut only applies when
      // the focus is not sitting on something that already answers to Space.
      if (tag === "BUTTON" || tag === "AUDIO" || tag === "A") return;
      e.preventDefault();
      togglePlayback();
    } else if (e.key === "Escape" && S.job && S.job.job_id) { $("#btn-cancel-match").click(); }
  });

  /* ==================================================================== */
  /* boot                                                                 */
  /* ==================================================================== */

  if (MOCK) $("#mock-banner").hidden = false;
  updateMatchButton();
  client.connect();
  window.EZ = { client, state: S };          // a console handle while developing
})();
