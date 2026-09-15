# ToneHound UI protocol

Contract between the **audio/engine backend** (Python) and the **browser UI**
(HTML/JS). Both sides implement exactly what is written here. Anything not in
this document does not exist yet; add it here first, then build it.

Written 2026-09-07 from measurements on the dev machine (i7-9700-class, 8
threads, Komplete Audio 2). Every number quoted below was measured, not
estimated; the raw figures are in [Appendix B](#appendix-b--measured-basis).

---

## 0. Why it is shaped this way

Three measurements decided the architecture.

1. **A `.nam` profile can be played in real time from Python, but only with a
   streaming path.** `NamModel.render()` is a whole-signal valid convolution.
   Re-running it per block over `receptive_field - 1 = 4092` samples of history
   wastes 9x the arithmetic and clears only 1.03x realtime at a 512-sample
   block — unusable. A stateful path that keeps each dilated conv's own history
   (Appendix A) is arithmetically identical (max deviation 4.7e-16 in float64)
   and clears **2.7x realtime at 512 samples**.

2. **The heavy work cannot share a process with the audio callback.** A single
   match run costs ~15 s of all-core CPU for separation alone, and the profile
   index costs ~45 s to fingerprint. An audio callback in the same interpreter
   would drop out continuously. So the audio engine gets its own OS process.

3. **The I/O layer, not the model, dominates latency.** PortAudio WASAPI
   reports 47-55 ms round trip on this machine no matter what you ask for;
   WDM-KS honours the latency hint and reports 10-20 ms. The block size is
   worth ~10 ms; the host API is worth ~40 ms. The UI must therefore let the
   user pick the host API, and must show the reported latency.

```
┌──────────────────────────────┐        ┌──────────────────────────────┐
│  browser UI                  │  ws    │  server process              │
│  waveform, crop, tuner dial, │◄──────►│  websocket + static + blobs  │
│  meters, profile list        │  http  │  import, crop, match,        │
│                              │◄──────►│  separation, ranking         │
└──────────────────────────────┘  blobs │  (heavy work in a pool)      │
                                        └──────────────┬───────────────┘
                                                       │ mp pipe
                                                       │ control down,
                                                       │ telemetry up
                                        ┌──────────────▼───────────────┐
                                        │  audio process               │
                                        │  sounddevice duplex stream   │
                                        │  streaming NAM · tuner ·     │
                                        │  meters. Nothing else.       │
                                        └──────────────────────────────┘
```

The UI is HTML/JS because the eventual plugin is JUCE + React WebView, so this
work transfers. The pipe between the two Python processes is an implementation
detail — the UI never sees it, and this document does not specify it.

---

## 1. Transport

| | |
|---|---|
| HTTP + WebSocket | `127.0.0.1:8730` (override with `--port`) |
| WebSocket | `ws://127.0.0.1:8730/ws` |
| Frames | UTF-8 text, one JSON **object** per frame. No binary frames. |
| Bulk data | never over the socket — see [§3 HTTP endpoints](#3-http-endpoints) |
| Sample rate | fixed at **48000 Hz** (`tonehound.config.SAMPLE_RATE`) |
| Origin | server binds loopback only and rejects `Origin` headers it did not serve |

Multiple sockets may be open at once (a second browser tab). Every client
receives every broadcast; command replies go only to the sender.

### Keepalive

The server sends a WebSocket ping every 15 s. A client that has heard nothing
for 30 s should reconnect. On reconnect the client re-runs the [hello
handshake](#41-session); it must not assume any state survived, because the
`hello` response carries a full snapshot anyway.

---

## 2. Envelope and conventions

Every frame is a JSON object with a `type` string.

**Client → server** frames are *commands*. Each carries an `id`: a
client-generated string unique within the connection (a counter is fine).

```json
{"type": "profile.load", "id": "c17", "profile_id": "9f2a1c3b7e004d51"}
```

**Server → client** frames are either *replies* (echo the command's `id`) or
*events* (no `id` field at all, or `"id": null`). Exactly one terminal reply —
a success reply or an `error` — is sent per command. Long commands may emit any
number of non-terminal `*.progress` events before it.

```json
{"type": "profile.loaded", "id": "c17", "profile_id": "9f2a1c3b7e004d51", ...}
{"type": "telemetry", "input_peak_db": -14.2, ...}
```

### Types

`float` is a JSON number; `null` is used for "not known" and is always
explicitly allowed where it can occur. Durations and positions are **seconds**
(`float`) unless the field name ends in `_ms` or `_samples`. Levels are
**dBFS** (`float`, `-120.0` for digital silence, may exceed `0.0`). Gains are
**dB** (`float`). Times of day are not sent.

Field names are `snake_case`. Unknown fields must be ignored, not rejected — it
is how this protocol grows.

### Errors

```json
{
  "type": "error",
  "id": "c17",
  "code": "profile_unsupported",
  "message": "unsupported .nam model (architecture 'A2'): this loader implements Linear, LSTM, WaveNet",
  "detail": {"profile_id": "9f2a1c3b7e004d51"}
}
```

| field | type | |
|---|---|---|
| `id` | `string \| null` | echoes the failed command, `null` for a spontaneous fault |
| `code` | `string` | from the table below; stable, machine-readable |
| `message` | `string` | human sentence, safe to show verbatim |
| `detail` | `object` | code-specific, may be `{}` |

| code | meaning |
|---|---|
| `bad_request` | malformed frame, missing field, wrong type |
| `unknown_type` | `type` the server does not implement |
| `not_found` | `profile_id` / `snippet_id` / `crop_id` / `di_id` / `job_id` does not exist |
| `device_open_failed` | PortAudio refused the stream (bad rate, channel count, format) |
| `device_busy` | device held by another application (common with WDM-KS) |
| `device_rate_unsupported` | device cannot do 48000 Hz |
| `stream_not_running` | command needs a running stream |
| `profile_unsupported` | `nam_render.UnsupportedArchitecture` — skipped, never guessed at |
| `profile_load_failed` | file unreadable, bad JSON, weight-count mismatch |
| `unsupported_media` | import file the decoder cannot read |
| `snippet_too_long` | over `max_snippet_s` (see `hello`) |
| `crop_too_short` | crop under `min_crop_s` (see `hello`) |
| `index_not_ready` | match requested before the profile index finished |
| `separation_failed` | UVR produced no guitar stem |
| `job_cancelled` | terminal reply to a cancelled long job |
| `busy` | a job of this kind is already running |
| `internal` | unexpected; `detail.traceback_id` points at a server log line |

An `error` is always terminal for its `id`. The UI must clear any spinner tied
to that `id`.

---

## 3. HTTP endpoints

Audio and waveform data never travel over the WebSocket. The server hands out
URLs; the browser fetches them with `<audio src>` or `fetch()`. All paths are
relative to the server origin. All are `GET` unless stated.

| endpoint | returns |
|---|---|
| `GET /` | the UI |
| `POST /upload` | `multipart/form-data`, one field `file`. → `{"snippet_id": "..."}` (HTTP 200) or `{"code": ..., "message": ...}` (HTTP 4xx). Also broadcasts [`snippet.ready`](#46-snippet-import-and-crop). |
| `GET /audio/{snippet_id}.wav` | the decoded snippet, 48 kHz mono float→PCM16 |
| `GET /audio/{crop_id}.wav` | the cropped region |
| `GET /audio/{render_id}.wav` | an audition render or an isolated stem |
| `GET /peaks/{id}?bins=N` | waveform envelope, see below |

`render_id`, `crop_id` and `snippet_id` share one namespace of opaque
lowercase-hex strings; any of them works in `/audio/{id}.wav`.

### `GET /peaks/{id}?bins=N`

`N` defaults to 2000, capped at 20000. One JSON object:

```json
{"id": "a3f1...", "bins": 2000, "duration_s": 15.0,
 "min": [-0.81, ...], "max": [0.79, ...], "rms": [0.12, ...]}
```

Three arrays of exactly `bins` floats in `[-1, 1]` (`rms` in `[0, 1]`), evenly
spaced over the whole item. Computed server-side so the UI never downloads
whole waveforms.

---

## 4. Message catalogue

### 4.1 Session

#### `hello` — C→S

```json
{"type": "hello", "id": "c1", "protocol": 1}
```

#### `hello` — S→C (reply)

Full snapshot; the UI can render its whole first frame from this.

| field | type | |
|---|---|---|
| `protocol` | `int` | `1`. Mismatch → `error` `bad_request`. |
| `server_version` | `string` | |
| `sample_rate` | `int` | always `48000` |
| `default_block_size` | `int` | `512` (see Appendix B for why) |
| `allowed_block_sizes` | `int[]` | `[256, 512, 1024, 2048]`. 128 is excluded: measured at 1.15x realtime p95, it cannot clear the budget. |
| `max_snippet_s` | `float` | e.g. `120.0` |
| `min_crop_s` | `float` | e.g. `3.0` |
| `max_crop_s` | `float` | e.g. `30.0` |
| `stream` | `StreamStatus` | see [§4.3](#43-stream-control) |
| `settings` | `StreamSettings` | see [§4.3](#43-stream-control) |
| `devices` | `DeviceList` | see [§4.2](#42-devices) |
| `profile` | `LoadedProfile \| null` | currently loaded, if any |
| `index` | `IndexStatus` | see [§4.5](#45-profiles) |
| `caveat` | `string` | the honesty text — see [§5](#5-what-the-ui-must-say) |

---

### 4.2 Devices

#### `devices.list` — C→S
```json
{"type": "devices.list", "id": "c2", "refresh": false}
```
`refresh` (`bool`, default `false`) forces PortAudio to re-scan, which briefly
blocks; use it behind an explicit "rescan" button, not on every open.

#### `devices.list` — S→C (reply). Object type `DeviceList`.

| field | type | |
|---|---|---|
| `hostapis` | `HostApi[]` | |
| `inputs` | `Device[]` | devices with `max_channels > 0` for capture |
| `outputs` | `Device[]` | |
| `default_input` | `int \| null` | device index |
| `default_output` | `int \| null` | |
| `recommended` | `Recommended \| null` | server's pick, see below |

`HostApi`:

| field | type | |
|---|---|---|
| `index` | `int` | |
| `name` | `string` | `"Windows WASAPI"`, `"Windows WDM-KS"`, ... |
| `supports_exclusive` | `bool` | WASAPI only, today |
| `honours_latency_hint` | `bool` | `false` for WASAPI on this machine — see Appendix B |
| `latency_note` | `string \| null` | one line the UI shows next to the API name |

`Device`:

| field | type | |
|---|---|---|
| `index` | `int` | PortAudio device index; the id used in `devices.select` |
| `name` | `string` | |
| `hostapi` | `int` | index into `hostapis` |
| `hostapi_name` | `string` | denormalised for convenience |
| `max_channels` | `int` | input or output, whichever list it is in |
| `default_samplerate` | `float` | |
| `supports_48k` | `bool` | probed with `sounddevice.check_input_settings` |
| `low_latency_ms` | `float` | PortAudio's `default_low_*_latency`, advisory only |

`Recommended`: `{"input": int, "output": int, "hostapi": int, "block_size": int,
"latency_ms": float, "exclusive": bool, "reason": string}` — the server's best
guess (prefer a host API with `honours_latency_hint`, then the lowest reported
round trip). The UI should preselect it and let the user override everything.

#### `devices.select` — C→S

Stops the stream if running; the caller must `stream.start` again.

| field | type | |
|---|---|---|
| `input_device` | `int` | |
| `input_channel` | `int` | 0-based channel on that device carrying the guitar |
| `output_device` | `int` | |
| `output_channels` | `int[]` | 1 or 2 entries, 0-based; `[0, 1]` for normal stereo |
| `block_size` | `int` | must be in `allowed_block_sizes` |
| `latency_ms` | `float \| null` | PortAudio suggested latency; `null` = `"low"` |
| `exclusive` | `bool` | WASAPI exclusive mode; ignored by other host APIs |

#### `devices.selected` — S→C (reply)

| field | type | |
|---|---|---|
| `input_device`, `output_device` | `int` | |
| `input_channel` | `int` | |
| `output_channels` | `int[]` | |
| `block_size` | `int` | |
| `sample_rate` | `int` | |
| `reported_latency` | `{input_ms: float, output_ms: float, roundtrip_ms: float}` | PortAudio's numbers |
| `estimated_total_ms` | `float` | `roundtrip_ms + block_size / 48000 * 1000` |
| `warnings` | `string[]` | e.g. `"WASAPI ignores the latency hint on this machine"` |

`reported_latency` is what the **driver claims**, not an acoustic measurement.
The UI must label it as such (see [§5](#5-what-the-ui-must-say)).

Failure → `error` with `device_open_failed`, `device_busy` or
`device_rate_unsupported`.

---

### 4.3 Stream control

#### `stream.start` / `stream.stop` — C→S
```json
{"type": "stream.start", "id": "c3"}
```
Both reply with `stream.status`. Starting an already-running stream is a no-op
success. Priming the streaming NAM state costs ~13 ms and happens inside
`stream.start`, before the first callback.

#### `stream.status` — S→C (reply **and** event)

Broadcast on every transition, and on any change to `xruns`.

| field | type | |
|---|---|---|
| `running` | `bool` | |
| `sample_rate` | `int` | |
| `block_size` | `int` | |
| `block_budget_ms` | `float` | `block_size / 48000 * 1000` |
| `reported_latency_ms` | `float \| null` | round trip, driver-reported |
| `estimated_total_ms` | `float \| null` | + block |
| `xruns` | `int` | cumulative since `stream.start` |
| `last_xrun_at_s` | `float \| null` | seconds since stream start |
| `uptime_s` | `float` | |
| `device_names` | `{input: string, output: string} \| null` | |

#### `stream.set` — C→S

Every field is optional; omitted fields are unchanged. Takes effect on the next
block. Gain changes are ramped over one block inside the engine, so the UI may
send these continuously from a slider (throttle to ~30 Hz).

| field | type | range |
|---|---|---|
| `input_gain_db` | `float` | `-24.0 .. +24.0` |
| `output_gain_db` | `float` | `-60.0 .. +12.0` |
| `bypass` | `bool` | `true` routes the input to the output untouched by the profile |
| `monitor` | `string` | `"amp"` (default) · `"dry"` · `"mute"` |
| `tuner_enabled` | `bool` | |
| `tuner_reference_hz` | `float` | `410.0 .. 470.0`, default `440.0` |

`bypass` and `monitor` overlap deliberately: `bypass` is the footswitch the UI
puts next to the profile name, `monitor` is the routing selector. `bypass:
true` is equivalent to `monitor: "dry"` and the server keeps them consistent —
setting one updates the other in the broadcast.

#### `stream.settings` — S→C (reply **and** event)

Object type `StreamSettings`. Broadcast to *all* clients on every change, so a
second tab stays in sync. Carries all seven fields above, always.

---

### 4.4 Telemetry

#### `telemetry` — S→C (event)

One message, **20 Hz**, only while the stream is running. Meters and tuner are
merged into one frame to keep the socket quiet. Never has an `id`. Dropped
silently if a client's send buffer is full — the UI must tolerate gaps.

| field | type | |
|---|---|---|
| `t_s` | `float` | seconds since `stream.start`, monotonic |
| `input_peak_db` | `float` | max abs over the window, post `input_gain_db` |
| `input_rms_db` | `float` | |
| `output_peak_db` | `float` | post `output_gain_db` |
| `output_rms_db` | `float` | |
| `input_clipped` | `bool` | any sample \|x\| ≥ 1.0 since the last frame |
| `output_clipped` | `bool` | |
| `tuner` | `Tuner \| null` | `null` when `tuner_enabled` is false |

`Tuner`:

| field | type | |
|---|---|---|
| `detected` | `bool` | `false` when nothing pitched is playing |
| `frequency_hz` | `float \| null` | `null` iff `detected` is false |
| `note` | `string \| null` | `"E"`, `"C#"` — sharps only, never flats |
| `octave` | `int \| null` | scientific pitch notation; low E of a guitar is `E2` |
| `cents` | `float \| null` | `-50.0 .. +50.0`, signed offset from `note`+`octave` |
| `confidence` | `float` | `0.0 .. 1.0` |
| `reference_hz` | `float` | echoes `tuner_reference_hz` |

The detector is YIN over an 85 ms window, refreshed at 20 Hz, run on the
engine's telemetry thread — never in the audio callback. Measured accuracy is
better than 0.1 cents from B1 (61.7 Hz, baritone) to E5, and it costs 0.55 ms
per evaluation (~1 % of one core at 20 Hz). Plain autocorrelation was tried and
rejected: it octave-fails on low E and low B *with high confidence*, so a
confidence gate would not have caught it.

`detected` is `false` when confidence is below the engine's threshold or the
input is below about -45 dBFS. The UI should hold the last reading for ~500 ms
before blanking, so the needle does not flicker between picks.

#### `perf` — S→C (event)

One message per second while running. This is how the user finds out the
machine cannot keep up, so the UI must surface it, not hide it.

| field | type | |
|---|---|---|
| `block_ms_mean` | `float` | wall time of the processing callback |
| `block_ms_p95` | `float` | |
| `block_ms_max` | `float` | over the last second |
| `budget_ms` | `float` | `block_size / 48000 * 1000` |
| `headroom_x` | `float` | `budget_ms / block_ms_p95` — below 1.5 means trouble |
| `xruns` | `int` | cumulative |

---

### 4.5 Profiles

#### `profiles.list` — C→S
```json
{"type": "profiles.list", "id": "c5", "query": "5150"}
```
`query` (`string`, optional) is a case-insensitive substring filter on `name`.

#### `profiles.list` — S→C (reply)

| field | type | |
|---|---|---|
| `profiles` | `Profile[]` | |
| `total` | `int` | before filtering |
| `unsupported` | `int` | files present but not loadable |

`Profile`:

| field | type | |
|---|---|---|
| `profile_id` | `string` | first 16 hex of SHA-1 of the file bytes; stable across restarts |
| `name` | `string` | file stem |
| `architecture` | `string` | `"WaveNet"`, `"LSTM"`, `"Linear"`, or the raw string |
| `sample_rate` | `int` | as declared by the file |
| `receptive_field` | `int \| null` | `null` if unloadable |
| `weight_count` | `int \| null` | |
| `loadable` | `bool` | |
| `unsupported_reason` | `string \| null` | the `UnsupportedArchitecture` detail |
| `rate_mismatch` | `bool` | `sample_rate != 48000` |
| `metadata` | `object` | the file's `metadata` block, verbatim, possibly `{}` |

A profile with `loadable: false` stays in the list. The UI shows it greyed with
its reason — hiding it would look like the file is missing.

#### `profile.load` — C→S
```json
{"type": "profile.load", "id": "c6", "profile_id": "9f2a1c3b7e004d51"}
```
Safe while the stream is running. The engine builds and primes the new
streaming state off the audio thread, then swaps it in at a block boundary, so
there is no dropout and no `xrun`. The tail of the previous profile is *not*
crossfaded; expect an audible click if a note is ringing.

Measured cost, so the UI knows whether to bother with a spinner: reading and
parsing the JSON dominates for large files, building the streaming state is
0.8 ms, TorchScripting it is 61 ms, priming is 13 ms. Budget ~100 ms and show
progress through the three `profile.loading` stages.

#### `profile.loading` — S→C (event, non-terminal)
`{"type": "profile.loading", "id": "c6", "profile_id": "...", "stage": "reading" | "building" | "priming"}`

#### `profile.loaded` — S→C (reply). Object type `LoadedProfile`.

| field | type | |
|---|---|---|
| `profile_id` | `string` | |
| `name` | `string` | |
| `architecture` | `string` | |
| `sample_rate` | `int` | |
| `receptive_field` | `int` | |
| `rate_mismatch` | `bool` | |
| `warnings` | `string[]` | |

A profile captured at 44100 is played at 48000 anyway, with `rate_mismatch:
true` and a warning — it will sound slightly bright and fast. It is not
silently resampled, because that would hide the problem from the person
choosing between candidates. The UI shows the warning next to the profile name.

Also broadcast as an event (`"id": null`) to other clients.

#### `index.status` — S→C (event) and a field of `hello`. Object type `IndexStatus`.

Fingerprinting the profile index takes about **45 s** on this machine. It runs
at server start. Matching before it finishes returns `error` `index_not_ready`.

| field | type | |
|---|---|---|
| `ready` | `bool` | |
| `profiles` | `int` | indexed so far |
| `total` | `int` | |
| `progress` | `float` | `0.0 .. 1.0` |
| `eta_s` | `float \| null` | |
| `di_name` | `string` | the DI take the index was rendered through |

---

### 4.6 Snippet import and crop

Import is `POST /upload` ([§3](#3-http-endpoints)) because a browser cannot
hand the server a file path. The upload response carries `snippet_id`; the
server then broadcasts:

#### `snippet.ready` — S→C (event)

| field | type | |
|---|---|---|
| `snippet_id` | `string` | |
| `filename` | `string` | as uploaded, for display only |
| `duration_s` | `float` | after decode to 48 kHz mono |
| `source_sample_rate` | `int` | before resampling |
| `source_channels` | `int` | |
| `audio_url` | `string` | `/audio/{snippet_id}.wav` |
| `peaks_url` | `string` | `/peaks/{snippet_id}` |

#### `snippet.crop` — C→S

Non-destructive: the full decode is kept, and every crop is a new derived item.
Re-cropping the same snippet is cheap and expected as the user drags handles.

| field | type | |
|---|---|---|
| `snippet_id` | `string` | |
| `start_s` | `float` | `>= 0` |
| `end_s` | `float` | `> start_s`, `<= duration_s` |

`end_s - start_s` must be within `[min_crop_s, max_crop_s]` from `hello`, else
`error` `crop_too_short` / `snippet_too_long`.

#### `snippet.cropped` — S→C (reply)

| field | type | |
|---|---|---|
| `snippet_id` | `string` | |
| `crop_id` | `string` | pass this to `match.run` |
| `start_s`, `end_s`, `duration_s` | `float` | as applied (may be clamped) |
| `audio_url` | `string` | `/audio/{crop_id}.wav` |
| `peaks_url` | `string` | `/peaks/{crop_id}` |

#### `snippet.forget` — C→S
`{"type": "snippet.forget", "id": "c9", "snippet_id": "..."}` → reply
`{"type": "snippet.forgotten", "id": "c9", "snippet_id": "..."}`. Drops the
decode and every crop derived from it. Their URLs 404 afterwards.

#### `di.list` — C→S / S→C (reply)

The user's own DI takes, used to audition candidates.

```json
{"type": "di.list", "id": "c10",
 "takes": [{"di_id": "d1", "name": "Djent", "duration_s": 41.2,
            "audio_url": "/audio/d1.wav", "peaks_url": "/peaks/d1"}],
 "default_di_id": "d1"}
```

A DI take uploaded through `POST /upload` with form field `kind=di` gets a
`di_id` instead of a `snippet_id` and appears in this list.

---

### 4.7 Match

One match may run at a time. A second `match.run` → `error` `busy`.

#### `match.run` — C→S

| field | type | |
|---|---|---|
| `crop_id` | `string` | the region with the guitar in it. Required. |
| `di_id` | `string \| null` | DI to audition candidates through; `null` = `default_di_id` |
| `top` | `int` | candidates to return and render, `1..10`, default `5` |
| `render_auditions` | `bool` | default `true`; `false` returns the ranking without rendering |

#### `match.progress` — S→C (event, non-terminal)

| field | type | |
|---|---|---|
| `job_id` | `string` | also echoed on the terminal reply |
| `stage` | `string` | `"separating"` · `"fingerprinting"` · `"ranking"` · `"rendering"` |
| `progress` | `float` | `0.0 .. 1.0` across the whole job, not the stage |
| `message` | `string` | short, showable, e.g. `"rendering 2/5: Helga B 5150 BlockLetter"` |
| `eta_s` | `float \| null` | |

Measured stage costs, for sizing the progress bar: separation runs at about
**1.0x realtime** (14.7 s for a 15 s crop) and dominates; query fingerprinting
is 0.2 s; ranking is instant; each audition render is about **1.3 s per 8 s of
DI**. A 15 s crop with 5 auditions is roughly 22 s end to end.

#### `match.result` — S→C (reply)

| field | type | |
|---|---|---|
| `job_id` | `string` | |
| `crop_id` | `string` | |
| `stem_audio_url` | `string` | **the isolated guitar** — see [§5](#5-what-the-ui-must-say) |
| `stem_peaks_url` | `string` | |
| `separation_ok` | `bool` | `false` = the separator returned nothing and the raw mix was matched instead |
| `di_id` | `string` | what the auditions were rendered through |
| `di_audio_url` | `string` | the dry DI, so the user can hear what went in |
| `index_size` | `int` | profiles ranked |
| `query_seconds` | `float` | length of audio actually fingerprinted |
| `distance_range` | `{min: float, max: float, median: float}` | over the whole index, so the UI can show where the top-5 sit |
| `candidates` | `Candidate[]` | length `top`, ascending `distance` |
| `caveat` | `string` | must be displayed — see [§5](#5-what-the-ui-must-say) |

`Candidate`:

| field | type | |
|---|---|---|
| `rank` | `int` | 1-based |
| `profile_id` | `string` | loadable with `profile.load` to play through it live |
| `name` | `string` | |
| `distance` | `float` | cosine distance, `0.0` = identical. **Show this number.** |
| `percentile` | `float` | `0.0 .. 1.0`, position within the full index |
| `audition_url` | `string \| null` | `null` when `render_auditions` was `false` |
| `metadata` | `object` | the profile's metadata, verbatim |

#### `match.cancel` — C→S
`{"type": "match.cancel", "id": "c12", "job_id": "..."}`. The running job's
terminal reply becomes `error` `job_cancelled` against its **original** `id`;
`match.cancel` gets its own `{"type": "match.cancelled", "id": "c12"}`.

---

### 4.8 Audition

Auditions arrive with `match.result`. This area re-renders one on demand — a
different DI, or a profile the user picked from the full list rather than from
the ranking.

#### `audition.render` — C→S

| field | type | |
|---|---|---|
| `profile_id` | `string` | |
| `di_id` | `string \| null` | `null` = `default_di_id` |
| `seconds` | `float \| null` | DI length to render, `null` = server default (8.0) |

#### `audition.progress` — S→C (event)
`{"type": "audition.progress", "id": "c13", "progress": 0.4}`

#### `audition.ready` — S→C (reply)

| field | type | |
|---|---|---|
| `render_id` | `string` | |
| `profile_id` | `string` | |
| `di_id` | `string` | |
| `audio_url` | `string` | |
| `peaks_url` | `string` | |
| `seconds` | `float` | |

To hear a candidate **live** under your own hands, send `profile.load` with the
candidate's `profile_id` — that is the whole mechanism. There is no separate
"audition live" command.

---

## 5. What the UI must say

The matcher is weak and the UI must not paper over it. Measured properly (query
= a *different* performance from the one in the index), top-1 accuracy is about
**0.176** on the dev corpus, and every same-amp pair in that corpus shares a
capturer, so the ranking may partly be sorting by who made the capture rather
than by how the amp sounds.

Three requirements, all enforceable by review:

1. **The isolated guitar stem gets a player at least as prominent as the
   ranking.** `stem_audio_url` is the user's own check on whether separation
   worked at all; if it is buried, a bad match looks like a bad matcher when it
   was a bad stem. If `separation_ok` is `false`, say so loudly.
2. **Show `distance` next to every candidate**, and place it against
   `distance_range` so the user can see whether rank 1 is meaningfully closer
   than rank 5 or whether the whole index is a wash.
3. **Show `caveat` verbatim**, near the results, not behind a tooltip or an
   "info" icon. The server owns this string so both UIs say the same thing:

   > Ranking is a rough draft. Measured against unseen performances it puts the
   > right amp first about 18% of the time, and the profiles it learned from
   > share capturers, so it may be sorting by who made the capture as much as
   > by how the amp sounds. Listen to the isolated guitar above, then listen to
   > the candidates. Trust your ears over the order.

Similarly for latency: `reported_latency_ms` is what the **driver claims**, not
a measured round trip. Label it "reported by driver". Nothing in this system
has measured the true acoustic round trip, and the UI must not imply otherwise.

---

## 6. Worked flows

### First run

```
C: hello                       -> S: hello {index: {ready:false, progress:0.3}, ...}
S: index.status {progress: 0.6}                                    (event, ~45s)
S: index.status {ready: true, profiles: 34, total: 34}
C: devices.list                -> S: devices.list {recommended: {...}}
C: devices.select {...}        -> S: devices.selected {reported_latency: {...}}
C: profiles.list               -> S: profiles.list {profiles: [...]}
C: profile.load {profile_id}   -> S: profile.loading / profile.loaded
C: stream.start                -> S: stream.status {running: true}
S: telemetry × 20/s, perf × 1/s
```

### Import, crop, match, audition

```
HTTP POST /upload              -> {"snippet_id": "s1"}
S: snippet.ready {duration_s: 214.0, peaks_url: "/peaks/s1"}       (event)
    UI: GET /peaks/s1?bins=2000, draws the waveform, user drags a region
C: snippet.crop {s1, 61.5, 76.5}  -> S: snippet.cropped {crop_id: "k1"}
C: match.run {crop_id: "k1", top: 5}
S: match.progress {stage: "separating", progress: 0.05}
S: match.progress {stage: "rendering", progress: 0.7, message: "rendering 2/5: ..."}
S: match.result {stem_audio_url, candidates: [...], caveat: "..."}
    UI: plays the stem, then each audition_url
C: profile.load {candidates[2].profile_id}   -> plays it live, no dropout
```

### Losing the device

```
S: error {id: null, code: "device_busy", message: "..."}            (event)
S: stream.status {running: false, xruns: 3}
    UI: shows the message, offers devices.list + devices.select again
```

---

## Appendix A — the streaming `NamModel` API

To be added to `engine/tonehound/nam_render.py`. **Not implemented yet.** This is
the exact surface to build against; the offline `render()` is unchanged.

```python
class NamModel:
    # ---- existing, unchanged --------------------------------------------
    def render(self, x: np.ndarray, pad_start: bool = True) -> np.ndarray: ...

    # ---- new ------------------------------------------------------------
    @property
    def streamable(self) -> bool: ...
    @property
    def stream_block_size(self) -> int | None: ...
    @property
    def stream_dtype(self) -> str: ...

    def prepare_stream(self, block_size: int, *, dtype: str = "float32") -> None: ...
    def reset_stream(self) -> None: ...
    def render_block(self, x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray: ...
```

**`prepare_stream(block_size, *, dtype="float32")`**
Allocates the per-convolution history buffers and, for `dtype="float32"`, a
float32 copy of the weights. `dtype` is `"float32"` or `"float64"`.
Idempotent for the same arguments. Calls `reset_stream()`. **Allocates and
takes tens of milliseconds — never call it from an audio callback.** Raises
`UnsupportedArchitecture` if `streamable` is `False`.

**`reset_stream()`**
Zeroes every history buffer, then primes the network with `receptive_field - 1`
zero samples. Priming is what makes the first `render_block` output match
`render(x, pad_start=True)` sample for sample: the network's internal DC state
after silence is *not* all zeros, because `layer1x1` carries a bias. Measured
cost about 13 ms for a 4093-sample receptive field. **Not real-time safe.**
For `LSTM` models it restores the file's stored initial `(h, c)`, which is the
trained settled state, not zeros.

**`render_block(x, out=None)`**
`x`: 1-D, C-contiguous, exactly `stream_block_size` samples, dtype matching
`stream_dtype`. Returns `block_size` samples; writes into `out` and returns it
when `out` is given. Allocation-free after `prepare_stream` when `out` is
supplied. Raises `RuntimeError` if `prepare_stream` was not called, `ValueError`
on a wrong length or dtype.

**Latency: zero.** Output sample *n* is the model's response to input sample
*n*. The streaming path adds no delay of its own; all the latency in the system
is the device buffers plus the block size.

**Equivalence guarantee** (the property the new tests must assert):

```python
m.prepare_stream(512, dtype="float64"); m.reset_stream()
streamed = np.concatenate([m.render_block(c) for c in chunks_of(x, 512)])
assert np.allclose(streamed, m.render(x, pad_start=True), atol=1e-12)
```

Measured on `Helga B 5150 BlockLetter - Boosted.nam` with a prototype of exactly
this design: **4.7e-16** max absolute deviation in float64 (block sizes 64, 256
and 1024 all identical), and **5.7e-07** in float32 against the float64 offline
render, on a signal of RMS 0.047. The float32 figure is the same order as the
1.1e-06 parity this renderer already holds against neural-amp-modeler 0.13.0,
so float32 is the right default for playing and float64 stays the default for
offline rendering.

**Thread and real-time notes**
- One `NamModel` streaming state, one thread. Swap profiles by preparing a
  second `NamModel` off-thread and publishing the reference; do not mutate one
  in place from two threads.
- Call `torch.set_num_threads(2)` in the audio process. Measured: 1 → 2 threads
  is worth ~15 %, 2 → 8 is worth nothing and adds jitter.
- Wrap `render_block` in `torch.no_grad()` internally so callers cannot forget.

**Implementation sketch** (verified as a prototype, ~70 lines)
For each convolution with kernel `k` and dilation `d`, keep the trailing
`(k - 1) * d` samples of *its own input*; each block concatenates history with
the new block, runs the same valid conv, and keeps the new tail. Every stage
then produces exactly `block_size` samples and the receptive field costs
nothing per block. `mixer`, `layer1x1`, `head1x1` and `rechannel` are all 1×1
and need no history; the dilated `conv` in each layer, the array
`head_rechannel` when its kernel is >1, and each network-head conv do.

**Known optimisation, not required for v1**: `torch.conv1d` is 4x slower at
dilation 512 than at dilation 1 (measured 122.8 µs vs 29.6 µs for 16 channels,
kernel 3, 512 samples). Expressing the dilated conv as three matmuls on strided
views removes that penalty entirely — a numpy sketch of the full model ran at
9x realtime versus the 3.1x measured for the TorchScript version. Ship the
torch version, keep this in the back pocket.

---

## Appendix B — measured basis

Machine: Intel 8-thread, Windows 10 19045, Python 3.10, torch 2.11.0+cpu (MKL
2025.3, MKL-DNN 3.10.2), PortAudio V19.7.0-devel via sounddevice 0.5.6,
Komplete Audio 2. Profile: `Helga B 5150 BlockLetter - Boosted.nam`, WaveNet,
13802 weights, receptive field 4093 — 26 of the 34 dev profiles have exactly
this shape, and all 34 have receptive field 4093.

The corpus's two largest profiles were measured separately, because the block
size has to hold for the worst file the user can load, not the typical one
(scripted float32, 2 threads, 512-sample block):

| profile | layers × channels | p95 | xRT p95 | xRT worst |
|---|---|---:|---:|---:|
| 13802 weights (26 of 34) | 10×16 + 10×8 | 3.31 ms | 3.22x | 2.77x |
| 41288 weights | 10×28 + 10×14 | 4.54 ms | 2.35x | 2.00x |
| 48854 weights (largest) | 10×32 + 10×12 | 4.77 ms | 2.24x | 1.91x |

3.5x the weights costs only 1.4x the time, because the per-block cost is still
part fixed dispatch. The largest profile in the corpus clears 2.24x p95 at a
512-sample block, so the choice holds across the whole corpus. It does not
generalise to arbitrary user profiles, so the backend reports
`perf.headroom_x` from live measurement rather than assuming a shape.

### Why naive streaming was rejected

Prepending `receptive_field - 1` samples to every block, float32, 1 thread:

| block | budget | mean | p95 | xRT mean | xRT p95 |
|---:|---:|---:|---:|---:|---:|
| 512 | 10.67 ms | 8.30 ms | 8.67 ms | 1.28 | 1.23 |
| 1024 | 21.33 ms | 8.79 ms | 9.56 ms | 2.43 | 2.23 |
| 2048 | 42.67 ms | 9.96 ms | 11.03 ms | 4.28 | 3.87 |

Offline batch throughput is 10.4x realtime (float32) / 5.5x (float64), so the
naive path burns 9x the work for a 512-sample block and lands at 1.23x p95 —
below the 2x floor, at a block size that is otherwise ideal.

### Stateful streaming (Appendix A design), float32

| block | 1 thread p95 | 2 threads p95 | 4 threads p95 |
|---:|---:|---:|---:|
| 256 | 1.58x | 1.62x | 1.52x |
| 512 | 2.61x | 2.59x | 2.68x |
| 1024 | 3.27x | 4.37x | 4.40x |

TorchScripting the same graph is worth a further ~25 % (512 samples, 1 thread:
3.90 ms → 2.74x p95) and, more importantly, releases the GIL for the whole
forward pass instead of holding it through ~100 Python-level op dispatches.

### Host APIs, this machine

Reported round trip, duplex, 48 kHz, guitar in / monitors out:

| host API | 128 | 256 | 512 | honours `latency` hint |
|---|---:|---:|---:|---|
| WASAPI shared | 46.7 ms | 49.3 ms | 54.7 ms | no |
| WASAPI exclusive | 44.7 ms | 47.3 ms | 52.7 ms | no |
| **WDM-KS** | **4-20 ms** | **4-20 ms** | **4-20 ms** | **yes** (2 ms → 4 ms, 5 ms → 10 ms) |
| DirectSound | 9.0 ms | 17.0 ms | 33.0 ms | yes, output only |
| MME | 5.3 ms | 10.7 ms | 21.3 ms | yes, but 17-54 xruns |

No ASIO: this PortAudio build does not expose an ASIO host API. WDM-KS is the
low-latency path here, but it takes the device exclusively and **failed to open
on the first attempt** in testing before succeeding on a retry — the backend
must retry once and fall back to WASAPI exclusive with a warning.

### End to end, streaming NAM inside the callback

5 s runs. `hog` = a busy pure-Python thread competing for the GIL, standing in
for the WebSocket server and telemetry.

| host API | block | hog | xruns | proc p99 | budget | headroom | est. total RT |
|---|---:|---|---:|---:|---:|---:|---:|
| WDM-KS | 256 | yes | 0 | 3.36 ms | 5.33 ms | 1.59x | 15.3 ms |
| **WDM-KS** | **512** | **yes** | **0** | **3.96 ms** | **10.67 ms** | **2.69x** | **20.7 ms** |
| WDM-KS | 1024 | yes | 0 | 5.06 ms | 21.33 ms | 4.22x | 31.3 ms |
| WASAPI excl | 512 | yes | 0 | 4.12 ms | 10.67 ms | 2.59x | 63.3 ms |

**512 samples is the choice.** 256 gives only 1.59x — under the 2x floor — and
produced a 95 ms outlier in one run. 1024 buys headroom nobody needs and costs
10.7 ms. At 512 the processing headroom is 2.7x with the interpreter under
load, and the block contributes 10.7 ms to a ~21 ms round trip that is
otherwise all driver.

### Match pipeline, 15 s crop

| stage | cost |
|---|---|
| decode + resample | 0.45 s |
| UVR separation (htdemucs_6s) | 14.67 s |
| fingerprint query | 0.19 s |
| fingerprint 34-profile index | 44.56 s — must be done once at startup, not per match |
| render 8 s of DI through one profile | 1.33 s |

This is why the audio engine is a separate process: separation alone is 15 s of
all-core work, and it would take the audio callback down with it.
