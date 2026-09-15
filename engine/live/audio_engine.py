"""The duplex audio path: guitar in, streaming NAM, monitors out.

This module owns the only hard real-time code in the project. At a 512-sample
block the callback has 10.67 ms of wall clock to return in, and the profile
itself takes 3.0-4.8 ms of that depending on the capture. Everything else in
the callback has to be effectively free, which drives every decision here:

*Nothing is allocated.* Every buffer -- the mono input, the wet output, the
gain ramp, the tuner ring, the meter accumulator, the timing ring -- is created
in ``configure`` and only written into afterwards, with ``np.copyto`` and
``out=`` forms rather than expressions that build temporaries. An allocation is
not slow by itself; it is slow when it triggers a garbage collection, and a GC
pause is measured in milliseconds against a 10.67 ms budget.

*Nothing is locked.* The callback never takes a lock the WebSocket or telemetry
threads could be holding, because a lock held for 20 ms by a thread doing JSON
is a dropout. Cross-thread state is passed as single attribute assignments,
which are atomic under the GIL: settings are plain floats and bools, and a
profile swap publishes a *fully built and primed* model object in one
assignment. The callback reads each into a local at the top of the block, so
a change landing mid-block cannot tear one block across two configurations.

*Meters are read destructively, not shared.* The callback accumulates into a
small float array and the telemetry thread copies it and zeroes it. A race
there loses at most one block of one meter, which is invisible; a lock there
would cost a dropout.

*Timing is measured, not assumed.* The block-size choice was validated against
this machine's dev corpus, but a user profile deeper than anything in it could
fall under the 2x headroom floor. So the callback records its own wall time and
``perf`` reports the p95 and the headroom, which is what the UI shows rather
than a promise.

The two things this module deliberately does not do: it does not resample (a
44.1 kHz profile is played at 48 kHz with a warning, because silently
resampling would hide the difference from the person comparing candidates), and
it does not crossfade a profile swap (the new state is primed and swapped at a
block boundary, so there is no dropout, but a ringing note will click).
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Callable

import numpy as np
import sounddevice as sd
import torch

from tonehound import nam_render
from tonehound.config import SAMPLE_RATE

from . import devices as device_info
from .tuner import Tuner, WINDOW_S

ALLOWED_BLOCK_SIZES = (256, 512, 1024, 2048)
"""128 is excluded on purpose: measured 1.15x realtime p95, it cannot clear
its own budget. 512 is the default -- 256 gives only 1.59x headroom with the
interpreter under load and produced a 95 ms outlier in testing, and 1024 buys
headroom nobody needs for an extra 10.7 ms of latency."""

DEFAULT_BLOCK_SIZE = 512

_START_ATTEMPTS = 3
_START_RETRY_PAUSE_S = 0.15
_START_PROOF_S = 0.25
"""How long to wait for the first callback before calling a stream dead. At a
512-sample block that is ~23 blocks, so a driver taking a few blocks to warm up
still passes, while the WDM-KS cold-start failure -- which delivers nothing at
all, ever -- is caught in a quarter of a second."""

_MONITOR_MODES = ("amp", "dry", "mute")

# Enough for the tuner's 85 ms window plus the largest block, rounded up. Held
# twice end to end so the telemetry thread can always take a contiguous slice.
_RING = 8192


class DeviceError(RuntimeError):
    """A stream that would not open. ``code`` maps onto the wire protocol."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _normalise(name: str) -> str:
    """Strip the decorations Windows adds per host API, for device matching.

    The same jack is ``Line (Komplete Audio 2)`` under WDM-KS and
    ``Line (2- Komplete Audio 2)`` under WASAPI. Matching on the raw strings
    fails; matching on letters alone with the enumerator digits removed works.
    """
    return re.sub(r"[^a-z]", "", re.sub(r"\b\d+\s*-\s*", "", name.lower()))


class AudioEngine:
    """Owns the PortAudio stream and everything that runs inside its callback."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        # Measured: 1 -> 2 threads is worth ~15% of the block time, 2 -> 8 is
        # worth nothing and adds jitter that shows up as p99 outliers.
        torch.set_num_threads(2)

        self.sample_rate = sample_rate
        self.block_size = DEFAULT_BLOCK_SIZE
        self._stream: sd.Stream | None = None
        self._selection: dict[str, Any] | None = None
        self._reported: dict[str, float] | None = None
        self._started_at: float | None = None

        # --- settings, written by the control thread, read by the callback ---
        self.input_gain_db = 0.0
        self.output_gain_db = 0.0
        self.bypass = False
        self.monitor = "amp"
        self.tuner_enabled = True
        self.tuner_reference_hz = 440.0
        self._in_gain = 1.0     # linear, what the last block ended on
        self._out_gain = 1.0

        # --- the profile, published as one atomic assignment -----------------
        self._model: nam_render.NamModel | None = None
        self.profile: dict[str, Any] | None = None
        self._load_lock = threading.Lock()

        # --- counters read by telemetry -------------------------------------
        self.xruns = 0
        self.last_xrun_at_s: float | None = None
        self._tuner = Tuner(sample_rate)

        self._alloc(DEFAULT_BLOCK_SIZE)

    # ---- buffers ---------------------------------------------------------

    def _alloc(self, block: int) -> None:
        """Everything the callback touches, allocated once, off the audio thread."""
        self.block_size = block
        self._mono = np.zeros(block, dtype=np.float32)
        self._wet = np.zeros(block, dtype=np.float32)
        self._gain = np.zeros(block, dtype=np.float32)
        # 0..1 across the block: a gain change is ramped over exactly one block,
        # so a UI slider can stream at 30 Hz without zipper noise.
        self._ramp = np.linspace(0.0, 1.0, block, endpoint=False, dtype=np.float32)
        # Written twice, _RING apart, so any recent window is one contiguous slice.
        self._ring = np.zeros(_RING * 2, dtype=np.float32)
        self._ring_pos = 0
        self._ring_filled = 0
        # [input peak, input sum of squares, output peak, output sum of squares, n]
        self._meters = np.zeros(5, dtype=np.float64)
        self._block_ms = np.zeros(512, dtype=np.float64)
        self._block_i = 0
        self._block_n = 0

    # ---- device selection ------------------------------------------------

    def configure(self, *, input_device: int, input_channel: int, output_device: int,
                  output_channels: list[int], block_size: int,
                  latency_ms: float | None, exclusive: bool) -> dict[str, Any]:
        """Open (and immediately close) a stream to validate and measure it.

        The stream is opened here rather than at ``start`` so a bad combination
        fails while the user is still looking at the device panel, and so the
        driver's reported latency can be returned with the selection.
        """
        if block_size not in ALLOWED_BLOCK_SIZES:
            raise DeviceError("bad_request",
                              f"block_size must be one of {list(ALLOWED_BLOCK_SIZES)}")
        if not output_channels or len(output_channels) > 2:
            raise DeviceError("bad_request", "output_channels must hold 1 or 2 entries")

        self.stop()
        warnings: list[str] = []
        selection = {
            "input_device": int(input_device), "input_channel": int(input_channel),
            "output_device": int(output_device),
            "output_channels": [int(c) for c in output_channels],
            "block_size": int(block_size), "latency_ms": latency_ms,
            "exclusive": bool(exclusive),
        }
        stream = self._open(selection, warnings)

        latency = {
            "input_ms": float(stream.latency[0]) * 1000.0,
            "output_ms": float(stream.latency[1]) * 1000.0,
            "roundtrip_ms": float(sum(stream.latency)) * 1000.0,
        }
        stream.close()

        api = sd.query_hostapis(sd.query_devices(selection["output_device"])["hostapi"])
        honours = device_info._HOST_API_NOTES.get(str(api["name"]), (True, None))[0]
        if latency_ms is not None and not honours:
            warnings.append(f"{api['name']} ignores the latency setting on this "
                            "machine; the reported figure will not move")
        warnings.append("latency figures are reported by the driver, not measured "
                        "acoustically -- the real round trip may be worse")

        self._alloc(selection["block_size"])
        self._selection = selection
        self._reported = latency
        if self._model is not None:
            # The block size may have changed under it.
            self._model.prepare_stream(selection["block_size"], dtype="float32")
        return {
            **selection,
            "sample_rate": self.sample_rate,
            "reported_latency": latency,
            "estimated_total_ms": latency["roundtrip_ms"]
            + selection["block_size"] / self.sample_rate * 1000.0,
            "warnings": warnings,
        }

    def _open(self, selection: dict[str, Any], warnings: list[str]) -> sd.Stream:
        """Open the stream, with the retry and fallback WDM-KS needs.

        WDM-KS is the only low-latency path on this machine, and in testing it
        **failed to open on the first attempt and succeeded on the second**. So
        one retry is not defensive coding, it is the difference between the app
        working on first launch and looking broken. If it still will not open,
        the same jack is found on WASAPI and used with a warning, because an
        app that plays at 50 ms beats an app that does not play.
        """
        try:
            return self._try_open(selection)
        except Exception as first:
            time.sleep(0.25)
            try:
                return self._try_open(selection)
            except Exception as second:
                fallback = self._wasapi_equivalent(selection)
                if fallback is None:
                    raise self._device_error(second) from second
                try:
                    stream = self._try_open(fallback)
                except Exception:
                    raise self._device_error(second) from second
                warnings.append(
                    f"{sd.query_hostapis(sd.query_devices(selection['output_device'])['hostapi'])['name']}"
                    f" would not open ({second}); fell back to WASAPI, which reports "
                    "about 50 ms round trip instead of 20 ms")
                selection.update(fallback)
                del first
                return stream

    def _try_open(self, selection: dict[str, Any]) -> sd.Stream:
        extra = None
        if selection["exclusive"]:
            api = sd.query_hostapis(
                sd.query_devices(selection["output_device"])["hostapi"])
            if str(api["name"]) == "Windows WASAPI":
                extra = sd.WasapiSettings(exclusive=True)
        latency = ("low" if selection["latency_ms"] is None
                   else float(selection["latency_ms"]) / 1000.0)
        stream = sd.Stream(
            samplerate=self.sample_rate,
            blocksize=selection["block_size"],
            device=(selection["input_device"], selection["output_device"]),
            channels=(selection["input_channel"] + 1,
                      max(selection["output_channels"]) + 1),
            dtype="float32",
            latency=latency,
            extra_settings=(None, extra) if extra is not None else None,
            callback=self._callback,
        )
        return stream

    def _wasapi_equivalent(self, selection: dict[str, Any]) -> dict[str, Any] | None:
        """The same physical jacks, reached through WASAPI instead."""
        apis = {str(a["name"]): i for i, a in enumerate(sd.query_hostapis())}
        wasapi = apis.get("Windows WASAPI")
        if wasapi is None:
            return None
        devices = sd.query_devices()
        if devices[selection["input_device"]]["hostapi"] == wasapi:
            return None  # already there; nothing to fall back to

        def match(index: int, is_input: bool) -> int | None:
            want = _normalise(str(devices[index]["name"]))
            key = "max_input_channels" if is_input else "max_output_channels"
            for i, dev in enumerate(devices):
                if dev["hostapi"] != wasapi or int(dev[key]) <= 0:
                    continue
                have = _normalise(str(dev["name"]))
                if want and (want in have or have in want):
                    return i
            return None

        got_in = match(selection["input_device"], True)
        got_out = match(selection["output_device"], False)
        if got_in is None or got_out is None:
            return None
        return {**selection, "input_device": got_in, "output_device": got_out,
                "latency_ms": None, "exclusive": True}

    @staticmethod
    def _device_error(exc: Exception) -> DeviceError:
        text = str(exc)
        low = text.lower()
        if "invalid sample rate" in low or "sample rate" in low:
            return DeviceError("device_rate_unsupported",
                               f"device cannot run at {SAMPLE_RATE} Hz: {text}")
        if "unanticipated" in low or "in use" in low or "busy" in low or "device unavail" in low:
            return DeviceError("device_busy",
                               f"the device is held by another application: {text}")
        return DeviceError("device_open_failed", text)

    # ---- run -------------------------------------------------------------

    def start(self) -> None:
        """Start the stream, and prove it is actually running before saying so.

        The proof is not optional here. Measured on this machine: the first
        WDM-KS stream started in a fresh process opens without error, reports
        ``stream.active == True``, and then **never calls the callback at all**.
        A second attempt works, and so does every attempt after that. So the
        only trustworthy signal is a callback having actually run, and this
        waits for one and retries if none arrives.

        Without this the app comes up saying "running", with meters at -120 and
        a headroom reading of zero, and looks like a broken amp rather than a
        stream that failed to start -- which is the worst version of this bug,
        because there is nothing on screen to act on.
        """
        if self._stream is not None:
            return
        if self._selection is None:
            raise DeviceError("bad_request", "select devices before starting the stream")

        last: Exception | None = None
        for attempt in range(_START_ATTEMPTS):
            warnings: list[str] = []
            try:
                stream = self._open(self._selection, warnings)
            except Exception as exc:            # already a DeviceError from _open
                last = exc
                break

            self._reset_run_state()
            stream.start()
            if self._wait_for_first_block(stream):
                self._stream = stream
                self._started_at = time.monotonic()
                return

            # Dead: `active` lies, so the only way out is to throw it away.
            try:
                stream.stop()
            finally:
                stream.close()
            last = DeviceError(
                "device_open_failed",
                "the stream opened but the driver never delivered a block")
            time.sleep(_START_RETRY_PAUSE_S)

        if isinstance(last, DeviceError):
            raise last
        raise DeviceError(
            "device_open_failed",
            f"the device would not start after {_START_ATTEMPTS} attempts"
            + (f": {last}" if last else ""))

    def _reset_run_state(self) -> None:
        self._meters[:] = 0.0
        self._block_n = 0
        self._ring_filled = 0
        self.xruns = 0
        self.last_xrun_at_s = None
        self._tuner.reset()
        if self._model is not None:
            # Start every session from the primed silent state rather than
            # wherever the last session left off.
            self._model.reset_stream()
        self._in_gain = 10.0 ** (self.input_gain_db / 20.0)
        self._out_gain = 10.0 ** (self.output_gain_db / 20.0)

    def _wait_for_first_block(self, stream: sd.Stream) -> bool:
        """Has the callback run yet? Polled, because there is nothing to await.

        ``_block_n`` is incremented at the *end* of the callback, so a non-zero
        count means a whole block was processed -- not merely that PortAudio
        called us. The wait is generous relative to a 10.7 ms block so a slow
        first block (the driver's own warm-up) is not mistaken for a dead one.
        """
        deadline = time.monotonic() + _START_PROOF_S
        while time.monotonic() < deadline:
            if self._block_n > 0:
                return True
            time.sleep(0.005)
        return self._block_n > 0

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        self._started_at = None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()

    @property
    def running(self) -> bool:
        return self._stream is not None

    # ---- the callback ----------------------------------------------------

    def _callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int,
                  _time: Any, status: sd.CallbackFlags) -> None:
        t0 = time.perf_counter()
        if status.input_overflow or status.output_underflow:
            self.xruns += 1
            self.last_xrun_at_s = self.uptime_s

        # One read each, so a setting landing mid-block cannot split it.
        model = self._model
        monitor = "dry" if self.bypass else self.monitor
        mono, wet = self._mono, self._wet

        if frames != self.block_size:
            # PortAudio may hand a short final block on stop; refuse to guess.
            outdata.fill(0.0)
            return

        np.copyto(mono, indata[:, self._selection["input_channel"]])

        target = 10.0 ** (self.input_gain_db / 20.0)
        self._apply_gain(mono, self._in_gain, target)
        self._in_gain = target

        self._accumulate(mono, 0)
        self._push_ring(mono)

        if monitor == "amp" and model is not None:
            model.render_block(mono, out=wet)
        elif monitor == "mute":
            wet.fill(0.0)
        else:
            np.copyto(wet, mono)

        target = 10.0 ** (self.output_gain_db / 20.0)
        self._apply_gain(wet, self._out_gain, target)
        self._out_gain = target
        self._accumulate(wet, 2)

        outdata.fill(0.0)
        for channel in self._selection["output_channels"]:
            outdata[:, channel] = wet

        self._block_ms[self._block_i] = (time.perf_counter() - t0) * 1000.0
        self._block_i = (self._block_i + 1) % self._block_ms.size
        self._block_n += 1

    def _apply_gain(self, buf: np.ndarray, current: float, target: float) -> None:
        if current == target:
            if target != 1.0:
                np.multiply(buf, target, out=buf)
            return
        # ramp = current + (target - current) * t, built in place.
        np.multiply(self._ramp, target - current, out=self._gain)
        np.add(self._gain, current, out=self._gain)
        np.multiply(buf, self._gain, out=buf)

    def _accumulate(self, buf: np.ndarray, base: int) -> None:
        peak = float(np.abs(buf).max())
        meters = self._meters
        if peak > meters[base]:
            meters[base] = peak
        meters[base + 1] += float(np.dot(buf, buf))
        if base == 0:
            meters[4] += buf.size

    def _push_ring(self, buf: np.ndarray) -> None:
        """Keep the last ``_RING`` samples, stored twice so reads are contiguous."""
        pos, n = self._ring_pos, buf.size
        end = pos + n
        if end <= _RING:
            self._ring[pos:end] = buf
            self._ring[_RING + pos:_RING + end] = buf
        else:
            split = _RING - pos
            self._ring[pos:_RING] = buf[:split]
            self._ring[_RING + pos:2 * _RING] = buf[:split]
            self._ring[0:end - _RING] = buf[split:]
            self._ring[_RING:_RING + end - _RING] = buf[split:]
        self._ring_pos = end % _RING
        self._ring_filled = min(_RING, self._ring_filled + n)

    # ---- read-side (telemetry thread) ------------------------------------

    @property
    def uptime_s(self) -> float:
        return 0.0 if self._started_at is None else time.monotonic() - self._started_at

    def telemetry(self) -> dict[str, Any]:
        """Meters and one tuner reading. Destructive: it drains the meters."""
        meters = self._meters.copy()
        self._meters[:] = 0.0
        n = max(meters[4], 1.0)

        def db(value: float) -> float:
            # float() is load-bearing: np.log10 returns np.float64, which
            # `json.dumps` refuses, and one of those in a telemetry frame kills
            # the frame rather than degrading it.
            return float(max(-120.0, 20.0 * np.log10(value + 1e-12)))

        pitch = None
        if self.tuner_enabled:
            self._tuner.reference_hz = self.tuner_reference_hz
            window = self.recent(int(WINDOW_S * self.sample_rate))
            result = self._tuner(window)
            pitch = {
                "detected": result.detected,
                "frequency_hz": result.frequency_hz,
                "note": result.note,
                "octave": result.octave,
                "cents": result.cents,
                "confidence": result.confidence,
                "reference_hz": result.reference_hz,
            }
        return {
            "type": "telemetry",
            "t_s": self.uptime_s,
            "input_peak_db": db(meters[0]),
            "input_rms_db": db(float(np.sqrt(meters[1] / n))),
            "output_peak_db": db(meters[2]),
            "output_rms_db": db(float(np.sqrt(meters[3] / n))),
            "input_clipped": bool(meters[0] >= 1.0),
            "output_clipped": bool(meters[2] >= 1.0),
            "tuner": pitch,
        }

    def recent(self, n: int) -> np.ndarray:
        """The last ``n`` samples of (post-gain) input, as one contiguous slice."""
        n = min(n, self._ring_filled)
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        end = _RING + self._ring_pos
        return self._ring[end - n:end]

    def perf(self) -> dict[str, Any]:
        """Block timing. This is the channel that tells the user about dropouts.

        The block size was validated against the dev corpus, but a deeper user
        profile could fall under the 2x headroom floor, so this is measured
        every second rather than assumed once.
        """
        budget = self.block_size / self.sample_rate * 1000.0
        count = min(self._block_n, self._block_ms.size)
        if count == 0:
            return {"type": "perf", "block_ms_mean": 0.0, "block_ms_p95": 0.0,
                    "block_ms_max": 0.0, "budget_ms": budget, "headroom_x": 0.0,
                    "xruns": self.xruns}
        # Most recent `count` entries, in whatever order the ring holds them --
        # order does not matter for mean/percentile/max.
        window = (self._block_ms[:count] if count < self._block_ms.size
                  else self._block_ms)
        p95 = float(np.percentile(window, 95))
        return {
            "type": "perf",
            "block_ms_mean": float(window.mean()),
            "block_ms_p95": p95,
            "block_ms_max": float(window.max()),
            "budget_ms": budget,
            "headroom_x": budget / p95 if p95 > 0 else 0.0,
            "xruns": self.xruns,
        }

    def status(self) -> dict[str, Any]:
        names = None
        if self._selection is not None:
            names = {
                "input": str(sd.query_devices(self._selection["input_device"])["name"]),
                "output": str(sd.query_devices(self._selection["output_device"])["name"]),
            }
        roundtrip = None if self._reported is None else self._reported["roundtrip_ms"]
        return {
            "type": "stream.status",
            "running": self.running,
            "sample_rate": self.sample_rate,
            "block_size": self.block_size,
            "block_budget_ms": self.block_size / self.sample_rate * 1000.0,
            "reported_latency_ms": roundtrip,
            "estimated_total_ms": (None if roundtrip is None else
                                   roundtrip + self.block_size / self.sample_rate * 1000.0),
            "xruns": self.xruns,
            "last_xrun_at_s": self.last_xrun_at_s,
            "uptime_s": self.uptime_s,
            "device_names": names,
        }

    # ---- settings and profiles -------------------------------------------

    def apply_settings(self, **kwargs: Any) -> dict[str, Any]:
        """Update settings. ``bypass`` and ``monitor`` are kept consistent.

        They overlap deliberately -- ``bypass`` is the footswitch next to the
        profile name, ``monitor`` is the routing selector -- so setting either
        moves the other, and the broadcast carries both.
        """
        if "input_gain_db" in kwargs:
            self.input_gain_db = float(np.clip(kwargs["input_gain_db"], -24.0, 24.0))
        if "output_gain_db" in kwargs:
            self.output_gain_db = float(np.clip(kwargs["output_gain_db"], -60.0, 12.0))
        if "tuner_enabled" in kwargs:
            self.tuner_enabled = bool(kwargs["tuner_enabled"])
            self._tuner.reset()
        if "tuner_reference_hz" in kwargs:
            self.tuner_reference_hz = float(np.clip(kwargs["tuner_reference_hz"],
                                                    410.0, 470.0))
        if "monitor" in kwargs:
            mode = str(kwargs["monitor"])
            if mode not in _MONITOR_MODES:
                raise ValueError(f"monitor must be one of {list(_MONITOR_MODES)}")
            self.monitor = mode
            self.bypass = mode == "dry"
        if "bypass" in kwargs:
            self.bypass = bool(kwargs["bypass"])
            if self.bypass:
                self.monitor = "dry"
            elif self.monitor == "dry":
                self.monitor = "amp"
        return self.settings()

    def settings(self) -> dict[str, Any]:
        return {
            "type": "stream.settings",
            "input_gain_db": self.input_gain_db,
            "output_gain_db": self.output_gain_db,
            "bypass": self.bypass,
            "monitor": self.monitor,
            "tuner_enabled": self.tuner_enabled,
            "tuner_reference_hz": self.tuner_reference_hz,
        }

    def load_profile(self, path: str, profile_id: str,
                     progress: Callable[[str], None] | None = None) -> dict[str, Any]:
        """Build and prime a profile off the audio thread, then swap it in.

        Safe while the stream is running: the callback keeps playing the old
        model until the single assignment at the end, and the new one is
        already primed, so there is no dropout and no xrun. A ringing note will
        click -- the tail is not crossfaded, and pretending otherwise would
        cost a block of latency for a transition nobody holds a note through.
        """
        with self._load_lock:
            if progress:
                progress("reading")
            model = nam_render.load(path)
            if progress:
                progress("building")
            model.prepare_stream(self.block_size, dtype="float32")
            if progress:
                progress("priming")
            model.reset_stream()

            warnings: list[str] = []
            rate_mismatch = model.sample_rate != self.sample_rate
            if rate_mismatch:
                warnings.append(
                    f"captured at {model.sample_rate} Hz and played at "
                    f"{self.sample_rate} Hz without resampling -- it will sound "
                    "slightly bright and fast. Resampling it silently would hide "
                    "that from a comparison, so it is shown instead.")
            if not model.stream_scripted:
                warnings.append("TorchScript compilation failed; running the slower "
                                "eager path, so watch the headroom reading")

            info = {
                "profile_id": profile_id,
                "name": path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1].removesuffix(".nam"),
                "architecture": model.architecture,
                "sample_rate": model.sample_rate,
                "receptive_field": model.receptive_field,
                "rate_mismatch": rate_mismatch,
                "warnings": warnings,
            }
            self._model = model  # atomic publish; the callback picks it up next block
            self.profile = info
            return info

    def close(self) -> None:
        self.stop()
