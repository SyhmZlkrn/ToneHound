"""PortAudio device enumeration, with the host-API facts the UI has to show.

The measurement that shaped this module: on this machine the block size is
worth about 10 ms of round trip and the **host API is worth about 40**. WASAPI
reports 47-55 ms whatever latency you ask it for -- it ignores the hint
entirely -- while WDM-KS honours the hint and reports 4-20 ms. A UI that only
exposes device and block size therefore hides the one control that matters, so
``honours_latency_hint`` and ``latency_note`` are part of the wire format and
are filled in here.

Every latency number that leaves this module is what the **driver claims**.
Nothing in this system has measured a true acoustic round trip -- that needs a
physical loopback cable -- so nothing here may be presented as measured, and
the field names and notes say so.

There is no ASIO host API in this PortAudio build, so the user's interface is
reachable only through the Windows APIs above. That is a build problem, not a
code problem, and it is surfaced as a note rather than hidden.
"""

from __future__ import annotations

from typing import Any

import sounddevice as sd

from tonehound.config import SAMPLE_RATE

# What each host API cost on the dev machine, duplex at 48 kHz. These are
# reported round trips, and they are stable enough per API to be worth telling
# the user before they try one, rather than after.
_HOST_API_NOTES = {
    "Windows WDM-KS": (
        True, "lowest latency here (4-20 ms reported) and it honours the "
              "latency setting, but it takes the device exclusively"),
    "Windows WASAPI": (
        False, "reports 47-55 ms round trip whatever latency you ask for; "
               "exclusive mode saves only about 2 ms"),
    "Windows DirectSound": (
        True, "9-33 ms with an explicit latency, but 125-141 ms on 'low'"),
    "MME": (
        True, "low reported latency but it dropped out constantly in testing "
              "(17-54 xruns in 5 s); avoid"),
    "ASIO": (True, "your interface's own driver, normally the best option"),
}

_PREFERENCE = ["Windows WDM-KS", "ASIO", "Windows WASAPI", "Windows DirectSound", "MME"]


def _supports_48k(index: int, channels: int, is_input: bool) -> bool:
    try:
        if is_input:
            sd.check_input_settings(device=index, channels=min(channels, 2),
                                    samplerate=SAMPLE_RATE)
        else:
            sd.check_output_settings(device=index, channels=min(channels, 2),
                                     samplerate=SAMPLE_RATE)
        return True
    except Exception:
        return False


def host_apis() -> list[dict[str, Any]]:
    out = []
    for i, api in enumerate(sd.query_hostapis()):
        name = str(api["name"])
        honours, note = _HOST_API_NOTES.get(name, (True, None))
        out.append({
            "index": i,
            "name": name,
            # PortAudio exposes exclusive mode only through WasapiSettings.
            "supports_exclusive": name == "Windows WASAPI",
            "honours_latency_hint": honours,
            "latency_note": note,
        })
    return out


def _device(index: int, dev: dict[str, Any], apis: list[dict[str, Any]],
            is_input: bool) -> dict[str, Any]:
    channels = int(dev["max_input_channels" if is_input else "max_output_channels"])
    latency = dev["default_low_input_latency" if is_input else "default_low_output_latency"]
    return {
        "index": index,
        "name": str(dev["name"]),
        "hostapi": int(dev["hostapi"]),
        "hostapi_name": apis[int(dev["hostapi"])]["name"],
        "max_channels": channels,
        "default_samplerate": float(dev["default_samplerate"]),
        "supports_48k": _supports_48k(index, channels, is_input),
        "low_latency_ms": float(latency) * 1000.0,
    }


def device_list(refresh: bool = False,
                probe: bool = True) -> dict[str, Any]:
    """The full ``DeviceList`` of the protocol.

    ``refresh`` terminates and re-initialises PortAudio, which is the only way
    to see a device plugged in after start. It briefly blocks and it invalidates
    every open stream, so it is behind an explicit button in the UI rather than
    on every panel open.
    """
    if refresh:
        sd._terminate()
        sd._initialize()

    apis = host_apis()
    devices = sd.query_devices()
    inputs, outputs = [], []
    for i, dev in enumerate(devices):
        if int(dev["max_input_channels"]) > 0:
            inputs.append(_device(i, dev, apis, True))
        if int(dev["max_output_channels"]) > 0:
            outputs.append(_device(i, dev, apis, False))

    default_in, default_out = sd.default.device
    return {
        "hostapis": apis,
        "inputs": inputs,
        "outputs": outputs,
        "default_input": int(default_in) if default_in is not None and default_in >= 0 else None,
        "default_output": int(default_out) if default_out is not None and default_out >= 0 else None,
        "recommended": recommend(apis, inputs, outputs, probe=probe),
    }


# A guitar arrives on a line or instrument input, never on a webcam or a
# virtual microphone endpoint, and WDM-KS in particular exposes a dozen of
# those at identical reported latency -- so without this nudge the first guess
# is whichever virtual device PortAudio happened to enumerate first.
_INPUT_HINTS = ("line", "instrument", "analog", "input")
_INPUT_PENALTIES = ("stereo mix", "what u hear", "loopback", "virtual", "voicemod",
                    "steam", "nvidia broadcast", "webcam", "rtx")
_OUTPUT_PENALTIES = ("voicemod", "steam", "nvidia broadcast", "dummy", "rtx",
                     "digital output", "spdif")

# Onboard codecs rank below a dedicated interface in both directions. Someone
# plugging a guitar in has an interface; someone who does not still gets these,
# one rank down, and the open-probe below has the final say either way.
_ONBOARD = ("realtek", "high definition audio", "hd audio", "sound mapper",
            "nvidia high definition")

# Words that appear on every endpoint and so identify nothing. What is left of
# a name after removing them is the product, which is how an input and an
# output belonging to the same physical interface are recognised.
_GENERIC = frozenset((
    "line", "in", "out", "input", "output", "microphone", "mic", "speaker",
    "speakers", "analog", "analogue", "digital", "audio", "hd", "high",
    "definition", "stereo", "wave", "device", "primary", "sound", "driver",
    "mapper", "the", "and", "port", "front", "rear", "left", "right", "ch"))

_MAX_PROBES = 12
"""Ceiling on open attempts across the whole recommendation. Measured at
0.5-75 ms each on this machine, so the worst case stays well under a second."""


def _score(name: str, penalties: tuple[str, ...], hints: tuple[str, ...] = ()) -> int:
    lower = name.lower()
    return (sum(-4 for p in penalties if p in lower)
            + sum(-3 for p in _ONBOARD if p in lower)
            + sum(2 for h in hints if h in lower))


def _identity(name: str) -> frozenset:
    """The tokens of a device name that name the *product* rather than the jack."""
    words = "".join(c if c.isalnum() else " " for c in name.lower()).split()
    return frozenset(w for w in words
                     if w not in _GENERIC and not w.isdigit() and len(w) > 1)


def can_open(input_device: int, output_device: int, block_size: int,
             latency_ms: float | None, exclusive: bool) -> bool:
    """Does PortAudio actually give us this duplex pair?

    This exists because nothing cheaper answers the question.
    ``check_input_settings`` passes for a Realtek line-in jack with nothing
    connected, and the duplex open then fails with ``Invalid device`` -- so a
    recommendation built from names and capability flags alone can, and on this
    machine did, name a pair that cannot be opened at all. Opening costs
    0.5-75 ms and is the only signal here that does not lie.

    A callback is passed because WDM-KS refuses a blocking stream outright
    ("Blocking API not supported yet"), which would otherwise make every
    WDM-KS pair look broken. The stream is opened and closed, never started.
    """
    def silent(indata: Any, outdata: Any, frames: int, time_info: Any,
               status: Any) -> None:
        outdata.fill(0)

    extra = None
    if exclusive:
        try:
            api = sd.query_hostapis(sd.query_devices(output_device)["hostapi"])
            if str(api["name"]) == "Windows WASAPI":
                extra = sd.WasapiSettings(exclusive=True)
        except Exception:
            return False
    try:
        stream = sd.Stream(
            samplerate=SAMPLE_RATE, blocksize=block_size,
            device=(input_device, output_device), channels=(1, 2),
            dtype="float32",
            latency="low" if latency_ms is None else float(latency_ms) / 1000.0,
            extra_settings=(None, extra) if extra is not None else None,
            callback=silent)
        stream.close()
        return True
    except Exception:
        return False


def recommend(apis: list[dict[str, Any]], inputs: list[dict[str, Any]],
              outputs: list[dict[str, Any]],
              probe: bool = True) -> dict[str, Any] | None:
    """Pick a starting point, and where possible one that has been opened once.

    The host-API choice is the measured part and it is the part worth ~40 ms.
    The *device* choice inside an API is a heuristic and on its own it is often
    wrong -- a virtual capture endpoint and a real interface look identical to
    PortAudio, and this machine lists eleven WDM-KS inputs at the same reported
    latency. So the heuristic only *orders* the candidates and then each pair is
    opened until one works. That turns "usually right" into "known to open",
    which is the difference between an app that plays on first launch and one
    that shows an error over a pair it preselected itself.

    ``probe=False`` skips the opening and must be passed while a stream is
    running: a WDM-KS probe takes the device exclusively and would interrupt
    the audio it is meant to be helping set up.
    """
    by_api: dict[int, str] = {a["index"]: a["name"] for a in apis}
    ranked = sorted(
        by_api,
        key=lambda i: (_PREFERENCE.index(by_api[i]) if by_api[i] in _PREFERENCE
                       else len(_PREFERENCE)))

    budget = _MAX_PROBES
    fallback: dict[str, Any] | None = None

    for api_index in ranked:
        ins = [d for d in inputs if d["hostapi"] == api_index and d["supports_48k"]]
        outs = [d for d in outputs if d["hostapi"] == api_index and d["supports_48k"]]
        if not ins or not outs:
            continue
        name = by_api[api_index]
        honours = _HOST_API_NOTES.get(name, (True, None))[0]
        # Asking WDM-KS for 5 ms yields ~10 ms reported and no dropouts; asking
        # an API that ignores the hint costs nothing either way.
        latency_ms = 5.0 if honours else None
        exclusive = name == "Windows WASAPI"

        pairs = []
        for source in ins:
            for sink in outs:
                score = (_score(source["name"], _INPUT_PENALTIES, _INPUT_HINTS)
                         + _score(sink["name"], _OUTPUT_PENALTIES)
                         # One physical interface carrying both directions is
                         # the strongest cheap signal that this is the box the
                         # guitar is actually plugged into.
                         + (3 if _identity(source["name"]) & _identity(sink["name"])
                            else 0))
                pairs.append((score,
                              -source["low_latency_ms"] - sink["low_latency_ms"],
                              source, sink))
        pairs.sort(key=lambda pair: (pair[0], pair[1]), reverse=True)

        def result(source: dict[str, Any], sink: dict[str, Any], verified: bool,
                   _api=api_index, _name=name, _lat=latency_ms,
                   _exc=exclusive) -> dict[str, Any]:
            return {
                "input": source["index"], "output": sink["index"],
                "hostapi": _api, "block_size": 512,
                "latency_ms": _lat, "exclusive": _exc,
                "reason": (f"{_name} is the lowest-latency host API available here"
                           + (" and this pair was opened once to check it works"
                              if verified else
                              "; the devices are a guess -- check them")),
            }

        if fallback is None and pairs:
            fallback = result(pairs[0][2], pairs[0][3], False)
        if not probe:
            continue

        for _s, _l, source, sink in pairs:
            if budget <= 0:
                break
            budget -= 1
            if can_open(source["index"], sink["index"], 512, latency_ms, exclusive):
                return result(source, sink, True)

    return fallback
