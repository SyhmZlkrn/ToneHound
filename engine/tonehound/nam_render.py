"""Self-contained ``.nam`` loading and offline rendering.

The indexer has to read whatever a user's profile folder happens to contain --
captures exported over several years by several tools -- so it owns its reader
instead of tracking someone else's config schema. neural-amp-modeler 0.13.0
refuses every legacy (``"version": "0.5.0"``) capture in ``assets/dev_profiles``
with::

    KeyError: "Each layer array config must include a 'head' object with
               'out_channels', 'kernel_size', and 'bias'"

because the modern schema nests per-layer-array fields that the legacy schema
stored flat. Only the *config* changed between generations; the flat weights
vector did not. So this module normalises legacy configs onto the modern shape
and runs one renderer over both.

The weight layout is the exporter's concatenation order, and the loader asserts
it consumed exactly ``len(weights)``. That catches a *truncated* layout but not a
*permuted* one: the per-layer blocks sum to the same total in any order, so
reading layer1x1 before the mixer -- or a conv kernel transposed -- consumes
every weight and renders plausible, wrong audio that nothing downstream can
catch. The real guard is therefore external: ``test_nam_render_parity.py``
compares this renderer sample-for-sample against neural-amp-modeler itself,
which is the only offline oracle for the layout.

Offline rendering uses float64 for headroom against the float32 reference.
CPU is the default; opt-in CUDA rendering keeps the same precision and uses
overlapping chunks to bound activation memory when indexing long DI takes.
"""

from __future__ import annotations

import copy
import json
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from . import nam_stream
from .config import SAMPLE_RATE

DEFAULT_SAMPLE_RATE = SAMPLE_RATE
"""Assumed rate for the many captures that carry no ``sample_rate`` key."""

_ACTIVATION_FN = Callable[[torch.Tensor], torch.Tensor]

# Every FiLM slot a modern layer array can declare, in the order the exporter
# writes their weights. All are inactive in every file seen so far; an active
# one is refused rather than skipped, because skipping would silently drop a
# conditioning path *and* desynchronise the weight cursor.
_FILM_KEYS = (
    "conv_pre_film",
    "conv_post_film",
    "input_mixin_pre_film",
    "input_mixin_post_film",
    "activation_pre_film",
    "activation_post_film",
    "layer1x1_post_film",
    "head1x1_post_film",
)


class UnsupportedArchitecture(RuntimeError):
    """A file this renderer refuses to guess at.

    Raised for unknown architecture strings and for renderable-looking configs
    that use a feature this module does not implement. Both mean the same thing
    to the indexer: skip and report, never render.
    """

    def __init__(self, architecture: str, detail: str) -> None:
        super().__init__(
            f"unsupported .nam model (architecture {architecture!r}): {detail}. "
            "The plugin's C++ NeuralAmpModelerCore may well play this file; the "
            "offline indexer must skip it rather than render it wrongly."
        )
        self.architecture = architecture
        self.detail = detail


class WeightCountMismatch(ValueError):
    """The parsed layout did not land exactly on the end of the weights list."""

    def __init__(self, consumed: int, available: int) -> None:
        super().__init__(
            f"weight layout consumed {consumed} weights but the file holds "
            f"{available} (expected consumed == available); the config was read "
            "with the wrong layout, so any render from it would be wrong"
        )
        self.consumed = consumed
        self.available = available


class _Weights:
    """Sequential reader over the flat weights vector.

    Every tensor is taken in export order and the cursor is checked at the end,
    so a layout error surfaces as a count mismatch rather than as audio.
    """

    def __init__(self, weights: Sequence[float]) -> None:
        self._w = np.asarray(weights, dtype=np.float64)
        self._i = 0

    def take(self, *shape: int) -> torch.Tensor:
        n = int(np.prod(shape)) if shape else 1
        if self._i + n > self._w.size:
            raise WeightCountMismatch(self._i + n, self._w.size)
        out = torch.from_numpy(self._w[self._i : self._i + n].reshape(shape).copy())
        self._i += n
        return out

    def scalar(self) -> float:
        if self._i >= self._w.size:
            raise WeightCountMismatch(self._i + 1, self._w.size)
        value = float(self._w[self._i])
        self._i += 1
        return value

    @property
    def consumed(self) -> int:
        return self._i

    def finish(self) -> None:
        if self._i != self._w.size:
            raise WeightCountMismatch(self._i, self._w.size)


# --------------------------------------------------------------------------- #
# Activations
# --------------------------------------------------------------------------- #

def _simple_activation(spec: str | dict[str, Any]) -> _ACTIVATION_FN:
    """Build one activation from either schema's spelling.

    The parsing and the arithmetic both live in ``nam_stream`` so that the
    offline and streaming paths cannot drift: there is one implementation of
    every activation, and this is a closure over it.
    """
    return _activation_fn(nam_stream.parse_activation(spec))


def _activation_fn(spec: nam_stream.ActSpec, device: str = "cpu") -> _ACTIVATION_FN:
    w = spec.tensor().to(device)
    return lambda x: nam_stream.apply_activation(
        x, spec.code, spec.p0, spec.p1, spec.p2, spec.p3, w)


_GATING_CODES = {"none": nam_stream.GATING_NONE,
                 "gated": nam_stream.GATING_GATED,
                 "blended": nam_stream.GATING_BLENDED}


def _layer_activation_specs(primary: str | dict, gating_mode: str,
                            secondary: str | dict | None,
                            ) -> tuple[int, nam_stream.ActSpec, nam_stream.ActSpec]:
    """Read the (primary, gating, secondary) triple as one unit.

    A gated layer's exported ``activation`` entry holds *only* the primary, so
    reading it alone yields a correct weight count and silently wrong audio.
    Keeping the triple together in one function is what stops that.
    """
    if gating_mode not in _GATING_CODES:
        raise ValueError(f"unknown gating_mode {gating_mode!r}")
    gating = _GATING_CODES[gating_mode]
    prim = nam_stream.parse_activation(primary)
    if gating == nam_stream.GATING_NONE:
        return gating, prim, nam_stream.IDENTITY
    if secondary is None:
        raise ValueError(f"gating_mode {gating_mode!r} without a secondary activation")
    return gating, prim, nam_stream.parse_activation(secondary)


def _layer_activation(gating: int, prim: nam_stream.ActSpec,
                      sec: nam_stream.ActSpec, device: str = "cpu") -> _ACTIVATION_FN:
    """Fold gating into a single channel-halving callable."""
    pw, sw = prim.tensor().to(device), sec.tensor().to(device)
    return lambda x: nam_stream.apply_gated(
        x, gating,
        prim.code, prim.p0, prim.p1, prim.p2, prim.p3, pw,
        sec.code, sec.p0, sec.p1, sec.p2, sec.p3, sw)


# --------------------------------------------------------------------------- #
# Convolutions
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _Conv:
    """A valid (unpadded) 1-D convolution with its weights already loaded."""

    weight: torch.Tensor  # (out, in // groups, kernel_size)
    bias: torch.Tensor | None
    dilation: int = 1
    groups: int = 1

    def to(self, device: str) -> "_Conv":
        return replace(self, weight=self.weight.to(device),
                       bias=None if self.bias is None else self.bias.to(device))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, self.weight, self.bias, dilation=self.dilation,
                        groups=self.groups)


def _read_conv(w: _Weights, out_channels: int, in_channels: int, kernel_size: int,
               *, bias: bool, dilation: int = 1, groups: int = 1) -> _Conv:
    if in_channels % groups or out_channels % groups:
        raise ValueError(
            f"conv channels {in_channels}->{out_channels} not divisible by groups {groups}"
        )
    weight = w.take(out_channels, in_channels // groups, kernel_size)
    return _Conv(weight, w.take(out_channels) if bias else None, dilation, groups)


# --------------------------------------------------------------------------- #
# WaveNet
# --------------------------------------------------------------------------- #

@dataclass
class _Layer:
    conv: _Conv           # dilated, carries the only bias of the pre-activation sum
    mixer: _Conv          # 1x1 over the condition, no bias
    activation: _ACTIVATION_FN
    layer1x1: _Conv | None
    head1x1: _Conv | None
    # The activation kept as data as well as a closure, so ``nam_stream`` can
    # rebuild it inside a TorchScript module (which cannot hold a closure).
    gating: int = nam_stream.GATING_NONE
    act_spec: nam_stream.ActSpec = nam_stream.IDENTITY
    sec_spec: nam_stream.ActSpec = nam_stream.IDENTITY

    def __call__(self, x: torch.Tensor, c: torch.Tensor,
                 head_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.conv(x)
        # The condition is convolved at full length and only then trimmed, so the
        # sum stays right-aligned when arrays are chained.
        m = self.mixer(c)[..., -z.shape[-1] :]
        n = min(z.shape[-1], m.shape[-1])
        post = self.activation(z[..., -n:] + m[..., -n:])

        # The head branch taps post-activation, *before* layer1x1.
        head_term = post if self.head1x1 is None else self.head1x1(post)
        head_term = head_term[..., -head_length:]

        layer_out = post if self.layer1x1 is None else self.layer1x1(post)
        residual = x[..., -layer_out.shape[-1] :] + layer_out
        return residual, head_term


@dataclass
class _LayerArray:
    rechannel: _Conv
    layers: list[_Layer]
    head_rechannel: _Conv
    receptive_field: int
    receptive_field_no_head: int
    head_channels: int
    head_term_channels: int

    def __call__(self, x: torch.Tensor, c: torch.Tensor,
                 head_input: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        available = min(x.shape[-1], c.shape[-1])
        out_length = available - (self.receptive_field - 1)
        head_length = available - (self.receptive_field_no_head - 1)

        x = self.rechannel(x)
        for layer in self.layers:
            x, head_term = layer(x, c, head_length)
            head_input = (head_term if head_input is None
                          else head_input[..., -head_length:] + head_term)
        head_out = self.head_rechannel(head_input)
        return head_out, x[..., -out_length:]


def _normalized_layer_array(cfg: dict[str, Any], index: int) -> dict[str, Any]:
    """Map either schema generation onto one canonical description.

    Dispatch is on the *shape* of the layer array, not on the file's version
    string: version strings drift between exporters and the legacy files carry
    no metadata at all, while the flat-vs-nested split is exactly what the
    package itself trips on.
    """
    modern = isinstance(cfg.get("head"), dict)
    legacy = "head_size" in cfg
    if modern == legacy:
        raise ValueError(
            f"layer array {index} matches neither schema: expected a nested 'head' "
            "object (modern) or a flat 'head_size' (legacy 0.5.0), got keys "
            f"{sorted(cfg)}"
        )

    n_layers = len(cfg["dilations"])
    if legacy:
        kernel_sizes = [int(cfg["kernel_size"])] * n_layers
        gating = "gated" if cfg.get("gated", False) else "none"
        gating_modes = [gating] * n_layers
        secondary = [{"type": "Sigmoid"} if gating == "gated" else None] * n_layers
        activations = [cfg["activation"]] * n_layers
        head = {"out_channels": int(cfg["head_size"]), "kernel_size": 1,
                "bias": bool(cfg["head_bias"])}
        head1x1 = {"active": False}
        layer1x1 = {"active": True, "groups": 1}
    else:
        raw_ks = cfg.get("kernel_sizes", cfg.get("kernel_size"))
        if raw_ks is None:
            raise ValueError(f"layer array {index} has neither kernel_sizes nor kernel_size")
        kernel_sizes = ([int(raw_ks)] * n_layers if isinstance(raw_ks, int)
                        else [int(k) for k in raw_ks])
        activations = cfg["activation"]
        if isinstance(activations, (str, dict)):
            activations = [activations] * n_layers
        gating_modes = cfg.get("gating_mode") or ["none"] * n_layers
        secondary = cfg.get("secondary_activation") or [None] * n_layers
        head = cfg["head"]
        head1x1 = cfg.get("head1x1") or {"active": False}
        layer1x1 = cfg.get("layer1x1") or {"active": True, "groups": 1}

    # Read the capability flags out of either generation: a legacy file has no
    # business carrying them, but refusing an unexpected one beats ignoring it.
    film = {k: cfg[k] for k in _FILM_KEYS if k in cfg}
    slimmable = cfg.get("slimmable")
    packing = cfg.get("packing")

    if not (len(kernel_sizes) == len(activations) == len(gating_modes)
            == len(secondary) == n_layers):
        raise ValueError(
            f"layer array {index}: per-layer lists disagree with {n_layers} dilations"
        )

    channels = int(cfg["channels"])
    return {
        "input_size": int(cfg["input_size"]),
        "condition_size": int(cfg["condition_size"]),
        "channels": channels,
        "bottleneck": int(cfg.get("bottleneck", channels)),
        "dilations": [int(d) for d in cfg["dilations"]],
        "kernel_sizes": kernel_sizes,
        "activations": activations,
        "gating_modes": list(gating_modes),
        "secondary": list(secondary),
        "head": head,
        "head1x1": head1x1,
        "layer1x1": layer1x1,
        "groups_input": int(cfg.get("groups_input", 1)),
        "groups_input_mixin": int(cfg.get("groups_input_mixin", 1)),
        "film": film,
        "slimmable": slimmable,
        "packing": packing,
    }


def _reject_unimplemented(spec: dict[str, Any], index: int) -> None:
    if spec["slimmable"] is not None:
        raise UnsupportedArchitecture("WaveNet", f"layer array {index} is slimmable")
    if spec["packing"] is not None:
        raise UnsupportedArchitecture("WaveNet", f"layer array {index} is packed")
    for key, block in spec["film"].items():
        if isinstance(block, dict) and block.get("active", False):
            raise UnsupportedArchitecture(
                "WaveNet", f"layer array {index} has FiLM conditioning active ({key})"
            )


def _build_layer_array(spec: dict[str, Any], w: _Weights, index: int) -> _LayerArray:
    _reject_unimplemented(spec, index)

    channels = spec["channels"]
    bottleneck = spec["bottleneck"]
    condition_size = spec["condition_size"]
    layer1x1_active = spec["layer1x1"].get("active", True)
    head1x1_active = spec["head1x1"].get("active", False)

    rechannel = _read_conv(w, channels, spec["input_size"], 1, bias=False)

    layers: list[_Layer] = []
    for k, dilation, act, gating, sec in zip(
        spec["kernel_sizes"], spec["dilations"], spec["activations"],
        spec["gating_modes"], spec["secondary"],
    ):
        mid = 2 * bottleneck if gating != "none" else bottleneck
        conv = _read_conv(w, mid, channels, k, bias=True, dilation=dilation,
                          groups=spec["groups_input"])
        mixer = _read_conv(w, mid, condition_size, 1, bias=False,
                           groups=spec["groups_input_mixin"])
        layer1x1 = (_read_conv(w, channels, bottleneck, 1, bias=True,
                               groups=spec["layer1x1"].get("groups", 1))
                    if layer1x1_active else None)
        head1x1 = (_read_conv(w, int(spec["head1x1"]["out_channels"]), bottleneck, 1,
                              bias=True, groups=spec["head1x1"].get("groups", 1))
                   if head1x1_active else None)
        gating_code, prim_spec, sec_spec = _layer_activation_specs(act, gating, sec)
        layers.append(_Layer(conv, mixer,
                             _layer_activation(gating_code, prim_spec, sec_spec),
                             layer1x1, head1x1, gating_code, prim_spec, sec_spec))

    # When head1x1 is inactive its exported out_channels/groups are placeholders
    # (the C++ core requires non-null fields there) and must not size anything.
    head_term_channels = (int(spec["head1x1"]["out_channels"]) if head1x1_active
                          else bottleneck)
    head_cfg = spec["head"]
    head_rechannel = _read_conv(w, int(head_cfg["out_channels"]), head_term_channels,
                                int(head_cfg["kernel_size"]),
                                bias=bool(head_cfg["bias"]))

    rf_no_head = 1 + sum((k - 1) * d for k, d in zip(spec["kernel_sizes"], spec["dilations"]))
    return _LayerArray(
        rechannel=rechannel,
        layers=layers,
        head_rechannel=head_rechannel,
        receptive_field=rf_no_head + int(head_cfg["kernel_size"]) - 1,
        receptive_field_no_head=rf_no_head,
        head_channels=int(head_cfg["out_channels"]),
        head_term_channels=head_term_channels,
    )


@dataclass
class _Head:
    """The optional network head: repeated (activation, valid conv) blocks.

    One activation is shared by every block, so it is stored once -- as a
    callable for the offline path and as a spec for the streaming twin.
    """

    activation: _ACTIVATION_FN
    act_spec: nam_stream.ActSpec
    convs: list[_Conv]
    receptive_field: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(self.activation(x))
        return x


def _build_head(cfg: dict[str, Any], in_channels: int, w: _Weights) -> _Head:
    kernel_sizes = [int(k) for k in cfg["kernel_sizes"]]
    if not kernel_sizes:
        raise ValueError("network head kernel_sizes must be non-empty")
    act_spec = nam_stream.parse_activation(cfg["activation"])
    activation = _activation_fn(act_spec)
    channels = int(cfg["channels"])
    out_channels = int(cfg["out_channels"])

    convs = []
    cin = in_channels
    for i, k in enumerate(kernel_sizes):
        cout = channels if i != len(kernel_sizes) - 1 else out_channels
        convs.append(_read_conv(w, cout, cin, k, bias=True))
        cin = channels
    return _Head(activation, act_spec, convs, 1 + sum(k - 1 for k in kernel_sizes))


class _WaveNet:
    def __init__(self, config: dict[str, Any], w: _Weights) -> None:
        if config.get("condition_dsp") is not None:
            # Its weights live in a separate nested blob, so the outer count check
            # still passes -- the package's own reader drops it and renders wrong.
            raise UnsupportedArchitecture("WaveNet", "config carries a condition_dsp")

        specs = [_normalized_layer_array(cfg, i) for i, cfg in enumerate(config["layers"])]
        if not specs:
            raise ValueError("WaveNet config has no layer arrays")

        self.arrays = [_build_layer_array(s, w, i) for i, s in enumerate(specs)]
        for i, (prev, cur) in enumerate(zip(self.arrays, self.arrays[1:]), start=1):
            # The head accumulator is threaded across arrays, so widths must line
            # up; otherwise this surfaces as an opaque shape error mid-forward.
            if cur.head_term_channels != prev.head_channels:
                raise ValueError(
                    f"layer array {i} contributes {cur.head_term_channels} head channels "
                    f"but array {i - 1} hands over {prev.head_channels}"
                )
            if specs[i]["input_size"] != specs[i - 1]["channels"]:
                raise ValueError(
                    f"layer array {i} takes {specs[i]['input_size']} input channels "
                    f"but array {i - 1} outputs {specs[i - 1]['channels']}"
                )

        head_cfg = config.get("head")
        self.head = (None if head_cfg is None
                     else _build_head(head_cfg, self.arrays[-1].head_channels, w))

        # head_scale is stored twice, in the config and as the final weight, and
        # the two are allowed to disagree: an export hook that compensates output
        # level rewrites weights[-1] without touching the config copy. The weight
        # is authoritative -- that is what the reference reader and the C++ core
        # play -- so a disagreement is recorded for the indexer to log, never
        # refused. Cursor drift is caught by the exact count check instead.
        self.head_scale = w.scalar()
        config_scale = config.get("head_scale")
        self.head_scale_config_mismatch: float | None = None
        if config_scale is not None:
            # Relative, because the weight is a float32 round of the config value
            # and that rounding error grows with magnitude.
            if abs(self.head_scale - float(config_scale)) > 1e-5 * max(1.0, abs(float(config_scale))):
                self.head_scale_config_mismatch = float(config_scale)

        self.input_channels = specs[0]["input_size"]
        self.condition_channels = specs[0]["condition_size"]
        self.receptive_field = 1 + sum(a.receptive_field - 1 for a in self.arrays)
        if self.head is not None:
            self.receptive_field += self.head.receptive_field - 1

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        c = x  # legacy and plain modern files condition on the raw input
        y, head_input = x, None
        for array in self.arrays:
            head_input, y = array(y, c, head_input)
        head_input = self.head_scale * head_input
        return head_input if self.head is None else self.head(head_input)


# --------------------------------------------------------------------------- #
# LSTM / Linear
# --------------------------------------------------------------------------- #

class _LSTM:
    """Stacked unidirectional LSTM plus a Linear(hidden, 1) head.

    Per layer the file stores ``cat([W_ih, W_hh], dim=1)`` -- a *horizontal*
    concatenation flattened row-major, so the two matrices interleave row by row
    and must be recovered by column slicing, not by reading two blocks. The
    stored initial (h, c) are the trained model's settled DC state, not zeros.
    """

    def __init__(self, config: dict[str, Any], w: _Weights) -> None:
        self.input_channels = int(config["input_size"])
        hidden = int(config["hidden_size"])
        num_layers = int(config["num_layers"])
        self.hidden_size = hidden
        self.receptive_field = 1

        self.cells: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for layer in range(num_layers):
            d = self.input_channels if layer == 0 else hidden
            fused = w.take(4 * hidden, d + hidden).numpy()
            bias = w.take(4 * hidden).numpy()
            h0 = w.take(hidden).numpy()
            c0 = w.take(hidden).numpy()
            self.cells.append((fused[:, :d].copy(), fused[:, d:].copy(), bias, h0, c0))
        self.head_weight = w.take(1, hidden).numpy()
        self.head_bias = w.take(1).numpy()

    def initial_state(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """The trained settled DC state, which is what the file stores -- not zeros."""
        return [(h0.copy(), c0.copy()) for *_, h0, c0 in self.cells]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = self.run(x[0].numpy(), self.initial_state())
        return torch.from_numpy(out).reshape(1, 1, len(out))

    def run(self, signal: np.ndarray,
            state: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Advance the recurrence over ``signal`` (input_size, L), mutating ``state``.

        Offline rendering and block streaming share this loop, so a streamed
        LSTM cannot diverge from a rendered one: streaming an LSTM *is* just
        not resetting the state between calls.
        """
        n = signal.shape[1]
        out = np.empty(n, dtype=np.float64)
        h_size = self.hidden_size
        for t in range(n):
            v = signal[:, t]
            for i, (w_ih, w_hh, bias, _, _) in enumerate(self.cells):
                h, c = state[i]
                z = w_ih @ v + w_hh @ h + bias
                gate_i = _sigmoid(z[:h_size])
                gate_f = _sigmoid(z[h_size : 2 * h_size])
                cell = np.tanh(z[2 * h_size : 3 * h_size])
                gate_o = _sigmoid(z[3 * h_size :])
                c = gate_f * c + gate_i * cell
                h = gate_o * np.tanh(c)
                state[i] = (h, c)
                v = h
            out[t] = self.head_weight[0] @ v + self.head_bias[0]
        return out


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # overflow-free logistic


class _Linear:
    def __init__(self, config: dict[str, Any], w: _Weights) -> None:
        self.receptive_field = int(config["receptive_field"])
        self.input_channels = 1
        self.conv = _read_conv(w, 1, 1, self.receptive_field,
                               bias=bool(config.get("bias", False)))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

_BUILDERS = {"WaveNet": _WaveNet, "LSTM": _LSTM, "Linear": _Linear}

DEFAULT_RENDER_CHUNK = 65536
"""Output samples per CUDA chunk; each also reads receptive_field - 1 history."""


def render_device(requested: str) -> str:
    """Resolve an offline device, allowing a CUDA request on a CPU-only host."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"render device must be cpu, cuda[:index] or auto, got {requested!r}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            return "cpu"
        if device.index is not None and device.index >= torch.cuda.device_count():
            return "cpu"
    return str(device)


def _device_net(net: Any, device: str) -> Any:
    """Copy the offline graph, including activation tensors, leaving CPU/stream state intact."""
    out = copy.copy(net)
    if isinstance(net, _Linear):
        out.conv = net.conv.to(device)
    elif isinstance(net, _WaveNet):
        out.arrays = []
        for array in net.arrays:
            layers = [replace(
                layer, conv=layer.conv.to(device), mixer=layer.mixer.to(device),
                layer1x1=None if layer.layer1x1 is None else layer.layer1x1.to(device),
                head1x1=None if layer.head1x1 is None else layer.head1x1.to(device),
                activation=_layer_activation(layer.gating, layer.act_spec,
                                             layer.sec_spec, device))
                for layer in array.layers]
            out.arrays.append(replace(array, rechannel=array.rechannel.to(device),
                                      head_rechannel=array.head_rechannel.to(device),
                                      layers=layers))
        if net.head is not None:
            out.head = replace(net.head,
                               activation=_activation_fn(net.head.act_spec, device),
                               convs=[conv.to(device) for conv in net.head.convs])
    else:
        raise TypeError("CUDA rendering supports WaveNet and Linear captures")
    return out


class _TorchStream:
    """Streaming state for the convolutional architectures.

    Holds one TorchScripted copy of the network with per-convolution history
    buffers. TorchScript is worth ~25% of the raw time, but the reason it is not
    optional is the GIL: eager, one block is ~100 Python op dispatches that each
    take and release it, and the audio callback shares its interpreter with a
    telemetry thread. Scripted, the whole forward releases the GIL once. If a
    future torch breaks scripting the eager module still runs -- slower and more
    exposed to interpreter jitter, but not dead -- so the failure is recorded,
    not raised.
    """

    def __init__(self, net: Any, torch_dtype: torch.dtype, in_channels: int,
                 block_size: int) -> None:
        module = nam_stream.build(net).to(torch_dtype).eval()
        self.scripted = True
        self.script_error: str | None = None
        try:
            self._module = torch.jit.script(module)
        except Exception as exc:  # pragma: no cover - version-dependent
            self._module = module
            self.scripted = False
            self.script_error = f"{type(exc).__name__}: {exc}"
        self._block = block_size
        # Pre-allocated so a steady-state block costs one memcpy and no
        # allocation; only priming, which is not real-time, takes another path.
        self._in = torch.zeros(1, in_channels, block_size, dtype=torch_dtype)
        self._in_np = self._in.numpy()[0]

    def reset(self) -> None:
        nam_stream.reset_history(self._module)

    def run(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] == self._block:
            np.copyto(self._in_np, x)
            t = self._in
        else:
            t = torch.from_numpy(np.ascontiguousarray(x))[None]
        with torch.no_grad():
            y = self._module(t)
        return y.numpy().reshape(-1)


class _LSTMStream:
    """Streaming state for LSTM captures.

    An LSTM's receptive field is 1, so there is no history to keep and no
    priming to do: streaming is simply not resetting ``(h, c)`` between blocks.
    ``reset`` restores the file's stored initial state, which is the trained
    settled DC state rather than zeros. The recurrence itself is the same
    ``_LSTM.run`` the offline render calls, so the two cannot drift.
    """

    scripted = False
    script_error = None

    def __init__(self, net: "_LSTM", np_dtype: np.dtype) -> None:
        self._net = net
        self._dtype = np_dtype
        self._state = net.initial_state()

    def reset(self) -> None:
        self._state = self._net.initial_state()

    def run(self, x: np.ndarray) -> np.ndarray:
        out = self._net.run(np.asarray(x, dtype=np.float64), self._state)
        return out.astype(self._dtype, copy=False)


_STREAM_DTYPES = {"float32": (np.float32, torch.float32),
                  "float64": (np.float64, torch.float64)}


class NamModel:
    """A loaded ``.nam`` capture, ready to render offline."""

    def __init__(self, data: dict[str, Any], source: str | None = None) -> None:
        self.source = source
        self.architecture = str(data.get("architecture", ""))
        self.version = data.get("version")
        self.metadata: dict[str, Any] = data.get("metadata") or {}

        raw_rate = data.get("sample_rate")
        self.sample_rate = (DEFAULT_SAMPLE_RATE if raw_rate is None
                            else int(round(float(raw_rate))))

        builder = _BUILDERS.get(self.architecture)
        if builder is None:
            raise UnsupportedArchitecture(
                self.architecture,
                f"this loader implements {', '.join(sorted(_BUILDERS))}",
            )

        weights = _Weights(data["weights"])
        with torch.no_grad():
            self._net = builder(data["config"], weights)
        weights.finish()
        self.weight_count = weights.consumed

        self._stream: Any = None
        self._stream_block: int | None = None
        self._stream_dtype: str = "float32"
        self._scratch: np.ndarray | None = None
        self.last_render_device: str | None = None
        self.last_render_note: str | None = None

    @property
    def receptive_field(self) -> int:
        return self._net.receptive_field

    @property
    def head_scale_config_mismatch(self) -> float | None:
        """The config's ``head_scale`` when it disagrees with the trailing weight.

        ``None`` for the usual case and for architectures that have no head scale.
        The render always uses the weight; this is here so the indexer can report
        a file whose two copies drifted apart rather than silently accepting it.
        """
        return getattr(self._net, "head_scale_config_mismatch", None)

    def render(self, x: np.ndarray, pad_start: bool = True, *,
               device: str = "cpu", chunk_samples: int = DEFAULT_RENDER_CHUNK) -> np.ndarray:
        """Run a signal through the capture.

        With ``pad_start`` the input is prefixed with ``receptive_field - 1``
        zeros so the output is sample-aligned and the same length as the input,
        which is what the plugin does at the start of a stream. Without it the
        output is shorter by exactly that much.

        ``device="auto"`` or ``"cuda"`` opts into float64 CUDA for WaveNet and
        Linear. LSTM retains its CPU recurrence. Unavailable CUDA or a CUDA
        runtime failure retries on CPU, reported in ``last_render_note`` and
        a warning. Streaming state is independent of this offline device.
        """
        self.last_render_device = None
        self.last_render_note = None
        resolved = render_device(device)
        if not isinstance(chunk_samples, (int, np.integer)) or chunk_samples < 1:
            raise ValueError("chunk_samples must be a positive integer")
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        elif arr.ndim != 2:
            raise ValueError(f"expected a (L,) or (channels, L) signal, got {arr.shape}")
        if arr.shape[0] != self._net.input_channels:
            raise ValueError(
                f"model takes {self._net.input_channels} input channel(s), got {arr.shape[0]}"
            )

        pad = self.receptive_field - 1 if pad_start else 0
        if arr.shape[1] + pad < self.receptive_field:
            raise ValueError(
                f"input of {arr.shape[1]} samples is shorter than the "
                f"{self.receptive_field}-sample receptive field"
            )
        if pad:
            arr = np.concatenate([np.zeros((arr.shape[0], pad)), arr], axis=1)

        if resolved.startswith("cuda") and isinstance(self._net, _LSTM):
            self.last_render_note = "LSTM offline rendering uses the CPU recurrence"
            resolved = "cpu"
        elif resolved == "cpu" and device not in {"cpu", "auto"}:
            self.last_render_note = f"render device {device} is unavailable; using CPU float64"

        if resolved.startswith("cuda"):
            try:
                out = self._render_cuda(arr, resolved, int(chunk_samples))
                self.last_render_device = resolved
                return out
            except RuntimeError as exc:
                self.last_render_note = f"CUDA render failed ({exc}); retrying on CPU float64"
            # Outside the except block so the traceback releases CUDA tensors
            # before reclaiming allocator memory and attempting the CPU render.
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass  # a broken CUDA context must not prevent the CPU retry
        if self.last_render_note:
            warnings.warn(self.last_render_note, RuntimeWarning, stacklevel=2)

        with torch.no_grad():
            y = self._net(torch.from_numpy(arr)[None])
        if y.shape[1] != 1:
            raise ValueError(f"model emits {y.shape[1]} output channels, expected 1")
        self.last_render_device = "cpu"
        return y[0, 0].numpy()

    def _render_cuda(self, arr: np.ndarray, device: str,
                     chunk_samples: int) -> np.ndarray:
        net = _device_net(self._net, device)
        history = self.receptive_field - 1
        n = arr.shape[1] - history
        out = np.empty(n, dtype=np.float64)
        with torch.no_grad():
            for start in range(0, n, chunk_samples):
                end = min(start + chunk_samples, n)
                # The full receptive-field overlap reproduces all intermediate
                # bias/activation history, including at chunk boundaries.
                x = torch.from_numpy(np.ascontiguousarray(arr[:, start:end + history]))
                y = net(x.to(device)[None])
                if y.shape[1] != 1:
                    raise ValueError(f"model emits {y.shape[1]} output channels, expected 1")
                out[start:end] = y[0, 0].cpu().numpy()
        return out


    # ---- streaming -------------------------------------------------------
    #
    # ``render`` needs the whole signal; a duplex callback has 512 samples and
    # ~10 ms to work in. The streaming path keeps each convolution's own input
    # history, so a block costs O(block) instead of O(block + receptive_field)
    # and is arithmetically *identical* to ``render(x, pad_start=True)`` rather
    # than an approximation of it. See ``nam_stream`` for the mechanism and for
    # what priming buys.

    @property
    def streamable(self) -> bool:
        """Whether this capture can be played block at a time."""
        if self._net.input_channels != 1:
            return False
        return isinstance(self._net, (_LSTM, _Linear, _WaveNet))

    @property
    def stream_block_size(self) -> int | None:
        """Block size ``prepare_stream`` was last called with, or ``None``."""
        return self._stream_block

    @property
    def stream_dtype(self) -> str:
        """``"float32"`` or ``"float64"``; the default until prepared."""
        return self._stream_dtype

    @property
    def stream_scripted(self) -> bool:
        """Whether the streaming graph is TorchScripted (see ``_TorchStream``)."""
        return bool(self._stream is not None and self._stream.scripted)

    def prepare_stream(self, block_size: int, *, dtype: str = "float32") -> None:
        """Allocate streaming state for a fixed block size.

        Allocates, builds a second copy of the weights and TorchScripts it:
        tens of milliseconds. **Never call this from an audio callback.** Calls
        ``reset_stream`` on the way out, and is idempotent for the same
        arguments (it re-resets but does not rebuild).
        """
        if not self.streamable:
            raise UnsupportedArchitecture(
                self.architecture,
                "no streaming path for this network "
                f"({self._net.input_channels} input channel(s))")
        if not isinstance(block_size, (int, np.integer)) or block_size < 1:
            raise ValueError(f"block_size must be a positive int, got {block_size!r}")
        if dtype not in _STREAM_DTYPES:
            raise ValueError(f"dtype must be 'float32' or 'float64', got {dtype!r}")
        block_size = int(block_size)

        if self._stream is not None and (block_size, dtype) == (self._stream_block,
                                                                self._stream_dtype):
            self.reset_stream()
            return

        np_dtype, torch_dtype = _STREAM_DTYPES[dtype]
        if isinstance(self._net, _LSTM):
            stream: Any = _LSTMStream(self._net, np.dtype(np_dtype))
        else:
            stream = _TorchStream(self._net, torch_dtype,
                                  self._net.input_channels, block_size)
        self._stream = stream
        self._stream_block = block_size
        self._stream_dtype = dtype
        self._scratch = np.zeros((self._net.input_channels, block_size), dtype=np_dtype)
        self.reset_stream()

    def reset_stream(self) -> None:
        """Return the streaming state to its start-of-signal condition.

        Zeroes every history buffer and *then* pushes at least
        ``receptive_field - 1`` zero samples through. The second half is what
        makes the first ``render_block`` match ``render(x, pad_start=True)``:
        ``layer1x1`` carries a bias, so the network's settled state after
        silence is not all zeros, and starting from zeroed buffers gets the
        first ~4092 samples wrong -- exactly the attack of the first note.
        Measured across the dev corpus, skipping the prime costs 5.4e-02 to
        2.6e-01 of absolute error there. Costs ~10 ms for a 4093-sample
        receptive field, so it is **not real-time safe**.

        The zeros go through *one block at a time* rather than in one long
        push, for two reasons. Extra zeros are free: once every history buffer
        holds the DC response, another zero sample reproduces it bit for bit,
        so over-priming to a block boundary changes nothing. And TorchScript's
        profiling executor re-optimises when the input shape changes -- priming
        with one 4092-sample tensor and then playing 512-sample blocks would
        pay a ~600 ms recompile on the first *note*, inside the callback. Same
        shape throughout means the recompile happens here, where it is allowed
        to be slow.
        """
        if self._stream is None:
            raise RuntimeError("prepare_stream() has not been called")
        self._stream.reset()
        prime = self.receptive_field - 1
        if prime <= 0:
            return  # an LSTM has no history; its reset restored the stored state
        block = self._stream_block
        np_dtype = _STREAM_DTYPES[self._stream_dtype][0]
        silence = np.zeros((self._net.input_channels, block), dtype=np_dtype)
        # A floor of 4 runs so a large block still gives the JIT profiler the
        # two or three passes it wants before the first real audio arrives.
        for _ in range(max(4, -(-prime // block))):
            self._stream.run(silence)

    def render_block(self, x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """Run exactly one block through the capture, continuing from the last.

        ``x`` is 1-D, ``stream_block_size`` samples, dtype ``stream_dtype``.
        With ``out`` supplied the call allocates nothing in steady state.

        Latency is zero: output sample *n* is the model's response to input
        sample *n*. All the latency in a live system is device buffers and the
        block size, none of it this.
        """
        if self._stream is None:
            raise RuntimeError("prepare_stream() has not been called")
        arr = np.asarray(x)
        if arr.ndim != 1:
            raise ValueError(f"render_block takes a 1-D block, got shape {arr.shape}")
        if arr.shape[0] != self._stream_block:
            raise ValueError(
                f"block must be exactly {self._stream_block} samples, got {arr.shape[0]}")
        want = np.dtype(_STREAM_DTYPES[self._stream_dtype][0])
        if arr.dtype != want:
            raise ValueError(f"block dtype must be {want}, got {arr.dtype}")

        self._scratch[0, :] = arr
        y = self._stream.run(self._scratch)
        if out is None:
            return y.copy()
        if out.shape != (self._stream_block,) or out.dtype != want:
            raise ValueError(
                f"out must be a {want} array of {self._stream_block} samples, "
                f"got {out.dtype} {out.shape}")
        np.copyto(out, y)
        return out


def load(path_or_dict: str | Path | dict[str, Any]) -> NamModel:
    """Load a ``.nam`` file (or an already-parsed dict) into a renderer."""
    if isinstance(path_or_dict, dict):
        return NamModel(path_or_dict)
    path = Path(path_or_dict)
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    return NamModel(data, source=str(path))
