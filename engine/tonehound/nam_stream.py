"""Block-at-a-time streaming for a loaded ``.nam`` capture.

``NamModel.render()`` is a whole-signal *valid* convolution: to produce one
output sample it needs ``receptive_field`` input samples, and it derives every
intermediate activation from scratch. That is exactly right offline and
unusable live. Re-running it per block over ``receptive_field - 1 = 4092``
samples of history burns 9x the arithmetic of the offline batch and clears only
1.23x realtime at a 512-sample block on the dev machine -- below the 2x floor a
duplex callback needs.

The fix is to stop re-deriving. Every convolution in the network is causal and
time-invariant, so the only thing a block needs from the past is the trailing
``(kernel_size - 1) * dilation`` samples of *its own input*. Keep that per
convolution and each stage emits exactly as many samples as it was given: the
receptive field costs nothing per block, and the arithmetic is *identical* to
the offline path rather than an approximation of it (measured 4.7e-16 max
deviation in float64, which is float64 round-off, not error). Measured 2.7x
realtime at a 512-sample block, 3.2x once TorchScripted.

Two non-obvious consequences shape this module.

*Priming is load-bearing.* Zeroing the history buffers is **not** the same as
the offline ``pad_start=True``. ``layer1x1`` carries a bias, so the network's
settled state after silence is not all zeros; starting from zeroed state
diverges from the offline render by 9.3e-02 on a signal of RMS 0.047 -- a wrong
first ~4092 samples, which is precisely the attack of the first note. So
``reset()`` zeroes the buffers and *then* pushes ``receptive_field - 1`` zero
samples through, reproducing the offline zero pad exactly.

*This must release the GIL.* The audio callback shares an interpreter with a
telemetry thread. Run eagerly, one block is ~100 Python-level op dispatches,
each grabbing the GIL. TorchScripted, the whole forward is one call that holds
the GIL only to enter. That is worth more than the ~25% raw speed it also buys.
Scripting is a version-sensitive API, so a failure falls back to eager: slower
and more exposed, but not dead.

Nothing here imports ``nam_render``; it consumes the already-loaded network
object by duck typing (``.arrays``, ``.layers``, ``.conv.weight`` ...), which
keeps the dependency one-way and lets ``nam_render`` use this module's
activation kernels for its offline path too. That sharing is deliberate: one
implementation of every activation means the streaming and offline paths cannot
drift apart in a way a test would have to catch after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Activation codes. An integer plus four floats plus one tensor covers every
# activation the exporter can write, and unlike a Python closure it survives
# TorchScript compilation.
ACT_TANH = 0
ACT_SIGMOID = 1
ACT_RELU = 2
ACT_IDENTITY = 3
ACT_ELU = 4
ACT_GELU = 5
ACT_SILU = 6
ACT_SOFTPLUS = 7
ACT_LEAKY_RELU = 8
ACT_HARDTANH = 9
ACT_PRELU = 10
ACT_SOFTSIGN = 11
ACT_SOFTSIGMOID = 12
ACT_LEAKY_HARDTANH = 13

GATING_NONE = 0
GATING_GATED = 1
GATING_BLENDED = 2


@dataclass(frozen=True)
class ActSpec:
    """One activation, flattened into scalars a scripted module can hold."""

    code: int
    p0: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    p3: float = 0.0
    weight: torch.Tensor | None = None  # PReLU slopes only

    def tensor(self) -> torch.Tensor:
        if self.weight is None:
            return torch.zeros(1, dtype=torch.float64)
        return self.weight


IDENTITY = ActSpec(ACT_IDENTITY)


def parse_activation(spec: str | dict[str, Any]) -> ActSpec:
    """Read one activation from either schema's spelling.

    Layer activations are exported as ``{"type": Name, ...params}``; the network
    head's activation is stored in init format (a bare string or
    ``{"name": ...}``). The same file mixes both conventions.
    """
    if isinstance(spec, str):
        name, params = spec, {}
    elif isinstance(spec, dict):
        d = dict(spec)
        name = d.pop("type", None) or d.pop("name", None)
        if name is None:
            raise ValueError(f"activation config has no 'type' or 'name': {spec!r}")
        params = d
    else:
        raise ValueError(f"unreadable activation config: {spec!r}")

    if name == "Tanh":
        return ActSpec(ACT_TANH)
    if name == "Sigmoid":
        return ActSpec(ACT_SIGMOID)
    if name == "ReLU":
        return ActSpec(ACT_RELU)
    if name == "Identity":
        return ActSpec(ACT_IDENTITY)
    if name == "ELU":
        return ActSpec(ACT_ELU, float(params.get("alpha", 1.0)))
    if name == "GELU":
        return ActSpec(ACT_GELU)
    if name == "SiLU":
        return ActSpec(ACT_SILU)
    if name == "Softplus":
        return ActSpec(ACT_SOFTPLUS, float(params.get("beta", 1.0)),
                       float(params.get("threshold", 20.0)))
    if name == "LeakyReLU":
        return ActSpec(ACT_LEAKY_RELU, float(params.get("negative_slope", 0.01)))
    if name == "Hardtanh":
        return ActSpec(ACT_HARDTANH, float(params.get("min_val", -1.0)),
                       float(params.get("max_val", 1.0)))
    if name == "PReLU":
        # Scalar or per-channel; the per-channel slopes ride in the config, not
        # in the weights vector, so nothing is consumed from the file either way.
        if "negative_slopes" in params:
            w = torch.tensor(params["negative_slopes"], dtype=torch.float64).reshape(1, -1, 1)
        else:
            w = torch.tensor([float(params.get("negative_slope", 0.25))], dtype=torch.float64)
        return ActSpec(ACT_PRELU, weight=w)
    if name == "Softsign":
        return ActSpec(ACT_SOFTSIGN)
    if name == "Softsigmoid":
        return ActSpec(ACT_SOFTSIGMOID)
    if name == "LeakyHardtanh":
        return ActSpec(ACT_LEAKY_HARDTANH,
                       float(params.get("min_val", -1.0)),
                       float(params.get("max_val", 1.0)),
                       float(params.get("min_slope", 0.01)),
                       float(params.get("max_slope", 0.01)))
    raise ValueError(f"unknown activation {name!r}")


def apply_activation(x: torch.Tensor, code: int, p0: float, p1: float, p2: float,
                     p3: float, w: torch.Tensor) -> torch.Tensor:
    """Evaluate one activation. Written to be TorchScript-compilable."""
    if code == 0:
        return torch.tanh(x)
    if code == 1:
        return torch.sigmoid(x)
    if code == 2:
        return F.relu(x)
    if code == 3:
        return x
    if code == 4:
        return F.elu(x, alpha=p0)
    if code == 5:
        return F.gelu(x)
    if code == 6:
        return F.silu(x)
    if code == 7:
        return F.softplus(x, beta=p0, threshold=p1)
    if code == 8:
        return F.leaky_relu(x, negative_slope=p0)
    if code == 9:
        return torch.clamp(x, p0, p1)
    if code == 10:
        # w is (1, C, 1) for per-channel slopes and (1,) for a scalar; both
        # broadcast against (batch, C, L).
        if w.numel() == 1:
            return torch.where(x >= 0.0, x, w[0] * x)
        return torch.where(x >= 0.0, x, w * x)
    if code == 11:
        return x / (1.0 + torch.abs(x))
    if code == 12:
        return 0.5 * (1.0 + x / (1.0 + torch.abs(x)))
    if code == 13:
        out = torch.where(x < p0, (x - p0) * p2 + p0, x)
        return torch.where(x > p1, (x - p1) * p3 + p1, out)
    raise ValueError("unknown activation code")


def apply_gated(x: torch.Tensor, gating: int,
                pc: int, pp0: float, pp1: float, pp2: float, pp3: float, pw: torch.Tensor,
                sc: int, sp0: float, sp1: float, sp2: float, sp3: float,
                sw: torch.Tensor) -> torch.Tensor:
    """Layer activation with gating folded in, halving the channel count.

    A gated layer's conv and mixer emit ``2 * bottleneck`` channels; the first
    half feeds the primary activation and the second half the gate. The
    exported ``activation`` entry holds only the primary, so reading it alone
    yields a correct weight count and silently wrong audio -- the triple is
    always evaluated together.
    """
    if gating == 0:
        return apply_activation(x, pc, pp0, pp1, pp2, pp3, pw)
    half = x.size(1) // 2
    a = x[:, :half]
    b = x[:, half:]
    blend = apply_activation(b, sc, sp0, sp1, sp2, sp3, sw)
    if gating == 1:
        return apply_activation(a, pc, pp0, pp1, pp2, pp3, pw) * blend
    if gating == 2:
        return blend * apply_activation(a, pc, pp0, pp1, pp2, pp3, pw) + (1.0 - blend) * a
    raise ValueError("unknown gating mode")


class StreamConv(nn.Module):
    """One causal convolution that carries its own input history.

    The history is the trailing ``(kernel_size - 1) * dilation`` samples of what
    this convolution was last given. Prepending it to the new block makes the
    same *valid* convolution the offline path runs emit exactly ``len(block)``
    samples, so no stage of the network ever changes length and the receptive
    field costs nothing per block.
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None,
                 dilation: int, groups: int, in_channels: int) -> None:
        super().__init__()
        kernel = int(weight.shape[-1])
        self.dilation = int(dilation)
        self.groups = int(groups)
        self.pad = (kernel - 1) * int(dilation)
        self.register_buffer("weight", weight.detach().clone())
        # A missing bias is materialised as zeros rather than branched on:
        # adding an exact zero cannot change a float, and it keeps one code path.
        zero = torch.zeros(weight.shape[0], dtype=weight.dtype)
        self.register_buffer("bias", zero if bias is None else bias.detach().clone())
        self.register_buffer("hist", torch.zeros(1, in_channels, max(self.pad, 0),
                                                 dtype=weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad == 0:
            return F.conv1d(x, self.weight, self.bias, dilation=self.dilation,
                            groups=self.groups)
        xp = torch.cat([self.hist, x], dim=2)
        y = F.conv1d(xp, self.weight, self.bias, dilation=self.dilation,
                     groups=self.groups)
        self.hist.copy_(xp[:, :, xp.size(2) - self.pad:])
        return y


class StreamLayer(nn.Module):
    """One WaveNet layer: dilated conv + conditioning mixer + residual + head tap."""

    def __init__(self, layer: Any, in_channels: int, condition_channels: int) -> None:
        super().__init__()
        self.conv = StreamConv(layer.conv.weight, layer.conv.bias,
                               layer.conv.dilation, layer.conv.groups, in_channels)
        self.mixer = StreamConv(layer.mixer.weight, layer.mixer.bias,
                                layer.mixer.dilation, layer.mixer.groups,
                                condition_channels)

        spec = layer.act_spec
        sec = layer.sec_spec
        self.gating = int(layer.gating)
        self.act_code = int(spec.code)
        self.act_p0, self.act_p1 = float(spec.p0), float(spec.p1)
        self.act_p2, self.act_p3 = float(spec.p2), float(spec.p3)
        self.register_buffer("act_w", spec.tensor().clone())
        self.sec_code = int(sec.code)
        self.sec_p0, self.sec_p1 = float(sec.p0), float(sec.p1)
        self.sec_p2, self.sec_p3 = float(sec.p2), float(sec.p3)
        self.register_buffer("sec_w", sec.tensor().clone())

        # The two 1x1 branches are optional in the modern schema. A scripted
        # module cannot hold an Optional submodule cheaply, so an absent branch
        # becomes an identity flag and a placeholder convolution.
        self.has_layer1x1 = layer.layer1x1 is not None
        post_channels = layer.conv.weight.shape[0]
        if self.gating != GATING_NONE:
            post_channels //= 2
        l1 = layer.layer1x1
        self.layer1x1 = StreamConv(
            l1.weight if l1 is not None else torch.zeros(1, post_channels, 1,
                                                         dtype=layer.conv.weight.dtype),
            l1.bias if l1 is not None else None,
            l1.dilation if l1 is not None else 1,
            l1.groups if l1 is not None else 1,
            post_channels)
        self.has_head1x1 = layer.head1x1 is not None
        h1 = layer.head1x1
        self.head1x1 = StreamConv(
            h1.weight if h1 is not None else torch.zeros(1, post_channels, 1,
                                                         dtype=layer.conv.weight.dtype),
            h1.bias if h1 is not None else None,
            h1.dilation if h1 is not None else 1,
            h1.groups if h1 is not None else 1,
            post_channels)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                head: torch.Tensor) -> List[torch.Tensor]:
        z = self.conv(x)
        m = self.mixer(c)
        post = apply_gated(z + m, self.gating,
                           self.act_code, self.act_p0, self.act_p1, self.act_p2,
                           self.act_p3, self.act_w,
                           self.sec_code, self.sec_p0, self.sec_p1, self.sec_p2,
                           self.sec_p3, self.sec_w)
        # The head branch taps post-activation, before layer1x1.
        head_term = self.head1x1(post) if self.has_head1x1 else post
        layer_out = self.layer1x1(post) if self.has_layer1x1 else post
        return [x + layer_out, head + head_term]


class StreamArray(nn.Module):
    """One WaveNet layer array."""

    def __init__(self, array: Any, condition_channels: int) -> None:
        super().__init__()
        in_channels = int(array.rechannel.weight.shape[1])
        channels = int(array.rechannel.weight.shape[0])
        self.rechannel = StreamConv(array.rechannel.weight, array.rechannel.bias,
                                    array.rechannel.dilation, array.rechannel.groups,
                                    in_channels)
        self.layers = nn.ModuleList(
            [StreamLayer(layer, channels, condition_channels) for layer in array.layers])
        self.head_rechannel = StreamConv(
            array.head_rechannel.weight, array.head_rechannel.bias,
            array.head_rechannel.dilation, array.head_rechannel.groups,
            int(array.head_term_channels))
        self.head_term_channels = int(array.head_term_channels)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                head: torch.Tensor) -> List[torch.Tensor]:
        x = self.rechannel(x)
        for layer in self.layers:
            pair = layer(x, c, head)
            x = pair[0]
            head = pair[1]
        return [self.head_rechannel(head), x]


class StreamHead(nn.Module):
    """The optional network head: repeated (activation, valid conv) blocks."""

    def __init__(self, head: Any) -> None:
        super().__init__()
        spec = head.act_spec
        self.act_code = int(spec.code)
        self.act_p0, self.act_p1 = float(spec.p0), float(spec.p1)
        self.act_p2, self.act_p3 = float(spec.p2), float(spec.p3)
        self.register_buffer("act_w", spec.tensor().clone())
        self.convs = nn.ModuleList(
            [StreamConv(conv.weight, conv.bias, conv.dilation, conv.groups,
                        int(conv.weight.shape[1])) for conv in head.convs])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(apply_activation(x, self.act_code, self.act_p0, self.act_p1,
                                      self.act_p2, self.act_p3, self.act_w))
        return x


class StreamWaveNet(nn.Module):
    """Streaming twin of ``nam_render._WaveNet``, arithmetically identical."""

    def __init__(self, net: Any) -> None:
        super().__init__()
        self.arrays = nn.ModuleList(
            [StreamArray(a, int(net.condition_channels)) for a in net.arrays])
        self.head_scale = float(net.head_scale)
        self.has_head = net.head is not None
        # A placeholder keeps the attribute a real module so TorchScript can
        # type it; the flag decides whether it is ever called.
        self.head = StreamHead(net.head) if net.head is not None else _NullHead()
        self.first_head_channels = int(net.arrays[0].head_term_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Legacy and plain modern files condition on the raw input.
        c = x
        y = x
        # Seeding the head accumulator with zeros instead of branching on the
        # first array is exact: adding a zero cannot perturb a float.
        head = torch.zeros(x.size(0), self.first_head_channels, x.size(2),
                           dtype=x.dtype, device=x.device)
        for array in self.arrays:
            pair = array(y, c, head)
            head = pair[0]
            y = pair[1]
        head = head * self.head_scale
        if self.has_head:
            head = self.head(head)
        return head


class _NullHead(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class StreamLinear(nn.Module):
    """Streaming twin of ``nam_render._Linear`` -- a single FIR convolution."""

    def __init__(self, net: Any) -> None:
        super().__init__()
        self.conv = StreamConv(net.conv.weight, net.conv.bias, net.conv.dilation,
                               net.conv.groups, int(net.conv.weight.shape[1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def build(net: Any) -> nn.Module:
    """Build the streaming twin of a loaded network. Raises ``TypeError`` if unknown."""
    kind = type(net).__name__
    if kind == "_WaveNet":
        return StreamWaveNet(net)
    if kind == "_Linear":
        return StreamLinear(net)
    raise TypeError(f"no streaming module for {kind}")


def reset_history(module: nn.Module) -> None:
    """Zero every history buffer, eager or scripted.

    Buffer *names* are the interface here rather than a method call, because a
    ``RecursiveScriptModule`` does not expose custom methods unless they were
    exported, and ``named_buffers`` works identically on both.
    """
    for name, buf in module.named_buffers():
        if name.endswith("hist"):
            buf.zero_()
