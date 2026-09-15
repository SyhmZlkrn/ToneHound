"""MERT tone embeddings.

The matcher's coordinate system. A signal goes in; a unit vector comes out, and
cosine distance between two of those vectors is what the ranking sorts by. Both
sides of the comparison are embedded here -- the guitar separated out of a song,
and a DI rendered through a `.nam` profile -- so that the query and the index
live in one space by construction.

**The classification head is not used.** The checkpoint is fine-tuned on GTZAN,
so its head emits ten genre logits; "Rock" is not a tone. What is wanted is the
representation underneath, so this loads the encoder only and reads its
`hidden_states`.

**Stock HuBERT, no `trust_remote_code`.** The fine-tuned repo's `auto_map`
points at a `modeling_MERT.py` that the repo does not actually contain, so
`AutoModel` fails outright there. Reading the upstream MERT source settles what
to do instead: `MERTModel` subclasses `HubertModel`, and its two departures are
both switched off by this config -- `feature_extractor_cqt` is false, which makes
`MERTFeatureProjection` identical to `HubertFeatureProjection` down to the
parameter names, and `do_stable_layer_norm` is true, which selects the *stock*
`HubertEncoderStableLayerNorm` and leaves the custom `deepnorm` and
`attention_relax` encoders unreachable. (`attention_relax` is in any case only a
softmax stability shift, which softmax is invariant to.) So the weights are
loaded into a plain `transformers.HubertModel`, and it is checked: the 403
encoder tensors land with zero missing and zero unexpected keys. That drops a
dependency on unpinned remote code that targets an older transformers than the
one installed.

**Which layers -- the setting that decides whether this works at all.**
`hidden_states` has 25 entries: the feature projection output and then one per
transformer layer. The obvious choice is the last few, and on this task the last
four retrieve *at chance*. `tools/gate_mert.py --sweep` measured every
four-layer band over the dev corpus, querying each profile with a take it had
never seen (34 profiles, chance top-1 = 0.029):

    layers 8-11    top-1 0.118   top-5 0.441   top-10 0.706   median rank  6
    layers 4-15    top-1 0.118   top-5 0.500   top-10 0.676   median rank  6
    layers 4-7     top-1 0.059   top-5 0.294   top-10 0.529   median rank  9
    layers 21-24   top-1 0.029   top-5 0.176   top-10 0.294   median rank 16

Single layers peak around 9-11 and decay to chance by layer 24. That is the
shape the MERT papers describe and the reason this checkpoint makes it so
stark: it is fine-tuned for *genre*, so its top layers are tuned to encode what
is being played, and the query and the index are deliberately different
performances. Timbre -- which amp, not which riff -- lives in the lower middle of
the stack.

Read top-5 and top-10 rather than top-1 when re-running this. With 34 queries a
top-1 of 0.118 is four profiles; the same sweep with separation in the chain
moves the *top-1* peak up to layers 12-20 while its top-5 and top-10 still put
5-11 well ahead. The band is a real effect, the exact layer within it is not
something a corpus this size can resolve.

Each layer is L2-normalised *before* the layers are concatenated: mean
activation magnitude climbs from 4.1 at the projection to 8.4 at layer 24, so a
raw concatenation would be an expensive way of using only the top layer.

**Level, not loudness.** A separated stem comes out of a mastered record and is
heavily limited; a DI through a profile is not. Peak-normalising the two would
leave the record much louder in RMS, and MERT is not level-invariant, so the
ranking would partly be measuring mastering. Inputs are matched on RMS instead,
with a peak clamp so nothing arrives clipped.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .audioio import to_rate
from .config import SAMPLE_RATE

DEFAULT_MODEL = "dzur658/MERT-v1-330M-finetuned-gtzan"
MERT_SAMPLE_RATE = 24_000
"""The rate the checkpoint was trained at. Feeding it 48 kHz would halve every
frequency the model thinks it is hearing."""

_EPS = 1e-12
TARGET_RMS = 0.1
"""About -20 dBFS. Low enough that a peaky DI does not need clamping to reach it."""

PEAK_CEILING = 0.99


@dataclass(frozen=True)
class EmbedConfig:
    """Everything that changes an embedding, and therefore invalidates a cache."""

    model_id: str = DEFAULT_MODEL
    revision: str | None = None

    layers: tuple[int, ...] = (8, 9, 10, 11)
    """Which `hidden_states` to keep. Negative indexes from the last layer.

    Measured, not chosen -- see the module docstring. `(-4, -3, -2, -1)` is the
    intuitive setting and scores at chance.
    """

    chunk_s: float = 5.0
    """Analysis window. MERT is self-attentive, so cost is quadratic in this."""

    hop_s: float = 5.0
    """Stride between windows. Equal to `chunk_s` means no overlap."""

    max_chunks: int = 24
    """Ceiling on how much of a long signal is embedded, for predictable cost."""

    batch: int = 4
    device: str = "auto"
    level: str = "rms"          # "rms" | "peak" | "none"
    dtype: str = "float32"

    def key(self) -> str:
        """A short hash of this configuration, for cache keys."""
        payload = json.dumps({
            "model": self.model_id, "revision": self.revision,
            "layers": list(self.layers), "chunk_s": self.chunk_s,
            "hop_s": self.hop_s, "max_chunks": self.max_chunks,
            "level": self.level, "dtype": self.dtype, "v": 1,
        }, sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:16]


MIN_SECONDS = 1.0
"""Below this there is not enough for the encoder's receptive field to mean
anything, and an embedding of it would be confident noise."""


class EmbedError(Exception):
    """Something that stopped an embedding being produced."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# --------------------------------------------------------------------------
# level matching and chunking
# --------------------------------------------------------------------------


def match_level(x: np.ndarray, mode: str = "rms") -> np.ndarray:
    """Bring a signal to a fixed level so two of them are comparable."""
    x = np.asarray(x, dtype=np.float64)
    if mode == "none" or x.size == 0:
        return x.astype(np.float32)
    if mode == "peak":
        peak = float(np.abs(x).max())
        return (x / peak * PEAK_CEILING if peak > 0 else x).astype(np.float32)

    rms = float(np.sqrt(np.mean(np.square(x))))
    if rms <= _EPS:
        return x.astype(np.float32)
    y = x * (TARGET_RMS / rms)
    peak = float(np.abs(y).max())
    if peak > PEAK_CEILING:
        y *= PEAK_CEILING / peak
    return y.astype(np.float32)


def chunks(x: np.ndarray, sr: int, cfg: EmbedConfig) -> list[np.ndarray]:
    """Split into equal analysis windows, longest-first coverage of the signal.

    A trailing fragment shorter than half a window is dropped rather than
    zero-padded: padding would put silence into the mean, and silence embeds to
    something, just not to anything about the tone.
    """
    size = int(cfg.chunk_s * sr)
    hop = max(1, int(cfg.hop_s * sr))
    if len(x) < size:
        return [x]

    out = []
    for start in range(0, len(x) - size + 1, hop):
        out.append(x[start:start + size])
        if len(out) >= cfg.max_chunks:
            break
    return out


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


def available_devices() -> dict[str, bool]:
    """What this machine can actually run on."""
    import torch

    mps = getattr(torch.backends, "mps", None)
    return {"cuda": bool(torch.cuda.is_available()),
            "mps": bool(mps is not None and mps.is_available()),
            "cpu": True}


def _resolve_device(spec: str) -> str:
    """The GPU when there is one, the CPU when there is not.

    `auto` is the default and prefers CUDA, then Apple's MPS, then CPU. An
    explicit request the machine cannot honour is *downgraded* rather than
    raised: a matcher that refuses to start because a laptop has no GPU would be
    worse than a slow one, and the downgrade is printed so it is not mistaken
    for the GPU simply being slow.

    Note that `torch.cuda.is_available()` is false for a CPU-only torch build
    even on a machine with a working card -- which is the usual reason a GPU
    goes unused here, and not something any amount of configuration fixes.
    """
    have = available_devices()
    wanted = (spec or "auto").lower()

    if wanted == "auto":
        return "cuda" if have["cuda"] else "mps" if have["mps"] else "cpu"

    family = wanted.split(":")[0]
    if family in have and not have[family]:
        print(f"[tonehound] {wanted} was requested but is not available "
              f"(torch {_torch_build()}); falling back to cpu")
        return "cpu"
    return wanted


def _torch_build() -> str:
    import torch

    return f"{torch.__version__}, cuda={torch.version.cuda or 'not built'}"


def load_encoder(cfg: EmbedConfig) -> tuple[Any, Any]:
    """Build a `HubertModel` and pour the checkpoint's encoder weights into it.

    Returns ``(model, hubert_config, device)``. The device is returned rather
    than assumed, because moving 1.3 GB onto a GPU can fail after
    `torch.cuda.is_available()` said yes -- a card busy with something else, a
    driver mismatch -- and the caller has to send its input tensors wherever the
    weights actually landed.

    Raises `EmbedError` with something actionable rather than a `KeyError` from
    three libraries down.
    """
    try:
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import HubertConfig, HubertModel
    except ImportError as exc:
        raise EmbedError("missing_dependency",
                         f"MERT embedding needs torch, transformers, "
                         f"huggingface_hub and safetensors: {exc}") from exc

    kw = {"revision": cfg.revision} if cfg.revision else {}
    try:
        config_path = hf_hub_download(cfg.model_id, "config.json", **kw)
        weights_path = hf_hub_download(cfg.model_id, "model.safetensors", **kw)
    except Exception as exc:
        raise EmbedError("model_unavailable",
                         f"could not fetch {cfg.model_id}: {exc}") from exc

    raw = json.loads(pathlib.Path(config_path).read_text("utf-8"))

    # The checkpoint's config carries MERT-only keys (`attention_relax`,
    # `feature_extractor_cqt`, the old `mask_channel_*` spelling) that
    # HubertConfig would reject or silently stash. Only the keys HubertConfig
    # declares are passed through; every one dropped here is either inert for
    # this configuration or off.
    allowed = set(HubertConfig().to_dict())
    ignore = {"architectures", "model_type", "auto_map", "dtype",
              "transformers_version", "torch_dtype"}
    hub_cfg = HubertConfig(**{k: v for k, v in raw.items()
                              if k in allowed and k not in ignore})

    state = load_file(weights_path)
    body = {k[len("hubert."):]: v for k, v in state.items() if k.startswith("hubert.")}
    if not body:
        # An unprefixed checkpoint is a bare encoder; take it as-is minus the head.
        body = {k: v for k, v in state.items()
                if not k.startswith(("classifier.", "projector."))}

    model = HubertModel(hub_cfg)
    result = model.load_state_dict(body, strict=False)
    if result.missing_keys:
        raise EmbedError(
            "weight_mismatch",
            f"{cfg.model_id} is missing {len(result.missing_keys)} encoder tensors "
            f"(first: {result.missing_keys[0]}); this loader expects a HuBERT-shaped "
            "MERT checkpoint")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    device = _resolve_device(cfg.device)
    dtype = getattr(torch, cfg.dtype)
    if device != "cpu" and dtype == torch.float16:
        pass          # half precision is the point of asking for a GPU
    elif device == "cpu" and dtype == torch.float16:
        # float16 on CPU is emulated and slower than float32, not faster.
        dtype = torch.float32

    try:
        model.to(device=device, dtype=dtype)
    except Exception as exc:
        if device == "cpu":
            raise EmbedError("model_load_failed",
                             f"could not place the model: {exc}") from exc
        print(f"[tonehound] could not move MERT onto {device} ({exc}); using cpu")
        device = "cpu"
        model.to(device=device, dtype=torch.float32)
    return model, hub_cfg, device


class MertEmbedder:
    """Audio in, one unit vector out. Load once; embedding is the cheap part."""

    def __init__(self, cfg: EmbedConfig | None = None, *, model: Any = None) -> None:
        self.cfg = cfg or EmbedConfig()
        self._model = model
        self._hub_cfg: Any = None
        # Provisional: replaced by wherever `load_encoder` actually placed the
        # weights, which is not always what was asked for.
        self.device = _resolve_device(self.cfg.device)

    # -- lifecycle -------------------------------------------------------

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model, self._hub_cfg, self.device = load_encoder(self.cfg)
        return self._model

    @property
    def n_layers(self) -> int:
        return len(self.cfg.layers)

    @property
    def hidden_size(self) -> int:
        if self._hub_cfg is not None:
            return int(self._hub_cfg.hidden_size)
        return int(getattr(self.model.config, "hidden_size", 1024))

    @property
    def n_hidden_states(self) -> int:
        """Feature-projection output plus one per transformer layer: 25 here."""
        if self._hub_cfg is not None:
            return int(self._hub_cfg.num_hidden_layers) + 1
        return int(getattr(self.model.config, "num_hidden_layers", 24)) + 1

    @property
    def dim(self) -> int:
        """Width of the vectors this produces."""
        return self.n_layers * self.hidden_size

    # -- embedding -------------------------------------------------------

    def embed(self, x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        """One L2-normalised embedding of a whole signal, over `cfg.layers`."""
        return combine_layers(self.embed_layers(x, sr, self.cfg.layers))

    def embed_layers(self, x: np.ndarray, sr: int = SAMPLE_RATE,
                     layers: Sequence[int] | None = None) -> np.ndarray:
        """Per-layer pooled vectors, ``(len(layers), hidden_size)``.

        One forward pass answers every layer question, which is the whole point:
        choosing which layers carry amp identity is an experiment, and re-running
        a 330M-parameter model once per candidate group would make that
        experiment cost more than it is worth. `combine_layers` turns any subset
        of these rows into the vector `embed` would have produced.

        Rows are the mean over windows of per-window (time-mean, L2-normalised)
        vectors -- deliberately *not* renormalised afterwards, so that stacking
        them and normalising once is arithmetically identical to embedding
        directly.

        Averaging in the normalised space means a window of near-silence pulls
        the result toward nothing in particular instead of toward zero.
        """
        import torch

        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2:
            # MERT consumes mono windows. Average the channel representations,
            # not the waveforms, so opposite-polarity / wide guitars survive.
            energy = np.mean(np.square(x), axis=0)
            active = np.flatnonzero(energy > max(float(energy.max()) * 1e-4, _EPS))
            if len(active) > 1:
                return np.mean([self.embed_layers(x[:, c], sr, layers) for c in active], axis=0)
            x = x[:, int(active[0]) if len(active) else 0]
        if x.size < MIN_SECONDS * sr:
            raise EmbedError("too_short",
                             f"{x.size / sr:.2f}s of audio is under the "
                             f"{MIN_SECONDS:.0f}s minimum for an embedding")

        y = to_rate(x, sr, MERT_SAMPLE_RATE)
        y = match_level(y, self.cfg.level)
        windows = chunks(y, MERT_SAMPLE_RATE, self.cfg)

        model = self.model
        wanted = list(range(self.n_hidden_states) if layers is None else layers)
        torch_dtype = getattr(torch, self.cfg.dtype)

        pooled: list[np.ndarray] = []
        for start in range(0, len(windows), self.cfg.batch):
            batch = windows[start:start + self.cfg.batch]
            # Equal lengths inside a batch, so no attention mask is needed; a
            # short tail is embedded on its own rather than padded into company.
            for group in _by_length(batch):
                pooled.append(self._forward(np.stack(group), wanted, torch_dtype))

        return np.concatenate(pooled, axis=1).mean(axis=1)

    def _forward(self, group: np.ndarray, wanted: Sequence[int],
                 torch_dtype: Any) -> np.ndarray:
        """One batch through the encoder, pooled to ``(layers, batch, hidden)``.

        A CUDA out-of-memory is treated as a reason to move to the CPU and carry
        on, not as a failed match. It is the one GPU error that is both likely
        on a small card and completely recoverable -- another process taking the
        VRAM mid-run should cost seconds, not the answer.
        """
        import torch

        try:
            return self._pool_batch(group, wanted, torch_dtype)
        except torch.cuda.OutOfMemoryError:
            if self.device == "cpu":
                raise
            print(f"[tonehound] {self.device} ran out of memory; "
                  "moving MERT to the cpu for the rest of this run")
            torch.cuda.empty_cache()
            self.model.to(device="cpu", dtype=torch.float32)
            self.device = "cpu"
            return self._pool_batch(group, wanted, torch.float32)

    def _pool_batch(self, group: np.ndarray, wanted: Sequence[int],
                    torch_dtype: Any) -> np.ndarray:
        """Mean over time, then L2 per layer per window.

        Without that per-layer step the deepest layer, whose activations are
        about twice the magnitude of the shallowest on this checkpoint, would
        decide the ranking on its own.
        """
        import torch

        tensor = torch.from_numpy(group).to(device=self.device, dtype=torch_dtype)
        with torch.no_grad():
            out = self.model(tensor, output_hidden_states=True)
        return np.stack([
            _unit_rows(out.hidden_states[i].mean(dim=1).float().cpu().numpy())
            for i in wanted])

    def embed_many(self, signals: Sequence[np.ndarray], sr: int = SAMPLE_RATE,
                   progress: Callable[[int, int], None] | None = None) -> np.ndarray:
        """Embed a list, reporting progress. Returns ``(n, dim)``."""
        out = np.empty((len(signals), self.dim), dtype=np.float32)
        for i, sig in enumerate(signals):
            out[i] = self.embed(sig, sr)
            if progress:
                progress(i + 1, len(signals))
        return out


def combine_layers(per_layer: np.ndarray,
                   pick: Sequence[int] | None = None) -> np.ndarray:
    """Stack per-layer vectors into one unit vector.

    Concatenating equally-weighted unit rows makes the cosine between two
    results the plain average of their per-layer cosines, which is what "use
    these layers" should mean.
    """
    rows = per_layer if pick is None else per_layer[list(pick)]
    return _unit(np.concatenate(np.asarray(rows, dtype=np.float64)))


def _by_length(batch: Sequence[np.ndarray]) -> list[list[np.ndarray]]:
    """Group equal-length windows so a batch never needs padding."""
    groups: dict[int, list[np.ndarray]] = {}
    for item in batch:
        groups.setdefault(len(item), []).append(item)
    return list(groups.values())


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + _EPS)).astype(np.float32)


def _unit_rows(m: np.ndarray) -> np.ndarray:
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + _EPS)


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Similarity of one unit vector against rows of a unit matrix, in [-1, 1]."""
    q = query / (np.linalg.norm(query) + _EPS)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + _EPS)
    return m @ q
