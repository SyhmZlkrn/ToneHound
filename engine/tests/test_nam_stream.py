"""Pin the streaming path to the offline render, sample for sample.

The streaming path exists because ``render()`` cannot run inside an audio
callback. It is worth having only if it is the *same function*: an off-by-one
in a history buffer does not crash and does not sound broken, it sounds
almost right, and nothing downstream would ever catch it. So the property
under test is equality with ``render(x, pad_start=True)`` at float64
round-off, across every block size, on every profile in the corpus.

Two failure modes get their own tests because equality on one long signal can
hide them:

* **Priming.** Zeroing the history buffers is not the same as the offline zero
  pad -- ``layer1x1`` carries a bias, so the settled state after silence is not
  zeros. Skipping the prime leaves the first ``receptive_field - 1`` samples
  wrong, which is exactly the attack of the first note, and the rest of the
  signal is perfect. ``test_priming_is_load_bearing`` measures that gap
  directly rather than trusting the code that closes it.
* **Block-boundary alignment.** A history buffer one sample short is invisible
  in the first block and only shows up at the seam, so every test here streams
  many blocks and the boundary test looks specifically at the samples either
  side of one.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from tonehound.nam_render import UnsupportedArchitecture, load

PROFILE_DIR = Path(__file__).resolve().parents[2] / "assets" / "dev_profiles"
PROFILES = sorted(PROFILE_DIR.glob("*.nam")) if PROFILE_DIR.is_dir() else []

# ``_roundtrip_modern.nam`` is a tiny synthetic fixture that happens to sort
# first; anything measuring a real magnitude wants a real capture instead.
REAL = [p for p in PROFILES if not p.stem.startswith("_")]

needs_profiles = pytest.mark.skipif(not PROFILES, reason="assets/dev_profiles is absent")
needs_real = pytest.mark.skipif(not REAL, reason="no real captures in assets/dev_profiles")
each_profile = pytest.mark.parametrize("path", PROFILES, ids=[p.stem[:40] for p in PROFILES])

BLOCK = 512
# Enough blocks that the deepest dilation (512 samples on these captures) has
# to be served out of history rather than out of the current block.
_N = BLOCK * 16


@lru_cache(maxsize=None)
def _model(path: Path):
    return load(path)


def _signal(n: int = _N, seed: int = 0) -> np.ndarray:
    """Broadband and loud enough to sit well into the nonlinearity."""
    rng = np.random.default_rng(seed)
    return 0.3 * rng.standard_normal(n)


def _stream(model, x: np.ndarray, block: int, dtype: str = "float64") -> np.ndarray:
    model.prepare_stream(block, dtype=dtype)
    n = (len(x) // block) * block
    xs = x.astype(dtype)
    return np.concatenate([model.render_block(xs[i:i + block]) for i in range(0, n, block)])


@needs_profiles
@each_profile
def test_streamed_equals_offline_render(path: Path) -> None:
    model = _model(path)
    x = _signal()
    got = _stream(model, x, BLOCK)
    want = model.render(x, pad_start=True)[:len(got)]
    # 1e-12 is four orders of magnitude above the ~6e-16 actually measured, so
    # this fails on a real defect and not on float64 reassociation.
    assert np.allclose(got, want, atol=1e-12, rtol=0.0), np.abs(got - want).max()


@needs_profiles
@pytest.mark.parametrize("block", [1, 64, 256, 512, 1024, 4093, 8192])
def test_block_size_does_not_change_the_output(block: int) -> None:
    """The block size is a scheduling choice, not a signal-processing one."""
    model = _model(PROFILES[0])
    x = _signal(n=max(8192, block * 2))
    got = _stream(model, x, block)
    want = model.render(x, pad_start=True)[:len(got)]
    assert np.allclose(got, want, atol=1e-12, rtol=0.0), np.abs(got - want).max()


@needs_profiles
def test_block_boundary_is_seamless() -> None:
    """A one-sample-short history shows up only at the seam, so look there."""
    model = _model(PROFILES[0])
    x = _signal()
    got = _stream(model, x, BLOCK)
    want = model.render(x, pad_start=True)[:len(got)]
    for seam in range(BLOCK, len(got), BLOCK):
        window = slice(seam - 4, seam + 4)
        assert np.allclose(got[window], want[window], atol=1e-12, rtol=0.0), seam


@needs_real
def test_priming_is_load_bearing() -> None:
    """Zeroed history is not the offline zero pad, and the difference is audible.

    This asserts the *size* of the error priming removes, so a refactor that
    drops the prime fails here with a reason rather than only failing the
    equality test with a number. Measured across the dev corpus the unprimed
    error is 5.4e-02 to 2.6e-01 -- eleven orders of magnitude above the 1e-12
    the primed path holds, and squarely audible.
    """
    from tonehound import nam_stream

    model = _model(REAL[0])
    x = _signal()
    want = model.render(x, pad_start=True)

    model.prepare_stream(BLOCK, dtype="float64")
    # Reach past the public API on purpose: the point is to compare against the
    # state the public API refuses to leave you in.
    nam_stream.reset_history(model._stream._module)
    unprimed = np.concatenate(
        [model.render_block(x[i:i + BLOCK]) for i in range(0, len(x), BLOCK)])

    head = model.receptive_field - 1
    assert np.abs(unprimed[:head] - want[:head]).max() > 1e-2
    # ...and it is confined to the first receptive field: the state converges
    # on its own once real samples arrive, which is why this is so easy to miss.
    assert np.allclose(unprimed[head:], want[head:], atol=1e-12, rtol=0.0)


@needs_profiles
def test_reset_returns_to_the_start_of_signal_state() -> None:
    model = _model(PROFILES[0])
    x = _signal()
    model.prepare_stream(BLOCK, dtype="float64")
    first = model.render_block(x[:BLOCK]).copy()
    model.render_block(x[BLOCK:2 * BLOCK])
    model.reset_stream()
    assert np.array_equal(model.render_block(x[:BLOCK]), first)


@needs_profiles
def test_float32_tracks_the_float64_render() -> None:
    """float32 is the play-time default; this bounds what that costs.

    Measured 7.7e-07 on a signal of RMS 0.3 -- the same order as the 1.1e-06
    parity this renderer holds against neural-amp-modeler itself, so float32
    is not the weakest link in the chain.
    """
    model = _model(PROFILES[0])
    x = _signal()
    got = _stream(model, x, BLOCK, dtype="float32")
    want = model.render(x, pad_start=True)[:len(got)]
    assert np.abs(got - want).max() < 1e-5


@needs_profiles
def test_out_parameter_writes_in_place_and_matches() -> None:
    model = _model(PROFILES[0])
    x = _signal().astype(np.float32)
    model.prepare_stream(BLOCK, dtype="float32")
    out = np.zeros(BLOCK, dtype=np.float32)
    returned = model.render_block(x[:BLOCK], out=out)
    assert returned is out
    model.reset_stream()
    assert np.array_equal(model.render_block(x[:BLOCK]), out)


@needs_profiles
def test_prepare_stream_is_idempotent_and_resets() -> None:
    model = _model(PROFILES[0])
    x = _signal()
    model.prepare_stream(BLOCK, dtype="float64")
    first = model.render_block(x[:BLOCK]).copy()
    model.render_block(x[BLOCK:2 * BLOCK])
    model.prepare_stream(BLOCK, dtype="float64")
    assert model.stream_block_size == BLOCK
    assert model.stream_dtype == "float64"
    assert np.array_equal(model.render_block(x[:BLOCK]), first)


@needs_profiles
def test_render_block_rejects_wrong_shapes_and_dtypes() -> None:
    model = _model(PROFILES[0])
    model.prepare_stream(BLOCK, dtype="float32")
    with pytest.raises(ValueError):
        model.render_block(np.zeros(BLOCK - 1, dtype=np.float32))
    with pytest.raises(ValueError):
        model.render_block(np.zeros(BLOCK, dtype=np.float64))
    with pytest.raises(ValueError):
        model.render_block(np.zeros((2, BLOCK), dtype=np.float32))
    with pytest.raises(ValueError):
        model.render_block(np.zeros(BLOCK, dtype=np.float32),
                           out=np.zeros(BLOCK, dtype=np.float64))


@needs_profiles
def test_unprepared_stream_raises_rather_than_guessing() -> None:
    model = load(PROFILES[0])  # a fresh model, deliberately not the cached one
    with pytest.raises(RuntimeError):
        model.render_block(np.zeros(BLOCK, dtype=np.float32))
    with pytest.raises(RuntimeError):
        model.reset_stream()


@needs_profiles
def test_prepare_stream_validates_its_arguments() -> None:
    model = _model(PROFILES[0])
    with pytest.raises(ValueError):
        model.prepare_stream(0)
    with pytest.raises(ValueError):
        model.prepare_stream(512, dtype="float16")


def test_linear_capture_streams() -> None:
    """A Linear capture is one long FIR, the degenerate case of the same design."""
    rf = 64
    rng = np.random.default_rng(1)
    taps = rng.standard_normal(rf) * 0.1
    model = load({"architecture": "Linear", "version": "0.5.0",
                  "config": {"receptive_field": rf, "bias": True},
                  "weights": [*taps, 0.05]})
    x = _signal(n=1024)
    got = _stream(model, x, 128)
    assert np.allclose(got, model.render(x, pad_start=True)[:len(got)], atol=1e-12)


def test_lstm_capture_streams() -> None:
    """An LSTM has no history buffers -- streaming it is just keeping (h, c).

    ``reset_stream`` must restore the *stored* initial state, not zeros: that
    state is the trained settled DC condition, and zeroing it would put a
    transient on the front of every stream.
    """
    hidden, layers = 4, 2
    rng = np.random.default_rng(2)
    w = []
    for layer in range(layers):
        d = 1 if layer == 0 else hidden
        w += list(rng.standard_normal(4 * hidden * (d + hidden)) * 0.2)  # fused W
        w += list(rng.standard_normal(4 * hidden) * 0.1)                 # bias
        w += list(rng.standard_normal(hidden) * 0.3)                     # h0
        w += list(rng.standard_normal(hidden) * 0.3)                     # c0
    w += list(rng.standard_normal(hidden) * 0.5) + [0.01]                # head
    model = load({"architecture": "LSTM", "version": "0.5.0",
                  "config": {"input_size": 1, "hidden_size": hidden,
                             "num_layers": layers},
                  "weights": w})
    assert model.receptive_field == 1
    x = _signal(n=600)
    got = _stream(model, x, 100)
    assert np.allclose(got, model.render(x, pad_start=True)[:len(got)], atol=1e-12)

    model.reset_stream()
    assert np.allclose(model.render_block(x[:100].astype(np.float64)), got[:100],
                       atol=1e-12)


def test_multichannel_input_is_refused_rather_than_streamed_wrongly() -> None:
    """`render_block` takes a 1-D block, so a 2-input capture has no meaning here."""
    rf, cin = 8, 2
    rng = np.random.default_rng(3)
    cfg = {"layers": [{"input_size": cin, "condition_size": cin, "channels": 4,
                       "dilations": [1, 2, 4], "kernel_size": 2, "activation": "Tanh",
                       "gated": False, "head_size": 1, "head_bias": True}],
           "head": None, "head_scale": 1.0}
    # rechannel + 3 layers (conv+bias, mixer, layer1x1+bias) + head + head_scale
    n = 4 * cin + 3 * (4 * 4 * 2 + 4 + 4 * cin + 4 * 4 + 4) + (1 * 4 + 1) + 1
    model = load({"architecture": "WaveNet", "version": "0.5.0", "config": cfg,
                  "weights": list(rng.standard_normal(n) * 0.1)})
    assert model.streamable is False
    with pytest.raises(UnsupportedArchitecture):
        model.prepare_stream(64)


@needs_profiles
def test_every_profile_reports_a_streamable_shape() -> None:
    """No file in the corpus needs the fallback, so a regression here is visible."""
    for path in PROFILES:
        assert _model(path).streamable, path.name


@needs_profiles
def test_the_modern_schema_capture_streams_too() -> None:
    """The one modern file exercises an active head1x1 and a >1 head kernel.

    Those are the two places in a layer array where history is easy to forget,
    and every legacy capture in the corpus has neither.
    """
    modern = [p for p in PROFILES if p.stem == "_roundtrip_modern"]
    if not modern:
        pytest.skip("_roundtrip_modern.nam is absent")
    with open(modern[0], "r", encoding="utf-8") as fp:
        cfg = json.load(fp)["config"]
    assert any(la.get("head1x1", {}).get("active") for la in cfg["layers"]) or \
        any(int(la["head"]["kernel_size"]) > 1 for la in cfg["layers"])
    model = _model(modern[0])
    x = _signal()
    got = _stream(model, x, BLOCK)
    assert np.allclose(got, model.render(x, pad_start=True)[:len(got)], atol=1e-12)
