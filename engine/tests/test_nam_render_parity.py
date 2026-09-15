"""Pin the weight layout against neural-amp-modeler itself.

``test_nam_render.py`` cannot catch a permuted layout, and that is the failure
mode that matters: the per-layer blocks (conv, mixer, layer1x1) sum to the same
total whatever order they are read in, and a conv kernel read transposed
consumes the same count too. Both produce a model that loads with an exact
weight count and renders finite, bounded, level-dependent, per-profile-distinct
audio -- everything the behavioural suite asks for -- while being a different
amplifier. Verified by doing it: with the mixer and layer1x1 swapped, the whole
suite stayed green.

The only offline oracle for the layout is the reference implementation, so these
tests drive it directly, two ways:

* round-trip -- build a model with the package, randomise every parameter,
  export a ``.nam`` through the package's own exporter, and compare renders. The
  randomisation matters: at init many tensors are near-symmetric and a
  transposed read is nearly a no-op.
* real weights -- take a legacy capture's untouched weights vector, translate
  only its *config* onto the modern schema, and hand both to the package. This
  is the one that covers the 0.5.0 files the indexer actually reads.

Skipped wholesale when neural-amp-modeler is not installed, since it is a
development dependency, not a runtime one.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("nam", reason="neural-amp-modeler is a dev-only dependency")

from nam.models._from_nam import init_from_nam  # noqa: E402
from nam.models.linear import Linear  # noqa: E402
from nam.models.recurrent import LSTM  # noqa: E402
from nam.models.wavenet import WaveNet  # noqa: E402

from tonehound.nam_render import load  # noqa: E402

PROFILE_DIR = Path(__file__).resolve().parents[2] / "assets" / "dev_profiles"
LEGACY = sorted(p for p in PROFILE_DIR.glob("*.nam") if not p.name.startswith("_"))

# The reference runs in float32, so its own accumulation noise sets the floor.
# Every layout error measured while building these tests landed above 1e-3.
TOLERANCE = 1e-4


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"))])
def render_device(request):
    return request.param


def _signal(n: int = 4096, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n) * 0.3


def _layer_array(**overrides) -> dict:
    spec = {
        "input_size": 1,
        "condition_size": 1,
        "channels": 8,
        "head": {"out_channels": 1, "kernel_size": 1, "bias": False},
        "kernel_sizes": 3,
        "dilations": [1, 2, 4],
        "activation": "Tanh",
    }
    spec.update(overrides)
    return spec


# Each case isolates one layout decision the loader makes. Names say which.
WAVENET_CASES = {
    "plain": dict(layers_configs=[_layer_array()], head=None, head_scale=0.02),
    # The exact topology of the legacy dev profiles: two chained arrays.
    "two_arrays": dict(
        layers_configs=[
            _layer_array(channels=16, dilations=[1, 2, 4, 8, 16],
                         head={"out_channels": 8, "kernel_size": 1, "bias": False}),
            _layer_array(input_size=16, channels=8, dilations=[1, 2, 4, 8, 16],
                         head={"out_channels": 1, "kernel_size": 1, "bias": True}),
        ],
        head=None, head_scale=0.02,
    ),
    # Gating doubles the conv and mixer output widths and halves them again in
    # the activation, so a wrong split shows up here and nowhere else.
    "gated": dict(
        layers_configs=[_layer_array(
            activation={"name": "PairMultiply", "primary": "Tanh", "secondary": "Sigmoid"})],
        head=None, head_scale=0.02,
    ),
    "blended": dict(
        layers_configs=[_layer_array(
            activation={"name": "PairBlend", "primary": "Tanh", "secondary": "Sigmoid"})],
        head=None, head_scale=0.02,
    ),
    # head1x1 active changes what the head rechannel is sized from.
    "head1x1": dict(
        layers_configs=[_layer_array(
            head_1x1_config={"active": True, "out_channels": 5, "groups": 1})],
        head=None, head_scale=0.02,
    ),
    "no_layer1x1": dict(
        layers_configs=[_layer_array(layer_1x1_config={"active": False})],
        head=None, head_scale=0.02,
    ),
    "bottleneck": dict(
        layers_configs=[_layer_array(channels=8, bottleneck=4)],
        head=None, head_scale=0.02,
    ),
    # Grouping divides the conv weight's input dimension; a transposed read of a
    # grouped kernel is not even the same shape.
    "groups": dict(
        layers_configs=[_layer_array(channels=8, groups_input=2,
                                     layer_1x1_config={"active": True, "groups": 2})],
        head=None, head_scale=0.02,
    ),
    # Non-square kernels: the only case where (out, in, k) and (out, k, in)
    # differ in *shape* rather than just in content.
    "mixed_kernels": dict(
        layers_configs=[_layer_array(kernel_sizes=[1, 3, 5],
                                     activation=["Tanh", "ReLU", "Softsign"])],
        head=None, head_scale=0.02,
    ),
    "head_kernel_3": dict(
        layers_configs=[_layer_array(head={"out_channels": 1, "kernel_size": 3, "bias": True})],
        head=None, head_scale=0.5,
    ),
    "network_head": dict(
        layers_configs=[_layer_array(head={"out_channels": 4, "kernel_size": 1, "bias": False})],
        head={"channels": 6, "activation": "ReLU", "out_channels": 1, "kernel_sizes": [3, 1]},
        head_scale=0.02,
    ),
}


def _randomised(model: torch.nn.Module, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.3)
    return model


def _export(model: torch.nn.Module, tmp_path: Path, name: str) -> Path:
    model.export(tmp_path, basename=name)
    return tmp_path / f"{name}.nam"


@pytest.mark.parametrize("name", sorted(WAVENET_CASES), ids=sorted(WAVENET_CASES))
def test_wavenet_render_matches_the_reference(name: str, tmp_path: Path,
                                             render_device: str) -> None:
    model = _randomised(WaveNet.init_from_config(dict(WAVENET_CASES[name])), seed=len(name))
    path = _export(model, tmp_path, name)
    x = _signal()

    with torch.no_grad():
        reference = model(torch.tensor(x, dtype=torch.float32)[None], pad_start=True)[0].numpy()
    ours = load(path)

    assert ours.weight_count == len(json.loads(path.read_text(encoding="utf-8"))["weights"])
    assert ours.receptive_field == model.receptive_field
    assert np.max(np.abs(ours.render(x, device=render_device, chunk_samples=997)
                         - reference)) < TOLERANCE
    assert ours.last_render_device == render_device


@pytest.mark.parametrize("num_layers", [1, 2])
def test_lstm_render_matches_the_reference(num_layers: int, tmp_path: Path) -> None:
    """The fused ``cat([W_ih, W_hh], dim=1)`` split and the exported (h, c).

    The reference has to be driven with ``_get_initial_state()`` explicitly: its
    ``forward`` starts from the learned ``_initial_hidden``/``_initial_cell``,
    while it *exports* the state a 48k-sample zero burn-in settles to. Comparing
    without it shows a ~1e-1 disagreement that says nothing about the loader --
    the file's state is the one the plugin uses, and the one loaded here.
    """
    model = _randomised(LSTM(hidden_size=12, num_layers=num_layers, sample_rate=48000.0),
                        seed=num_layers)
    path = _export(model, tmp_path, f"lstm_{num_layers}")
    x = _signal(3000, seed=num_layers)

    with torch.no_grad():
        reference = model(torch.tensor(x, dtype=torch.float32)[None], pad_start=True,
                          initial_state=model._get_initial_state())[0].numpy()
    ours = load(path)

    assert ours.weight_count == len(json.loads(path.read_text(encoding="utf-8"))["weights"])
    assert np.max(np.abs(ours.render(x) - reference)) < TOLERANCE


@pytest.mark.parametrize("bias", [False, True])
def test_linear_render_matches_the_reference(bias: bool, tmp_path: Path,
                                            render_device: str) -> None:
    model = _randomised(Linear(receptive_field=21, bias=bias, sample_rate=48000.0), seed=3)
    path = _export(model, tmp_path, f"linear_{bias}")
    x = _signal(3000, seed=4)

    with torch.no_grad():
        reference = model(torch.tensor(x, dtype=torch.float32)[None], pad_start=True)[0].numpy()
    ours = load(path)

    assert ours.receptive_field == 21
    assert np.max(np.abs(ours.render(x, device=render_device, chunk_samples=997)
                         - reference)) < TOLERANCE
    assert ours.last_render_device == render_device


def _legacy_config_to_modern(data: dict) -> dict:
    """Rewrite a 0.5.0 config onto the modern schema, weights untouched.

    This is the same normalisation ``nam_render`` performs internally, written
    out again here against the package's own init schema. Two independent
    spellings of the same mapping is the point: if the loader's version drifts,
    only one of them moves.
    """
    out = copy.deepcopy(data)
    for array in out["config"]["layers"]:
        n_layers = len(array["dilations"])
        gated = bool(array.pop("gated"))
        activation = array.pop("activation")
        array["kernel_sizes"] = [array.pop("kernel_size")] * n_layers
        array["activation"] = [{"type": activation}] * n_layers
        array["gating_mode"] = ["gated" if gated else "none"] * n_layers
        array["secondary_activation"] = [{"type": "Sigmoid"} if gated else None] * n_layers
        array["head"] = {"out_channels": array.pop("head_size"), "kernel_size": 1,
                         "bias": array.pop("head_bias")}
        array["bottleneck"] = array["channels"]
        array["head1x1"] = {"active": False, "out_channels": 1, "groups": 1}
        array["layer1x1"] = {"active": True, "groups": 1}
        array["groups_input"] = 1
        array["groups_input_mixin"] = 1
        array["slimmable"] = None
    out["version"] = "0.7.0"
    return out


@pytest.mark.skipif(not LEGACY, reason="assets/dev_profiles is absent")
@pytest.mark.parametrize("path", LEGACY[:4], ids=[p.stem[:40] for p in LEGACY[:4]])
def test_legacy_capture_renders_like_the_reference(path: Path,
                                                  render_device: str) -> None:
    """Real 0.5.0 weights, read by both implementations.

    The package refuses the legacy config outright, which is why this loader
    exists -- but it accepts the *weights* once the config is translated, so it
    can still serve as the oracle for the files that matter most.

    Both run in float64 here, which leaves the two renders separated only by the
    float32 rounding of the JSON weights themselves, so the tolerance is far
    tighter than the round-trip tests can manage.
    """
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    reference = init_from_nam(_legacy_config_to_modern(data)).double().eval()
    ours = load(path)
    x = _signal(9000, seed=7)

    assert ours.receptive_field == reference.receptive_field
    with torch.no_grad():
        expected = reference(torch.from_numpy(x), pad_start=True).numpy()
    assert np.max(np.abs(ours.render(x, device=render_device, chunk_samples=997)
                         - expected)) < 1e-7
    assert ours.last_render_device == render_device


@pytest.mark.skipif(not LEGACY, reason="assets/dev_profiles is absent")
def test_a_permuted_layout_is_caught_by_parity_though_not_by_the_count() -> None:
    """The mutation the rest of the suite cannot see.

    Rebuilds a real capture's weights vector with layer1x1 read before the mixer
    -- a within-layer permutation, so the total is unchanged. The loader accepts
    it and renders plausible audio; only the reference disagrees.
    """
    with open(LEGACY[0], "r", encoding="utf-8") as fp:
        data = json.load(fp)

    weights = data["weights"]
    cursor = 0

    def take(n: int) -> list[float]:
        nonlocal cursor
        cursor += n
        return weights[cursor - n : cursor]

    permuted: list[float] = []
    for array in data["config"]["layers"]:
        channels = array["channels"]
        permuted += take(channels * array["input_size"])
        for _ in array["dilations"]:
            conv = take(channels * channels * array["kernel_size"]) + take(channels)
            mixer = take(channels * array["condition_size"])
            layer1x1 = take(channels * channels) + take(channels)
            permuted += conv + layer1x1 + mixer
        permuted += take(array["head_size"] * channels)
        if array["head_bias"]:
            permuted += take(array["head_size"])
    permuted += take(1)
    assert cursor == len(weights), "the permutation must not change the layout's total"

    x = _signal(9000, seed=7)
    wrong = load({**data, "weights": permuted})
    assert wrong.weight_count == len(weights)  # the count check is blind to this

    reference = init_from_nam(_legacy_config_to_modern(data)).double().eval()
    with torch.no_grad():
        expected = reference(torch.from_numpy(x), pad_start=True).numpy()
    assert np.max(np.abs(wrong.render(x) - expected)) > 1e-3
    assert np.max(np.abs(load(LEGACY[0]).render(x) - expected)) < 1e-7
