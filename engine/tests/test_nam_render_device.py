"""Offline GPU execution must preserve the CPU oracle and live streaming state."""

import numpy as np
import pytest
import torch

from tonehound import nam_render, nam_stream


def _linear():
    return nam_render.load({
        "architecture": "Linear", "config": {"receptive_field": 3, "bias": True},
        "weights": [0.2, -0.5, 0.9, 0.01]})


def test_explicit_cuda_request_falls_back_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = _linear()
    x = np.arange(100) / 100
    expected = model.render(x)
    with pytest.warns(RuntimeWarning, match="unavailable"):
        actual = model.render(x, device="cuda")
    np.testing.assert_array_equal(actual, expected)
    assert model.last_render_device == "cpu"
    assert nam_render.render_device("auto") == "cpu"


@pytest.mark.parametrize("error", [torch.cuda.OutOfMemoryError("test OOM"),
                                       RuntimeError("test CUDA kernel failure")])
def test_cuda_failure_restarts_the_whole_render_on_cpu(monkeypatch, error):
    model = _linear()
    x = np.arange(101) / 100
    expected = model.render(x)
    monkeypatch.setattr(nam_render, "render_device", lambda _: "cuda")
    released = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: released.append(True))

    def fail(*args):
        raise error

    monkeypatch.setattr(model, "_render_cuda", fail)
    with pytest.warns(RuntimeWarning, match="retrying on CPU float64"):
        actual = model.render(x, device="auto")
    np.testing.assert_array_equal(actual, expected)
    assert model.last_render_device == "cpu" and released == [True]
    monkeypatch.undo()
    model.render(x)
    assert model.last_render_note is None


def test_lstm_gpu_request_keeps_the_trained_cpu_recurrence(monkeypatch):
    model = nam_render.load({
        "architecture": "LSTM",
        "config": {"input_size": 1, "hidden_size": 1, "num_layers": 1},
        "weights": [0.1] * 16})
    x = np.linspace(-0.5, 0.5, 100)
    expected = model.render(x)
    monkeypatch.setattr(nam_render, "render_device", lambda _: "cuda")
    with pytest.warns(RuntimeWarning, match="LSTM.*CPU"):
        actual = model.render(x, device="cuda")
    np.testing.assert_array_equal(actual, expected)
    assert model.last_render_device == "cpu"


@pytest.mark.parametrize("chunk", [0, -1, 1.5])
def test_invalid_chunk_size_is_rejected(chunk):
    with pytest.raises(ValueError, match="positive integer"):
        _linear().render(np.zeros(10), chunk_samples=chunk)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("pad_start", [False, True])
@pytest.mark.parametrize("chunk", [1, 31, 65536])
def test_cuda_chunks_match_cpu_with_and_without_start_padding(pad_start, chunk):
    model = _linear()
    x = np.random.default_rng(4).standard_normal(101)
    expected = model.render(x, pad_start=pad_start)
    actual = model.render(x, pad_start=pad_start, device="cuda", chunk_samples=chunk)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    assert actual.dtype == np.float64
    assert model.last_render_device == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_offline_gpu_render_does_not_reset_or_move_live_stream_state():
    model, control = _linear(), _linear()
    for m in (model, control):
        m.prepare_stream(32, dtype="float64")
        m.render_block(np.ones(32))
    x = np.linspace(-0.5, 0.5, 71)
    expected = model.render(x)
    model.render(x, device="cuda", chunk_samples=17)
    np.testing.assert_array_equal(model.render(x), expected)
    np.testing.assert_array_equal(model.render_block(np.zeros(32)),
                                  control.render_block(np.zeros(32)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_per_channel_activation_weights_move_to_cuda():
    prim = nam_stream.ActSpec(nam_stream.ACT_PRELU,
                              weight=torch.tensor([0.1, 0.3], dtype=torch.float64).reshape(1, 2, 1))
    sec = nam_stream.ActSpec(nam_stream.ACT_SIGMOID)
    x = torch.randn(1, 4, 101, dtype=torch.float64)
    cpu = nam_render._layer_activation(nam_stream.GATING_GATED, prim, sec)(x)
    gpu = nam_render._layer_activation(nam_stream.GATING_GATED, prim, sec, "cuda")(x.cuda())
    torch.testing.assert_close(gpu.cpu(), cpu, atol=1e-12, rtol=0)
