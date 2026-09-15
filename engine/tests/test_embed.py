"""The MERT embedder: the coordinate system everything else sorts by.

Two kinds of test here. The cheap ones cover level matching, chunking and cache
keys, and run without a model. The rest load the real checkpoint, because the
claims that matter cannot be tested against a stub:

  * the fine-tuned repo's `auto_map` points at a `modeling_MERT.py` it does not
    ship, so the loader deliberately builds a stock `HubertModel` instead --
    ``test_the_checkpoint_loads_into_stock_hubert_with_no_missing_weights`` is
    the check that this substitution is exact rather than approximately right,
  * embedding is deterministic in eval mode (spec-augment is training-only, and
    a matcher whose answer changes per run is not a matcher), and
  * the same performance at two sample rates embeds to nearly the same vector,
    which is what makes a 48 kHz render and a 24 kHz model comparable at all.

The model is ~1.3 GB. These skip rather than fail when it cannot be fetched.
"""

from __future__ import annotations

import numpy as np
import pytest

from tonehound import embed
from tonehound.config import SAMPLE_RATE
from tonehound.embed import EmbedConfig, EmbedError, MertEmbedder

pytestmark = pytest.mark.filterwarnings("ignore")


def _pluck(seconds: float, sr: int = SAMPLE_RATE, hz: float = 110.0) -> np.ndarray:
    """A signal with harmonics and an envelope -- something MERT can hold onto."""
    t = np.arange(int(seconds * sr)) / sr
    tone = sum(np.sin(2 * np.pi * hz * k * t) / k for k in (1, 2, 3, 4, 5))
    env = np.exp(-1.5 * (t % 0.5))
    return (0.3 * tone * env).astype(np.float32)


@pytest.fixture(scope="module")
def embedder() -> MertEmbedder:
    """The one loaded model in this file.

    Module-scoped and shared on purpose: the checkpoint is 1.3 GB, and loading
    it per test crashed the full suite with a Windows access violation once the
    earlier test modules had taken their share of memory.
    """
    e = MertEmbedder(EmbedConfig(max_chunks=2))
    try:
        _ = e.model
    except EmbedError as exc:
        pytest.skip(f"MERT checkpoint unavailable: {exc.message}")
    return e


# -- level matching --------------------------------------------------------


def test_rms_matching_puts_a_quiet_and_a_loud_take_at_the_same_level() -> None:
    quiet, loud = _pluck(1.0) * 0.01, _pluck(1.0)
    a, b = embed.match_level(quiet), embed.match_level(loud)
    assert np.sqrt(np.mean(a**2)) == pytest.approx(embed.TARGET_RMS, rel=0.01)
    assert np.sqrt(np.mean(b**2)) == pytest.approx(embed.TARGET_RMS, rel=0.01)


def test_rms_matching_never_lets_a_peaky_signal_clip() -> None:
    spiky = np.zeros(SAMPLE_RATE, dtype=np.float32)
    spiky[::1000] = 1.0                      # crest factor far above any real take
    assert np.abs(embed.match_level(spiky)).max() <= embed.PEAK_CEILING + 1e-6


def test_peak_matching_is_available_and_does_what_it_says() -> None:
    out = embed.match_level(_pluck(0.5) * 0.05, "peak")
    assert np.abs(out).max() == pytest.approx(embed.PEAK_CEILING, rel=1e-3)


def test_silence_survives_level_matching_instead_of_becoming_infinity() -> None:
    assert np.all(embed.match_level(np.zeros(1000, dtype=np.float32)) == 0.0)


# -- chunking --------------------------------------------------------------


def test_chunking_cuts_whole_windows_and_drops_no_full_one() -> None:
    cfg = EmbedConfig(chunk_s=5.0, hop_s=5.0)
    got = embed.chunks(np.zeros(24_000 * 17), 24_000, cfg)
    assert [len(c) for c in got] == [24_000 * 5] * 3


def test_chunking_respects_the_ceiling_on_long_signals() -> None:
    cfg = EmbedConfig(chunk_s=5.0, hop_s=5.0, max_chunks=2)
    assert len(embed.chunks(np.zeros(24_000 * 60), 24_000, cfg)) == 2


def test_a_signal_shorter_than_one_window_is_kept_whole_not_padded() -> None:
    got = embed.chunks(np.zeros(24_000 * 3), 24_000, EmbedConfig(chunk_s=5.0))
    assert len(got) == 1 and len(got[0]) == 24_000 * 3


# -- configuration keys ----------------------------------------------------


def test_the_cache_key_changes_when_the_layers_do() -> None:
    assert EmbedConfig().key() != EmbedConfig(layers=(-1,)).key()
    assert EmbedConfig(chunk_s=5.0).key() != EmbedConfig(chunk_s=8.0).key()
    assert EmbedConfig().key() == EmbedConfig().key()


def test_the_cache_key_ignores_things_that_cannot_change_a_vector() -> None:
    # Batch size and device are throughput, not arithmetic.
    assert EmbedConfig(batch=1).key() == EmbedConfig(batch=8).key()


# -- the real model --------------------------------------------------------


def test_the_checkpoint_loads_into_stock_hubert_with_no_missing_weights(
        embedder: MertEmbedder) -> None:
    """The substitution this module is built on, checked rather than assumed.

    Reads the model the fixture already built rather than loading a second one.
    `load_encoder` raises on any missing tensor, so reaching this assertion at
    all is the "0 missing keys" check; a second 1.3 GB copy would only prove it
    twice, and did in fact exhaust memory when the whole suite ran in one
    process.
    """
    model, cfg, device = embedder.model, embedder._hub_cfg, embedder.device
    assert type(model).__name__ == "HubertModel"
    assert (cfg.hidden_size, cfg.num_hidden_layers) == (1024, 24)
    assert cfg.do_stable_layer_norm is True   # what makes MERT's encoder stock
    assert not model.training
    # The device is reported, not assumed: the weights may not have reached the
    # one that was asked for.
    assert device in ("cuda", "mps", "cpu") or device.startswith("cuda:")
    assert str(next(model.parameters()).device).startswith(device.split(":")[0])


def test_an_embedding_is_a_unit_vector_of_the_advertised_width(
        embedder: MertEmbedder) -> None:
    v = embedder.embed(_pluck(6.0), SAMPLE_RATE)
    assert v.shape == (embedder.dim,) == (4 * 1024,)
    assert v.dtype == np.float32
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-4)


def test_embedding_is_deterministic_so_a_ranking_does_not_move_between_runs(
        embedder: MertEmbedder) -> None:
    x = _pluck(6.0)
    assert np.allclose(embedder.embed(x, SAMPLE_RATE),
                       embedder.embed(x, SAMPLE_RATE), atol=1e-6)


def test_the_same_take_at_two_rates_embeds_to_nearly_the_same_vector(
        embedder: MertEmbedder) -> None:
    """A 48 kHz profile render and a 24 kHz model only compare because of this."""
    x48 = _pluck(6.0, SAMPLE_RATE)
    x24 = _pluck(6.0, 24_000)
    a = embedder.embed(x48, SAMPLE_RATE)
    b = embedder.embed(x24, 24_000)
    assert float(a @ b) > 0.99


def test_level_is_not_what_is_being_measured(embedder: MertEmbedder) -> None:
    x = _pluck(6.0)
    assert float(embedder.embed(x, SAMPLE_RATE) @
                 embedder.embed(x * 0.05, SAMPLE_RATE)) > 0.999


def test_opposite_polarity_stereo_keeps_the_guitar_representation(
        embedder: MertEmbedder) -> None:
    x = _pluck(6.0)
    stereo = np.column_stack((x, -x))
    assert np.max(np.abs(stereo.mean(axis=1))) == 0
    got = embedder.embed(stereo, SAMPLE_RATE)
    # A waveform downmix would be silence; both audible channels contain guitar.
    assert float(got @ embedder.embed(x, SAMPLE_RATE)) > 0.95
    assert np.linalg.norm(got) == pytest.approx(1.0, abs=1e-4)


def test_a_silent_stereo_channel_does_not_dilute_the_active_guitar(
        embedder: MertEmbedder) -> None:
    x = _pluck(6.0)
    stereo = np.column_stack((np.zeros_like(x), x))
    assert np.allclose(embedder.embed(stereo, SAMPLE_RATE),
                       embedder.embed(x, SAMPLE_RATE), atol=1e-6)


def test_two_different_timbres_are_further_apart_than_one_from_itself(
        embedder: MertEmbedder) -> None:
    clean = _pluck(6.0)
    fuzz = np.tanh(clean * 40.0).astype(np.float32)   # heavy clipping
    a, b = embedder.embed(clean, SAMPLE_RATE), embedder.embed(fuzz, SAMPLE_RATE)
    assert float(a @ b) < 0.999


def test_audio_under_the_minimum_is_refused_rather_than_guessed_at(
        embedder: MertEmbedder) -> None:
    with pytest.raises(EmbedError) as exc:
        embedder.embed(_pluck(0.2), SAMPLE_RATE)
    assert exc.value.code == "too_short"


def test_selecting_fewer_layers_narrows_the_vector(
        embedder: MertEmbedder) -> None:
    """Taken off the shared model: a second embedder means a second 1.3 GB of
    weights, which is what made the full suite fall over."""
    per_layer = embedder.embed_layers(_pluck(6.0), SAMPLE_RATE)
    assert embed.combine_layers(per_layer, (-1,)).shape == (1024,)
    assert embed.combine_layers(per_layer, (8, 9)).shape == (2048,)


# -- device selection ------------------------------------------------------


def test_auto_prefers_a_gpu_and_settles_for_the_cpu() -> None:
    """The whole device policy: use the GPU when there is one, run anyway when
    there is not."""
    have = embed.available_devices()
    assert have["cpu"] is True
    resolved = embed._resolve_device("auto")
    if have["cuda"]:
        assert resolved == "cuda"
    elif have["mps"]:
        assert resolved == "mps"
    else:
        assert resolved == "cpu"


def test_a_device_this_machine_lacks_is_downgraded_rather_than_raised(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embed, "available_devices",
                        lambda: {"cuda": False, "mps": False, "cpu": True})
    assert embed._resolve_device("cuda") == "cpu"
    assert embed._resolve_device("mps") == "cpu"
    assert embed._resolve_device("auto") == "cpu"


def test_an_explicit_cpu_request_is_honoured_even_with_a_gpu_present(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embed, "available_devices",
                        lambda: {"cuda": True, "mps": False, "cpu": True})
    assert embed._resolve_device("cpu") == "cpu"
    assert embed._resolve_device("auto") == "cuda"
    assert embed._resolve_device("cuda:1") == "cuda:1"   # index kept intact


def test_the_device_does_not_change_a_cached_vector() -> None:
    """A cache built on a GPU has to stay valid on a CPU, so the key ignores it."""
    assert EmbedConfig(device="cuda").key() == EmbedConfig(device="cpu").key()
    # Precision does change the arithmetic, so that one must invalidate.
    assert EmbedConfig(dtype="float16").key() != EmbedConfig(dtype="float32").key()


def test_gpu_and_cpu_embeddings_agree(embedder: MertEmbedder) -> None:
    """If these diverged, an index built on one machine would be worthless on
    another -- which is exactly what the cache assumes cannot happen.

    The only test that needs a second copy of the weights, so it is also the
    only one that has to hand them back afterwards.
    """
    import gc

    import torch

    if not embed.available_devices()["cuda"]:
        pytest.skip("no CUDA device")

    x = _pluck(8.0)
    reference = embedder.embed(x, SAMPLE_RATE)

    other = "cpu" if embedder.device.startswith("cuda") else "cuda"
    twin = MertEmbedder(EmbedConfig(device=other, max_chunks=2))
    try:
        assert float(reference @ twin.embed(x, SAMPLE_RATE)) > 0.9999
    finally:
        del twin
        gc.collect()
        torch.cuda.empty_cache()


# -- layer selection -------------------------------------------------------


def test_one_pass_over_all_layers_reproduces_a_direct_embedding(
        embedder: MertEmbedder) -> None:
    """`embed_layers` + `combine_layers` is what makes a layer sweep affordable;
    it is only useful if it agrees with `embed` exactly."""
    x = _pluck(8.0)
    direct = embedder.embed(x, SAMPLE_RATE)
    per_layer = embedder.embed_layers(x, SAMPLE_RATE)
    assert per_layer.shape == (embedder.n_hidden_states, 1024)
    combined = embed.combine_layers(per_layer, embedder.cfg.layers)
    assert np.allclose(direct, combined, atol=1e-5)


def test_the_default_layers_are_the_measured_ones_not_the_last_ones() -> None:
    """Layers 8-11 retrieve at 0.118 top-1 on the dev corpus; the last four sit
    at chance. `tools/gate_mert.py --sweep` is what re-checks this."""
    assert EmbedConfig().layers == (8, 9, 10, 11)
