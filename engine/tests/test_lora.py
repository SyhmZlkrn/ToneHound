"""Exercise installed adapters through real tiny HuBERT inference and the app boundary."""
import copy
from dataclasses import replace
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from tonehound.embed import EmbedConfig, EmbedError, match_level
from tonehound.lora import LoRAEmbedder, config_for, create_embedder
from tonehound.pipeline import Pipeline, PipelineConfig
from tonehound.tone_index import ToneIndex, ToneEntry
from training import models
from tools.install_lora import install


@pytest.fixture()
def exported(tmp_path, monkeypatch):
    from transformers import HubertConfig, HubertModel
    import yaml
    torch.set_num_threads(2)
    torch.manual_seed(73)
    hub = HubertConfig(hidden_size=8, num_hidden_layers=3, num_attention_heads=2,
                       intermediate_size=16, conv_dim=(4, 4, 4), conv_stride=(10, 10, 5),
                       conv_kernel=(11, 11, 7), num_conv_pos_embeddings=4,
                       num_conv_pos_embedding_groups=2, mask_time_prob=0)
    hub._attn_implementation = 'eager'
    backbone = HubertModel(hub)
    monkeypatch.setattr(models, 'load_backbone', lambda spec: copy.deepcopy(backbone))
    source = tmp_path / 'run'
    (source / 'checkpoint').mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((root / 'configs/mert_amp_lora.yaml').read_text())
    cfg['model'].update(layers=[1, 2], train_blocks=1, lora_rank=2, head_hidden=8, embedding_dim=4)
    (source / 'config.yaml').write_text(yaml.safe_dump(cfg))
    model = models.ToneEncoder(cfg['model']).eval()
    # Nonzero adapter updates exercise the trained path, not just its base encoder.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith('.b'):
                param.fill_(.2)
    save_file(model.trainable_state(), str(source / 'checkpoint/best.safetensors'))
    return source, model


def signal(seconds=5):
    t = np.arange(round(seconds * 24000)) / 24000
    return (.1 * np.sin(2*np.pi*137*t) + .03*np.sin(2*np.pi*281*t)).astype('float32')


def test_install_and_inference_match_training_forward(exported, tmp_path):
    source, model = exported
    manifest = install(source, tmp_path)
    cfg = config_for(tmp_path, 'lora', device='cpu', dtype='float16')
    assert cfg.checkpoint_sha256 == manifest['checkpoint_sha256']
    embedder = create_embedder(cfg)
    x = signal()
    with torch.inference_mode():
        expected = model(torch.from_numpy(match_level(x)).unsqueeze(0))[0].numpy()
    np.testing.assert_allclose(embedder.embed(x, 24000), expected, atol=1e-6)
    assert embedder.model.training is False
    assert not any(p.requires_grad for p in embedder.model.parameters())
    assert embedder.retrieval['mode'] == 'lora'


def test_window_pooling_and_antiphase_stereo_remain_finite(exported, tmp_path):
    install(exported[0], tmp_path)
    embedder = LoRAEmbedder(config_for(tmp_path, 'lora', device='cpu', batch=1))
    a, b = signal(), signal() * .01
    whole = embedder.embed(np.concatenate([a, b]), 24000)
    np.testing.assert_allclose(whole, embedder.embed(a, 24000), atol=1e-5)
    stereo = embedder.embed(np.column_stack([a, -a]), 24000)
    assert np.isfinite(stereo).all() and np.linalg.norm(stereo) == pytest.approx(1, abs=1e-6)
    for x in [np.zeros(24000), np.full(24000, np.nan), np.zeros((24000, 2)), signal(.3)]:
        with pytest.raises(EmbedError):
            embedder.embed(x, 24000)


def test_keys_include_weights_and_spec_but_not_install_location(exported, tmp_path):
    install(exported[0], tmp_path)
    cfg = config_for(tmp_path, 'lora')
    assert replace(cfg, checkpoint='elsewhere').key() == cfg.key()
    assert replace(cfg, checkpoint_sha256='0'*64).key() != cfg.key()
    assert replace(cfg, spec_json=cfg.spec_json.replace('"lora_alpha": 16', '"lora_alpha": 8')).key() != cfg.key()
    assert cfg.key() != EmbedConfig().key()


def test_missing_or_tampered_model_never_silently_uses_standard(exported, tmp_path):
    assert type(config_for(tmp_path)) is EmbedConfig
    with pytest.raises(EmbedError, match='choose Standard'):
        config_for(tmp_path, 'lora')
    install(exported[0], tmp_path)
    cfg = config_for(tmp_path, 'lora')
    Path(cfg.checkpoint).write_bytes(b'broken')
    with pytest.raises(EmbedError, match='checksum'):
        config_for(tmp_path, 'lora')
    with pytest.raises(EmbedError, match='changed'):
        LoRAEmbedder(cfg).model
    with pytest.raises(ValueError, match='Choose'):
        config_for(tmp_path, 'unknown')


def test_zip_install_and_invalid_weights_preserve_previous_install(exported, tmp_path):
    source, _ = exported
    archive = tmp_path / 'export.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.write(source / 'config.yaml', 'run/config.yaml')
        z.write(source / 'checkpoint/best.safetensors', 'run/checkpoint/best.safetensors')
        z.writestr('../unexpected.txt', 'not extracted')
    installed = install(archive, tmp_path)
    path = tmp_path / '.cache/matching_models/lora/model.json'
    before = path.read_bytes()
    save_file({'head.0.weight': torch.ones(1)}, str(source / 'checkpoint/best.safetensors'))
    with pytest.raises(ValueError, match='parameters'):
        install(source, tmp_path)
    assert path.read_bytes() == before
    assert config_for(tmp_path, 'lora').checkpoint_sha256 == installed['checkpoint_sha256']
    assert not (tmp_path / 'unexpected.txt').exists()


def test_lora_uses_raw_cosine_and_never_standard_projection(exported, tmp_path, monkeypatch):
    install(exported[0], tmp_path)
    pipe = Pipeline(PipelineConfig(cache_dir=tmp_path / '.cache', embed=config_for(tmp_path, 'lora')))
    vectors = np.array([[1, 0, 0, 0], [.8, .6, 0, 0]], dtype='float32')
    entries = [ToneEntry(str(i), str(i), tmp_path / f'{i}.nam') for i in range(2)]
    from tonehound import tone_learning
    monkeypatch.setattr(tone_learning, 'apply_active', lambda *args: pytest.fail('Wrong projection'))
    result = pipe._retrieval_index(ToneIndex(entries, vectors), None)
    np.testing.assert_allclose(result.similarities(vectors[0]), [1, .8])
    assert result.learning['mode'] == 'lora' and result.dim == 4


def test_native_worker_selects_lora_and_keeps_model_provenance(exported, tmp_path, monkeypatch):
    import soundfile as sf
    from types import SimpleNamespace
    from tonehound import native_bridge, reference_features
    install(exported[0], tmp_path)
    path = tmp_path / 'song.wav'
    sf.write(path, signal(30), 24000)
    configs = []
    class StubPipeline:
        def __init__(self, cfg):
            configs.append(cfg)
        def match(self, *args, **kwargs):
            return SimpleNamespace(to_json=lambda: dict(matches=[], caveat='Audition.',
                retrieval=LoRAEmbedder(configs[-1].embed).retrieval),
                stem=SimpleNamespace(samples=signal()), source=SimpleNamespace(samples=signal()))
    monkeypatch.setattr('tonehound.pipeline.Pipeline', StubPipeline)
    monkeypatch.setattr(reference_features, 'analyse_reference', lambda *args: ({}, {}))
    result = native_bridge.run(dict(action='match', path=str(path), start=0, end=30, matching_model='lora'), tmp_path)
    assert configs[0].embed.checkpoint_sha256 == result['retrieval']['checkpoint_sha256']
    assert 'LoRA' in result['caveat'] and 'Experimental' in result['caveat']
