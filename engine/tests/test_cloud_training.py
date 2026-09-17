"""Exercise real optimizer/checkpoint paths without network or large weights."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from training.build_dataset import build, prepare
from training.common import read_config, read_json, signature, write_json
from training.data import AudioDataset, ToneBatchSampler
from training.evaluate_mert import evaluate
from training.losses import tone_contrastive_loss
from training.models import LoRALinear, ToneEncoder
from training.train_mert import train

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def dataset(tmp_path):
    """Real WAVs through the production dataset reader; separate DI groups."""
    root = tmp_path/'dataset'; root.mkdir()
    rigs = [dict(id=f'rig-{i}', rig='full-rig', cabinet_included=True, amp_family='unknown', gain_class='unknown') for i in range(3)]
    takes = [dict(id=f'di-{i}', performance_group=f'take-{i}', split=s, audio_sha256=f'hash-{i}')
             for i,s in enumerate(('train','train','validation','test'))]
    rows = []
    from training.common import digest
    for rig in rigs:
        for take in takes:
            name = rig['id']+'-'+take['id']
            path = root/(name+'.wav')
            n = np.arange(24000)
            freq = 80+int(rig['id'][-1])*150+int(take['id'][-1])*3
            sf.write(path, .1*np.sin(2*np.pi*freq*n/24000), 24000, subtype='PCM_16')
            rows.append(dict(id=name, path=path.name, nam_id=rig['id'], di_id=take['id'],
                             performance_group=take['performance_group'], split=take['split'], audio_sha256=digest(path)))
    meta = dict(schema=1, sample_rate=24000, seconds=1, profiles=rigs, di=takes, records=rows)
    meta['dataset_sha256'] = signature(meta)
    write_json(root/'metadata.json', meta)
    return root


def tiny_factory(spec):
    from transformers import HubertConfig, HubertModel
    cfg = HubertConfig(hidden_size=8, num_hidden_layers=3, num_attention_heads=2, intermediate_size=16,
                       conv_dim=(4,4,4), conv_stride=(10,10,5), conv_kernel=(11,11,7),
                       num_conv_pos_embeddings=4, num_conv_pos_embedding_groups=2,
                       mask_time_prob=0, layerdrop=0, hidden_dropout=0, attention_dropout=0,
                       feat_proj_dropout=0, activation_dropout=0)
    cfg._attn_implementation = 'eager'
    return ToneEncoder(spec, backbone=HubertModel(cfg))


def small_config(mode='lora'):
    cfg = read_config(ROOT/'configs/mert_amp_lora.yaml')
    cfg['model'].update(mode=mode, layers=[1,2], train_blocks=1, lora_rank=2,
                        head_hidden=8, embedding_dim=4, gradient_checkpointing=True)
    cfg['training'].update(epochs=2, classes_per_batch=2, takes_per_class=2,
                           steps_per_epoch=3, eval_batch_size=4, accumulation=2,
                           precision='float32', checkpoint_every_steps=1)
    return cfg


def test_loss_requires_cross_performance_and_separates_rigs():
    labels = torch.tensor([0,0,1,1])
    groups = ['a','b','a','b']
    good = torch.tensor([[1.,0],[1.,.1],[0.,1],[.1,1.]], requires_grad=True)
    bad = good[[0,2,1,3]]
    loss = tone_contrastive_loss(good, labels, groups)
    assert loss < tone_contrastive_loss(bad, labels, groups)
    loss.backward(); assert torch.isfinite(good.grad).all()
    with pytest.raises(ValueError, match='positive'):
        tone_contrastive_loss(good, labels, ['a']*4)


def test_batch_sampler_independent_takes_and_all_rigs(dataset):
    data = AudioDataset(dataset)
    batches = list(ToneBatchSampler(data,2,2,719).batches(0))
    seen = set()
    for batch in batches:
        rigs = {data.rows[i]['nam_id'] for i in batch}
        assert len(rigs) == 2
        seen |= rigs
        for rig in rigs:
            assert len({data.rows[i]['performance_group'] for i in batch if data.rows[i]['nam_id']==rig}) == 2
    assert seen == set(data.rigs)


@pytest.mark.parametrize('mode', ['frozen','lora','unfreeze_last'])
def test_only_intended_parameters_update(mode):
    torch.manual_seed(719)
    model = tiny_factory(small_config(mode)['model'])
    model.train()
    before = {k:v.detach().clone() for k,v in model.named_parameters()}
    loss = tone_contrastive_loss(model(torch.randn(4,24000)), torch.tensor([0,0,1,1]), ['a','b','a','b'])
    loss.backward()
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    optim.step()
    changed = {k for k,v in model.named_parameters() if not torch.equal(before[k],v)}
    assert any(k.startswith('head.') for k in changed)
    assert all(model.get_parameter(k).requires_grad for k in changed)
    if mode == 'frozen':
        assert all(k.startswith('head.') for k in changed)
    else:
        assert any(k.startswith('backbone.encoder.layers.1.') for k in changed)
        assert not any(k.startswith('backbone.encoder.layers.2.') for k in changed)
        if mode == 'lora':
            assert all(k.startswith('head.') or k.endswith(('.a','.b')) for k in changed)


def test_lora_initially_exact_base():
    base = torch.nn.Linear(8,8)
    lora = LoRALinear(copy.deepcopy(base),2,4)
    x = torch.randn(3,8)
    assert torch.equal(base(x), lora(x))


def test_resume_matches_uninterrupted_and_test_stays_closed(dataset, tmp_path, monkeypatch):
    torch.set_num_threads(2)
    cfg = small_config()
    full = tmp_path/'full'; resumed = tmp_path/'resumed'
    original_get = AudioDataset.__getitem__
    def no_test(self, i):
        assert self.rows[i]['split'] != 'test', 'Training read held-out test audio'
        return original_get(self, i)
    monkeypatch.setattr(AudioDataset, '__getitem__', no_test)
    train(cfg, dataset, full, device='cpu', model_factory=tiny_factory)
    train(cfg, dataset, resumed, device='cpu', model_factory=tiny_factory, stop_after_steps=3)
    with pytest.raises(ValueError, match='Finish training'):
        evaluate(resumed, dataset, 'cpu', tiny_factory)
    train(cfg, dataset, resumed, device='cpu', resume=True, model_factory=tiny_factory)
    a = torch.load(full/'checkpoint/last.pt', weights_only=True)
    b = torch.load(resumed/'checkpoint/last.pt', weights_only=True)
    assert a['global_step'] == b['global_step'] == 4
    for key in a['weights']:
        torch.testing.assert_close(a['weights'][key], b['weights'][key], rtol=0, atol=0)
    assert read_json(full/'metrics.json')['test'] is None
    monkeypatch.setattr(AudioDataset, '__getitem__', original_get)
    result = evaluate(resumed, dataset, 'cpu', tiny_factory)
    assert result['test']['candidate']['queries'] == 3
    assert result['test']['candidate']['family_top5'] is None
    assert evaluate(resumed, dataset, 'cpu', tiny_factory) == result
    assert result['activation'] is False


def test_reject_modified_audio_and_group_leakage(dataset):
    meta = read_json(dataset/'metadata.json')
    meta['di'][-1]['performance_group'] = meta['di'][0]['performance_group']
    meta['dataset_sha256'] = signature({k:v for k,v in meta.items() if k!='dataset_sha256'})
    write_json(dataset/'metadata.json', meta)
    with pytest.raises(ValueError, match='leakage'):
        AudioDataset(dataset)


def test_audio_hash_guard(dataset):
    meta = read_json(dataset/'metadata.json')
    with (dataset/meta['records'][0]['path']).open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        AudioDataset(dataset)


def test_source_builder_real_linear_nam_resume_and_head_guard(tmp_path):
    root = tmp_path/'source'; root.mkdir()
    profiles, takes = [], []
    for i in range(2):
        # A small valid NAM fixture exercises the actual renderer and writer.
        name = f'rig-{i}.nam'
        write_json(root/name, dict(version='0.5.4', architecture='Linear',
                                   config={'receptive_field': 1, 'bias': False}, weights=[.5+i*.1], sample_rate=48000))
        profiles.append(dict(id=f'rig-{i}', path=name, rig='full-rig', cabinet_included=True,
                             rig_evidence='Test fixture, not a real amp claim', training_use_authorized=True))
    for i,split in enumerate(('train','train','validation','test')):
        name = f'di-{i}.wav'
        sf.write(root/name, .1*np.sin(2*np.pi*(100+i*41)*np.arange(48000)/48000),48000)
        takes.append(dict(id=f'di-{i}', path=name, performance_group=f'take-{i}', split=split, channel='mono'))
    source = dict(schema=1, render=dict(seconds=1, starts_s=[0], input_gains_db=[0]), profiles=profiles, di=takes)
    write_json(root/'source.json', source)
    output = tmp_path/'rendered'
    a = build(root/'source.json', output)
    assert a == build(root/'source.json', output)
    assert len(AudioDataset(output)) == 8
    source['profiles'][0]['cabinet_included'] = False
    write_json(root/'source.json', source)
    with pytest.raises(ValueError, match='full-rig'):
        prepare(root/'source.json')


def test_unpack_rejects_paths_outside_source(tmp_path):
    import zipfile
    from training.unpack_source import unpack
    archive = tmp_path/'bad.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('ToneHound-ML/source/../../escape.txt','bad')
    with pytest.raises(ValueError, match='relative path'):
        unpack(archive, tmp_path/'destination')
    assert not (tmp_path/'destination/escape.txt').exists()


def test_selection_uses_validation_only_and_rejects_exposed_tests(tmp_path):
    from training.select_experiment import select
    a, b = tmp_path/'a', tmp_path/'b'
    for path,top5,mrr in [(a,.6,.4),(b,.7,.5)]:
        write_json(path/'metrics.json', dict(dataset_sha256='same', validation=dict(top5=top5,mrr=mrr),
                                           best_epoch=3, test=None))
    assert select([a,b])['selected']['run'] == 'b'
    write_json(a/'metrics.json', dict(dataset_sha256='same', validation=dict(top5=.6,mrr=.4),
                                    best_epoch=3, test={'top5':.99}))
    with pytest.raises(ValueError, match='already exposed'):
        select([a,b])


def test_private_collection_order_spreads_packs():
    from training.prepare_collection import diverse_order
    rows = [dict(category='clean',tone_id=str(t),creator='creator',model_id=f'{t}-{i}')
            for t in range(3) for i in range(4)]
    first = list(diverse_order(rows,719))[:3]
    assert len({r['tone_id'] for r in first}) == 3


def test_native_renderer_receipt_guard(tmp_path):
    from training.render import verify_renderer, CORE_REVISION
    binary = tmp_path/'renderer.exe'; binary.write_bytes(b'test')
    write_json(binary.with_suffix('.json'), dict(revision=CORE_REVISION,binary_sha256='wrong'))
    with pytest.raises(ValueError, match='receipt/hash'):
        verify_renderer(binary)


def test_notebooks_are_valid_empty_and_code_compiles():
    for path in (ROOT/'notebooks').glob('*.ipynb'):
        notebook = read_json(path)
        assert notebook['nbformat'] == 4
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                assert cell['outputs'] == [] and cell['execution_count'] is None
                compile(''.join(cell['source']),str(path),'exec')


@pytest.mark.parametrize('name', ['ToneHound_Colab.ipynb', 'ToneHound_Kaggle.ipynb'])
def test_notebook_runner_streams_output_and_propagates_failure(name, capsys):
    import ast
    import subprocess
    import sys
    notebook = read_json(ROOT/'notebooks'/name)
    first = next(c for c in notebook['cells'] if c['cell_type'] == 'code')
    tree = ast.parse(''.join(first['source']))
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    namespace = {'subprocess': subprocess}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), name, 'exec'), namespace)
    with pytest.raises(subprocess.CalledProcessError) as error:
        namespace['run']([sys.executable, '-u', '-c',
                          'import sys; print("progress"); print("failure", file=sys.stderr); sys.exit(7)'])
    assert error.value.returncode == 7
    output = capsys.readouterr().out
    assert 'progress' in output and 'failure' in output
