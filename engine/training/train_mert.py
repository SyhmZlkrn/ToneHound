"""Train a contrastive MERT retrieval model; select on validation only."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import uuid

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from training.common import read_config, read_json, signature, write_json
from training.data import AudioDataset, ToneBatchSampler
from training.losses import tone_contrastive_loss
from training.metrics import fit_projection, retrieval, unit
from training.models import ToneEncoder


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        torch.save(payload, temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def rng_state():
    np_state = np.random.get_state()
    return dict(python=random.getstate(), numpy=[np_state[0], np_state[1].tolist(), *np_state[2:]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']; np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def precision_context(device, precision):
    if device.type != 'cuda' or precision == 'float32':
        return nullcontext()
    return torch.autocast('cuda', dtype=getattr(torch, precision))


@torch.no_grad()
def extract(model, data, indices, device, batch_size, precision, *, features=False):
    model.eval()
    dim = model.backbone.config.hidden_size * len(model.layers) if features else model.spec['embedding_dim']
    vectors = np.zeros((len(data), dim), np.float32)
    for start in range(0, len(indices), batch_size):
        ids = indices[start:start+batch_size]
        x = torch.stack([data[i][0] for i in ids]).to(device)
        with precision_context(device, precision):
            z = model.features(x) if features else model(x)
        z = z.float().cpu().numpy()
        if not np.isfinite(z).all() or (np.linalg.norm(z, axis=1) < 1e-8).any():
            raise ValueError('Invalid encoder output')
        vectors[ids] = z
    return vectors


def baseline(model, data, output, cfg, device):
    """Cache only train/validation features; test audio is never embedded here."""
    from tonehound.tone_learning import save_arrays
    path = output / 'baseline.npz'
    cache_key = signature(dict(dataset=data.meta['dataset_sha256'], model=cfg['model']['id'],
                               revision=cfg['model']['revision'], layers=cfg['model']['layers'],
                               precision=cfg['training']['precision'], method='projection-v1'))
    if path.exists():
        with np.load(path, allow_pickle=False) as f:
            if str(f['key']) != cache_key:
                raise ValueError('Baseline cache configuration mismatch')
            vectors, mean, projection = (f[k].copy() for k in ('vectors','mean','projection'))
            details = json.loads(str(f['details']))
        if (vectors.shape != (len(data), model.backbone.config.hidden_size*len(model.layers)) or
                not all(np.isfinite(a).all() for a in (vectors, mean, projection)) or
                np.any(vectors[data.indices('test')])):
            raise ValueError('Invalid baseline cache or test contamination')
    else:
        print('Embedding frozen train/validation baseline', flush=True)
        vectors = extract(model, data, data.indices('train')+data.indices('validation'), device,
                          cfg['training']['eval_batch_size'], cfg['training']['precision'], features=True)
        mean, projection, details = fit_projection(data, vectors)
        save_arrays(path, key=cache_key, vectors=vectors, mean=mean, projection=projection,
                    details=json.dumps(details))
    write_json(output / 'baseline.json', dict(model=cfg['model'], method='Frozen encoder + training-only covariance projection',
                                            dataset_sha256=data.meta['dataset_sha256'], **details))
    return vectors


def run_info(cfg, data, model, device):
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=Path(__file__).resolve().parents[2],
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    versions = {}
    for name in ('torch','transformers','numpy','scipy','soundfile','safetensors','huggingface-hub'):
        versions[name] = importlib.metadata.version(name)
    return dict(schema=1, git_commit=commit, python=platform.python_version(), packages=versions,
                dataset_sha256=data.meta['dataset_sha256'], captures=len(data.rigs),
                split_performances={s: sorted({r['performance_group'] for r in data.rows if r['split']==s})
                                    for s in ('train','validation','test')},
                examples={s:len(data.indices(s)) for s in ('train','validation','test')},
                device=str(device), gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                trainable_blocks_zero_based=model.trainable_blocks, model=cfg['model'],
                test_accessed=False, activation='none; experiment output only')


def train(cfg, dataset, output, *, device='cuda', resume=False, model_factory=ToneEncoder, stop_after_steps=None):
    """An interruption restarts at the last saved optimizer boundary."""
    data = AudioDataset(dataset)
    t = cfg['training']
    sampler = ToneBatchSampler(data, t['classes_per_batch'], t['takes_per_class'], cfg['seed'], t['steps_per_epoch'])
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('No CUDA GPU: select a GPU runtime or explicitly use --device cpu')
    if device.type == 'cuda' and t['precision'] == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise ValueError('GPU does not support bfloat16; use a float16 experiment config')
    output = Path(output).resolve()
    fingerprint = signature(dict(config=cfg, dataset=data.meta['dataset_sha256']))
    last = output / 'checkpoint' / 'last.pt'
    if resume:
        if not last.is_file():
            raise ValueError('No last.pt checkpoint to resume')
        state = torch.load(last, map_location='cpu', weights_only=True)
        if state.get('fingerprint') != fingerprint:
            raise ValueError('Resume requires the identical config and dataset')
        if state['epoch'] >= t['epochs'] and all((output/p).is_file() for p in
                ('metrics.json','training_history.json','checkpoint/best.safetensors')):
            print('This run has already finished training.', flush=True)
            return dict(status='already_complete', steps=state['global_step'])
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Experiment folder is not empty; use --resume or a new run name')
        state = None
    output.mkdir(parents=True, exist_ok=True)
    import yaml
    (output/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    seed_all(cfg['seed'])
    model = model_factory(cfg['model']).to(device)
    frozen = baseline(model, data, output, cfg, device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=t['learning_rate'], weight_decay=t['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda' and t['precision']=='float16', init_scale=1024)
    info = run_info(cfg, data, model, device)
    write_json(output/'run_info.json', info)
    epoch, next_batch, global_step, loss_sum, loss_count = 0, 0, 0, 0., 0
    best_score = (-1., -1.)
    history = []
    if state:
        model.load_trainable(state['weights'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        epoch, next_batch, global_step = state['epoch'], state['next_batch'], state['global_step']
        best_score, history = tuple(state['best_score']), state['history']
        loss_sum, loss_count = state['loss_sum'], state['loss_count']
        restore_rng(state['rng'])

    def checkpoint():
        atomic_torch(last, dict(schema=1, fingerprint=fingerprint, weights=model.trainable_state(),
                               optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), rng=rng_state(),
                               epoch=epoch, next_batch=next_batch, global_step=global_step,
                               best_score=best_score, history=history, loss_sum=loss_sum, loss_count=loss_count))

    checkpoint()
    started = time.monotonic()
    while epoch < t['epochs']:
        batches = list(sampler.batches(epoch))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for b in range(next_batch, len(batches)):
            ids = batches[b]
            labels = torch.tensor([data.labels[data.rows[i]['nam_id']] for i in ids], device=device)
            groups = [data.rows[i]['performance_group'] for i in ids]
            group_begin = (b // t['accumulation']) * t['accumulation']
            accumulation = min(t['accumulation'], len(batches)-group_begin)
            with precision_context(device, t['precision']):
                if model.mode == 'frozen':
                    z = F.normalize(model.head(torch.from_numpy(frozen[ids]).to(device)), dim=-1)
                else:
                    z = model(torch.stack([data[i][0] for i in ids]).to(device))
                loss = tone_contrastive_loss(z, labels, groups, t['temperature'])
            scaler.scale(loss/accumulation).backward()
            loss_sum += float(loss.detach()); loss_count += 1
            if (b+1) % t['accumulation'] == 0 or b+1 == len(batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, t['max_grad_norm'], error_if_nonfinite=True)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                next_batch = b+1
                if global_step % t['checkpoint_every_steps'] == 0:
                    checkpoint()
                if stop_after_steps is not None and global_step >= stop_after_steps:
                    checkpoint()
                    return dict(status='interrupted_for_smoke_test', steps=global_step)
        if model.mode == 'frozen':
            model.eval()
            with torch.no_grad():
                # Head projection is small; keep even the feature path batched.
                v = np.zeros((len(data), cfg['model']['embedding_dim']), np.float32)
                indices = data.indices('train')+data.indices('validation')
                for begin in range(0, len(indices), t['eval_batch_size']):
                    ids = indices[begin:begin+t['eval_batch_size']]
                    with precision_context(device, t['precision']):
                        v[ids] = F.normalize(model.head(torch.from_numpy(frozen[ids]).to(device)), dim=-1).float().cpu().numpy()
        else:
            v = extract(model, data, data.indices('train')+data.indices('validation'), device,
                        t['eval_batch_size'], t['precision'])
        metric, _, _ = retrieval(data, v, 'validation')
        score = (metric['top5'], metric['mrr'])
        history.append(dict(epoch=epoch+1, steps=global_step, loss=loss_sum/max(loss_count,1), validation=metric))
        if score > best_score:
            best_score = score
            atomic_torch(output/'checkpoint'/'best.pt', dict(schema=1, fingerprint=fingerprint,
                         config=cfg, dataset_sha256=data.meta['dataset_sha256'], weights=model.trainable_state(),
                         epoch=epoch+1, validation=metric))
        print(json.dumps(history[-1]), flush=True)
        epoch += 1; next_batch = 0; loss_sum = 0.; loss_count = 0
        checkpoint()
        write_json(output/'training_history.json', history)
    best = torch.load(output/'checkpoint'/'best.pt', map_location='cpu', weights_only=True)
    from safetensors.torch import save_file
    temp = output/'checkpoint'/'best.safetensors.tmp'
    save_file(best['weights'], str(temp), metadata={'fingerprint': fingerprint})
    temp.replace(output/'checkpoint'/'best.safetensors')
    metrics = dict(schema=1, status='trained; test not evaluated', best_epoch=best['epoch'],
                   validation=best['validation'], dataset_sha256=data.meta['dataset_sha256'],
                   selection='validation top5, then MRR; earliest epoch on ties', test=None)
    if not (output/'retrieval_test.json').exists():
        write_json(output/'metrics.json', metrics)
    write_json(output/'training_history.json', history)
    print('Training complete. Run evaluate_mert.py separately for the held-out test.', flush=True)
    return dict(status='complete', steps=global_step, seconds=time.monotonic()-started)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda', choices=('cuda','cpu'))
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    print(json.dumps(train(read_config(a.config), a.dataset, a.output, device=a.device, resume=a.resume)))


if __name__ == '__main__':
    main()
