"""Evaluate the validation-selected checkpoint once against its frozen baseline."""
from pathlib import Path
import argparse
import json
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from training.common import digest, read_config, read_json, signature, write_json
from training.data import AudioDataset
from training.metrics import retrieval, unit
from training.models import ToneEncoder
from training.train_mert import extract, seed_all


def evaluate(run, dataset, device='cuda', model_factory=ToneEncoder, reference_run=None):
    run = Path(run)
    cfg = read_config(run/'config.yaml')
    data = AudioDataset(dataset)
    checkpoint = run/'checkpoint'/'best.pt'
    reference = None
    if reference_run is not None:
        reference_run = Path(reference_run)
        reference = read_config(reference_run/'config.yaml')
        if read_json(reference_run/'baseline.json')['dataset_sha256'] != data.meta['dataset_sha256']:
            raise ValueError('Reference baseline must use exactly the same dataset and split')
    key = signature(dict(dataset=data.meta['dataset_sha256'], checkpoint=digest(checkpoint),
                         reference_config=reference,
                         reference_baseline=digest(reference_run/'baseline.npz') if reference else None))
    if (run/'retrieval_test.json').exists():
        existing = read_json(run/'retrieval_test.json')
        if existing['evaluation_key'] != key:
            raise ValueError('This run already tested another checkpoint/dataset; use a new experiment')
        return read_json(run/'metrics.json')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['fingerprint'] != signature(dict(config=cfg, dataset=data.meta['dataset_sha256'])):
        raise ValueError('Evaluation config/dataset differs from training')
    last = torch.load(run/'checkpoint'/'last.pt', map_location='cpu', weights_only=True)
    if last['epoch'] != cfg['training']['epochs']:
        raise ValueError('Finish training and checkpoint selection before opening the test set')
    del last
    device = torch.device(device)
    seed_all(cfg['seed'])
    model = model_factory(cfg['model']).to(device)
    t = cfg['training']
    with np.load(run/'baseline.npz', allow_pickle=False) as f:
        frozen = f['vectors'].copy(); mean = f['mean'].copy(); projection = f['projection'].copy()
    test_ids = data.indices('test')
    # Backbone still has its ORIGINAL weights here. Baseline and candidate use
    # the same test queries and gallery identities, each in its own coordinates.
    test_features = extract(model, data, test_ids, device, t['eval_batch_size'], t['precision'], features=True)
    frozen[test_ids] = test_features[test_ids]
    transform = lambda x: unit((x-mean) @ projection if projection.shape[1] else x-mean)
    raw_metric, _, _ = retrieval(data, frozen, 'test')
    frozen_metric, _, _ = retrieval(data, frozen, 'test', transform=transform)
    model.load_trainable(saved['weights'])
    vectors = extract(model, data, data.indices('train')+test_ids, device,
                      t['eval_batch_size'], t['precision'])
    metric, details, confusion = retrieval(data, vectors, 'test', include_rows=True)
    result = dict(schema=1, status='evaluated; not activated', best_epoch=saved['epoch'],
                  dataset_sha256=data.meta['dataset_sha256'], validation=saved['validation'],
                  test=dict(candidate=metric, frozen_multi_di=raw_metric, frozen_projection=frozen_metric),
                  comparison='Same dataset/captures/DI split/encoder revision; projection fitted on train and selected on validation',
                  delta_vs_projection=dict(top5=metric['top5']-frozen_metric['top5'], mrr=metric['mrr']-frozen_metric['mrr']),
                  activation=False)
    if reference is not None:
        del model, vectors, frozen, test_features
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        seed_all(reference['seed'])
        ref_model = model_factory(reference['model']).to(device)
        with np.load(reference_run/'baseline.npz', allow_pickle=False) as f:
            ref_vectors = f['vectors'].copy(); ref_mean = f['mean'].copy(); ref_proj = f['projection'].copy()
        ref_t = reference['training']
        ref_test = extract(ref_model, data, test_ids, device, ref_t['eval_batch_size'], ref_t['precision'], features=True)
        ref_vectors[test_ids] = ref_test[test_ids]
        ref_transform = lambda x: unit((x-ref_mean) @ ref_proj if ref_proj.shape[1] else x-ref_mean)
        ref_metric, _, _ = retrieval(data, ref_vectors, 'test', transform=ref_transform)
        result['test']['reference_projection'] = ref_metric
        result['reference_model'] = reference['model']['id']
        result['delta_vs_reference'] = dict(top5=metric['top5']-ref_metric['top5'], mrr=metric['mrr']-ref_metric['mrr'])
    write_json(run/'metrics.json', result)
    write_json(run/'confusion.json', dict(evaluation_key=key, captures=confusion))
    write_json(run/'retrieval_test.json', dict(evaluation_key=key, queries=details))
    info = read_json(run/'run_info.json'); info['test_accessed'] = True
    write_json(run/'run_info.json', info)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--device', default='cuda', choices=('cuda','cpu'))
    p.add_argument('--reference-run', type=Path, help='Also compare a previously fitted baseline, e.g. the current 330M encoder')
    a = p.parse_args()
    print(json.dumps(evaluate(a.run, a.dataset, a.device, reference_run=a.reference_run), indent=2))


if __name__ == '__main__':
    main()
