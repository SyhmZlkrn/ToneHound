"""Small, supervised retrieval models on frozen MERT features.

Whole DI recordings define the split, never randomly shuffled audio chunks.
Only compact vectors are cached; renders live in memory for one capture/take.
The held-out test is evaluated after validation selects the metric strength.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import uuid
import zipfile

import numpy as np

from .tone_index import ToneIndex

VERSION = 1


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def validate_splits(takes):
    """Fail before feature extraction if the same audio is in multiple splits."""
    import soundfile as sf
    seen = set()
    for take in takes:
        audio, sr = sf.read(take['path'], dtype='float32', always_2d=True)
        # Hash decoded samples too: changing container tags must not bypass this.
        key = hashlib.sha256(audio.tobytes() + str(sr).encode()).hexdigest()
        if key in seen:
            raise ValueError('Duplicate DI recording: ' + str(take['path']))
        seen.add(key)
        if len(audio) < 18 * sr or not np.isfinite(audio).all():
            raise ValueError('Each DI must contain at least 18 seconds of finite audio')
        if np.max(np.abs(audio)) < 1e-5:
            raise ValueError('Silent DI recording')
        take['sha256'] = digest(take['path'])
        take['audio_sha256'] = key
    splits = [t['split'] for t in takes]
    if (splits.count('train') < 2 or 'validation' not in splits or 'test' not in splits or
        set(splits) - {'train','validation','test'}):
        raise ValueError('Use at least two training DIs, one validation DI and one test DI')


def mastered(x, sr):
    """Deterministic modest EQ/limiting stress case, not a simulated full mix."""
    from scipy.signal import butter, sosfilt
    low = sosfilt(butter(2, 320, fs=sr, output='sos'), x)
    high = x - sosfilt(butter(2, 2800, fs=sr, output='sos'), x)
    y = x + 0.25 * low - 0.25 * high
    rms = max(float(np.sqrt(np.mean(y*y))), 1e-8)
    return np.tanh(y / (rms * 5)) * (rms * 5)


def extract(entries, takes, embedder, cache, device='auto', progress=print):
    from .audioio import to_rate
    from . import nam_render
    import soundfile as sf

    validate_splits(takes)
    vectors, labels, splits, views, take_ids = [], [], [], [], []
    cfg = {'version': VERSION, 'encoder': embedder.cfg.key(),
           'windows': [[0, 8, 0], [10, 8, -6]], 'augmentation': 'eq-limit-v1'}
    for i, entry in enumerate(entries):
        model = None
        for take_id, take in enumerate(takes):
            signature = json.dumps(dict(cfg, model=digest(entry.path), di=take['sha256']), sort_keys=True)
            path = Path(cache) / (hashlib.sha256(signature.encode()).hexdigest() + '.npz')
            if path.is_file():
                with np.load(path, allow_pickle=False) as saved:
                    batch = saved['vectors']
                if batch.shape != (4, embedder.dim) or not np.isfinite(batch).all():
                    raise ValueError('Invalid cached training features: ' + str(path))
            else:
                model = model or nam_render.load(entry.path)
                audio, sr = sf.read(take['path'], dtype='float64', always_2d=True)
                # Raw DI gain is preserved. Do not peak-normalise before the NAM.
                audio = to_rate(audio.mean(axis=1), sr, model.sample_rate)
                batch = []
                for start, duration, gain in cfg['windows']:
                    dry = audio[int(start*model.sample_rate):int((start+duration)*model.sample_rate)]
                    dry = dry * 10**(gain/20)
                    if np.sqrt(np.mean(dry*dry)) < 1e-5:
                        raise ValueError('Silent DI window: ' + take['name'])
                    wet = model.render(dry, device=device)
                    if not np.isfinite(wet).all() or np.max(np.abs(wet)) < 1e-7:
                        raise ValueError('Invalid NAM render: ' + entry.name)
                    batch.extend([embedder.embed(wet, model.sample_rate),
                                  embedder.embed(mastered(wet, model.sample_rate), model.sample_rate)])
                batch = np.asarray(batch, dtype=np.float32)
                save_arrays(path, vectors=batch, signature=signature)
            vectors.extend(batch)
            labels.extend([i]*4)
            splits.extend([take['split']]*4)
            views.extend(['clean-0dB', 'mastered-0dB', 'clean-minus6dB', 'mastered-minus6dB'])
            take_ids.extend([take_id]*4)
        progress(f'Features {i+1}/{len(entries)}: {entry.name}', flush=True)
    return dict(vectors=np.asarray(vectors), labels=np.asarray(labels),
                splits=np.asarray(splits), views=np.asarray(views), takes=np.asarray(take_ids))


def metrics(scores, truth):
    order = np.argsort(-scores, axis=1, kind='stable')
    ranks = np.argmax(order == np.asarray(truth)[:, None], axis=1) + 1
    failed = ~np.isfinite(scores).any(axis=1)
    ranks[failed] = scores.shape[1]+1
    return dict(queries=len(ranks), top1=float(np.mean((ranks <= 1) & ~failed)),
                top5=float(np.mean((ranks <= 5) & ~failed)), top10=float(np.mean((ranks <= 10) & ~failed)),
                mrr=float(np.mean(np.where(failed,0.,1/ranks))), median_rank=float(np.median(ranks)),
                separation_failures=int(failed.sum()))


def mark_failures(scores, queries):
    scores[np.linalg.norm(queries,axis=1)<1e-8] = -np.inf
    return scores


def prototypes(x, labels, count):
    return np.asarray([x[labels == i].mean(axis=0) for i in range(count)])


def fit(data, entries, embed_key, takes, *, selection_condition=None):
    """Learn within-capture nuisance covariance on training performances only.

    A regularised inverse square root reduces directions that vary between
    performances of the same capture. It is a fitted supervised projection,
    not fine-tuning the 330M-parameter MERT backbone.
    """
    x = unit(data['vectors'])
    labels = data['labels']
    train = data['splits'] == 'train'
    train &= np.linalg.norm(data['vectors'],axis=1)>1e-8
    val = data['splits'] == 'validation'
    test = data['splits'] == 'test'
    count = len(entries)
    if selection_condition:
        val &= np.char.startswith(data['views'], selection_condition)
        if not val.any():
            raise ValueError('No validation queries for the requested condition')
    base = prototypes(x[train], labels[train], count)
    mean = base.mean(axis=0)
    # SVD and covariance see training data only.
    _, _, vt = np.linalg.svd(x[train] - mean, full_matrices=False)
    basis = vt[:min(64, count*2, int(train.sum())-1)].T
    z = (x[train] - mean) @ basis
    centres = prototypes(z, labels[train], count)
    residual = z - centres[labels[train]]
    cov = residual.T @ residual / len(residual)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    scale = max(float(np.mean(eigenvalues)), 1e-8)
    candidates = []
    # Identity is included so a projection is never enabled just because it trained.
    for strength in (0.0, 0.1, 1.0, 10.0, 100.0):
        projection = (np.zeros((x.shape[1], 0), np.float32) if strength == 0 else
                      basis @ (eigenvectors * (1 + strength*eigenvalues/scale)**-0.5))
        def transform(a):
            return unit((a-mean) @ projection if projection.shape[1] else a-mean)
        gallery = transform(base)
        result = metrics(mark_failures(transform(x[val]) @ gallery.T, x[val]), labels[val])
        candidates.append((result['top5'], result['mrr'], -strength, projection, result))
    selected = max(candidates, key=lambda c: c[:3])
    projection = selected[3]
    def score(query, gallery, centre, proj=None):
        q, g = query-centre, gallery-centre
        if proj is not None and proj.shape[1]:
            q, g = q@proj, g@proj
        return mark_failures(unit(q) @ unit(g).T, query)
    # Same inputs/library size for every baseline. A single original DI/clean view.
    first_take = next(i for i,t in enumerate(takes) if t['split']=='train')
    single = prototypes(x[(data['takes']==first_take) & (data['views']=='clean-0dB')],
                        labels[(data['takes']==first_take) & (data['views']=='clean-0dB')], count)
    report = {'version': VERSION, 'captures': count, 'encoder_key': embed_key,
              'trained': 'supervised regularised retrieval projection; MERT backbone frozen',
              'selection': f'validation {selection_condition or "all"} top-5, then MRR; test never used to tune strength',
              'strength': -selected[2], 'validation_candidates': [dict(strength=-c[2], **c[4]) for c in candidates],
              'takes': [{k:str(v) for k,v in t.items() if k != 'path'} for t in takes],
              'conditions': 'unseen DI; clean and modest EQ/limiting; no drums, vocals or Demucs in this benchmark',
              'test': {}}
    for name, gallery, centre, proj in [('single_di', single, single.mean(axis=0), None),
                                        ('multi_di', base, mean, None),
                                        ('selected', base, mean, projection)]:
        scores = score(x[test], gallery, centre, proj)
        report['test'][name] = {'all': metrics(scores, labels[test])}
        for condition in ('clean', 'mastered', 'separated'):
            mask = np.char.startswith(data['views'][test], condition)
            if mask.any():
                report['test'][name][condition] = metrics(scores[mask], labels[test][mask])
    # A fixed non-regression gate, not another hyperparameter search on the test.
    chosen = report['test']['selected']['all']
    original = report['test']['single_di']['all']
    report['activation_passed'] = bool(chosen['top5'] >= original['top5'] and chosen['mrr'] > original['mrr'])
    metadata = dict(version=VERSION, encoder_key=embed_key, keys=[e.key for e in entries],
                    profiles=[dict(key=e.key, sha256=digest(e.path), name=e.name) for e in entries],
                    strength=-selected[2], report=report, activation_passed=report['activation_passed'])
    return dict(metadata=json.dumps(metadata), mean=mean.astype(np.float32),
                projection=projection.astype(np.float32), gallery=base.astype(np.float32)), report


class LearnedToneIndex(ToneIndex):
    """Same public query dimension; transform both query and gallery internally."""
    def __init__(self, original, artifact):
        path = Path(artifact)
        if path.stat().st_size > 32*1024*1024:
            raise ValueError('Retrieval artifact exceeds 32 MiB')
        import zipfile
        with zipfile.ZipFile(path) as archive:
            if sum(i.file_size for i in archive.infolist()) > 64*1024*1024:
                raise ValueError('Retrieval artifact expands beyond 64 MiB')
        with np.load(path, allow_pickle=False) as saved:
            meta = json.loads(str(saved['metadata']))
            mean, projection, gallery = (saved[k].copy() for k in ('mean','projection','gallery'))
        if meta.get('version') != VERSION or meta.get('encoder_key') != original.embed_key:
            raise ValueError('Retrieval artifact encoder/version mismatch')
        keys = meta['keys']
        if (not meta.get('activation_passed') or len(keys) != len(set(keys)) or
            mean.shape != (original.dim,) or projection.ndim != 2 or
            projection.shape[0] != original.dim or gallery.shape != (len(keys), original.dim) or
            not all(np.issubdtype(a.dtype, np.floating) and np.isfinite(a).all() for a in (mean,projection,gallery))):
            raise ValueError('Invalid or unapproved retrieval artifact')
        lookup = dict(zip(keys, gallery))
        # Untrained captures retain their encoder descriptor in the same metric.
        vectors = np.asarray([lookup.get(e.key, v) for e,v in zip(original.entries,original.vectors)])
        super().__init__(original.entries, vectors, original.embed_key, original.di_key, original.centred)
        self.center, self.projection = mean, projection
        centred = vectors - mean
        self.matrix = unit(centred @ projection if projection.shape[1] else centred)
        self.learning = {'covered': sum(e.key in lookup for e in original.entries),
                         'total': len(original), 'strength': meta['strength']}

    def transform(self, x):
        x = unit(x) - self.center
        return unit(x @ self.projection if self.projection.shape[1] else x)

    def similarities(self, query):
        return self.matrix @ self.transform(query)


def apply_active(index, path, progress=None):
    """Corrupt, stale or missing optional training never disables basic matching."""
    if not Path(path).is_file():
        return index
    try:
        learned = LearnedToneIndex(index, path)
    except (ValueError, KeyError, OSError, TypeError, zipfile.BadZipFile) as error:
        if progress:
            progress('indexing', 1., 'Using base MERT descriptors: ' + str(error))
        return index
    if progress:
        progress('indexing', 1., f'Multi-DI retrieval ready: {learned.learning["covered"]}/{len(index)} trained captures')
    return learned
