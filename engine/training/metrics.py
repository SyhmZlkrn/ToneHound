"""Capture retrieval and optional, explicitly labelled family/gain metrics."""
from collections import Counter
import numpy as np


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def gallery(data, vectors):
    centres = []
    for rig in data.rigs:
        # Each performance gets equal weight even if it has more gain views.
        groups = data.gallery_groups[rig]
        centres.append(np.mean([unit(vectors[groups[take]].mean(axis=0))
                                for take in sorted(groups)], axis=0))
    return unit(centres)


def retrieval(data, vectors, split, transform=None, include_rows=False):
    keys = list(data.rigs)
    base = gallery(data, vectors)
    indices = data.indices(split)
    query = vectors[indices]
    if transform is not None:
        query, base = transform(query), transform(base)
    query, base = unit(query), unit(base)
    scores = query @ base.T
    valid = np.isfinite(query).all(axis=1) & (np.linalg.norm(query, axis=1) > 1e-8)
    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf), axis=1, kind='stable')
    truth = np.array([data.labels[data.rows[i]['nam_id']] for i in indices])
    ranks = np.argmax(order == truth[:, None], axis=1) + 1
    result = dict(queries=len(indices), captures=len(keys), failed_queries=int((~valid).sum()),
                  top1=float(np.mean((ranks == 1) & valid)),
                  top5=float(np.mean((ranks <= 5) & valid)),
                  mrr=float(np.mean(np.where(valid, 1/ranks, 0))))
    details, confusion = [], Counter()
    for field, metric, k in [('amp_family', 'family_top5', 5), ('gain_class', 'gain_top1', 1)]:
        labelled = [j for j,idx in enumerate(indices) if data.rigs[data.rows[idx]['nam_id']].get(field) not in (None,'','unknown')]
        hits = sum(bool(valid[j]) and any(data.rigs[keys[v]].get(field) == data.rigs[keys[truth[j]]][field]
                                       for v in order[j,:k]) for j in labelled)
        result[metric] = hits/len(labelled) if labelled else None
        result[metric + '_labelled_queries'] = len(labelled)
    for j,idx in enumerate(indices):
        row = data.rows[idx]
        predicted = keys[order[j,0]] if valid[j] else 'failed'
        confusion[(row['nam_id'], predicted)] += 1
        if include_rows:
            details.append(dict(id=row['id'], di_id=row['di_id'], expected=row['nam_id'],
                                rank=int(ranks[j]) if valid[j] else None,
                                top5=[keys[v] for v in order[j,:5]] if valid[j] else []))
    return result, details, [dict(expected=a, predicted=b, count=n) for (a,b),n in sorted(confusion.items())]


def fit_projection(data, vectors):
    """Re-fit the deployed projection method on this dataset, validation only."""
    train = np.asarray(data.indices('train'))
    x = unit(vectors)
    base = gallery(data, x)
    mean = base.mean(axis=0)
    _, _, vt = np.linalg.svd(x[train]-mean, full_matrices=False)
    basis = vt[:min(64, len(data.rigs)*2, len(train)-1)].T
    z = (x[train]-mean) @ basis
    labels = np.asarray([data.labels[data.rows[i]['nam_id']] for i in train])
    centres = np.stack([z[labels == c].mean(axis=0) for c in range(len(data.rigs))])
    residual = z - centres[labels]
    eigenvalues, eigenvectors = np.linalg.eigh(residual.T @ residual / len(residual))
    scale = max(float(eigenvalues.mean()), 1e-8)
    candidates = []
    for strength in (0., .1, 1., 10., 100.):
        projection = np.zeros((x.shape[1],0), np.float32) if strength == 0 else basis @ (eigenvectors * (1+strength*eigenvalues/scale)**-.5)
        transform = lambda v: unit((v-mean) @ projection if projection.shape[1] else v-mean)
        metric, _, _ = retrieval(data, x, 'validation', transform=transform)
        candidates.append((metric['top5'], metric['mrr'], -strength, projection, metric))
    best = max(candidates, key=lambda c:c[:3])
    return mean.astype('float32'), best[3].astype('float32'), dict(strength=-best[2], validation=best[4])
