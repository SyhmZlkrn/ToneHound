"""Prepare a diverse private full-rig pilot from a user-supplied local CSV."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import re
import shutil
import sys
import zipfile

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from training.common import digest, identifier, inside, read_json, signature, write_json
from training.render import CoreRenderer, verify_renderer

FULL_RIG_TYPES = {'amp_cab', 'amp-cab', 'full-rig', 'full_rig', 'amp_pedal_cab', 'full_rig_combo'}


def candidates(root, csv_path):
    """Only explicit embedded cabinet labels qualify; names are not evidence."""
    eligible, rejected, seen, captures = [], [], set(), set()
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r:(r.get('arch') != '2', r.get('tone_id',''), r.get('model_id','')))
    for i,row in enumerate(rows):
        path = inside(root, row['path'].replace('\\','/'))
        reason = None
        if row.get('status') != 'ok' or not path.is_file():
            reason = 'file missing or CSV status not ok'
        else:
            try:
                nam = read_json(path)
                meta = nam.get('metadata') or {}
                if meta.get('gear_type') not in FULL_RIG_TYPES:
                    reason = 'no explicit embedded full-rig metadata'
                elif re.search(r'\b(?:svt|bass\s+(?:amp|rig|preamp|head)|darkglass|gallien.?krueger|aguilar|b7k|b3k)\b',
                               ' '.join(str(row.get(k,'')) for k in ('tone','model')), re.IGNORECASE):
                    reason = 'explicitly bass-oriented capture; guitar pilot'
                elif nam.get('architecture') not in ('WaveNet','SlimmableContainer','LSTM','Linear'):
                    reason = 'unsupported architecture'
                else:
                    # Exclude metadata-only copies of the same exact model.
                    content = {k:nam.get(k) for k in ('architecture','config','weights','sample_rate')}
                    key = signature(content)
                    capture = (row['tone_id'], row['model'].casefold().strip())
                    if key in seen or capture in captures:
                        reason = 'identical model content'
                    else:
                        seen.add(key); captures.add(capture)
                        eligible.append(dict(row, nam_metadata=meta, content_sha256=key,
                                             nam_architecture=nam['architecture'], sha256=digest(path)))
            except (ValueError, KeyError, OSError) as error:
                reason = str(error)
        if reason:
            rejected.append(dict(path=row['path'], reason=reason))
        if (i+1) % 500 == 0:
            print(f'Inspected {i+1}/{len(rows)} capture records', flush=True)
    return eligible, rejected, len(rows)


def diverse_order(rows, seed):
    """Round-robin gain categories, favouring new tone packs and creators."""
    rng = random.Random(seed)
    pools = defaultdict(list)
    for row in rows:
        category = row['category']
        if category.startswith('multi'):
            category = 'mixed gain'
        pools[category].append(row)
    for pool in pools.values():
        rng.shuffle(pool)
    tones, creators = Counter(), Counter()
    categories = sorted(pools)
    while pools:
        for category in categories:
            if category not in pools:
                continue
            pool = pools[category]
            at = min(range(len(pool)), key=lambda j:(tones[pool[j]['tone_id']], creators[pool[j]['creator']]))
            row = pool.pop(at)
            tones[row['tone_id']] += 1; creators[row['creator']] += 1
            if not pool:
                del pools[category]
            yield row


def prepare_collection(root, csv_path, di_root, splits_path, output, count, renderer, seed=719):
    root, di_root, output = Path(root).resolve(), Path(di_root).resolve(), Path(output).resolve()
    renderer = verify_renderer(renderer)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty pilot output folder')
    output.mkdir(parents=True, exist_ok=True)
    eligible, rejected, csv_count = candidates(root, csv_path)
    write_json(output/'collection_audit.json', dict(csv_rows=csv_count, eligible=len(eligible), rejected=rejected))
    if len(eligible) < count:
        raise ValueError(f'Only {len(eligible)} distinct explicitly full-rig models; requested {count}')
    profiles, failures = [], []
    source = output/'source'
    (source/'nam').mkdir(parents=True, exist_ok=True)
    used_ids = set()
    for row in diverse_order(eligible, seed):
        path = inside(root, row['path'].replace('\\','/'))
        model = CoreRenderer(path, renderer)
        time = np.arange(round(model.sample_rate*.25)) / model.sample_rate
        probe = .02*np.sin(2*np.pi*110*time) + .005*np.sin(2*np.pi*997*time)
        try:
            wet = model.render(probe)
            if not np.isfinite(wet).all() or np.sqrt(np.mean(wet*wet)) < 1e-7 or np.abs(wet).max() > 100:
                raise ValueError('Non-finite, silent or unbounded probe render')
        except (ValueError, OSError, TimeoutError) as error:
            failures.append(dict(model_id=row['model_id'], reason=str(error)))
            continue
        nam_id = 't3k-' + row['model_id']
        if nam_id in used_ids:
            nam_id += '-' + row['sha256'][:8]
        identifier(nam_id); used_ids.add(nam_id)
        target = source/'nam'/(nam_id+'.nam')
        shutil.copy2(path, target)
        meta = row['nam_metadata']
        family = meta.get('gear_model') or 'unknown'
        if str(family).lower() in ('tz-model','model','n/a','none'):
            family = 'unknown'
        category = row['category']
        gain = category if category in ('clean','edge of breakup','crunch','high gain / distortion') else 'unknown'
        profiles.append(dict(id=nam_id, path='nam/'+target.name, sha256=row['sha256'],
                             content_sha256=row['content_sha256'], rig='full-rig', cabinet_included=True,
                             rig_evidence='Embedded NAM metadata.gear_type=' + meta['gear_type'],
                             training_use_authorized=True,
                             authorization_scope='User-provided collection for private training; redistribution not authorized',
                             name=row['model'], creator=row['creator'], license=row['license'],
                             tone_url=row['tone_url'], tone_id=row['tone_id'], model_id=row['model_id'],
                             amp_family=family, family_label_source='Creator NAM gear_model; aliases not adjudicated',
                             gain_class=gain, gain_label_source='User-supplied CSV category; mixed packs remain unknown',
                             manufacturer=meta.get('gear_make') or 'unknown', architecture=row['nam_architecture']))
        if len(profiles) % 25 == 0:
            print(f'Validated full rigs: {len(profiles)}/{count}', flush=True)
        if len(profiles) == count:
            break
    if len(profiles) != count:
        write_json(output/'render_failures.json', failures)
        raise ValueError(f'Only {len(profiles)} full rigs passed native rendering')
    takes = []
    (source/'di').mkdir(exist_ok=True)
    for item in read_json(splits_path)['di']:
        identifier(item['id'])
        original = inside(di_root, item['filename'])
        target = source/'di'/(item['id']+original.suffix.lower())
        shutil.copy2(original, target)
        takes.append(dict(item, path='di/'+target.name))
    manifest = dict(schema=1, renderer='nam-core-2563c0f-full',
                    render=dict(seconds=5, starts_s=[0], input_gains_db=[0]), profiles=profiles, di=takes)
    write_json(source/'manifest.json', manifest)
    from training.build_dataset import build
    size_plan = build(source/'manifest.json', output/'datasets'/'amp-tone-v1', plan_only=True)
    report = dict(csv_rows=csv_count, eligible_full_rigs=len(eligible), selected=len(profiles),
                  creators=len({p['creator'] for p in profiles}), tone_packs=len({p['tone_id'] for p in profiles}),
                  architectures=dict(Counter(p['architecture'] for p in profiles)),
                  declared_gain_classes=dict(Counter(p['gain_class'] for p in profiles)),
                  di_recordings=len(takes), split_counts=dict(Counter(t['split'] for t in takes)),
                  failed_probe_renders=failures, dataset=size_plan,
                  label_caveat='Full-rig status and gear labels are creator declarations, not independently measured facts',
                  scope='Private research input package. No NAM, DI or model weights are uploaded to GitHub.')
    write_json(output/'pilot_report.json', report)
    archive = output/'ToneHound-pilot-source.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(source.rglob('*')):
            if path.is_file():
                z.write(path, 'ToneHound-ML/'+path.relative_to(output).as_posix())
        z.write(output/'pilot_report.json','ToneHound-ML/pilot_report.json')
    write_json(output/'package.json', dict(file=archive.name, bytes=archive.stat().st_size, sha256=digest(archive)))
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nam-root', type=Path, required=True)
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--di-root', type=Path, required=True)
    p.add_argument('--di-splits', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=500)
    p.add_argument('--renderer', type=Path, required=True)
    p.add_argument('--seed', type=int, default=719)
    a = p.parse_args()
    if a.count < 2:
        p.error('Need at least two profiles')
    prepare_collection(a.nam_root, a.csv, a.di_root, a.di_splits, a.output, a.count, a.renderer, a.seed)


if __name__ == '__main__':
    main()
