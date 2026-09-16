"""Verified audio manifests and batches containing independent performances."""
from collections import defaultdict
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from training.common import digest, inside, read_json, signature


class AudioDataset(Dataset):
    def __init__(self, root, *, verify=True):
        self.root = Path(root)
        self.meta = read_json(self.root / 'metadata.json')
        payload = {k:v for k,v in self.meta.items() if k != 'dataset_sha256'}
        if self.meta.get('schema') != 1 or signature(payload) != self.meta['dataset_sha256']:
            raise ValueError('Dataset metadata hash/schema mismatch')
        if self.meta['sample_rate'] != 24000 or not 1 <= self.meta['seconds'] <= 30:
            raise ValueError('Expected 24 kHz fixed-duration training audio')
        self.rows = self.meta['records']
        self.rigs = {p['id']: p for p in self.meta['profiles']}
        self.takes = {p['id']: p for p in self.meta['di']}
        self.labels = {key:i for i,key in enumerate(self.rigs)}
        if len(self.rigs) < 2 or len(self.rigs) != len(self.meta['profiles']):
            raise ValueError('Need distinct full rigs')
        if any(p.get('rig') != 'full-rig' or p.get('cabinet_included') is not True for p in self.rigs.values()):
            raise ValueError('Dataset contains a head-only or unverified capture')
        groups, audio_splits, row_ids, paths = {}, {}, set(), set()
        for t in self.takes.values():
            group = t['performance_group']
            sha = t['audio_sha256']
            if ((group in groups and groups[group] != t['split']) or
                    (sha in audio_splits and audio_splits[sha] != t['split'])):
                raise ValueError('DI leakage across dataset splits')
            groups[group] = audio_splits[sha] = t['split']
        self.frames = round(self.meta['seconds'] * 24000)
        self.by_split = {s:[] for s in ('train','validation','test')}
        self.gallery_groups = defaultdict(lambda: defaultdict(list))
        coverage = defaultdict(set)
        for index, row in enumerate(self.rows):
            take = self.takes[row['di_id']]
            if (row['nam_id'] not in self.rigs or row['split'] != take['split'] or
                    row['performance_group'] != take['performance_group'] or
                    row['split'] not in ('train','validation','test')):
                raise ValueError('Invalid row identity or split')
            if row['id'] in row_ids or row['path'] in paths:
                raise ValueError('Duplicate audio row')
            row_ids.add(row['id']); paths.add(row['path'])
            self.by_split[row['split']].append(index)
            coverage[(row['split'], row['nam_id'])].add(row['performance_group'])
            if row['split'] == 'train':
                self.gallery_groups[row['nam_id']][row['performance_group']].append(index)
            path = inside(self.root, row['path'])
            if verify and digest(path) != row['audio_sha256']:
                raise ValueError('Audio checksum mismatch: ' + row['id'])
            info = sf.info(path)
            if info.samplerate != 24000 or info.frames != self.frames or info.channels != 1:
                raise ValueError('Training audio must have a fixed duration and one channel')
        for split in ('train','validation','test'):
            for rig in self.rigs:
                if len(coverage[(split, rig)]) < (2 if split == 'train' else 1):
                    raise ValueError('Every rig needs independent DIs in every split')

    def indices(self, split):
        return self.by_split[split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        x, _ = sf.read(inside(self.root, row['path']), dtype='float32')
        if not np.isfinite(x).all() or np.linalg.norm(x) < 1e-6:
            raise ValueError('Invalid/silent training audio: ' + row['id'])
        return torch.from_numpy(x), self.labels[row['nam_id']], row['performance_group']


class ToneBatchSampler:
    def __init__(self, data, classes, takes, seed, steps=0):
        self.pool = defaultdict(lambda: defaultdict(list))
        for i in data.indices('train'):
            r = data.rows[i]
            self.pool[r['nam_id']][r['performance_group']].append(i)
        if not 2 <= classes <= len(self.pool) or takes < 2:
            raise ValueError('Batch requires 2+ rigs, no more than the dataset contains')
        if any(len(groups) < takes for groups in self.pool.values()):
            raise ValueError('Not enough independent training performances per rig')
        self.classes, self.takes, self.seed = classes, takes, seed
        self.steps = steps or math.ceil(len(self.pool) / classes)

    def batches(self, epoch):
        rng = np.random.default_rng(self.seed + epoch)
        rigs = list(rng.permutation(sorted(self.pool)))
        cursor = 0
        for _ in range(self.steps):
            chosen = []
            while len(chosen) < self.classes:
                if cursor == len(rigs):
                    rigs = list(rng.permutation(sorted(self.pool))); cursor = 0
                rig = rigs[cursor]; cursor += 1
                if rig not in chosen:
                    chosen.append(rig)
            batch = []
            for rig in chosen:
                groups = rng.choice(sorted(self.pool[rig]), self.takes, replace=False)
                batch.extend(int(rng.choice(self.pool[rig][g])) for g in groups)
            yield batch
