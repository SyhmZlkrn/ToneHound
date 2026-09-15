"""Labelled remix training with real backing stems and htdemucs_6s.

Song recordings provide accompaniment, not unverified amp identity labels.
All mixed WAVs and six-stem outputs use an owned temporary directory.
"""
from pathlib import Path
import hashlib
import json
import random
import tempfile

import numpy as np
import soundfile as sf

from .audioio import to_rate
from .config import SAMPLE_RATE
from .tone_learning import digest, mastered, save_arrays


def song_plan(root):
    folder = Path(root)/'assets/song_train'
    plan = []
    # Periphery recordings stay together. These split labels carry no amp identity.
    for path in sorted(folder.glob('*.mp3')):
        lower = path.name.lower()
        split = ('validation' if lower.startswith('karmanjakah') else
                 'test' if lower.startswith(('spiritbox', 'panah')) else 'train')
        plan.append(dict(path=path, name=path.name, split=split, sha256=digest(path), start_s=40))
    if not all(any(s['split']==split for s in plan) for split in ('train','validation','test')):
        raise ValueError('Song corpus needs training, validation and held-out test songs')
    if len({s['sha256'] for s in plan}) != len(plan):
        raise ValueError('Duplicate song file in the separation split')
    return plan


def extract_separated(entries, takes, embedder, root, progress=print):
    from .separate import Separator, isolate
    from .nam_render import load
    import torch
    root = Path(root)
    songs = song_plan(root)
    cache = root/'.cache/tone_learning/separated_features'
    vectors, labels, splits, views, take_ids = [], [], [], [], []
    records = []
    with tempfile.TemporaryDirectory(prefix='tonehound_training_') as working:
        separator = Separator(root/'.cache/uvr_models', work_dir=working, autocast=True, shifts=2)
        beds = {}
        for song in songs:
            audio, sr = sf.read(song['path'], start=int(song['start_s']*sf.info(song['path']).samplerate),
                                frames=int(8*sf.info(song['path']).samplerate), always_2d=True)
            audio = to_rate(audio, sr, SAMPLE_RATE)
            if len(audio) < SAMPLE_RATE*8:
                raise ValueError('Song excerpt is too short: ' + song['name'])
            random.seed(719);torch.manual_seed(719)
            stems = separator.split(audio, SAMPLE_RATE, tag='backing')
            # Exclude guitar, other and piano to limit unknown original guitar leakage.
            bed = sum(stems[k] for k in ('drums','bass','vocals') if k in stems)
            rms = float(np.sqrt(np.mean(bed*bed)))
            if rms < 1e-5:
                raise ValueError('No usable accompaniment: ' + song['name'])
            beds[song['sha256']] = np.asarray(bed/rms*.07, dtype=np.float32)
            progress('Prepared song backing: ' + song['name'], flush=True)
        tasks = []
        for split in ('train','validation','test'):
            subset = [s for s in songs if s['split']==split]
            di_ids = [i for i,t in enumerate(takes) if t['split']==split]
            for i,song in enumerate(subset):
                # Both DI windows in validation; test songs use distinct DI windows.
                for window in ([0,1] if split=='validation' else [i%2]):
                    tasks.append((song,di_ids[i%len(di_ids)],window))
        for label,entry in enumerate(entries):
            model = None
            for song,take_id,window in tasks:
                take = takes[take_id]
                signature = dict(version=1, encoder=embedder.cfg.key(), model=digest(entry.path),
                                 song=song['sha256'], song_start=song['start_s'], di=take['sha256'],
                                 window=window, seconds=8, separator='htdemucs_6s', shifts=2,
                                 seed=719, mix='guitar0.1_backing0.07_stereo_eq_limit_mp3')
                key = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
                path = cache/(key+'.npz')
                if path.is_file():
                    with np.load(path, allow_pickle=False) as saved:
                        vector = saved['vector']; used = str(saved['stem'])
                    if vector.shape != (embedder.dim,) or not np.isfinite(vector).all():
                        raise ValueError('Invalid separated feature cache')
                else:
                    model = model or load(entry.path)
                    audio,sr = sf.read(take['path'], always_2d=True)
                    audio = to_rate(audio.mean(axis=1),sr,model.sample_rate)
                    start = window*10*model.sample_rate
                    dry = audio[start:start+8*model.sample_rate] * (1 if window==0 else 10**(-6/20))
                    wet = to_rate(model.render(dry, device='auto'),model.sample_rate,SAMPLE_RATE)
                    wet = wet / max(float(np.sqrt(np.mean(wet*wet))),1e-8)*.1
                    bed = beds[song['sha256']]
                    n = min(len(wet),len(bed)); mix = bed[:n] + wet[:n,None]
                    mix = np.stack([mastered(mix[:,c],SAMPLE_RATE) for c in range(2)],axis=1)
                    # Real lossy encode/decode; no copyrighted remix is retained.
                    mp3 = Path(working)/'remix.mp3'
                    sf.write(mp3, mix/max(float(np.abs(mix).max()),1.)*.95, SAMPLE_RATE, format='MP3')
                    mix,sr = sf.read(mp3,always_2d=True);mp3.unlink()
                    random.seed(719);torch.manual_seed(719)
                    recovered = isolate(mix,sr,separator=separator,tag='training',preserve_stereo=True)
                    if recovered.stem not in ('guitar','other'):
                        # Keep the failed query in the test denominator, never
                        # train a raw mix as if it were the known guitar.
                        vector=np.zeros(embedder.dim,dtype=np.float32);used='failed'
                    else:
                        vector = embedder.embed(recovered.samples,sr);used=recovered.stem
                    save_arrays(path, vector=vector, stem=used, signature=json.dumps(signature))
                vectors.append(vector);labels.append(label);splits.append(song['split'])
                views.append('separated-'+str(window));take_ids.append(take_id)
                records.append(dict(capture=entry.key,song=song['name'],split=song['split'],di=take['name'],window=window,stem=used))
            progress(f'Separated features {label+1}/{len(entries)}: {entry.name}', flush=True)
    return dict(vectors=np.asarray(vectors),labels=np.asarray(labels),splits=np.asarray(splits),
                views=np.asarray(views),takes=np.asarray(take_ids)), dict(
                    songs=[{k:str(v) for k,v in s.items() if k!='path'} for s in songs],records=records)
