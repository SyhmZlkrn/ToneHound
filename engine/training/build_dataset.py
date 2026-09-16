"""Render an explicitly reviewed full-rig manifest to a portable audio dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
from training.common import digest, identifier, inside, read_json, signature, write_json
from tonehound.audioio import to_rate
from tonehound.embed import match_level

SPLITS = ('train', 'validation', 'test')
SAMPLE_RATE = 24000


def read_di(path, channel):
    audio, rate = sf.read(path, dtype='float64', always_2d=True)
    if not np.isfinite(audio).all() or not len(audio):
        raise ValueError('DI must contain finite audio')
    if channel == 'mono':
        if audio.shape[1] > 1 and not np.allclose(audio, audio[:, :1], atol=1e-7):
            raise ValueError('Stereo DI has distinct channels; choose a zero-based DI channel')
        dry = audio[:, 0]
    elif isinstance(channel, int) and not isinstance(channel, bool) and 0 <= channel < audio.shape[1]:
        dry = audio[:, channel]
    else:
        raise ValueError('DI channel must be mono or a valid zero-based channel number')
    if np.sqrt(np.mean(dry * dry)) < 1e-6:
        raise ValueError('Silent DI recording')
    return dry, rate


def prepare(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    source = read_json(manifest_path)
    if source.get('schema') != 1:
        raise ValueError('Expected schema 1 source manifest')
    root = inside(manifest_path.parent, source.get('source_root', '.'))
    render = source['render']
    seconds = float(render['seconds'])
    starts = render['starts_s']
    gains = render['input_gains_db']
    if (not 1 <= seconds <= 30 or not starts or not gains or
            len(set(starts)) != len(starts) or len(set(gains)) != len(gains) or
            any(not np.isfinite(s) or s < 0 for s in starts) or
            any(not np.isfinite(g) or not -24 <= g <= 6 for g in gains)):
        raise ValueError('Invalid render windows or input gains')
    rigs, seen, ids = [], set(), set()
    for item in source['profiles']:
        identifier(item['id'])
        if (item.get('rig') != 'full-rig' or item.get('cabinet_included') is not True or
                not item.get('rig_evidence') or item.get('training_use_authorized') is not True):
            raise ValueError('Every capture needs full-rig/cabinet evidence and training-use authorization: ' + item['id'])
        path = inside(root, item['path'])
        if path.suffix.lower() != '.nam':
            raise ValueError('Expected a NAM file')
        sha = digest(path)
        if sha in seen or item['id'] in ids:
            raise ValueError('Duplicate NAM content or ID')
        if item.get('sha256') and sha != item['sha256']:
            raise ValueError('NAM hash mismatch: ' + item['id'])
        seen.add(sha); ids.add(item['id'])
        rigs.append(dict(item, sha256=sha))
    if len(rigs) < 2:
        raise ValueError('Need at least two distinct full rigs')
    takes, seen, ids, groups = [], set(), set(), {}
    for item in source['di']:
        identifier(item['id']); identifier(item['performance_group'])
        if item['split'] not in SPLITS:
            raise ValueError('Unknown DI split')
        path = inside(root, item['path'])
        audio, rate = read_di(path, item.get('channel', 'mono'))
        # Canonical decoded mono catches container/tag changes, sample-rate
        # changes and duplicate stereo wrappers. Group IDs catch edited takes.
        canonical = np.round(to_rate(audio, rate, SAMPLE_RATE), 6).astype('<f4')
        audio_sha = __import__('hashlib').sha256(canonical.tobytes()).hexdigest()
        if item['id'] in ids or audio_sha in seen:
            raise ValueError('Duplicate DI ID or decoded recording')
        seen.add(audio_sha); ids.add(item['id'])
        group = item['performance_group']
        if group in groups and groups[group] != item['split']:
            raise ValueError('A DI performance group crosses train/validation/test')
        groups[group] = item['split']
        for start in starts:
            clip = audio[round(start * rate):round((start + seconds) * rate)]
            if len(clip) != round(seconds * rate) or np.sqrt(np.mean(clip * clip)) < 1e-6:
                raise ValueError('Short or silent DI window: ' + item['id'])
        takes.append(dict(item, channel=item.get('channel', 'mono'),
                          sha256=digest(path), audio_sha256=audio_sha))
    if (sum(v == 'train' for v in groups.values()) < 2 or
            not all(s in groups.values() for s in SPLITS)):
        raise ValueError('Need >=2 independent training DI groups, >=1 validation and >=1 test')
    renderer = source.get('renderer', 'tonehound-nam-float64-v1')
    if renderer not in ('tonehound-nam-float64-v1', 'nam-core-2563c0f-full'):
        raise ValueError('Unknown dataset renderer')
    plan = dict(schema=1, renderer=renderer, sample_rate=SAMPLE_RATE,
                render=render, profiles=rigs, di=takes)
    # Paths may change when moving to Drive; provenance is based on content.
    portable = {**plan, 'profiles': [{k:v for k,v in r.items() if k != 'path'} for r in rigs],
                'di': [{k:v for k,v in t.items() if k != 'path'} for t in takes]}
    return root, plan, signature(portable)


def build(manifest, output, *, plan_only=False, device='cpu', renderer=None):
    root, plan, plan_hash = prepare(manifest)
    count = len(plan['profiles']) * len(plan['di']) * len(plan['render']['starts_s']) * len(plan['render']['input_gains_db'])
    summary = dict(profiles=len(plan['profiles']), di=len(plan['di']), examples=count,
                   estimated_pcm_bytes=round(count * plan['render']['seconds'] * SAMPLE_RATE * 2),
                   plan_sha256=plan_hash)
    print(__import__('json').dumps(summary), flush=True)
    if plan_only:
        return summary
    from tonehound.nam_render import load
    if plan['renderer'] == 'nam-core-2563c0f-full':
        from training.render import CoreRenderer, verify_renderer
        if renderer is None:
            raise ValueError('This A1/A2 dataset requires --renderer from scripts/setup_training_renderer.py')
        binary = verify_renderer(renderer)
        loader = lambda path: CoreRenderer(path, binary)
    else:
        loader = load
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'build_state.json'
    if state_path.exists():
        state = read_json(state_path)
        if state['plan_sha256'] != plan_hash:
            raise ValueError('Output belongs to a different dataset; choose a new folder')
    else:
        if any(output.iterdir()):
            raise ValueError('Output must be empty or a resumable ToneHound dataset')
        state = dict(plan_sha256=plan_hash, records={})
        write_json(state_path, state)
    rows = []
    for rig in plan['profiles']:
        model = None
        for take in plan['di']:
            audio = None
            for start in plan['render']['starts_s']:
                for gain in plan['render']['input_gains_db']:
                    key = signature([plan_hash, rig['id'], take['id'], start, gain])[:24]
                    relative = take['split'] + '/' + key + '.wav'
                    target = inside(output, relative)
                    record = state['records'].get(key)
                    if record is not None:
                        if not target.is_file() or digest(target) != record['audio_sha256']:
                            raise ValueError('Changed or missing cached render: ' + relative)
                    else:
                        model = model or loader(inside(root, rig['path']))
                        if audio is None:
                            audio, rate = read_di(inside(root, take['path']), take['channel'])
                            audio = to_rate(audio, rate, model.sample_rate)
                        # Input amplitude controls NAM distortion. Normalize only
                        # AFTER rendering; retain context before each excerpt.
                        begin = round(start * model.sample_rate)
                        end = begin + round(plan['render']['seconds'] * model.sample_rate)
                        warm = min(begin, max(model.receptive_field, model.sample_rate))
                        wet = model.render(audio[begin-warm:end] * 10 ** (gain/20), device=device)[warm:]
                        wet = to_rate(wet, model.sample_rate, SAMPLE_RATE)
                        if not np.isfinite(wet).all() or np.sqrt(np.mean(wet*wet)) < 1e-7:
                            raise ValueError('Invalid NAM render: ' + rig['id'])
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temp = target.with_suffix('.tmp')
                        sf.write(temp, match_level(wet), SAMPLE_RATE, format='WAV', subtype='PCM_16')
                        temp.replace(target)
                        record = dict(id=key, path=relative, nam_id=rig['id'], di_id=take['id'],
                                      performance_group=take['performance_group'], split=take['split'],
                                      start_s=start, input_gain_db=gain, condition='clean',
                                      audio_sha256=digest(target))
                        state['records'][key] = record
                        if len(state['records']) % 32 == 0:
                            write_json(state_path, state)
                    rows.append(record)
        write_json(state_path, state)
        print('Rendered ' + rig['id'], flush=True)
    metadata = dict(schema=1, sample_rate=SAMPLE_RATE, seconds=plan['render']['seconds'], renderer=plan['renderer'],
                    plan_sha256=plan_hash, profiles=[{k:v for k,v in p.items() if k != 'path'} for p in plan['profiles']],
                    di=[{k:v for k,v in p.items() if k != 'path'} for p in plan['di']], records=rows)
    metadata['dataset_sha256'] = signature(metadata)
    write_json(output / 'metadata.json', metadata)
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', action='store_true', help='Validate files and estimate size without rendering')
    p.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='cpu')
    p.add_argument('--renderer', type=Path, help='Pinned NAM Core renderer for A1/A2 collections')
    a = p.parse_args()
    build(a.manifest, a.output, plan_only=a.plan, device=a.device, renderer=a.renderer)


if __name__ == '__main__':
    main()
