"""File-based jobs for the native standalone/VST3 editor.

No listener or browser is involved. Each native instance launches its own job
process; matching never runs in the DAW's audio callback. Credentials remain in
the existing TONE3000 client and never appear in a native response.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np
import soundfile as sf

from . import ingest
from .config import SAMPLE_RATE
from .tone3000 import Tone3000Client, parse_model_reference

ROOT = Path(__file__).resolve().parents[2]
MIN_CROP_SECONDS = 30.0
DEFAULT_CROP_SECONDS = 40.0


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    for attempt in range(5):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.02)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def profile_row(path: Path, cache: Path) -> dict:
    tone_id, model_id = parse_model_reference(path)
    meta = read_json(cache.parent / 'tone3000/metadata' / f'{tone_id}.json') if tone_id else {}
    if not tone_id:
        from .community import metadata
        meta = metadata(path, cache.parent.parent)
    photo = read_json(cache / 'artwork' / f'{tone_id}.json') if tone_id else {}
    meta = {**meta, **photo}
    return {'path': str(path.resolve()), 'name': meta.get('title') or path.stem,
            'model_name': path.stem, 'tone_id': tone_id, 'model_id': model_id,
            'creator': meta.get('creator', ''), 'gear': meta.get('gear', ''),
            'description': meta.get('description', ''), 'makes': meta.get('makes', []),
            'metadata_version': meta.get('metadata_version', 0),
            'models': meta.get('models', []), 'tags': meta.get('tags', []),
            'image_path': meta.get('image_path', ''),
            'url': meta.get('url') or (f'https://www.tone3000.com/tones/{tone_id}' if tone_id else '')}


def library(root: Path) -> dict:
    cache = root / '.cache/native'
    directories = [root / '.cache/tone3000/profiles', root / 'assets/dev_profiles']
    rows = [profile_row(p, cache) for d in directories for p in sorted(d.glob('*.nam'))
            if not p.name.startswith('_')]
    from .community import entries as community_entries
    rows.extend(profile_row(e.path, cache) for e in community_entries(root))
    from .catalogue import Catalogue
    catalogue = Catalogue(root)
    known = {r['path'] for r in rows}
    for entry in catalogue.rows():
        path = catalogue.path(entry)
        if str(path.resolve()) not in known:
            rows.append(profile_row(path,cache))
    result = {'profiles': rows, 'min_crop_seconds': MIN_CROP_SECONDS,
              'default_crop_seconds': DEFAULT_CROP_SECONDS, 'storage': catalogue.summary()}
    write_json(cache / 'library.json', result)
    return result


def artwork(root: Path, tone_id: int) -> dict:
    import io
    import requests
    from PIL import Image
    from urllib.parse import urlparse

    directory = root / '.cache/native/artwork'
    cached = read_json(directory / f'{tone_id}.json')
    if cached.get('metadata_version') == 2 and (not cached.get('image_path') or Path(cached['image_path']).is_file()):
        return cached
    client = Tone3000Client(cache_dir=root / '.cache')
    try:
        tone = client.get_tone(tone_id)
    except Exception:
        # Existing artwork remains usable offline or if OAuth needs reconnecting.
        if cached:
            return cached
        raise
    from .library_manager import tone_metadata, _store_metadata
    _store_metadata(root, tone)
    result = dict(tone_metadata(tone), image_path=cached.get('image_path', ''),
                  image_url=cached.get('image_url', ''))
    for url in tone.raw.get('images') or []:
        if not isinstance(url, str) or urlparse(url).scheme != 'https':
            continue
        # Artwork requests deliberately carry no OAuth/Authorization headers.
        try:
            with requests.get(url, timeout=20, stream=True) as response:
                response.raise_for_status()
                data = bytearray()
                for chunk in response.iter_content(65536):
                    data.extend(chunk)
                    if len(data) > 12 * 1024 * 1024:
                        raise ValueError('Artwork exceeds 12 MB')
            with Image.open(io.BytesIO(data)) as original:
                original.thumbnail((1400, 1000))
                image = original.convert('RGB')
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f'{tone_id}.jpg'
                image.save(path, quality=94)
            result.update(image_path=str(path.resolve()), image_url=url)
            break
        except (OSError, ValueError, requests.RequestException):
            continue
    # A failed image download is retryable; a tone with no image is a valid state.
    if result['image_path'] or not tone.raw.get('images'):
        write_json(directory / f'{tone_id}.json', result)
    return result


def import_song(root: Path, spec: str, progress: Callable) -> dict:
    cache = root / '.cache/native'
    progress('importing', -1, 'Opening reference audio')
    source = ingest.load(spec, cache_dir=root / '.cache/downloads', seconds=None, mono=False,
                         progress=lambda stage, f: progress(stage, f, 'Downloading reference audio'))
    if source.duration_s < MIN_CROP_SECONDS:
        raise ValueError('Choose a song at least 30 seconds long.')
    key = hashlib.sha256(b'stereo-v2' + str(source.samples.shape).encode() + source.samples.tobytes()).hexdigest()[:24]
    path = cache / 'media' / f'{key}.wav'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        sf.write(path, source.samples, SAMPLE_RATE, subtype='FLOAT')
    return {'path': str(path.resolve()), 'name': source.name,
            'duration': source.duration_s, 'origin': source.origin,
            'channels': source.samples.shape[1] if source.samples.ndim == 2 else 1,
            'source_path': str(source.path.resolve()) if source.path else '',
            'url': source.url or '', 'start': 0.0,
            'end': min(DEFAULT_CROP_SECONDS, source.duration_s)}


def validate_crop(path: Path, start: float, end: float) -> float:
    if not math.isfinite(start) or not math.isfinite(end) or start < 0:
        raise ValueError('Selection times must be finite and non-negative.')
    duration = sf.info(path).duration
    if end - start < MIN_CROP_SECONDS - 1 / SAMPLE_RATE:
        raise ValueError('Select at least 30 seconds of the song.')
    if end > duration + 1 / SAMPLE_RATE:
        raise ValueError('Selection extends beyond the end of the song.')
    return end - start


def cache_stem(root: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    """Reuse identical stereo EQ targets, with atomic writes and verified audio.

    Identity covers the canonical float32 audio, its shape and sample rate.
    Old PID-named files remain valid for existing saved sessions and are kept.
    """
    audio = np.asarray(samples, dtype='<f4', order='C')
    if audio.ndim not in (1, 2) or not len(audio):
        raise ValueError('Separated guitar audio must contain samples.')
    channels = audio.shape[1] if audio.ndim == 2 else 1
    header = json.dumps({'version': 1, 'shape': audio.shape, 'sample_rate': sample_rate},
                        sort_keys=True).encode('ascii')
    digest = hashlib.sha256(header)
    digest.update(memoryview(audio).cast('B'))
    key = digest.hexdigest()
    path = root / '.cache/native/media' / f'stem-{key}.wav'
    if path.is_file():
        try:
            info = sf.info(path)
            if (info.samplerate == sample_rate and info.channels == channels and
                    info.frames == len(audio) and info.subtype == 'FLOAT'):
                cached_digest = hashlib.sha256(header)
                for block in sf.blocks(path, blocksize=65536, dtype='float32', always_2d=True):
                    cached_digest.update(memoryview(np.asarray(block, dtype='<f4', order='C')).cast('B'))
                if cached_digest.hexdigest() == key:
                    return path
        except (OSError, RuntimeError, ValueError):
            pass  # A truncated/invalid cache file is repaired atomically below.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.tmp.wav')
    try:
        sf.write(temporary, audio, sample_rate, subtype='FLOAT')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def match_song(root: Path, request: dict, progress: Callable) -> dict:
    from .lora import config_for
    from .pipeline import Pipeline, PipelineConfig
    from .index_probe import resolve_probe
    from .reference_features import analyse_reference

    path = Path(request['path'])
    start, end = float(request['start']), float(request['end'])
    duration = validate_crop(path, start, end)
    role = str(request.get('role', 'rhythm'))
    if role not in ('solo', 'rhythm'):
        raise ValueError('Choose either rhythm or solo for the selected passage.')
    profile_dir = root / '.cache/tone3000/profiles'
    from .catalogue import Catalogue
    if not any(profile_dir.glob('*.nam')) and not Catalogue(root).rows():
        profile_dir = root / 'assets/dev_profiles'
    cfg = PipelineConfig(profile_dir=profile_dir, cache_dir=root / '.cache',
                         di_path=resolve_probe(root),
                         seconds=duration, start_s=start, top=6,
                         embed=config_for(root, request.get('matching_model', 'standard'),
                                          dtype='float16', max_chunks=max(24, math.ceil(duration / 5))),
                         separator_autocast=True, render_device='auto', persistent_catalogue=True)
    result = Pipeline(cfg).match(path, progress=progress)
    payload = result.to_json()
    if payload.get('retrieval', {}).get('mode') == 'lora':
        payload['caveat'] = ('Matching model: ' + payload['retrieval']['name']
                            + '. Experimental; the pilot did not beat its frozen projection baseline. '
                            + payload['caveat'])
    for match in payload['matches']:
        match.update(profile_row(Path(match['path']), root / '.cache/native'))
    stem_path = cache_stem(root, result.stem.samples)
    payload['stem_path'] = str(stem_path.resolve())
    progress('tempo', .98, 'Estimating the passage tempo and preparing effects')
    features, suggestion = analyse_reference(result.source.samples, SAMPLE_RATE, role)
    payload['reference_features'] = features
    payload['effect_suggestion'] = suggestion
    return payload


def run(request: dict, root: Path = ROOT, progress: Callable | None = None) -> dict:
    progress = progress or (lambda *_: None)
    action = request.get('action')
    if action == 'library':
        return library(root)
    if action == 'profile_resolve':
        from .catalogue import Catalogue
        catalogue = Catalogue(root)
        path = catalogue.ensure(str(request['filename']))
        catalogue.prune(keep=(path.name,))
        return {'path': str(path.resolve())}
    if action == 'catalogue_index':
        from .pipeline import Pipeline, PipelineConfig
        from .lora import config_for
        from .index_probe import resolve_probe
        from .catalogue import Catalogue
        cfg=PipelineConfig(profile_dir=root/'.cache/tone3000/profiles',cache_dir=root/'.cache',
            di_path=resolve_probe(root),embed=config_for(root, request.get('matching_model', 'standard'), dtype='float16'),render_device='auto',persistent_catalogue=True,
            refresh_descriptors=bool(request.get('rebuild',False)))
        prepared = Pipeline(cfg).index(progress=progress)
        library(root)
        summary = dict(Catalogue(root).summary(), **Catalogue(root).prune())
        summary['tone3000_indexed_count'] = summary['indexed_count']
        summary['indexed_count'] = len(prepared)
        summary['local_full_rig_count'] = sum(e.source=='community' for e in prepared.entries)
        summary['retrieval'] = getattr(prepared, 'learning', {})
        return summary
    if action in ('catalogue_import', 'catalogue_export'):
        from .catalogue import Catalogue
        catalogue=Catalogue(root)
        path=Path(request['path'])
        if action=='catalogue_import':
            result=catalogue.import_pack(path)
            library(root)
            return result
        catalogue.export(path)
        return {'path':str(path.resolve()),'bytes':path.stat().st_size}
    if action == 'artwork':
        return artwork(root, int(request['tone_id']))
    if action == 'import':
        return import_song(root, str(request['source']), progress)
    if action == 'match':
        return match_song(root, request, progress)
    if action in ('library_status', 'library_search', 'library_fetch', 'library_connect', 'library_disconnect'):
        from . import library_manager
        from .index_probe import probe_info
        client = Tone3000Client(cache_dir=root / '.cache')
        if action == 'library_status':
            from .catalogue import Catalogue
            return dict(library_manager.status(root, client), index_source=probe_info(root), storage=Catalogue(root).summary())
        if action == 'library_search':
            progress('catalogue', -1, 'Searching TONE3000')
            return library_manager.search(root, client, request)
        if action == 'library_fetch':
            result = library_manager.fetch(root, client, request, progress)
            result['profiles'] = library(root)['profiles']
            return result
        if action == 'library_connect':
            progress('connecting', -1, 'Complete sign-in in your browser, then return to ToneHound')
            client.connect(print_url=False)
        else:
            client.disconnect()
        return dict(library_manager.status(root, client), index_source=probe_info(root))
    raise ValueError(f'Unknown native job: {action}')


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--response', type=Path, required=True)
    parser.add_argument('--progress', type=Path, required=True)
    args = parser.parse_args()
    os.environ['PATH'] = str(ROOT / '.cache/bin') + os.pathsep + os.environ.get('PATH', '')
    log = args.response.with_suffix('.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as handle, contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
        try:
            request = json.loads(args.request.read_text(encoding='utf-8'))
            def progress(stage, fraction, message):
                write_json(args.progress, {'stage': stage, 'fraction': fraction, 'message': message})
            payload = run(request, progress=progress)
            write_json(args.response, {'ok': True, 'result': payload})
            return 0
        except Exception as exc:
            import traceback
            traceback.print_exc()
            write_json(args.response, {'ok': False, 'message': str(exc),
                                      'code': getattr(exc, 'code', 'job_failed')})
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
