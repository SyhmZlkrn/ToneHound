"""Bounded, user-driven catalogue browsing and incremental local expansion.

Browsing never downloads captures. Fetching touches only selected Tone IDs and
stops at the requested model count. Rendering/MERT work happens on the next
match, where ToneIndex reuses every unchanged vector in its existing cache.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

from .tone3000 import Tone, Tone3000Client, Tone3000Error, parse_model_reference

MAX_PAGE_SIZE = 25
MAX_SELECTED_TONES = 5
MAX_MODELS_PER_BATCH = 25
MAX_MODELS_PER_TONE = 5


def _integer(value, name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ValueError(f'{name} must be a whole number from {lo} to {hi}.')
    return value


def tone_metadata(tone: Tone) -> dict:
    return {'tone_id': tone.id, 'id': tone.id, 'title': tone.title,
            'creator': tone.creator, 'description': tone.description,
            'gear': tone.gear, 'makes': tone.makes, 'models': tone.gear_models,
            'tags': tone.tags, 'url': tone.url or f'https://www.tone3000.com/tones/{tone.id}',
            'metadata_version': 2}


def _store_metadata(root: Path, tone: Tone) -> None:
    directory = root / '.cache/tone3000/metadata'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{tone.id}.json'
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(tone_metadata(tone), ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def status(root: Path, client: Tone3000Client) -> dict:
    profiles = list((root / '.cache/tone3000/profiles').glob('*.nam'))
    return {'configured': client.configured, 'connected': client.connected,
            'profile_count': len(profiles), 'size_bytes': sum(p.stat().st_size for p in profiles),
            'limits': {'page_size': MAX_PAGE_SIZE, 'selected_tones': MAX_SELECTED_TONES,
                       'models_per_batch': MAX_MODELS_PER_BATCH,
                       'models_per_tone': MAX_MODELS_PER_TONE}}


def search(root: Path, client: Tone3000Client, request: dict) -> dict:
    page = _integer(request.get('page', 1), 'Page', 1, 10000)
    size = _integer(request.get('page_size', 20), 'Page size', 1, MAX_PAGE_SIZE)
    query = str(request.get('query', '')).strip()[:200]
    sort = request.get('sort', 'best-match' if query else 'trending')
    if sort not in ('trending', 'newest', 'oldest', 'best-match', 'downloads-all-time'):
        raise ValueError('Unknown catalogue sort order.')
    payload = client.search_tones(query, page=page, page_size=size, sort=sort,
                                  architecture=1)
    tones = [Tone.from_json(row) for row in (payload.get('data') or [])[:size]]
    local_ids = {parse_model_reference(p)[0] for p in (root / '.cache/tone3000/profiles').glob('*.nam')}
    # Search metadata is sufficient for result labels; richer get_tone metadata
    # is fetched only for explicit downloads/artwork.
    return {'tones': [dict(tone_metadata(t), downloaded=t.id in local_ids) for t in tones],
            'page': page, 'page_size': size, 'total_pages': int(payload.get('total_pages') or 1),
            'query': query, 'note': 'Browse freely; only selected captures are downloaded.'}


def _validate_profile(path: Path) -> dict:
    from .nam_render import NamModel
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not isinstance(data.get('weights'), list):
        raise ValueError('Downloaded file is not a NAM capture.')
    weights = np.asarray(data['weights'], dtype=np.float64)
    if not weights.size or not np.all(np.isfinite(weights)):
        raise ValueError('Capture has missing or non-finite weights.')
    # Actual construction verifies architecture, configuration and exact weight
    # count, so a bad remote file cannot poison every subsequent library match.
    model = NamModel(data, source=str(path))
    if not 8000 <= model.sample_rate <= 192000:
        raise ValueError('Unsupported capture sample rate.')
    return {'architecture': model.architecture, 'sample_rate': model.sample_rate}


def fetch(root: Path, client: Tone3000Client, request: dict, progress: Callable) -> dict:
    ids = request.get('tone_ids')
    if not isinstance(ids, list) or not 1 <= len(ids) <= MAX_SELECTED_TONES:
        raise ValueError(f'Select between 1 and {MAX_SELECTED_TONES} tones.')
    ids = list(dict.fromkeys(_integer(v, 'Tone ID', 1, 2**63 - 1) for v in ids))
    per_tone = _integer(request.get('models_per_tone', 1), 'Models per tone', 1, MAX_MODELS_PER_TONE)
    limit = _integer(request.get('max_models', len(ids) * per_tone), 'Batch size', 1, MAX_MODELS_PER_BATCH)
    directory = root / '.cache/tone3000/profiles'
    incoming = root / '.cache/tone3000/incoming'
    directory.mkdir(parents=True, exist_ok=True)
    existing = {parse_model_reference(p)[1]: p for p in directory.glob('*.nam')}
    added, skipped, errors = [], [], []
    examined = 0
    for index, tone_id in enumerate(ids):
        if examined >= limit:
            break
        progress('catalogue', index / len(ids), f'Opening selected tone {index + 1}/{len(ids)}')
        tone = client.get_tone(tone_id)
        _store_metadata(root, tone)
        # One explicit bounded page: selecting a tone never pulls all its
        # variants. More can be selected in a subsequent batch.
        models = client.list_models(tone_id, architecture=1, page=1, page_size=50)
        accepted = 0
        for model in models:
            if examined >= limit or accepted >= per_tone:
                break
            if model.architecture not in ('', '1'):
                continue
            if not model.tone_id:
                model.tone_id = tone_id
            if model.tone_id != tone_id:
                errors.append({'tone_id': tone_id, 'model_id': model.id, 'message': 'Model belongs to a different tone.'})
                continue
            if model.id in existing:
                skipped.append({'tone_id': tone_id, 'model_id': model.id, 'reason': 'already_downloaded'})
                continue
            examined += 1
            progress('downloading', examined / limit, f'Downloading capture {examined}/{limit}: {model.name}')
            path = None
            try:
                path = client.download_model(model, incoming)
                details = _validate_profile(path)
                destination = directory / path.name
                path.replace(destination)
                existing[model.id] = destination
                added.append({'tone_id': tone_id, 'model_id': model.id, 'path': str(destination.resolve()),
                              'model_name': model.name, **details})
                if request.get('managed_cache', False):
                    from .catalogue import Catalogue
                    from .tone_index import ToneEntry, file_key
                    Catalogue(root).register(ToneEntry(file_key(destination), destination.stem, destination,
                                              'tone3000',tone_id,model.id),managed=True)
                accepted += 1
            except Tone3000Error:
                # A quota/auth/server error must stop the batch immediately;
                # retrying the rest would spend more requests without progress.
                raise
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                errors.append({'tone_id': tone_id, 'model_id': model.id,
                               'message': f'Capture could not be loaded: {type(exc).__name__}'})
            finally:
                # Remove only this rejected staging file. Existing user models
                # are never deleted or overwritten by an invalid download.
                if path is not None and path.exists():
                    path.unlink()
    progress('complete', 1.0, f'Added {len(added)} capture(s); new captures are indexed on your next match')
    return {'added': added, 'skipped': skipped, 'errors': errors,
            'profile_count': len(list(directory.glob('*.nam'))), 'needs_indexing': bool(added)}
