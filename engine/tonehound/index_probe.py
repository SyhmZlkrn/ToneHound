"""Resolve a user DI, existing development DI, or distributable synthetic probe."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import SAMPLE_RATE


def probe_info(root: Path) -> dict:
    configured = os.environ.get('TONEHOUND_DI_PATH', '').strip()
    if configured:
        path = Path(configured).expanduser().resolve()
        return {'path': str(path), 'kind': 'user_di', 'available': path.is_file(),
                'note': 'Custom dry guitar recording from TONEHOUND_DI_PATH.'}
    existing = root / 'assets/user_di/Djent DI.wav'
    if existing.is_file():
        return {'path': str(existing.resolve()), 'kind': 'user_di', 'available': True,
                'note': 'Existing local dry guitar recording.'}
    path = root / '.cache/probes/synthetic-guitar-v1.wav'
    return {'path': str(path.resolve()), 'kind': 'synthetic', 'available': path.is_file(),
            'note': 'Built-in synthetic guitar probe. Set TONEHOUND_DI_PATH to use your dry recording; '
                    'changing the probe rebuilds the index and can change rankings.'}


def resolve_probe(root: Path) -> Path:
    info = probe_info(root)
    path = Path(info['path'])
    if info['kind'] == 'user_di':
        if not path.is_file():
            raise FileNotFoundError('TONEHOUND_DI_PATH does not point to an existing audio file.')
        return path
    if not path.is_file():
        from .probe import build_phrase
        phrase = build_phrase(SAMPLE_RATE)
        # A consistent -9 dBFS peak with complete phrase coverage. This is an
        # openly generated fallback, not a claim of real-guitar equivalence.
        audio = np.tile(phrase, 2)[:15 * SAMPLE_RATE] * (10 ** (-9 / 20))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f'.{os.getpid()}.tmp.wav')
        sf.write(temporary, audio, SAMPLE_RATE, subtype='FLOAT')
        temporary.replace(path)
    return path
