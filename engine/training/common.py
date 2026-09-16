"""Portable paths, provenance and atomic experiment files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import uuid


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def inside(root, relative):
    root = Path(root).resolve()
    if '\\' in str(relative) or PureWindowsPath(relative).drive:
        raise ValueError('Use a portable relative path inside the data folder')
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Use a portable relative path inside the data folder')
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path escapes the data folder')
    return path


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', value):
        raise ValueError('IDs must be 1-80 letters, digits, dots, underscores or hyphens')
    return value


def read_config(path):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(cfg, dict) or cfg.get('schema') != 1:
        raise ValueError('Expected a schema 1 training config')
    if cfg['model']['mode'] not in ('frozen', 'lora', 'unfreeze_last'):
        raise ValueError('Unknown training mode')
    if not re.fullmatch('[0-9a-f]{40}', cfg['model']['revision']):
        raise ValueError('Pin the model revision to a Hugging Face commit hash')
    t = cfg['training']
    if (t['epochs'] < 1 or t['classes_per_batch'] < 2 or t['takes_per_class'] < 2
            or t['learning_rate'] <= 0 or t['temperature'] <= 0 or t['accumulation'] < 1
            or t['steps_per_epoch'] < 0 or t['checkpoint_every_steps'] < 1):
        raise ValueError('Invalid training settings; need >=2 rigs and >=2 DI takes per batch')
    if t['precision'] not in ('float32', 'float16', 'bfloat16'):
        raise ValueError('Unsupported precision')
    return cfg
