"""Inference for locally installed ToneHound LoRA exports.

The adapter and learned head are inseparable; both sides of retrieval use the
same checkpoint. Runtime manifests are JSON and weights are safetensors only.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .audioio import to_rate
from .config import SAMPLE_RATE
from .embed import (EmbedConfig, EmbedError, MERT_SAMPLE_RATE, MIN_SECONDS,
                    _resolve_device, _unit, chunks, match_level)
from training.common import digest, signature


@dataclass(frozen=True)
class LoRAConfig(EmbedConfig):
    checkpoint: str = ''
    checkpoint_sha256: str = ''
    spec_json: str = '{}'
    name: str = 'LoRA (pilot)'

    def key(self) -> str:
        return signature(dict(base=super().key(), weights=self.checkpoint_sha256,
                              spec=json.loads(self.spec_json), inference=1))[:16]


def config_for(root: Path, model: str = 'standard', **kwargs) -> EmbedConfig:
    if model == 'standard':
        return EmbedConfig(**kwargs)
    if model != 'lora':
        raise ValueError('Choose Standard or LoRA (pilot) as the matching model.')
    folder = Path(root) / '.cache/matching_models/lora'
    try:
        manifest = json.loads((folder / 'model.json').read_text(encoding='utf-8'))
        sha = manifest['checkpoint_sha256']
        if manifest['schema'] != 1 or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('Invalid LoRA manifest')
        spec = manifest['model']
        if spec['mode'] != 'lora':
            raise ValueError('Expected a LoRA adapter and head')
        path = folder / (sha + '.safetensors')
        if digest(path) != sha:
            raise ValueError('LoRA checkpoint checksum mismatch')
        return LoRAConfig(model_id=spec['id'], revision=spec['revision'],
                          layers=tuple(spec['layers']), checkpoint=str(path),
                          checkpoint_sha256=sha, spec_json=json.dumps(spec, sort_keys=True),
                          name=manifest.get('name', 'LoRA (pilot)'), **kwargs)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise EmbedError('lora_unavailable',
                         'LoRA is missing or invalid. Reinstall the LoRA export or choose Standard. '
                         + str(exc)) from exc


class LoRAEmbedder:
    trained = True

    def __init__(self, cfg: LoRAConfig):
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        self._model = None

    @property
    def dim(self):
        return int(json.loads(self.cfg.spec_json)['embedding_dim'])

    @property
    def model(self):
        if self._model is None:
            import torch
            from safetensors.torch import load_file
            from training.models import ToneEncoder
            try:
                if digest(self.cfg.checkpoint) != self.cfg.checkpoint_sha256:
                    raise ValueError('Checkpoint changed since configuration was loaded')
                spec = dict(json.loads(self.cfg.spec_json), gradient_checkpointing=False)
                model = ToneEncoder(spec)
                model.load_trainable(load_file(self.cfg.checkpoint))
                model.eval().requires_grad_(False)
                try:
                    model.to(self.device)
                except torch.cuda.OutOfMemoryError:
                    model.to('cpu')
                    self.device = 'cpu'
                    torch.cuda.empty_cache()
                self._model = model
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                raise EmbedError('lora_load_failed',
                                 'Could not load LoRA. Reinstall the export or choose Standard. '
                                 + str(exc)) from exc
        return self._model

    @property
    def retrieval(self):
        return dict(mode='lora', name=self.cfg.name,
                    checkpoint_sha256=self.cfg.checkpoint_sha256,
                    base_model=self.cfg.model_id, experimental=True,
                    comparison='cosine', embedding_dim=self.dim)

    def embed(self, x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim not in (1, 2) or not x.size or not np.isfinite(x).all():
            raise EmbedError('invalid_audio', 'LoRA needs finite mono or stereo audio.')
        if len(x) < MIN_SECONDS * sr:
            raise EmbedError('too_short', 'At least one second of audio is needed for LoRA.')
        if x.ndim == 2:
            energy = np.mean(np.square(x, dtype=np.float64), axis=0)
            active = np.flatnonzero(energy > max(float(energy.max()) * 1e-4, 1e-12))
            if not len(active):
                raise EmbedError('silent_audio', 'The selected audio is silent.')
            return _unit(np.mean([self.embed(x[:, c], sr) for c in active], axis=0))
        if np.mean(np.square(x, dtype=np.float64)) < 1e-12:
            raise EmbedError('silent_audio', 'The selected audio is silent.')
        windows = chunks(to_rate(x, sr, MERT_SAMPLE_RATE), MERT_SAMPLE_RATE, self.cfg)
        # Training normalises each five-second render separately before the head.
        vectors = []
        for offset in range(0, len(windows), self.cfg.batch):
            group = windows[offset:offset + self.cfg.batch]
            vectors.append(self._forward(np.stack([match_level(w, self.cfg.level) for w in group])))
        result = _unit(np.concatenate(vectors).mean(axis=0))
        if result.shape != (self.dim,) or not np.isfinite(result).all() or np.linalg.norm(result) < .99:
            raise EmbedError('invalid_embedding', 'LoRA produced an invalid tone description.')
        return result

    def _forward(self, batch):
        import torch
        model = self.model
        try:
            context = (torch.autocast('cuda', dtype=getattr(torch, self.cfg.dtype))
                       if self.device.startswith('cuda') and self.cfg.dtype != 'float32'
                       else nullcontext())
            with torch.inference_mode(), context:
                return model(torch.from_numpy(batch).to(self.device)).float().cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            if self.device == 'cpu':
                raise
            model.to('cpu')
            self.device = 'cpu'
            torch.cuda.empty_cache()
            return self._forward(batch)


def create_embedder(cfg):
    if isinstance(cfg, LoRAConfig):
        return LoRAEmbedder(cfg)
    from .embed import MertEmbedder
    return MertEmbedder(cfg)
