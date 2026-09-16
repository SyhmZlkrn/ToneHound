"""Pinned MERT encoders, explicit trainable blocks and compact LoRA updates."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from training.common import read_json

SUPPORTED = {
    'm-a-p/MERT-v1-95M': ('12af15fef9d0ac838c3f475bfbbf26d2060dd4f5', 'pytorch_model.bin'),
    'dzur658/MERT-v1-330M-finetuned-gtzan': ('160034a919fc6a85f8ba075562f762744fdf53e8', 'model.safetensors'),
}


def load_backbone(spec):
    """These two inspected configs use stock HuBERT; never execute Hub code."""
    from huggingface_hub import hf_hub_download
    from transformers import HubertConfig, HubertModel
    from safetensors.torch import load_file
    name, revision = spec['id'], spec['revision']
    if name not in SUPPORTED or revision != SUPPORTED[name][0]:
        raise ValueError('Unreviewed MERT checkpoint; add a verified architecture/revision before training')
    raw = read_json(hf_hub_download(name, 'config.json', revision=revision))
    if (raw.get('feature_extractor_cqt') or raw.get('deepnorm') or
            not raw.get('feat_proj_layer_norm', True) or
            raw.get('sample_rate', 24000) != 24000):
        raise ValueError('Checkpoint requires a different architecture')
    allowed = set(HubertConfig().to_dict())
    ignored = {'architectures', 'model_type', 'auto_map', 'dtype', 'torch_dtype', 'transformers_version'}
    config = HubertConfig(**{k:v for k,v in raw.items() if k in allowed and k not in ignored})
    config._attn_implementation = 'eager'
    model = HubertModel(config)
    filename = SUPPORTED[name][1]
    path = hf_hub_download(name, filename, revision=revision)
    state = load_file(path) if filename.endswith('.safetensors') else torch.load(path, map_location='cpu', weights_only=True)
    body = {k.removeprefix('hubert.'):v for k,v in state.items() if k.startswith('hubert.')}
    if not body:
        body = {k:v for k,v in state.items() if not k.startswith(('classifier.', 'projector.'))}
    # Older HF checkpoints store weight_norm's g/v; PyTorch parametrizations
    # use original0/original1. This is a naming migration, not a weight reshape.
    expected = model.state_dict()
    for old, new in [('encoder.pos_conv_embed.conv.weight_g', 'encoder.pos_conv_embed.conv.parametrizations.weight.original0'),
                     ('encoder.pos_conv_embed.conv.weight_v', 'encoder.pos_conv_embed.conv.parametrizations.weight.original1')]:
        if old in body and new in expected:
            body[new] = body.pop(old)
    model.load_state_dict(body, strict=True)
    return model


class LoRALinear(nn.Module):
    """W x + (alpha/r) B A x. Base weights frozen, zero initial update."""
    def __init__(self, base, rank, alpha):
        super().__init__()
        if not isinstance(base, nn.Linear) or not 1 <= rank <= min(base.in_features, base.out_features):
            raise ValueError('Invalid LoRA target or rank')
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = float(alpha) / rank

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale


class ToneEncoder(nn.Module):
    def __init__(self, spec, backbone=None):
        super().__init__()
        self.spec = dict(spec)
        self.backbone = backbone if backbone is not None else load_backbone(spec)
        self.mode = spec['mode']
        self.layers = tuple(spec['layers'])
        total = len(self.backbone.encoder.layers)
        if not self.layers or min(self.layers) < 1 or max(self.layers) > total or len(set(self.layers)) != len(self.layers):
            raise ValueError('Feature layers are unique 1-based transformer output indices')
        self.backbone.requires_grad_(False)
        self.backbone.feature_extractor._freeze_parameters()
        # Explicit waveform views drive augmentation. Keep the frozen baseline
        # deterministic and do not randomly drop a trainable layer.
        self.backbone.config.apply_spec_augment = False
        self.backbone.config.layerdrop = 0.0
        for module in self.backbone.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        self.trainable_blocks = []
        if self.mode != 'frozen':
            last = max(self.layers)
            count = spec['train_blocks']
            if not 1 <= count <= last:
                raise ValueError('Invalid train_blocks')
            # Tune the last USED blocks. Tuning 330M layers 23/24 while reading
            # only layers 8-11 would leave the retrieval loss disconnected.
            self.trainable_blocks = list(range(last-count, last))
            for i in self.trainable_blocks:
                block = self.backbone.encoder.layers[i]
                if self.mode == 'unfreeze_last':
                    block.requires_grad_(True)
                elif self.mode == 'lora':
                    for name in ('q_proj', 'v_proj'):
                        base = getattr(block.attention, name)
                        setattr(block.attention, name, LoRALinear(base, spec['lora_rank'], spec['lora_alpha']))
                else:
                    raise ValueError('Unknown model mode')
            if spec.get('gradient_checkpointing', True):
                self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        dim = self.backbone.config.hidden_size * len(self.layers)
        self.head = nn.Sequential(nn.Linear(dim, spec['head_hidden']), nn.GELU(),
                                  nn.Linear(spec['head_hidden'], spec['embedding_dim']))

    def train(self, mode=True):
        super().train(mode)
        if self.mode == 'frozen':
            self.backbone.eval()
        return self

    def features(self, x):
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.mode != 'frozen'):
            states = self.backbone(x, output_hidden_states=True, return_dict=True).hidden_states
            pooled = [F.normalize(states[i].float().mean(dim=1), dim=-1) for i in self.layers]
            return F.normalize(torch.cat(pooled, dim=-1), dim=-1)

    def forward(self, x):
        return F.normalize(self.head(self.features(x)), dim=-1)

    def trainable_state(self):
        return {name:p.detach().cpu().contiguous() for name,p in self.named_parameters() if p.requires_grad}

    def load_trainable(self, state):
        expected = self.trainable_state()
        if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in expected):
            raise ValueError('Checkpoint does not match trainable model parameters')
        if any(not torch.isfinite(t).all() for t in state.values()):
            raise ValueError('Checkpoint contains non-finite weights')
        self.load_state_dict(state, strict=False)
