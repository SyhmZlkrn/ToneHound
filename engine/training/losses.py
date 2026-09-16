"""Supervised contrastive learning with cross-performance positives."""
import torch
import torch.nn.functional as F


def tone_contrastive_loss(embeddings, labels, performances, temperature=0.1):
    if temperature <= 0 or len(embeddings) < 4:
        raise ValueError('Need positive temperature and >=4 examples')
    z = F.normalize(embeddings.float(), dim=-1)
    logits = z @ z.T / temperature
    same_rig = labels[:, None] == labels[None, :]
    different_take = torch.tensor([[a != b for b in performances] for a in performances], device=z.device)
    positive = same_rig & different_take
    # Same-rig/same-performance views are neither positives nor false negatives.
    allowed = ~same_rig | positive
    if not positive.any(dim=1).all() or not (~same_rig).any(dim=1).all():
        raise ValueError('Each anchor needs a same-rig different-DI positive and a different-rig negative')
    logp = logits - torch.logsumexp(logits.masked_fill(~allowed, -torch.inf), dim=1, keepdim=True)
    loss = -(logp.masked_fill(~positive, 0).sum(dim=1) / positive.sum(dim=1)).mean()
    if not torch.isfinite(loss):
        raise ValueError('Non-finite contrastive loss')
    return loss
