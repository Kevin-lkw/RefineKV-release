"""PyTorch reference for the same summaries and selection rule as CUDA."""

import math

import torch

from .summary import BlockSummary


def selection_statistics(query: torch.Tensor, summary: BlockSummary):
    q = query[:, :, 0].to(torch.bfloat16).float()
    q2 = q.square().to(torch.bfloat16).float()
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("hgd,hbd->hgb", q, summary.key_mean.float()) * scale
    projected = torch.einsum("hgd,hbd->hgb", q2, summary.ellipsoid.float())
    delta = projected.clamp_min(0).sqrt() * scale * (1 + 2**-8)
    z = scores + summary.counts.float().log()[None, None, :]
    # Retain the small-delta algebra of the production selector.
    clipped = delta.clamp(max=10)
    log_cosh_direct = ((clipped.exp() + (-clipped).exp()) * 0.5).log()
    log_cosh_large = delta + ((1 + (-2 * delta).exp()) * 0.5).log()
    log_cosh = torch.where(delta <= 10, log_cosh_direct, log_cosh_large)
    log_z = torch.logsumexp(z, -1, keepdim=True)
    log_c = torch.logsumexp(z + log_cosh, -1, keepdim=True)
    t = clipped.exp()
    direct = 2 * summary.max_value_norm[:, None, None] * (z - log_c).exp()
    direct = direct * (0.5 * (t + t.reciprocal()) - 1)
    exp_neg = (-delta).exp()
    log_cosh_minus_one = delta + 2 * (1 - exp_neg).clamp_min(1e-30).log() - math.log(2)
    large = 2 * summary.max_value_norm[:, None, None] * (z - log_c + log_cosh_minus_one).exp()
    mass = torch.where(delta <= 10, direct, large)
    tanh_half = torch.where(delta <= 10, (t - 1) / (t + 1), (1 - exp_neg) / (1 + exp_neg))
    value = 2 * summary.value_norm[:, None] * (z - log_z).exp() * tanh_half
    return z, mass + value, log_z, log_c


def select_blocks_reference(query, summary, config):
    z, condition, log_z, log_c = selection_statistics(query, summary)
    if config.gqa_reduction == "mean":
        selected = condition.mean(1) > config.eps
    elif config.gqa_reduction == "max":
        selected = condition.amax(1) > config.eps
    else:
        selected = (condition > config.eps).any(1)
    bad_global = (~torch.isfinite(log_z) | ~torch.isfinite(log_c)).any(1)
    bad_local = (~torch.isfinite(condition)).any(1)
    return selected | bad_global | bad_local, z


@torch.no_grad()
def sparse_attention_reference(query, summary, key_suffix, value_suffix, config, anchor_mask=None):
    """Materialize mixed logits for numerical validation, not latency measurement."""
    selected, z = select_blocks_reference(query, summary, config)
    if anchor_mask is not None:
        selected = selected | anchor_mask
    outputs = []
    q = query[:, :, 0].to(torch.bfloat16).float()
    scale = q.shape[-1] ** -0.5
    for head in range(q.shape[0]):
        exact_mask = selected[head, :, None] & (
            torch.arange(summary.block_size, device=q.device)[None] < summary.counts[:, None]
        )
        exact_keys = summary.keys[head][exact_mask].to(torch.bfloat16).float()
        exact_values = summary.values[head][exact_mask].to(torch.bfloat16).float()
        exact_keys = torch.cat([exact_keys, key_suffix[head].to(torch.bfloat16).float()])
        exact_values = torch.cat([exact_values, value_suffix[head].to(torch.bfloat16).float()])
        exact_logits = q[head] @ exact_keys.T * scale
        rep_logits = z[head, :, ~selected[head]]
        rep_values = summary.value_mean[head, ~selected[head]].float()
        logits = torch.cat([rep_logits, exact_logits], -1)
        values = torch.cat([rep_values, exact_values], 0)
        outputs.append(logits.softmax(-1) @ values)
    return torch.stack(outputs).unsqueeze(2).to(query.dtype), selected
