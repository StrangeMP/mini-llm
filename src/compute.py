import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_val = x.max(dim=dim, keepdim=True).values
    # If a whole row is -inf, clamp max_val to 0.0 to prevent (-inf) - (-inf) = NaN
    max_val = torch.where(torch.isneginf(max_val), torch.zeros_like(max_val), max_val)

    y = torch.exp(x - max_val)
    return y / (torch.sum(y, dim=dim, keepdim=True) + 1e-9)


def cross_entropy_loss(
    logits: torch.Tensor, targets: torch.Tensor, ignore_idx: int = -100
):
    # logits.shape = (*batch_dims, seq_len, vocab_size)
    # targets.shape = (*batch_dims, seq_len)
    M, _ = torch.max(logits, dim=-1, keepdim=True)
    logsumexp = torch.log(torch.sum(torch.exp(logits - M), dim=-1, keepdim=True)) + M
    logsumexp = logsumexp.squeeze(-1)

    safe_targets = targets.clamp(min=0)
    p_truth = torch.gather(logits, dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)

    token_loss = logsumexp - p_truth
    mask = (targets != ignore_idx).to(logits.dtype)
    active_tokens = mask.sum()

    if active_tokens > 0:
        avg_loss = torch.sum(token_loss * mask) / active_tokens
    else:
        avg_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
    return avg_loss
