from collections.abc import Callable, Iterable
from dataclasses import asdict
from typing import Optional
import torch
from torch import nn
import math
from transformer import ModelConfig, TransformerLM


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):  # type: ignore
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("step", 0)
                grad = p.grad.data
                p.data -= grad * lr / (math.sqrt(t + 1))
                state["step"] = t + 1

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, lmda=0.01):
        """
        lr: learning rate
        beta1: decay rate for the first moment
        beta2: decay rate for the second moment
        eps: small value to avoid division by zero
        lmda: weight decay
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {beta2}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= lmda:
            raise ValueError(f"Invalid weight decay value: {lmda}")

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, lmda=lmda)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):  # type: ignore
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            lmda = group["lmda"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 1
                    state["mv"] = (torch.zeros_like(grad), torch.zeros_like(grad))

                t = state["t"]
                m, v = state["mv"]

                beta1_t = beta1**t
                beta2_t = beta2**t
                sqrt_one_minus_beta2_t = math.sqrt(1 - beta2_t)
                lr_t = lr * sqrt_one_minus_beta2_t / (1 - beta1_t)  # bias correction
                p.data.mul_(1 - lr * lmda)  # weight decay
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                p.data.add_(
                    -(lr_t * m / (torch.sqrt(v) + eps * sqrt_one_minus_beta2_t))
                )

                state["t"] += 1
                state["mv"] = (m, v)

        return loss


class CosineAnnealingLRwithWarmup(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_warmup: int,
        T_decay_end: int,
        lr_min: float,
        last_epoch: int = -1,
    ):
        """
        Cosine learning rate schedule with warmup.

        T_warmup: number of warmup steps
        T_decay_end: step at which cosine annealing ends
        lr_min: minimum learning rate for all parameter groups
        """
        self.optimizer = optimizer
        self.T_warmup = T_warmup
        self.T_decay_end = T_decay_end
        self.lr_min = lr_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        t = self.last_epoch

        if t < self.T_warmup:
            factor = t / self.T_warmup
        elif t < self.T_decay_end:
            factor = 0.5 * (
                1
                + math.cos(
                    math.pi * (t - self.T_warmup) / (self.T_decay_end - self.T_warmup)
                )
            )
        else:
            factor = 0.0

        return [
            self.lr_min + (base_lr - self.lr_min) * factor for base_lr in self.base_lrs
        ]


def gradient_clipping_(
    parameters: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6
) -> None:
    """
    Clips the gradients of the given parameters to have a maximum norm of max_norm.
    parameters: iterable of parameters whose gradients will be clipped
    max_norm: maximum allowed norm of the gradients
    """
    parameters = list(parameters)
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += p.grad.norm(2).item() ** 2
    total_norm = total_norm**0.5
    clip_coef = max_norm / (total_norm + eps)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    out,  # file path or file-like object passed to torch.save
    train_config: Optional[dict] = None,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "train_config": train_config, # old checkpoints may not have this key
        "model_config": asdict(model.get_config()),
    }
    torch.save(checkpoint, out)


def create_training_objects(
    model_config: ModelConfig, train_config
) -> tuple[TransformerLM, AdamW, CosineAnnealingLRwithWarmup]:
    model = TransformerLM.from_config(model_config)
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.max_lr,
        beta1=train_config.beta1,
        beta2=train_config.beta2,
        eps=train_config.eps,
        lmda=train_config.weight_decay,
    )
    scheduler = CosineAnnealingLRwithWarmup(
        optimizer,
        train_config.warmup_steps,
        train_config.total_steps,
        train_config.min_lr,
    )
    return model, optimizer, scheduler


def _config_value(config, key: str):
    if hasattr(config, key):
        return getattr(config, key)
    return config.get(key)


def _configs_align(saved_config: dict, current_config) -> bool:
    trajectory_keys = (
        "train_token_budget",
        "batch_size",
        "gradient_accumulation_steps",
        "max_lr",
        "min_lr",
        "weight_decay",
        "beta1",
        "beta2",
        "eps",
    )
    return all(
        saved_config.get(key) == _config_value(current_config, key)
        for key in trajectory_keys
    )


def _infer_resume_step(
    checkpoint_step: int, saved_config: dict, train_config, context_length: int
) -> int:
    consumed_tokens = (
        (checkpoint_step + 1)
        * saved_config["batch_size"]
        * saved_config["gradient_accumulation_steps"]
        * context_length
    )
    current_tokens_per_step = (
        _config_value(train_config, "batch_size")
        * _config_value(train_config, "gradient_accumulation_steps")
        * context_length
    )
    return consumed_tokens // current_tokens_per_step - 1


def load_checkpoint(
    src,  # file path or file-like object passed to torch.load
    train_config,
    device: str | torch.device | None = None,
):
    target_device = torch.device(device) if device is not None else None
    default_model_config = ModelConfig(
        device=str(target_device) if target_device is not None else ModelConfig().device
    )
    checkpoint = torch.load(
        src,
        map_location=target_device or torch.device(default_model_config.device),
    )
    saved_model_config = checkpoint.get("model_config")
    model_config = ModelConfig(**saved_model_config) if saved_model_config else default_model_config
    if target_device is not None:
        model_config.device = str(target_device)
    model, optimizer, scheduler = create_training_objects(model_config, train_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    saved_config = checkpoint.get("train_config")
    if saved_config is None:
        saved_config = asdict(type(train_config)())

    checkpoint_step = checkpoint["step"]
    if _configs_align(saved_config, train_config):
        # Restore the initialized scheduler before optimizer state, as required
        # by PyTorch because optimizer restoration overwrites param-group rates.
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return model, optimizer, scheduler, checkpoint_step

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    step = _infer_resume_step(
        checkpoint_step, saved_config, train_config, model_config.context_length
    )
    for parameter_group in optimizer.param_groups:
        parameter_group.update(
            lr=train_config.max_lr,
            initial_lr=train_config.max_lr,
            beta1=train_config.beta1,
            beta2=train_config.beta2,
            eps=train_config.eps,
            lmda=train_config.weight_decay,
        )
    scheduler = CosineAnnealingLRwithWarmup(
        optimizer,
        train_config.warmup_steps,
        train_config.total_steps,
        train_config.min_lr,
        last_epoch=step,
    )
    return model, optimizer, scheduler, step
