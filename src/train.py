from dataclasses import asdict, dataclass
import argparse
import re
from pathlib import Path
from datasets import load_from_disk
import torch, wandb, os
from torch.utils.data import DataLoader
from transformer import TransformerLM, ModelConfig
from trainutils import (
    AdamW,
    CosineAnnealingLRwithWarmup,
    gradient_clipping_,
    load_checkpoint,
    save_checkpoint,
)


parser = argparse.ArgumentParser(description="Train or continue training the mini-LLM")
parser.add_argument(
    "--resume-from",
    type=str,
    default=None,
    help="Path to a checkpoint produced by this training script",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Torch device for the model, for example cuda or cpu",
)
parser.add_argument("--wandb-run-name", type=str, default=None)
args = parser.parse_args()

DISKROOT = "/FS1"
dataset_base_path = os.path.join(DISKROOT, "datasets/openwebtext_tokenized_chunked")

model_config = ModelConfig( # gpt-2 small configuration
    vocab_size=50257,
    context_length=1024,
    d_model=768,
    n_layer=12,
    n_head=12,
    rope_base=10000.0,
    device=args.device,
)

@dataclass
class TrainingConfig:
    train_token_budget: int = 20 * 162 * 10**6 # 20 * 162M tokens, where 162M is the size of model weights
    eval_token_budget: int = train_token_budget // 100
    train_examples_budget: int = train_token_budget // model_config.context_length
    eval_examples_budget: int = eval_token_budget // model_config.context_length
    batch_size: int = 32
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    warmup_steps: int = 1000
    total_steps: int = train_examples_budget // batch_size
    eval_num_batches: int = eval_examples_budget // batch_size
    save_interval: int = 1000
    eval_interval: int = 1000
    train_dataset_path: str = dataset_base_path + "_train"
    eval_dataset_path: str = dataset_base_path + "_eval"
    model_save_path: str = os.path.join(DISKROOT, "models/mini-llm")
    wandb_project: str = "mini-llm"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None

train_config = TrainingConfig(wandb_run_name=args.wandb_run_name)
os.makedirs(train_config.model_save_path, exist_ok=True)


def resolve_resume_checkpoint(resume_from: str | None, checkpoint_dir: str) -> str | None:
    if resume_from is None or resume_from != "latest":
        return resume_from

    checkpoint_pattern = re.compile(r"checkpoint_(?:interrupt_)?step_(\d+)\.pt$")
    candidates: list[tuple[int, float, Path]] = []
    for checkpoint_path in Path(checkpoint_dir).glob("checkpoint*.pt"):
        match = checkpoint_pattern.fullmatch(checkpoint_path.name)
        if match is not None:
            candidates.append(
                (int(match.group(1)), checkpoint_path.stat().st_mtime, checkpoint_path)
            )

    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    _, _, latest_checkpoint = max(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    return str(latest_checkpoint)


resume_from = resolve_resume_checkpoint(
    args.resume_from, train_config.model_save_path
)
wandb.init(
    project=train_config.wandb_project,
    entity=train_config.wandb_entity,
    name=train_config.wandb_run_name,
    config=asdict(train_config),
)


train_dataset = load_from_disk(train_config.train_dataset_path)
train_dataset.set_format(type="torch", columns=["input_ids"])
train_loader = DataLoader(
    train_dataset,
    batch_size=train_config.batch_size,
    shuffle=True,
)
eval_dataset = load_from_disk(train_config.eval_dataset_path)
eval_dataset.set_format(type="torch", columns=["input_ids"])
eval_loader = DataLoader(
    eval_dataset,
    batch_size=train_config.batch_size,
    shuffle=False,
)
model = TransformerLM.from_config(model_config)

optimizer = AdamW(
    model.parameters(),
    lr=train_config.max_lr,
    beta1=train_config.beta1,
    beta2=train_config.beta2,
    eps=train_config.eps,
    lmda=train_config.weight_decay,
)
lr_scheduler = CosineAnnealingLRwithWarmup(
    optimizer, train_config.warmup_steps, train_config.total_steps, train_config.min_lr
)
step = -1
if resume_from is not None:
    step = load_checkpoint(resume_from, model, optimizer, lr_scheduler)
    print(f"Resumed training from step {step}: {resume_from}")


def evaluate(model: TransformerLM, loader: DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch in loader:
            loss = model(batch["input_ids"], compute_loss=True).loss
            total_loss += loss.item()
            batch_count += 1

            if batch_count >= train_config.eval_num_batches:
                break
    if batch_count == 0:
        raise ValueError("Evaluation dataset is empty")
    return total_loss / batch_count


try:
    model.train()
    for step, batch in enumerate(train_loader, start=step + 1):
        if step >= train_config.total_steps:
            break
        loss: torch.Tensor = model(batch["input_ids"], compute_loss=True).loss
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

        wandb.log(
            {
                "train/loss": loss.item(),
                "train/learning_rate": lr_scheduler.get_last_lr()[0],
            },
            step=step,
        )

        if step % train_config.eval_interval == 0:
            wandb.log({"eval/loss": evaluate(model, eval_loader)}, step=step)
            model.train()

        if step % train_config.save_interval == 0:
            checkpoint_path = os.path.join(
                train_config.model_save_path, f"checkpoint_step_{step}.pt"
            )
            save_checkpoint(model, optimizer, lr_scheduler, step, checkpoint_path)
            wandb.save(checkpoint_path)
except KeyboardInterrupt:
    checkpoint_path = os.path.join(
        train_config.model_save_path, f"checkpoint_interrupt_step_{step}.pt"
    )
    save_checkpoint(model, optimizer, lr_scheduler, step, checkpoint_path)
    print(f"Interrupted; saved checkpoint to {checkpoint_path}")
finally:
    wandb.finish()
