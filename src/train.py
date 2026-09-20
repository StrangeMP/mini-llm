from dataclasses import asdict, dataclass
from datasets import load_dataset
import torch
import wandb
from torch.utils.data import DataLoader
from transformer import TransformerLM, ModelConfig
from trainutils import AdamW, CosineAnnealingLRwithWarmup, gradient_clipping_, save_checkpoint

model_config = ModelConfig( # gpt-2 small configuration
    vocab_size=50257,
    context_length=1024,
    d_model=768,
    n_layer=12,
    n_head=12,
    rope_base=10000.0,
)

@dataclass
class TrainingConfig:
    train_token_budget: int = 20 * 162 * 10**6 # 20 * 162M tokens, according to Chinchilla result
    eval_token_budget: int = 0.01 * train_token_budget
    batch_size: int = 32
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    warmup_steps: int = 1000
    total_steps: int = train_token_budget // batch_size // model_config.context_length
    eval_steps: int = eval_token_budget // batch_size // model_config.context_length
    save_interval: int = 1000
    eval_interval: int = 1000
    train_dataset_path: str = "../datasets/openwebtext_tokenized_chunked_train"
    eval_dataset_path: str = "../datasets/openwebtext_tokenized_chunked_eval"
    wandb_project: str = "mini-llm"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None

train_config = TrainingConfig()
wandb.init(
    project=train_config.wandb_project,
    entity=train_config.wandb_entity,
    name=train_config.wandb_run_name,
    config=asdict(train_config),
)


train_dataset = load_dataset(train_config.train_dataset_path)
train_dataset.set_format(type="torch", columns=["input_ids"])
train_loader = DataLoader(
    train_dataset,
    batch_size=train_config.batch_size,
    shuffle=True,
)
eval_dataset = load_dataset(train_config.eval_dataset_path)
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
)
lr_scheduler = CosineAnnealingLRwithWarmup(
    optimizer, train_config.warmup_steps, train_config.total_steps, train_config.min_lr
)


def evaluate(model: TransformerLM, loader: DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch in loader:
            loss = model(batch, compute_loss=True).loss
            total_loss += loss.item()
            batch_count += 1

            if batch_count >= train_config.eval_steps:
                break
    if batch_count == 0:
        raise ValueError("Evaluation dataset is empty")
    return total_loss / batch_count


model.train()
for step, batch in enumerate(train_loader):
    loss: torch.Tensor = model(batch, compute_loss=True).loss
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
        checkpoint_path = f"checkpoint_step_{step}.pt"
        save_checkpoint(model, optimizer, lr_scheduler, step, checkpoint_path)
        wandb.save(checkpoint_path)

wandb.finish()
