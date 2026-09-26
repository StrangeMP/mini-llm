from dataclasses import dataclass
from importlib import import_module
from linear import Linear
import torch, math
from torch import nn
from typing import TypedDict
from compute import softmax, cross_entropy_loss

try:
    flash_attn_module = import_module("flash_attn")
except ImportError:
    try:
        flash_attn_module = import_module("flash_attn_interface")
    except ImportError:
        flash_attn_module = None

flash_attn_qkvpacked_func = getattr(
    flash_attn_module, "flash_attn_qkvpacked_func", None
)
flash_attn_func = getattr(flash_attn_module, "flash_attn_func", None)


class Embedding(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, device=None, dtype=None
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device)
        )
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids, :]


class RMSLayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))
        self.eps = eps

    "Expected shape: (batch_size, sequence_length, d_model)"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        d_model = self.weight.shape[0]
        rms = torch.sqrt(
            torch.sum(torch.square(x), dim=2, keepdim=True) / d_model + self.eps
        )
        result = x / rms * self.weight
        return result.to(in_dtype)


def SiLU(x: torch.Tensor):
    return x * torch.sigmoid(x)


class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        multiple = 64
        d_ff = int(8 * d_model / 3)
        d_ff = ((d_ff + multiple - 1) // multiple) * multiple
        self.gate = Linear(d_model, d_ff, dtype=dtype, device=device)
        self.up = Linear(d_model, d_ff, dtype=dtype, device=device)
        self.down = Linear(d_ff, d_model, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(SiLU(self.gate(x)) * self.up(x))


class RoPE(nn.Module):
    cos_table: torch.Tensor
    sin_table: torch.Tensor

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k

        rows = torch.arange(start=0, end=d_k, step=2, dtype=torch.float, device=device)
        cols = torch.arange(
            start=0, end=max_seq_len, step=1, dtype=torch.float, device=device
        )
        rows = torch.pow(theta, exponent=-rows / d_k)
        outered = torch.outer(cols, rows)
        outered = torch.cat([outered, outered], dim=-1)
        cos = torch.cos(outered)
        sin = torch.sin(outered)
        self.register_buffer("cos_table", cos, persistent=False)
        self.register_buffer("sin_table", sin, persistent=False)

    def rotate_half(self, x: torch.Tensor):
        half_dim = self.d_k // 2
        first_half = x[..., :half_dim]
        second_half = x[..., half_dim:]

        return torch.cat([-second_half, first_half], dim=-1)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x is a tensor of shape (*batch_dims, seq_len, num_heads, d_k)
        token_positions are a tensor of shape (*batch_dims, seq_len) specifying the token positions of x along the sequence dimension.
        """
        if token_positions is None:
            token_positions = torch.arange(
                start=0, end=x.shape[-2], step=1, dtype=torch.long, device=x.device
            )
        # unsqueeze for broadcasting along the num_heads dimension
        # before unsqueezing, cos and sin have shape (*batch_dims, seq_len, d_k)
        # after unsqueezing, cos and sin have shape (*batch_dims, seq_len, 1, d_k)
        cos = self.cos_table[token_positions].unsqueeze(-2)
        sin = self.sin_table[token_positions].unsqueeze(-2)

        return x * cos + self.rotate_half(x) * sin


class ScaledDotProductAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.sqrt_d_model = math.sqrt(d_model)
        self.q = Linear(d_model, d_model, dtype=dtype, device=device)
        self.k = Linear(d_model, d_model, dtype=dtype, device=device)
        self.v = Linear(d_model, d_model, dtype=dtype, device=device)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x is a tensor of shape (batch, seq_len, d_model)
        mask is a boolean tensor of shape (seq_len, seq_len)
        """

        q: torch.Tensor = self.q(x)  # (batch, seq_len, d_model)
        k: torch.Tensor = self.k(x)
        v: torch.Tensor = self.v(x)

        # (batch, seq_len, seq_len)
        qk = torch.matmul(q, k.transpose(-1, -2)) / self.sqrt_d_model
        if mask is not None:
            qk = qk.masked_fill(~mask, torch.finfo(qk.dtype).min)
        qk = softmax(qk)

        # (batch, seq_len, d_model)
        result = torch.matmul(qk, v)
        return result


class LayerCache(TypedDict):
    """
    Tensor of shape (batch, seq_len, num_heads, d_k)
    """

    key: torch.Tensor
    value: torch.Tensor


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        RoPE_base: float = 10000.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model {d_model} is not a multiple of num_heads {num_heads}"
            )
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.sqrt_d_head = math.sqrt(self.d_head)
        self.rope = RoPE(
            theta=RoPE_base, d_k=self.d_head, max_seq_len=max_seq_len, device=device
        )

        self.q = Linear(d_model, d_model, dtype=dtype, device=device)
        self.k = Linear(d_model, d_model, dtype=dtype, device=device)
        self.v = Linear(d_model, d_model, dtype=dtype, device=device)
        self.o = Linear(d_model, d_model, dtype=dtype, device=device)

    def forward(
        self,
        x: torch.Tensor,
        mask: (
            torch.Tensor | None
        ) = None,  # padding mask over x and cache, shape (batch, seq_len + past_seq_len)
        token_positions: torch.Tensor | None = None,
        past_key_value: LayerCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, LayerCache | None]:
        *batch_dims, seq_len, _ = x.shape
        shape_in_heads = (*batch_dims, seq_len, self.num_heads, self.d_head)
        q: torch.Tensor = self.q(x).view(shape_in_heads)
        k: torch.Tensor = self.k(x).view(shape_in_heads)
        v: torch.Tensor = self.v(x).view(shape_in_heads)
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        if past_key_value is not None:
            k = torch.cat([past_key_value["key"], k], dim=-3)
            v = torch.cat([past_key_value["value"], v], dim=-3)

        flash_causal = False
        flash_mask_is_supported = mask is None
        if mask is not None and len(batch_dims) == 1:
            key_seq_len = k.shape[-3]
            if mask.shape == (seq_len, seq_len):
                mask_for_flash = mask
            elif mask.shape == (batch_dims[0], 1, seq_len, key_seq_len):
                mask_for_flash = mask[:, 0]
            else:
                mask_for_flash = None
            if mask_for_flash is not None:
                query_positions = torch.arange(
                    key_seq_len - seq_len,
                    key_seq_len,
                    device=mask.device,
                )
                key_positions = torch.arange(key_seq_len, device=mask.device)
                causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
                causal_mask = causal_mask.expand_as(mask_for_flash)
                flash_mask_is_supported = bool(
                    torch.all(mask_for_flash)
                    or torch.equal(mask_for_flash, causal_mask)
                )
                flash_causal = torch.equal(mask_for_flash, causal_mask)

        use_flash_attention = bool(
            (flash_attn_qkvpacked_func is not None or flash_attn_func is not None)
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16)
            and len(batch_dims) == 1
            and flash_mask_is_supported
        )
        if use_flash_attention and past_key_value is None and flash_attn_qkvpacked_func is not None:
            qkv = torch.stack((q, k, v), dim=-3).to(dtype=x.dtype)
            out = flash_attn_qkvpacked_func(
                qkv, softmax_scale=1 / self.sqrt_d_head, causal=flash_causal
            )
        elif use_flash_attention and flash_attn_func is not None:
            out = flash_attn_func(
                q.to(dtype=x.dtype),
                k.to(dtype=x.dtype),
                v.to(dtype=x.dtype),
                softmax_scale=1 / self.sqrt_d_head,
                causal=flash_causal,
            )
        else:
            qk = torch.einsum("...ihd,...jhd->...hij", q, k) / self.sqrt_d_head
            if mask is not None:
                qk = qk.masked_fill(~mask, -torch.inf)
            qk = softmax(qk, dim=-1).to(v.dtype)
            out = torch.einsum("...hij,...jhd->...ihd", qk, v)
        out = out.reshape(*batch_dims, seq_len, self.d_model)
        out = self.o(out)
        present = LayerCache({"key": k, "value": v}) if use_cache else None
        return out, present

def create_causal_mask(
    attention_mask: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    past_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Combine a batch-specific padding mask with the causal constraint."""
    total_seq_len = past_seq_len + seq_len
    if attention_mask is None:
        key_is_valid = torch.ones(
            (batch_size, total_seq_len), dtype=torch.bool, device=device
        )
    else:
        if attention_mask.ndim != 2 or attention_mask.shape != (
            batch_size,
            total_seq_len,
        ):
            raise ValueError(
                "attention mask must have shape "
                f"({batch_size}, {total_seq_len}), got {tuple(attention_mask.shape)}"
            )
        key_is_valid = attention_mask.to(device=device, dtype=torch.bool)

    query_positions = torch.arange(seq_len, device=device) + past_seq_len
    key_positions = torch.arange(total_seq_len, device=device)
    is_causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    return is_causal.unsqueeze(0).unsqueeze(1) & key_is_valid[:, None, None, :]

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 2048,
        RoPE_base: float = 10000.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.mha_norm = RMSLayerNorm(d_model, device=device, dtype=dtype)
        self.mha = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            RoPE_base=RoPE_base,
            device=device,
            dtype=dtype,
        )
        self.ffn_norm = RMSLayerNorm(d_model, device=device, dtype=dtype)
        self.ffn = FFN(d_model=d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        mask: (
            torch.Tensor | None
        ) = None,  # padding mask over x and cache, shape (batch, seq_len + past_seq_len)
        token_positions: torch.Tensor | None = None,
        past_key_value: LayerCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, LayerCache | None]:
        x_norm = self.mha_norm(x)
        x_mha, present = self.mha(
            x_norm,
            mask=mask,
            token_positions=token_positions,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + x_mha
        x_norm = self.ffn_norm(x)
        x_ffn = self.ffn(x_norm)
        return x_ffn + x, present


class LMOutput(dict):
    def __init__(
        self,
        logits: torch.Tensor,
        hidden_states: list[torch.Tensor],
        past_key_values: list[LayerCache] | None = None,
        loss: torch.Tensor | None = None,
    ):
        super().__init__()
        self.logits = logits
        self.hidden_states = hidden_states
        self.past_key_values = past_key_values
        self.loss = loss

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    d_model: int = 768
    n_layer: int = 12
    n_head: int = 12
    rope_base: float = 10000.0
    dtype: str = "bfloat16"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        RoPE_base: float = 10000.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = ModelConfig(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            n_layer=num_layers,
            n_head=num_heads,
            rope_base=RoPE_base,
            dtype=str(dtype).removeprefix("torch."),
            device=str(device),
        )
        self.emb = Embedding(vocab_size, d_model, dtype=dtype, device=device)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    max_seq_len=context_length,
                    RoPE_base=RoPE_base,
                    dtype=dtype,
                    device=device,
                )
                for i in range(num_layers)
            ]
        )
        self.post_norm = RMSLayerNorm(d_model, dtype=dtype, device=device)
        self.lm_head = nn.Linear(d_model, vocab_size, dtype=dtype, device=device)

    def get_config(self) -> ModelConfig:
        return self.config

    @classmethod
    def from_config(cls, config: ModelConfig):
        return cls(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.d_model,
            num_heads=config.n_head,
            num_layers=config.n_layer,
            RoPE_base=config.rope_base,
            dtype=getattr(torch, config.dtype),
            device=torch.device(config.device),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        # 2D padding mask over current and cached tokens: (batch, seq_len + past_seq_len)
        mask: torch.Tensor | None = None,
        past_key_values: list[LayerCache] | None = None,
        token_positions: torch.Tensor | None = None,
        use_cache: bool = False,
        compute_loss: bool = False,
    ) -> LMOutput:
        """
        token_ids: (batch, seq_len)
        mask: (batch, seq_len + past_seq_len)
        past_key_values: list of LayerCache, length num_layers
        seq_len is the length of the current input sequence, which may be less than the context length if we are doing incremental decoding.
        """
        *batch_dims, seq_len = token_ids.shape
        embeddings = self.emb(token_ids)
        use_cache = use_cache or past_key_values is not None
        past_seq_len = (
            past_key_values[0]["key"].shape[-3] if past_key_values is not None else 0
        )
        if past_seq_len + seq_len > self.config.context_length:
            raise ValueError(
                "sequence length including cached tokens must not exceed "
                f"context_length ({self.config.context_length}), got "
                f"{past_seq_len + seq_len}"
            )
        attention_mask = mask
        combined_mask = create_causal_mask(
            attention_mask=attention_mask,
            batch_size=token_ids.shape[0],
            seq_len=seq_len,
            past_seq_len=past_seq_len,
            device=token_ids.device,
        )
        if token_positions is None and attention_mask is not None:
            valid_tokens = attention_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
            token_positions = valid_tokens[:, -seq_len:].clamp_min(0)
        hidden_states: list[torch.Tensor] = []
        present_key_values: list[LayerCache] | None = [] if use_cache else None
        for layer_idx, block in enumerate(self.blocks):
            past_key_value = (
                past_key_values[layer_idx] if past_key_values is not None else None
            )
            embeddings, present = block(
                embeddings,
                mask=combined_mask,
                token_positions=(
                    token_positions
                    if token_positions is not None
                    else torch.arange(
                        start=past_seq_len,
                        end=past_seq_len + seq_len,
                        device=token_ids.device,
                    )
                ),
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            hidden_states.append(embeddings)
            if present_key_values is not None:
                assert present is not None
                present_key_values.append(present)
        embeddings = self.post_norm(embeddings)
        logits = self.lm_head(embeddings)

        loss = None
        if compute_loss:
            targets = token_ids[:, 1:]
            if attention_mask is not None:
                valid_predictions = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
                targets = targets.masked_fill(~valid_predictions, -100)
            loss = cross_entropy_loss(
                logits=logits[:, :-1, :], targets=targets, ignore_idx=-100
            )

        return LMOutput(
            logits=logits,
            hidden_states=hidden_states,
            past_key_values=present_key_values,
            loss=loss,
        )

    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        token_ids: (batch, seq_len)
        max_new_tokens: number of new tokens to generate
        temperature > 0 scales logits before sampling
        top_k limits sampling to the k highest-probability tokens
        top_p performs nucleus sampling on the remaining tokens
        returns: (batch, max_new_tokens)
        """
        if max_new_tokens <= 0:
            return torch.empty(
                token_ids.shape[:-1] + (0,),
                dtype=torch.long,
                device=token_ids.device,
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if top_p is not None and not (0.0 <= top_p <= 1.0):
            raise ValueError(f"top_p must be in [0, 1], got {top_p}")

        def sample_next(logits: torch.Tensor) -> torch.Tensor:
            scaled_logits = logits / temperature

            if top_k is not None:
                k = min(top_k, scaled_logits.size(-1))
                kth_value = torch.topk(scaled_logits, k=k, dim=-1).values[..., -1:]
                scaled_logits = scaled_logits.masked_fill(
                    scaled_logits < kth_value, torch.finfo(scaled_logits.dtype).min
                )

            if top_p is not None:
                probs = torch.softmax(scaled_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                keep_mask = (cumulative_probs - sorted_probs) < top_p
                keep_mask[..., :1] = True
                filtered = torch.zeros_like(probs, dtype=torch.bool)
                filtered.scatter_(-1, sorted_indices, keep_mask)
                probs = probs.masked_fill(~filtered, 0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(scaled_logits, dim=-1)

            return torch.multinomial(probs, num_samples=1).to(dtype=torch.long)

        with torch.inference_mode():
            if token_ids.shape[-1] + max_new_tokens > self.config.context_length:
                raise ValueError(
                    "prompt and generated tokens must fit within "
                    f"context_length ({self.config.context_length}), got "
                    f"{token_ids.shape[-1] + max_new_tokens}"
                )
            if attention_mask is None:
                attention_mask = torch.ones_like(token_ids, dtype=torch.bool)
            else:
                if attention_mask.shape != token_ids.shape:
                    raise ValueError(
                        "attention_mask must have the same shape as token_ids, got "
                        f"{tuple(attention_mask.shape)} and {tuple(token_ids.shape)}"
                    )
                attention_mask = attention_mask.to(
                    device=token_ids.device, dtype=torch.bool
                )
                if not attention_mask.any(dim=-1).all():
                    raise ValueError("each batch row must contain at least one real token")

            output = self.forward(
                token_ids=token_ids,
                mask=attention_mask,
                use_cache=True,
                compute_loss=False,
            )
            kv_cache = output.past_key_values
            last_positions = attention_mask.sum(dim=-1) - 1
            batch_positions = torch.arange(token_ids.shape[0], device=token_ids.device)
            logits = output.logits[batch_positions, last_positions, :]
            new_token = sample_next(logits)
            buffer = [new_token]

            for _ in range(1, max_new_tokens):
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (token_ids.shape[0], 1),
                            dtype=torch.bool,
                            device=token_ids.device,
                        ),
                    ],
                    dim=-1,
                )
                output = self.forward(
                    token_ids=new_token,
                    mask=attention_mask,
                    past_key_values=kv_cache,
                    use_cache=True,
                    compute_loss=False,
                )
                kv_cache = output.past_key_values
                logits = output.logits[..., -1, :]
                new_token = sample_next(logits)
                buffer.append(new_token)

        result = torch.concat(buffer, dim=-1)
        return result
