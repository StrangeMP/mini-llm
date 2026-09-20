# Mini-LLM: Transformer Implementation and Systems Exploration

## 1. Project Objective

Build a small but complete experimental codebase that develops a first-principles understanding of Transformer-based language models.

The central question is:

> What happens to a sequence of tokens as it passes through a modern Transformer language model, and what are the computational consequences of its major design choices?

The project should connect three levels:

1. Mathematical mechanisms
2. PyTorch implementations
3. Modern LLM implementations and systems behavior

The project is **not** intended to train a competitive language model.

The final deliverable should demonstrate that I can move comfortably between:

- mathematical formulations
- tensor shapes and operations
- PyTorch implementations
- modern LLM source code
- inference behavior
- memory and computational analysis

---

# 2. Scope

The project consists of six modules.

| Module | Topics | Priority |
|---|---|---:|
| A | Transformer fundamentals | ★★★★★ |
| B | Positional representations | ★★★★★ |
| C | Training stability | ★★★★☆ |
| D | Autoregressive inference | ★★★★★ |
| E | Modern LLM architecture | ★★★★★ |
| F | Parameter-efficient adaptation | ★★★★☆ |

---

# 3. Module A — Transformer from First Principles

## Goal

Implement a small decoder-only Transformer without using:

- `nn.Transformer`
- high-level Transformer implementations
- code copied directly from existing LLM implementations

The important operations should be implemented explicitly.

### Recommended model scale

- Vocabulary: 5k–20k
- Context length: 128–512
- Hidden dimension: 256–512
- Layers: 4–8
- Attention heads: 4–8

The model should remain small enough to train on a single consumer GPU.

---

## Required Components

Implement:

- Token embedding
- Q/K/V projections
- Multi-head self-attention
- Causal attention mask
- Attention softmax
- Output projection
- Feed-forward network
- Residual connections
- LayerNorm
- LM head
- Cross-entropy loss
- Autoregressive generation
- Training loop

The final model should be capable of training on a small text corpus and generating text.

---

# 4. Attention Investigation

Attention should be studied at the tensor level.

For an input:

    x: [B, T, D]

explicitly trace:

    Q/K/V: [B, T, D]

    split heads:
    [B, H, T, Dh]

    QK^T:
    [B, H, T, T]

    attention weights:
    [B, H, T, T]

    attention output:
    [B, H, T, Dh]

    merge heads:
    [B, T, D]

Be able to explain every dimension.

---

## Experiment A1 — Number of Attention Heads

Compare different numbers of heads under approximately fixed parameter count:

    H = 1
    H = 2
    H = 4
    H = 8

Record:

- training loss
- validation loss
- training speed
- GPU memory
- qualitative attention behavior if useful

Question:

> How does the number of attention heads affect representation and computation?

---

## Experiment A2 — Causal Masking

Verify experimentally that token `t` cannot attend to future tokens.

Do not only inspect the implementation.

Construct a small example and inspect the resulting attention matrix.

---

# 5. Module B — Positional Representations

Implement and compare:

1. Learned positional embeddings
2. Sinusoidal positional encoding
3. RoPE

Optional:

4. ALiBi

The goal is not simply to implement these methods.

Understand:

> How does each method inject positional information into the computation?

---

## Experiment B1 — Length Extrapolation

Train using:

    context length = 128

Evaluate using:

    128
    256
    512

Compare performance degradation.

---

## Experiment B2 — RoPE

Understand and implement the rotation applied to Q and K.

Be able to explain:

    q' = R_theta q
    k' = R_theta k

and why modifying Q/K in this way changes their attention dot product.

---

# 6. Module C — LayerNorm and Training Stability

Implement LayerNorm from its mathematical definition:

    mean = E[x]

    variance = E[(x - mean)^2]

    normalized_x =
        (x - mean) / sqrt(variance + epsilon)

Then compare:

## Pre-LN

    x
    ↓
    LayerNorm
    ↓
    Attention
    ↓
    Residual

## Post-LN

    x
    ↓
    Attention
    ↓
    Residual
    ↓
    LayerNorm

---

## Experiment C1 — Pre-LN vs Post-LN

Compare:

- convergence speed
- training stability
- validation loss
- gradient norms
- final performance

Question:

> Why does normalization placement affect optimization behavior?

---

# 7. Module D — KV Cache and Inference

First implement naive autoregressive generation.

Without KV cache:

    generate token 1
    → recompute previous tokens

    generate token 2
    → recompute previous tokens

    generate token 3
    → recompute previous tokens

Then implement KV caching:

    previous K/V → cache

    new token
    ↓
    new Q/K/V
    ↓
    reuse cached K/V

---

## Experiment D1 — Cached vs Uncached Decoding

Measure:

- generation latency
- tokens/second
- GPU memory
- scaling with sequence length

Test several context lengths, for example:

    32
    64
    128
    256
    512

Understand:

1. Why KV cache reduces repeated computation
2. Why KV cache consumes additional memory
3. How cache memory scales with sequence length

---

# 8. Module E — Modern LLM Implementation Study

After the toy Transformer works, study one modern decoder-only LLM implementation.

Preferably use a Llama-family architecture.

Trace the forward pass:

    ModelForCausalLM
        ↓
    Model
        ↓
    DecoderLayer
        ├── Attention
        └── MLP

Identify:

- token embedding
- positional representation
- RoPE
- Q/K/V projections
- GQA/MQA if applicable
- attention output
- RMSNorm
- FFN / SwiGLU
- residual stream
- LM head
- KV cache

The objective is:

> Can I open an unfamiliar modern LLM implementation and understand its forward pass without getting lost?

---

# 9. Activation Inspection

Use PyTorch hooks or equivalent mechanisms to inspect:

- embedding outputs
- Q/K/V
- attention outputs
- MLP outputs
- residual states
- final hidden states
- logits

For a single prompt, trace:

    tokens
      ↓
    embedding
      ↓
    layer 1
      ↓
    layer 2
      ↓
    ...
      ↓
    final hidden state
      ↓
    logits

The purpose is implementation literacy.

Do **not** turn this into a new representation-learning research project.

---

# 10. Module F — LoRA

Implement LoRA manually.

For a frozen weight matrix:

    W' = W + BA

where:

- W is frozen
- A and B are trainable low-rank matrices

Compare:

    Full fine-tuning
    vs.
    LoRA

Measure:

- trainable parameters
- GPU memory
- training speed
- task performance

Understand exactly which parameters are updated.

---

# 11. Engineering Specification

Recommended repository structure:

    mini-llm/
    ├── README.md
    ├── src/
    │   ├── attention.py
    │   ├── positional.py
    │   ├── normalization.py
    │   ├── transformer.py
    │   ├── generation.py
    │   └── lora.py
    │
    ├── experiments/
    │   ├── attention_heads.py
    │   ├── positional_encoding.py
    │   ├── pre_post_ln.py
    │   ├── kv_cache.py
    │   └── lora.py
    │
    ├── notebooks/
    │   └── visualization.ipynb
    │
    └── configs/

Each experiment should follow:

    Question
    → Hypothesis
    → Implementation
    → Experiment
    → Result
    → Interpretation

---

# 12. Acceptance Criteria

√ denotes completed items, [-] denotes unplanned items, and [ ] denotes uncompleted items.

## Architecture

- [√] Implement MHA from tensor operations
- [√] Explain every major tensor shape
- [√] Implement causal masking
- [√] Implement FFN
- [√] Implement residual connections
- [√] Implement LayerNorm
- [ ] Train a small GPT-like model successfully

## Positional Representations

- [-] Implement learned positional embeddings
- [-] Implement sinusoidal encoding
- [√] Implement RoPE
- [ ] Explain their differences

## Training

- [√] Understand Pre-LN vs Post-LN
- [ ] Inspect gradient norms
- [ ] Explain basic training stability differences

## Inference

- [√] Implement autoregressive generation
- [√] Implement KV cache
- [-] Benchmark cached vs uncached decoding
- [-] Understand cache memory scaling

## Modern LLMs

- [-] Trace a Llama-style implementation
- [√] Understand RMSNorm
- [√] Understand RoPE
- [ ] Understand GQA/MQA if present
- [√] Understand SwiGLU if present
- [-] Inspect internal activations

## Adaptation

- [-] Implement LoRA
- [-] Understand which parameters are trainable
- [-] Compare LoRA with full fine-tuning

