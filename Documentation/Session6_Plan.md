# Plan: Notebook 06 — Stage 2 LoRA Training

## Context

Stage 1 (audio projector training) is complete. The trained projector has been pushed to `DanJZY/Qwen2-VL-7B-Speech` on HuggingFace. Notebook 06 implements Stage 2: LoRA fine-tuning of the LLM layers on top of the trained projector. Using plain bf16 LoRA (no quantization) for the first run — simpler, faster, and our 49 GB GPUs have plenty of headroom. QLoRA is the fallback if we need more VRAM later.

---

## What Stage 2 Does

- Load the Stage 1 model (trained projector) in **bf16** (~16.7 GB)
- Apply **LoRA adapters** to all LLM attention + MLP projections across 28 decoder layers (~161M adapter params)
- Keep `audio_projector` trainable as full params via `modules_to_save` (~17M)
- Total trainable: ~178M (2.1% of 8.3B), lr=2e-5 (10x lower than Stage 1)
- Save LoRA adapters only (~700MB), not the full model

### What we're fine-tuning

| Component | Trainable? | Method |
|---|---|---|
| LLM decoder layers (28 layers) | Yes | LoRA on q/k/v/o/gate/up/down_proj |
| Audio projector | Yes | Full params (via `modules_to_save`) |
| Audio encoder (Whisper) | No | Frozen — already pretrained, no need to touch |
| Vision encoder | No | Frozen — we're not changing vision capabilities |

**Important**: Simple suffix matching (e.g., `["q_proj", "k_proj", "v_proj"]`) would also hit Whisper's 96 attention layers (`audio_encoder.layers.*.self_attn.{q,k,v}_proj`), not just the LLM. We scope LoRA to the LLM only using a regex pattern:
```python
target_modules=r"model\.language_model\.layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
```
This matches exactly the 196 LLM modules (28 layers × 7 projections) and excludes all audio_encoder and visual modules.

---

## Files to Create

| File | Description |
|------|-------------|
| `notebooks/06_training_stage2_lora.ipynb` | Training notebook (11 sections, mirrors Notebook 05 structure) |
| `notebooks/06_training_stage2_lora.py` | Jupytext percent-format sync |
| `scripts/train_stage2.py` | Multi-GPU DDP script (mirrors train_stage1.py) |

---

## Notebook Structure (11 Sections)

### Section 1: Environment Setup
Same as Notebook 05 + additional imports:
```python
from peft import LoraConfig, get_peft_model, PeftModel
```

### Section 2: Load Dataset (from cache)
Identical to Notebook 05. Same dataset, same 20 shards, same pre-filtering. Data is already cached from Stage 1 — `load_dataset()` reads from `./data` cache in seconds, no re-download.

### Section 3: Memory Cleanup Utility
Identical to Notebook 05.

### Section 4: Load Model + Apply LoRA (KEY SECTION)

**Step 4a** — Load in bf16 (same as Stage 1):
```python
model = Qwen2VLForConditionalGeneration.from_pretrained(
    REPO_ID, torch_dtype=torch.bfloat16, device_map=DEVICE,
)
model.config.use_cache = False
```

**Step 4b** — Freeze base model + apply LoRA:
```python
lora_config = LoraConfig(
    r=64, lora_alpha=128,
    # Regex scoped to LLM only — plain suffix matching would also hit Whisper's attention layers
    target_modules=r"model\.language_model\.layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    modules_to_save=["audio_projector"],  # keep full-param training
)
model = get_peft_model(model, lora_config)
```

No `BitsAndBytesConfig` or `prepare_model_for_kbit_training` needed — bf16 LoRA is simpler. The audio_projector stays in bf16 naturally (no quantization risk).

**Step 4c** — Verification: `model.print_trainable_parameters()`, check ~178M trainable (~2.1%).

If bf16 LoRA OOMs, see the **QLoRA Fallback Plan** section below for the pre-designed drop-in replacement.

### Section 5: Data Collator
Identical `AudioTextCollator` from Notebook 05. Same collator test cell.

### Section 6: Training Config
```python
TrainingArguments(
    output_dir="./checkpoints/stage2_lora",
    learning_rate=2e-5,       # 10x lower than Stage 1 (fine-tuning, not random init)
    per_device_eval_batch_size=2,  # explicit (HF defaults to 8)
    # ... rest same as Notebook 05
)
```

### Section 7: wandb Init
Same pattern, name="stage2-lora", log LoRA config in wandb.config.

### Section 8: Train or Load Checkpoint
Same dual-option pattern: commented training cell (Option A) + checkpoint loading cell (Option B).

For Option B, loading LoRA checkpoint:
```python
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    REPO_ID, torch_dtype=torch.bfloat16, device_map=DEVICE,
)
model = PeftModel.from_pretrained(base_model, "./checkpoints/stage2_lora")
```

### Section 9: Save LoRA Adapters
```python
model.save_pretrained("./checkpoints/stage2_lora")  # adapters only (~700MB)
model.push_to_hub("DanJZY/Qwen2-VL-7B-Speech-LoRA")  # separate repo
```

### Section 10: Post-Training Inference
Same `run_inference()` pattern. Compare output quality vs Stage 1.

### Section 11: Cleanup
`wandb.finish()`, `clear_memory()`.

---

## DDP Script (`scripts/train_stage2.py`)

Mirrors `scripts/train_stage1.py` with these changes:
- Additional CLI args: `--lora_r`, `--lora_alpha`, `--lora_dropout`
- Default `--lr 2e-5` (was 1e-4), `--output_dir ./checkpoints/stage2_lora`
- Model loading in bf16 + `get_peft_model` with regex-scoped LoRA
- `device_map={"": local_rank}` for DDP
- `ddp_find_unused_parameters=True` (PEFT's `modules_to_save` creates unused original copies)
- Save with `model.save_pretrained()` (adapters only)
- Dynamic eval via `--num_evals` (carried over from Stage 1)

---

## Hyperparameters: Locked vs Flexible

| Parameter | Value | Status |
|-----------|-------|--------|
| Base model precision | bf16 (QLoRA 4-bit as fallback) | Locked |
| target_modules (7 projections) | q/k/v/o/gate/up/down (regex-scoped to LLM) | Locked |
| modules_to_save | ["audio_projector"] | Locked |
| lora_r | 64 | Flexible |
| lora_alpha | 128 | Flexible |
| learning_rate | 2e-5 | Flexible |
| epochs | 3 | Flexible |
| Dataset shards | 20 (same as Stage 1) | Flexible (can scale to 72) |

---

## Memory Estimate (single RTX 6000 Ada, 49 GB)

| Component | Stage 1 (bf16) | Stage 2 LoRA (bf16) | Stage 2 QLoRA (4-bit, fallback) |
|-----------|---------------|--------------------|---------------------------------|
| Model weights | ~16.7 GB | ~16.7 GB | ~4.5 GB |
| Trainable params + optimizer | ~0.14 GB (17M) | ~1.4 GB (178M) | ~1.4 GB (178M) |
| Activations + grad ckpt | ~3-6 GB | ~3-6 GB | ~3-6 GB |
| **Total** | **~22-25 GB** | **~22-25 GB** | **~12-16 GB** |

bf16 LoRA uses similar VRAM to Stage 1 (~22-25 GB) — fits comfortably on our 49 GB GPUs. If we need more headroom (larger batches, more data), QLoRA drops it to ~12-16 GB.

**Key insight — why QLoRA uses LESS VRAM despite training 10x more params than Stage 1**: The dominant cost is model weight storage, not optimizer states. The full 8.3B model must be loaded regardless of how many params you train (forward pass needs the whole model). 4-bit quantization cuts model weights from 16.7 GB to 4.5 GB — a 12 GB saving that dwarfs the 1.3 GB increase in optimizer states. Stage 1 couldn't use 4-bit because the randomly-initialized projector needs precise bf16 gradients flowing through the full-precision model.

---

## Verification

1. Model memory ~16.7 GB after bf16 loading
2. `print_trainable_parameters()` shows ~178M trainable (~2.1%)
3. audio_projector params are bf16 and `requires_grad=True`
4. Collator test: decoded labels match ground truth (unchanged from Stage 1)
5. Training loss starts lower than Stage 1's starting loss (projector is trained)
6. Training loss decreases over epochs
7. Saved adapter directory is ~700MB (not full 16GB model)
8. Reload base + adapter, run inference, confirm identical output
9. Transcription quality improved vs Stage 1

---

## QLoRA Fallback Plan

If bf16 LoRA OOMs or we need more VRAM headroom, switch to QLoRA. Everything below is pre-designed and ready to swap in.

### Why QLoRA uses less VRAM despite training 10x more params than Stage 1

The dominant cost is model weight storage, not optimizer states. The full 8.3B model must be loaded regardless of how many params you train (forward pass needs the whole model). 4-bit quantization cuts model weights from 16.7 GB to 4.5 GB — a 12 GB saving that dwarfs the 1.3 GB increase in optimizer states. Stage 1 couldn't use 4-bit because the randomly-initialized projector needs precise bf16 gradients flowing through the full-precision model.

### What changes from bf16 LoRA to QLoRA

| Aspect | bf16 LoRA (primary) | QLoRA (fallback) |
|--------|--------------------|--------------------|
| Model loading | `from_pretrained(torch_dtype=bf16)` | `from_pretrained(quantization_config=bnb_config)` |
| Model weights VRAM | ~16.7 GB | ~4.5 GB |
| Total VRAM | ~22-25 GB | ~12-16 GB |
| Extra setup | None | `prepare_model_for_kbit_training()` |
| Speed | Faster (~10-20% per step) | Slower (4-bit dequantization overhead) |
| audio_projector risk | None (stays bf16) | May get quantized — must verify |
| Code complexity | Simple | More imports, more setup |

### QLoRA code (drop-in replacement for Section 4)

**Step 4a** — Load with 4-bit quantization:
```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    REPO_ID, quantization_config=bnb_config,
    torch_dtype=torch.bfloat16, device_map=DEVICE,
)
model.config.use_cache = False
```

**Step 4b** — Prepare for k-bit training + apply LoRA:
```python
model = prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

lora_config = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=r"model\.language_model\.layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    modules_to_save=["audio_projector"],
)
model = get_peft_model(model, lora_config)
```

### QLoRA-specific technical risk

`BitsAndBytesConfig` quantizes ALL `nn.Linear` layers, including those inside `audio_projector`. The projector's trained bf16 weights could be destroyed. Mitigations:
1. `modules_to_save` may handle this automatically (PEFT creates a fresh bf16 copy)
2. If not: manually rebuild audio_projector in bf16 and reload Stage 1 weights before `get_peft_model()`
3. Always verify by checking `audio_projector` param dtype after full setup

### QLoRA checkpoint loading (Section 8 Option B)
```python
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    REPO_ID, quantization_config=bnb_config,
    torch_dtype=torch.bfloat16, device_map=DEVICE,
)
model = PeftModel.from_pretrained(base_model, "./checkpoints/stage2_lora")
```

### QLoRA verification differences
- Model memory should be ~4.5 GB after loading (not ~16.7 GB)
- Check audio_projector params are bf16, not 4-bit quantized

---

## Implementation Order

1. Create `notebooks/06_training_stage2_lora.ipynb` (all 11 sections)
2. Sync to `.py` via jupytext
3. Create `scripts/train_stage2.py` (DDP script)
4. Test notebook sections 1-5 (up to collator) to verify setup
5. Run Stage 2 training, monitor eval metrics
6. Compare inference quality vs Stage 1
