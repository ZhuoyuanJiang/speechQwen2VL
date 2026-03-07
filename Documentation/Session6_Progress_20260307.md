# Session 6 Progress — 2026-03-07

## Goal

Stage 2: LoRA fine-tuning of the LLM decoder layers on top of the Stage 1 trained audio projector.

---

## What Stage 2 Does

- Load the Stage 1 model (trained projector from `DanJZY/Qwen2-VL-7B-Speech`) in **bf16** (~16.7 GB)
- Apply **LoRA adapters** to all LLM attention + MLP projections (28 layers × 7 modules = 196 targets, ~161M params)
- Keep `audio_projector` trainable as full params via `modules_to_save` (~17M)
- Total trainable: ~178,920,448 (2.1% of 8.3B), lr=2e-5 (10x lower than Stage 1)
- Audio encoder (Whisper) and vision encoder stay **frozen**
- Save LoRA adapters only (~700MB), not the full model

### What we're fine-tuning

| Component | Trainable? | Method |
|---|---|---|
| LLM decoder layers (28 layers) | Yes | LoRA on q/k/v/o/gate/up/down_proj |
| Audio projector | Yes | Full params (via `modules_to_save`) |
| Audio encoder (Whisper) | No | Frozen — already pretrained |
| Vision encoder | No | Frozen — not changing vision capabilities |

### Key design decisions

1. **bf16 LoRA (not QLoRA)**: Our 49 GB GPUs have plenty of headroom for bf16. Faster (~10-20% per step, no dequantization overhead), simpler code, no risk of audio_projector getting quantized. QLoRA is the documented fallback if VRAM becomes tight — see `Documentation/Session6_Plan.md` for the full QLoRA drop-in replacement code.

2. **Regex-scoped target_modules**: PEFT's suffix matching (`["q_proj", "k_proj", ...]`) would also hit Whisper's 96 attention layers (292 total modules instead of 196). We use a regex to scope LoRA to the LLM only:
   ```python
   target_modules=r"model\.language_model\.layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
   ```
   The `model.` prefix is required because `Qwen2VLForConditionalGeneration` wraps the LLM as `self.model.language_model`, and PEFT uses `re.fullmatch` against the full dotted module path. Verified: 196 LLM modules matched, 0 audio-encoder modules matched.

3. **Data strategy**: Start with 20 shards (~30K samples) for pipeline validation, then scale to full `small/` split (72 shards, ~107K samples) for actual training. More data matters more for Stage 2 than Stage 1 because we're adapting the LLM, not just aligning a projector.

---

## Step-by-step Reproduction

### Prerequisites

- Stage 1 complete: trained projector pushed to `DanJZY/Qwen2-VL-7B-Speech`
- Conda env `speech_qwen2vl` set up (see `Documentation/Session5_Progress_20260306.md` for full env setup)
- Forked libraries installed via `bash scripts/setup_forks.sh`
- `./data` and `./checkpoints` directories exist (symlinks to local SSD on our server)

### 1. Create the notebook and DDP script

The notebook source is `notebooks/06_training_stage2_lora.py` (jupytext percent-format). To generate the `.ipynb`:

```bash
conda activate speech_qwen2vl
jupytext --to notebook notebooks/06_training_stage2_lora.py --output notebooks/06_training_stage2_lora.ipynb
```

The DDP script is `scripts/train_stage2.py` — ready to run as-is.

### 2. Validate the pipeline in the notebook (optional but recommended)

Open `notebooks/06_training_stage2_lora.ipynb` in Jupyter and run Sections 1-5 sequentially:

```bash
jupyter lab notebooks/06_training_stage2_lora.ipynb
```

**What each section verifies**:

| Section | What to check |
|---|---|
| 1. Environment Setup | transformers `4.56.0.dev0` from `forks/`, GPU auto-selected, dataset cache points to `./data` |
| 2. Load Dataset | ~29,820 train samples (20 shards), 100 test samples, 0 dropped by pre-filter |
| 3. Memory Cleanup | `clear_memory()` defined (no output) |
| 4. Load Model + LoRA | `print_trainable_parameters()` shows ~178M trainable (2.1%), audio_encoder has 0 trainable params |
| 5. Data Collator | Decoded labels match ground truth transcript + `<\|im_end\|>` |

**Do NOT run Section 8 Option A (training)** in the notebook — that's for single-GPU debugging only. Use the DDP script for actual training.

### 3. Download the full dataset (if using all 72 shards)

The DDP script's `load_dataset()` automatically downloads any missing shards on first run. The 20 shards from Stage 1 are already cached — only the remaining ~52 shards need downloading.

To pre-download without starting training, you can run the notebook's Section 2 with Option B uncommented. Or just let the DDP script handle it — the download happens before training begins.

To check current cache size:
```bash
du -sh ./data
```

### 4. Choose dataset size in the DDP script

The script has two options in the dataset section:

```python
# Option A: 20 shards for pipeline validation (~30K samples)
# train_dataset = load_dataset(
#     "speechbrain/LargeScaleASR",
#     data_files=["small/train-0000*", "small/train-0001*"],
#     num_proc=12,
# )

# Option B: Full small split for actual training (~107K samples)
train_dataset = load_dataset(
    "speechbrain/LargeScaleASR",
    data_files=["small/train-*"],
    num_proc=12,
)
```

Currently Option B (full 72 shards) is active. To switch, comment/uncomment the appropriate block.

### 5. Kill any leftover GPU processes

Check for stale processes from previous notebook runs:

```bash
nvidia-smi
```

If you see `speech_qwen2vl` processes on any GPU, kill them:

```bash
kill <PID>
```

### 6. Run Stage 2 training

```bash
mkdir -p logs && python scripts/train_stage2.py --num_evals 10 2>&1 | tee logs/stage2_$(date +%Y%m%d_%H%M%S).log
```

The script:
1. Auto-detects idle GPUs (< 500 MiB used)
2. Sets `CUDA_VISIBLE_DEVICES` to those GPUs
3. Launches `torchrun --nproc_per_node=N scripts/train_stage2.py`
4. Each worker loads the model, applies LoRA, and trains with DDP

**CLI arguments** (all optional, showing defaults):

```bash
python scripts/train_stage2.py \
    --nproc 6 \              # max GPUs (default: all idle)
    --output_dir ./checkpoints/stage2_lora \
    --data_dir ./data \
    --epochs 3 \
    --batch_size 2 \         # per-device train batch size
    --eval_batch_size 2 \    # per-device eval batch size
    --grad_accum 8 \
    --lr 2e-5 \
    --num_evals 10 \         # number of evenly-spaced evals
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05
```

**To manually select GPUs**:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_stage2.py --num_evals 10
```

### 7. Monitor training

**wandb**: Training logs to wandb project `speechQwen2VL`, run name `stage2-lora`.

**Terminal**: Loss logged every 10 steps. Example output:
```
Using 6 GPUs: [0, 1, 2, 3, 6, 7]
Train: 107000 → 106500 samples
Test:  100 samples
trainable params: 178,920,448 || all params: 8,457,218,048 || trainable%: 2.1154
Total steps: 3345, eval every 334 steps (10 evals)
Training: 106500 samples, 3 epochs
Batch: 2/gpu × 8 accum × 6 GPUs = 96 effective
```

**GPU memory**: Check periodically with `nvidia-smi`. Expect ~30 GB typical, spikes up to ~40+ GB on GPUs that get longer audio batches.

**Log file**: Saved to `logs/stage2_YYYYMMDD_HHMMSS.log`.

### 8. After training: load checkpoint and run inference

Open the notebook and run Section 8 **Option B** (checkpoint loading):

```python
checkpoint_dir = "./checkpoints/stage2_lora"
checkpoint_folders = sorted(
    glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")),
    key=lambda x: int(x.rsplit("-", 1)[-1]),
)
load_path = checkpoint_folders[-1] if checkpoint_folders else checkpoint_dir
# ... loads base model + PeftModel.from_pretrained(base_model, load_path)
```

Then run Section 10 (inference) to compare transcription quality vs Stage 1.

### 9. Save and push LoRA adapters

Run Section 9 of the notebook:
```python
model.save_pretrained("./checkpoints/stage2_lora")  # adapters only (~700MB)
processor.save_pretrained("./checkpoints/stage2_lora")
model.push_to_hub("DanJZY/Qwen2-VL-7B-Speech-LoRA")  # separate repo from Stage 1
```

The saved directory contains only LoRA adapter weights + `modules_to_save` (audio_projector), NOT the full base model. To use the trained model later:
```python
base_model = Qwen2VLForConditionalGeneration.from_pretrained("DanJZY/Qwen2-VL-7B-Speech", ...)
model = PeftModel.from_pretrained(base_model, "DanJZY/Qwen2-VL-7B-Speech-LoRA")
```

---

## Files Created

| File | Description |
|------|-------------|
| `Documentation/Session6_Plan.md` | Full plan for Stage 2 LoRA training with QLoRA fallback |
| `Documentation/Lessons/session6_QA.md` | Q&A: LoRA target scoping, bf16 vs QLoRA tradeoffs |
| `notebooks/06_training_stage2_lora.py` | Training notebook (11 sections, jupytext percent-format) |
| `notebooks/06_training_stage2_lora.ipynb` | Generated from .py via jupytext |
| `scripts/train_stage2.py` | Multi-GPU DDP script with LoRA |
| `Documentation/Session6_Progress_20260307.md` | This file |

---

## Notebook Structure (11 Sections)

1. **Environment Setup** — chdir to project root, set HF_DATASETS_CACHE, auto-select GPU
2. **Load Dataset** — Option A (20 shards) / Option B (full 72 shards), pre-filter over-budget samples
3. **Memory Cleanup Utility** — `clear_memory()` function
4. **Load Model + Apply LoRA** — bf16 loading, regex-scoped LoRA, `modules_to_save=["audio_projector"]`, verification
5. **Data Collator** — `AudioTextCollator` (same as Stage 1), collator test cell
6. **Training Config** — `TrainingArguments` with lr=2e-5, eval schedule
7. **wandb Init** — project="speechQwen2VL", name="stage2-lora"
8. **Train or Load Checkpoint** — Option A (train, commented out) / Option B (load from DDP checkpoint)
9. **Save LoRA Adapters** — `save_pretrained` (adapters only ~700MB), `push_to_hub`
10. **Post-Training Inference** — `run_inference()` to compare quality vs Stage 1
11. **Cleanup** — `wandb.finish()`, `clear_memory()`

---

## DDP Script Details (`scripts/train_stage2.py`)

Mirrors `scripts/train_stage1.py` with these changes:

| Aspect | Stage 1 | Stage 2 |
|--------|---------|---------|
| Learning rate | 1e-4 | 2e-5 |
| Output dir | `./checkpoints/stage1_audio_projector` | `./checkpoints/stage2_lora` |
| Model setup | Freeze all except audio_projector | bf16 + `get_peft_model` with LoRA |
| Extra CLI args | — | `--lora_r`, `--lora_alpha`, `--lora_dropout` |
| DDP flag | — | `ddp_find_unused_parameters=True` |
| Save method | `trainer.save_model()` (full model ~17 GB) | `trainer.model.save_pretrained()` (adapters only ~700MB) |
| wandb run name | `stage1-audio-projector` | `stage2-lora` |

**Why `ddp_find_unused_parameters=True`**: Set as a precaution for PEFT's `modules_to_save`, which creates a copy of the original module alongside the trainable version. In practice, the live training log showed "did not find any unused parameters" on all 6 ranks — so PEFT may handle this internally. The flag adds minor overhead (DDP tracks which params were used each step) but ensures compatibility.

---

## Verification

`py_compile` passed on both `.py` files. PEFT config instantiation against the Stage 1 checkpoint confirmed:
- Regex hits **196 LLM modules** (28 layers × 7 projections) — correct
- Regex hits **0 audio-encoder modules** — correct
- Total trainable: **178,920,448 params** with audio_projector still trainable — correct

### Bug found and fixed

**Checkpoint sorting bug** (notebook Section 8, Option B): `sorted(glob.glob("checkpoint-*"))` uses lexicographic sort, so `checkpoint-672` sorts after `checkpoint-3360`. Fixed by sorting numerically:
```python
checkpoint_folders = sorted(
    glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")),
    key=lambda x: int(x.rsplit("-", 1)[-1]),
)
```

---

## Stage 2 Training Run #1 (in progress)

### Config

| Parameter | Value |
|-----------|-------|
| Server | vllab15 |
| GPUs | 6× RTX 6000 Ada (indices 0, 1, 2, 3, 6, 7) — GPUs 4, 5 in use by another user |
| Dataset | Full small split — 72 shards (~107K samples) |
| Per-device train batch size | 2 |
| Per-device eval batch size | 2 |
| Gradient accumulation | 8 |
| Effective batch size | 2 × 8 × 6 = 96 |
| Estimated steps/epoch | ~1115 |
| Total steps | ~3345 (3 epochs) |
| Learning rate | 2e-5, cosine scheduler, 3% warmup |
| Eval schedule | `--num_evals 10` → eval every ~335 steps + final eval |
| LoRA rank | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| Gradient checkpointing | Yes (`use_reentrant=False`) |
| Precision | bf16 |

### Command used

```bash
# Check for leftover GPU processes
nvidia-smi

# Kill any stale notebook kernels (we found PID 2450295 using 528 MiB on GPU 0)
kill 2450295

# Start training with timestamped log
mkdir -p logs && python scripts/train_stage2.py --num_evals 10 2>&1 | tee logs/stage2_$(date +%Y%m%d_%H%M%S).log
```

### VRAM observations (during training)

```
GPU 0: 30191 / 49140 MiB  (~61%)
GPU 1: 30547 / 49140 MiB  (~62%)
GPU 2: 29493 / 49140 MiB  (~60%)
GPU 3: 29817 / 49140 MiB  (~61%)  — spiked to 43943 MiB briefly from a long audio batch
GPU 6: 30411 / 49140 MiB  (~62%)
GPU 7: 39225 / 49140 MiB  (~80%)
```

Typical usage ~30 GB with ~18-19 GB free. GPU 7 runs higher at ~39 GB consistently.

**Why VRAM varies across GPUs**: Two factors combine:
1. **Variable-length audio**: Samples have different durations → different sequence lengths → different activation sizes. DDP distributes batches by index, not by length, so some GPUs get longer clips than others.
2. **CUDA caching allocator**: PyTorch's memory allocator reserves memory at peak usage and doesn't release it back to the OS. A GPU that processes one long batch will show high `nvidia-smi` usage even after subsequent shorter batches — the allocator holds the memory for future allocations. So persistent high readings may reflect a past spike, not ongoing high usage.

**Why VRAM spikes**: GPU 3 briefly hit 44 GB (only 5 GB free). This was a transient spike caused by a batch containing unusually long audio samples. It settled back to ~30 GB on the next step. With `eval_batch_size=2`, evaluation should fit within the available headroom (eval disables gradient checkpointing but processes fewer samples).

### Comparison with Stage 1 VRAM

| | Stage 1 (6 GPUs) | Stage 2 (6 GPUs) |
|---|---|---|
| Model weights | ~16.7 GB (bf16) | ~16.7 GB (bf16) |
| Trainable params | 17M (projector only) | 178M (LoRA + projector) |
| Optimizer states | ~0.07 GB | ~1.4 GB |
| Typical VRAM | ~25-35 GB | ~30-39 GB |
| Peak VRAM | 47 GB (GPU 7) | 44 GB (GPU 3, transient) |

Stage 2 uses slightly more VRAM due to 10x more optimizer states (178M vs 17M trainable params), but the difference is modest (~1.3 GB) because model weights dominate.

---

## Commits

1. **Stage 1 DDP fixes + docs** — Dynamic `--num_evals`, `--eval_batch_size`, notebook .py sync via jupytext
2. **Session 6 plan + Q&A** — `Documentation/Session6_Plan.md`, `Documentation/Lessons/session6_QA.md`
3. **Notebook 06 + DDP script** — (pending commit, waiting for training validation)

---

## Next Steps

1. Monitor Stage 2 training loss and eval loss on wandb
2. After training: load checkpoint in notebook Section 8 Option B, run inference in Section 10
3. Compare transcription quality vs Stage 1
4. If results are good: push LoRA adapters to `DanJZY/Qwen2-VL-7B-Speech-LoRA` (Section 9)
5. Commit notebook + DDP script
6. Consider whether to train longer (more epochs) or with different hyperparameters based on loss curves
