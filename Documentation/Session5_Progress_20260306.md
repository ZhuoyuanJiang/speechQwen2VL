# Session 5 Progress — 2026-03-06

## Server

**vllab15** (first session running on server instead of Mac/Colab):
- 8x NVIDIA RTX 6000 Ada Generation (49 GB each)
- CUDA driver 12.4 (driver version 550.163.01), runtime 12.1 (via conda)
- Local SSDs: `/ssd1/` through `/ssd4/` (7 TB each)
- User space on local disk: `/ssd1/zhuoyuan/`
- Home directory: `/home/zhuoyuan/` (NAS, 100 GB quota, shared across servers — NO large files here)
- Hostname: `vllab15`

---

## Step-by-step reproduction

### 1. Clone the repo (already done)

```bash
git clone https://github.com/ZhuoyuanJiang/speechQwen2VL.git
cd speechQwen2VL
```

### 2. Create conda environment (`speech_qwen2vl`)

**Attempt 1: `conda env create -f environment.yml`**

```bash
conda env create -f environment.yml
```

This created the conda env (Python 3.10, cuda-toolkit 12.1.1, ffmpeg, libsndfile) but the pip install step inside `environment.yml` failed. Specifically, `flash-attn==2.6.3` requires `torch` at build time, but pip tries to resolve all dependencies at once and torch wasn't installed yet when flash-attn's `setup.py` ran.

**Fallback: manual pip install in 3 stages**

```bash
conda activate speech_qwen2vl

# Verify conda deps were installed
conda install -c conda-forge -c nvidia cuda-toolkit=12.1.1 ffmpeg libsndfile -y
# Output: "All requested packages already installed."

# Stage 1: Install PyTorch first (flash-attn needs it at build time)
pip install torch==2.4.1+cu121 torchaudio==2.4.1+cu121 torchvision==0.19.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Stage 2: Install all other pip deps (excluding flash-attn)
pip install \
  "trl @ git+https://github.com/huggingface/trl.git@215294872e4853de6e90e9c4845ce4bfc1ba096d" \
  peft==0.17.1 datasets==4.0.0 accelerate==1.10.0 huggingface-hub==0.34.4 \
  tokenizers==0.21.4 safetensors==0.6.2 bitsandbytes==0.47.0 \
  "librosa>=0.10.0" "soundfile>=0.12.0" "jiwer>=3.0.0" \
  wandb==0.21.1 tensorboard==2.20.0 \
  matplotlib==3.10.5 numpy==2.1.2 pandas==2.3.1 pillow==11.0.0 \
  einops==0.8.1 pyyaml==6.0.2 tqdm==4.67.1 \
  jupyter==1.1.1 jupyterlab==4.4.6 ipykernel==6.30.1 ipywidgets==8.1.7

# Note: this also installed transformers==4.55.4 from PyPI as a dependency of trl/peft/accelerate.
# This is expected — the fork install in Step 3 will override it.

# Stage 3: Install flash-attn (now torch is available)
pip install flash-attn==2.6.3 --no-build-isolation --no-cache-dir
```

**Why `--no-build-isolation --no-cache-dir`?**
- `--no-build-isolation`: lets flash-attn's setup.py see the already-installed torch (without isolation, pip creates a clean venv for building where torch isn't present)
- `--no-cache-dir`: avoids `[Errno 18] Invalid cross-device link` error. pip's cache is on the NAS home dir, but temp build files are on local disk — `os.rename()` fails across filesystems.

**CUDA compatibility note**: Server has CUDA 12.4 driver. We install CUDA 12.1 runtime (via conda `cuda-toolkit=12.1.1` and PyTorch `cu121`). This works because NVIDIA drivers are backward-compatible: driver version >= runtime version.

### 3. Install forked libraries (editable mode)

```bash
bash scripts/setup_forks.sh
```

This script:
1. Clones `ZhuoyuanJiang/transformers` into `forks/transformers/`, checks out `speech-qwen2vl` branch
2. Runs `pip install -e forks/transformers/` → installs `transformers==4.56.0.dev0` (overrides the 4.55.4 from PyPI)
3. Clones `ZhuoyuanJiang/Qwen3-VL` into `forks/Qwen2-VL/`, checks out `speech-qwen2vl` branch
4. Runs `pip install -e forks/Qwen2-VL/qwen-vl-utils/` → installs `qwen_vl_utils==0.0.14`

**Verification**:
```bash
python -c "import transformers; print(transformers.__version__, transformers.__file__)"
# 4.56.0.dev0 /home/zhuoyuan/projects/speechQwen2VL/forks/transformers/src/transformers/__init__.py

python -c "import qwen_vl_utils; print(qwen_vl_utils.__file__)"
# /home/zhuoyuan/projects/speechQwen2VL/forks/Qwen2-VL/qwen-vl-utils/src/qwen_vl_utils/__init__.py
```

Both paths point to `forks/`, not `site-packages/`. Correct.

### 4. HuggingFace & wandb login

```bash
huggingface-cli whoami
# DanJZY
```

HF token was already configured in `~/.bashrc` as `HF_TOKEN`. wandb was already logged in (verified with `python -c "import wandb; print('logged in' if wandb.api.api_key else 'not logged in')"`).

### 5. Verify TRL compatibility

```bash
python -c "import trl; print(trl.__version__)"
# 0.22.0.dev0

python -c "from trl import SFTConfig; import inspect; sig = inspect.signature(SFTConfig); \
  print('max_seq_length:', 'max_seq_length' in sig.parameters); \
  print('max_length:', 'max_length' in sig.parameters); \
  print('dataset_text_field:', 'dataset_text_field' in sig.parameters); \
  print('dataset_kwargs:', 'dataset_kwargs' in sig.parameters)"
# max_seq_length: False
# max_length: True
# dataset_text_field: True
# dataset_kwargs: True
```

Result: TRL 0.22.0.dev0 uses `max_length` (not `max_seq_length`). Notebook was updated accordingly.

### 6. Set up directory structure

```bash
# Create directories on local SSD
mkdir -p /ssd1/zhuoyuan/speechQwen2VL/data
mkdir -p /ssd1/zhuoyuan/speechQwen2VL/checkpoints

# Create symlinks from project directory
ln -s /ssd1/zhuoyuan/speechQwen2VL/checkpoints /home/zhuoyuan/projects/speechQwen2VL/checkpoints
ln -s /ssd1/zhuoyuan/speechQwen2VL/data /home/zhuoyuan/projects/speechQwen2VL/data

# Symlink the HF-cached dataset into ./data for easy browsing
ln -s /ssd1/zhuoyuan/hf_cache/hub/datasets--speechbrain--LargeScaleASR \
      /ssd1/zhuoyuan/speechQwen2VL/data/LargeScaleASR
```

**Resulting layout**:

| Project path | Actual location on SSD | Purpose |
|---|---|---|
| `./data/` | `/ssd1/zhuoyuan/speechQwen2VL/data/` | Dataset browsing directory |
| `./data/LargeScaleASR` | → `/ssd1/zhuoyuan/hf_cache/hub/datasets--speechbrain--LargeScaleASR` | Symlink to HF-cached dataset (8.6 GB) |
| `./checkpoints/` | `/ssd1/zhuoyuan/speechQwen2VL/checkpoints/` | Training checkpoint saves |
| (HF model cache) | `/ssd1/zhuoyuan/hf_cache/hub/` | Model weights, configured via `HF_HOME` in `.bashrc` |
| (HF dataset cache) | `/ssd1/zhuoyuan/hf_cache/` | Dataset raw files, configured via `HF_DATASETS_CACHE` in `.bashrc` |

The notebook's `load_dataset()` uses the default HF cache (no `cache_dir` parameter) — the dataset is already there from the test run. The `./data/LargeScaleASR` symlink is purely for human browsing.

Added `data/` to `.gitignore` (`.checkpoints/` was already there).

### 7. Test run #1 — collator assertion error (fixed)

```bash
conda activate speech_qwen2vl
jupyter nbconvert --to script notebooks/05_training_stage1_adapter.ipynb
mv notebooks/05_training_stage1_adapter.txt notebooks/05_training_stage1_adapter.py
CUDA_VISIBLE_DEVICES=0 python notebooks/05_training_stage1_adapter.py
```

**Result**: Sections 1-4 passed, but Section 5 (collator test) hit an assertion:

```
AssertionError: Sample 0: last real label is 198, expected <|im_end|> (151645).
The transcript was likely partially truncated.
```

**Root cause**: The Qwen2-VL chat template produces `...TRANSCRIPT<|im_end|>\n`. Token 198 is `\n`. The collator's label masking included this trailing `\n` as a real label, but the assertion expected `<|im_end|>` (151645) to be the last real label.

**Fix**: After finding the assistant boundary, also find the `<|im_end|>` token after the assistant content and mask everything after it (the trailing `\n`). This way the model learns to stop generating at `<|im_end|>`, not at `\n`.

### 8. Test run #2 — training started successfully (aborted to reorganize)

Re-ran with the collator fix. All sanity checks passed:

| Section | Check | Result |
|---|---|---|
| 1 (Env) | transformers version and path | `4.56.0.dev0` from `forks/transformers/` |
| 2 (Dataset) | Sample counts | 29,820 train / 100 test, 0 dropped by pre-filter |
| 4 (Model) | GPU memory and trainable params | 16.67 GB, 17,439,744 trainable (0.19%) |
| 5 (Collator) | Decoded labels match ground truth | `AND WHAT ABOUT INTEROPERABILITY...<\|im_end\|>` — correct |
| 5 (Collator) | MASKED→real label transition | Position 882: `\n` (MASKED) → `AND` (real label) — correct |
| 8 (Training) | Training started | ~4.7s/step, 5,592 total steps (~7.3 hours) |

**Aborted after ~80 steps** to reorganize directory structure (set up `./data` and `./checkpoints` symlinks on local SSD).

### 9. Test run #3 — SFTTrainer RuntimeError (fixed)

User ran the notebook in Jupyter after directory reorganization. Training cell hit:

```
RuntimeError: The size of tensor a (2) must match the size of tensor b (16)
at non-singleton dimension 0
```

at `trl/trainer/sft_trainer.py:1073`:
```python
correct_predictions = (predictions == shift_labels) & mask
```

**Root cause**: SFTTrainer overrides `compute_loss()` to add a token accuracy metric. After the model forward pass, `outputs.logits` has batch_size=2 (per_device_train_batch_size), but `inputs["labels"]` has been internally resized to 16 (2 × 8 = batch_size × gradient_accumulation_steps) by the Trainer's batch accumulation. This is a bug in how SFTTrainer's accuracy computation interacts with gradient accumulation.

**Fix**: Replaced `SFTTrainer` + `SFTConfig` with plain `Trainer` + `TrainingArguments` from transformers.

This is not a workaround — it's the correct tool for our use case:
- SFTTrainer's main feature (automatic dataset preprocessing) was already bypassed via `skip_prepare_dataset=True`, because our custom `AudioTextCollator` handles all data processing (audio needs special handling SFTTrainer doesn't know about).
- SFTTrainer's other feature (token accuracy logging) is what caused the bug. Token accuracy is a nice-to-have monitoring metric (what % of tokens the model predicted correctly each step), but **loss** is the key training signal, and we still have that.
- The core training loop (loss computation, gradient backprop, optimizer step) is identical between `SFTTrainer` and `Trainer` — `SFTTrainer` inherits all of this from `Trainer`.

The Session 5 plan anticipated this: "If SFTTrainer conflicts persist, fall back to plain `Trainer` from transformers (same API)."

### 10. Multi-GPU and OOM fixes

**Problem 1: Trainer used all 8 GPUs.** HuggingFace `Trainer` automatically uses all visible GPUs via DataParallel. Even though `device_map=DEVICE` loads the model onto one GPU, the Trainer tries to distribute across all 8.

**Fix**: Set `os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)` right after GPU selection, before any CUDA initialization. After this, PyTorch sees exactly 1 GPU (as `cuda:0`).

**Problem 2: OOM during evaluation.** With 8-GPU DataParallel, each GPU runs `eval_batch_size=2`, and all logits get gathered to GPU 0: 8 × 2 × 2048 × 151K × 2 bytes ≈ 10 GB of logits on one card, plus model weights → exceeds 49 GB.

**Fix**: The `CUDA_VISIBLE_DEVICES` fix above resolved this. Measured single-GPU peak VRAM:
- train_batch=2, eval_batch=1 → ~26 GB peak
- train_batch=4, eval_batch=4 → ~43 GB peak (43892 MiB)

### 11. Dataset cache location fix

**Problem**: `HF_DATASETS_CACHE` was set in the dataset cell (cell-4), but the `datasets` library reads its cache path at import time (cell-2). So the setting had no effect — data kept going to `HF_HOME/hub/` inside hf_cache.

Additionally, `os.path.abspath("./data")` resolved relative to the notebook's working directory (`notebooks/`), not the project root — so data landed in `notebooks/data/` on the NAS home directory.

**Fix**: Set `os.environ["HF_DATASETS_CACHE"]` at the very top of cell-2, before any imports, using an absolute path: `/ssd1/zhuoyuan/speechQwen2VL/data`.

**Note on HF's two-layer cache**: `HF_DATASETS_CACHE` only controls processed arrow files (what training reads). Raw parquet downloads always go to `HF_HOME/hub/` — this is by HF's design and not configurable separately without changing `HF_HOME`. Both are on the local SSD, so this is fine.

### 12. Training config adjustments

- **`eval_steps`**: Changed from 100 to 500 (matching `save_steps`). Eval every 100 steps was too frequent — 55 evals × ~1-2 min each ≈ 15-25% of total training time wasted on eval.
- **`batch_size`**: Tested `per_device_train_batch_size=4` with `gradient_accumulation_steps=4` (same effective batch=16). Result: slower (~10+ hours vs ~7 hours) because larger batches cause more padding waste. Reverted to batch_size=2, grad_accum=8.

---

## Dataset details

**Source**: `speechbrain/LargeScaleASR` — a large-scale ASR dataset on HuggingFace.

Available configs: `large`, `clean`, `small`, `medium`.

We use the `small` config, but **not all of it**:

| What | Shards | Samples |
|---|---|---|
| `small/` split total | 72 shards | ~107K (estimated) |
| **What we download** | 20 shards | **29,820** |
| `test/` split total | 6 shards | ~1,348 |
| **What we use for eval** | 1 shard, first 100 | **100** |

The `data_files` glob patterns select specific shards:
- `"small/train-0000*"` → matches `train-00000-of-00072` through `train-00009-of-00072` (10 shards)
- `"small/train-0001*"` → matches `train-00010-of-00072` through `train-00019-of-00072` (10 shards)
- `"test/test-00000*"` → matches `test-00000-of-00006` (1 shard), then `.select(range(100))` takes first 100

**Why not use all 72 shards?** This follows the skeleton notebook (project advisor's original design). Stage 1 only trains a 17M-parameter projector to learn the basic audio→text mapping — it doesn't need massive data. 29,820 samples over 3 epochs is sufficient for this.

To use more data in the future (e.g., Stage 2), change the glob pattern:
- `"small/train-*"` → all 72 shards (~107K samples)
- `"small/train-000[0-3]*"` → first 40 shards (~60K samples)

---

## All notebook changes (cumulative)

| Cell | Original | Changed to | Reason |
|---|---|---|---|
| cell-2 (Section 1) | `from trl import SFTConfig, SFTTrainer` | `from transformers import Trainer, TrainingArguments` | SFTTrainer has accuracy computation bug with gradient accumulation |
| cell-2 (Section 1) | Static GPU info print | `get_free_gpu()` → `GPU_ID`, `DEVICE`; set `CUDA_VISIBLE_DEVICES` | Auto-select idle GPU, restrict to single GPU so Trainer doesn't use DataParallel |
| cell-2 (Section 1) | (no dataset cache config) | `os.environ["HF_DATASETS_CACHE"] = "/ssd1/zhuoyuan/speechQwen2VL/data"` before imports | Dataset arrow files go to local SSD, not hf_cache |
| cell-4 (Section 2) | No `cache_dir` | Removed redundant cache config; added `DATA_DIR` print | Cache is set in cell-2; cell-4 just prints for verification |
| cell-9 (Section 4) | `device_map="cuda:0"` | `device_map=DEVICE` (`"cuda:0"` after CUDA_VISIBLE_DEVICES restriction) | Use auto-selected GPU |
| cell-11 (Section 5) | Labels included trailing `\n` after `<\|im_end\|>` | Mask everything after `<\|im_end\|>` | Chat template adds `\n` after `<\|im_end\|>`; model should learn to stop at `<\|im_end\|>` |
| cell-13 (Section 6) | Markdown: "SFTConfig" | Markdown: "TrainingArguments" + explanation of why we switched | Documentation |
| cell-14 (Section 6) | `SFTConfig(... max_seq_length, dataset_text_field, dataset_kwargs ...)` | `TrainingArguments(...)` with `eval_steps=500` | Plain Trainer; eval every 500 steps (was 100, too frequent) |
| cell-16 (Section 7) | `sft_config.learning_rate` etc. | `training_args.learning_rate` etc. | Variable renamed |
| cell-18 (Section 8) | `SFTTrainer(... args=sft_config ...)` | `Trainer(... args=training_args ...)` | Use plain Trainer |

---

## Next steps

1. Run full training (~7 hours on single GPU, 5,592 steps)
2. Monitor loss on wandb (project: `speechQwen2VL`, run: `stage1-audio-projector`)
3. After training: run Section 9 (inference test) — should produce coherent transcription
4. If inference looks good: run Section 10 (push to HuggingFace `DanJZY/Qwen2-VL-7B-Speech`)
5. Convert notebook to training script with multi-GPU DDP support (`torchrun`)
6. Commit all changes
