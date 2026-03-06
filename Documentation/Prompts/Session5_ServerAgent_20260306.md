# Server Agent Prompt — Session 5: Train Audio Projector (Stage 1)

**Date**: 2026-03-06
**Target**: Server with 2x A6000 (48GB each)
**Goal**: Set up the environment from scratch, then run Notebook 05 to train the audio projector.

---

## Prerequisites

Before starting, ensure:
- **GitHub token auth** is configured on the server. The main repo (`ZhuoyuanJiang/speechQwen2VL`) and both fork repos are private. All clone URLs (in this prompt and in `scripts/setup_forks.sh`) use HTTPS, so you need a personal access token configured via `gh auth login`, `git credential store`, or the `GH_TOKEN` / `GITHUB_TOKEN` environment variable. SSH keys alone won't work with HTTPS URLs — if you prefer SSH, replace all `https://github.com/` URLs with `git@github.com:` in both this prompt's clone command and in `scripts/setup_forks.sh` before running.
- **conda** is available (Miniconda or Anaconda installed).

---

## Instructions

You are setting up and running a training notebook on a fresh server. The project fine-tunes Qwen2-VL (a vision-language model) to also understand speech, by adding a Whisper audio encoder and training a projection layer to map audio embeddings into the LLM's text space.

### Step 1: Clone the repository and read context

```bash
git clone https://github.com/ZhuoyuanJiang/speechQwen2VL.git
cd speechQwen2VL
```

Read these files to understand the full project context:

1. **`Documentation/Session5_Plan.md`** — The detailed plan for this session. This is the primary reference. Read it thoroughly before doing anything else.
2. **`Documentation/Prompts/Session5_ServerAgent_20260306.md`** — This file (you're reading it).
3. **`notebooks/05_training_stage1_adapter.ipynb`** (or its `.py` version `notebooks/05_training_stage1_adapter.py`) — The notebook you'll run. Read the code to understand what each section does.
4. **`fine_tuning_vlm_for_speech_understanding_trl_original.ipynb`** — The original skeleton notebook from the project advisor. This is the project's guideline — cells 58-73 correspond to what Notebook 05 implements. Useful if you need to understand the original intent.
5. **`Documentation/Session4_Plan.md`** and **`notebooks/04_inference_and_testing.ipynb`** — Previous session's work. Notebook 04 tested the inference pipeline and confirmed the audio projector produces garbage (expected — it's randomly initialized). After Stage 1 training, Notebook 05's verification step should show meaningful output instead.
6. **`Documentation/Lessons/general_QA.md`** — Explains the fork-based project structure (3 repos: main repo, transformers fork, Qwen3-VL fork).
7. **`scripts/setup_forks.sh`** — Script for cloning and installing the fork repos.
8. **`environment.yml`** and **`requirements.txt`** — All dependency versions are pinned here. The environment setup uses these files.

### Step 2: Set up the conda environment

The project provides `environment.yml` and `requirements.txt` with all versions pinned. Use them directly:

```bash
# Check CUDA version first — environment.yml expects CUDA 12.1
nvidia-smi

# Create the conda environment (installs Python 3.10, CUDA toolkit, ffmpeg, libsndfile, and all pip deps)
conda env create -f environment.yml
conda activate speech_qwen2vl
```

If `conda env create` fails (e.g., CUDA version mismatch), you can fall back to manual setup:

```bash
conda create -n speech_qwen2vl python=3.10 -y
conda activate speech_qwen2vl

# Install system-level dependencies that environment.yml normally provides via conda
conda install -c conda-forge -c nvidia cuda-toolkit=12.1.1 ffmpeg libsndfile -y

# Install pip dependencies
pip install -r requirements.txt
```

**Note**: `requirements.txt` intentionally excludes `transformers` and `qwen-vl-utils` — these are installed from forks in Step 3.

### Step 3: Install the forked libraries (editable mode)

The project modifies two HuggingFace libraries via forks. The setup script handles everything:

```bash
bash scripts/setup_forks.sh
```

This clones (or pulls) both forks into `forks/`, checks out the `speech-qwen2vl` branch, and installs them in editable mode. Forks are installed LAST to prevent other packages from overwriting them.

**Verify the installs are correct:**

```bash
python -c "import transformers; print(transformers.__version__, transformers.__file__)"
# Should print: 4.56.0.dev0 and a path inside forks/transformers/

python -c "import qwen_vl_utils; print(qwen_vl_utils.__file__)"
# Should print a path inside forks/Qwen2-VL/qwen-vl-utils/ (NOT a site-packages path)
```

If either path points to `site-packages` instead of `forks/`, the fork install was overwritten. Re-run `bash scripts/setup_forks.sh`.

### Step 4: HuggingFace and wandb login

```bash
huggingface-cli login
# Enter token for account: DanJZY

wandb login
# Enter API key
```

### Step 5: Verify TRL compatibility

Before running the notebook, check the TRL version and parameter names:

```bash
python -c "import trl; print(trl.__version__)"
python -c "from trl import SFTConfig; import inspect; sig = inspect.signature(SFTConfig); print('max_seq_length' in sig.parameters, 'max_length' in sig.parameters)"
```

- If it prints `True False` → `max_seq_length` is correct (notebook uses this).
- If it prints `False True` → rename `max_seq_length` to `max_length` in the notebook's SFTConfig cell.
- Also verify `dataset_text_field` and `dataset_kwargs` are valid parameters. If SFTTrainer's API has changed and these parameters cause errors, fall back to plain `Trainer` from transformers (same API, just replace `SFTTrainer` → `Trainer` and `SFTConfig` → `TrainingArguments`).

### Step 6: Run the notebook

You can run it as a Jupyter notebook or convert to a Python script:

**Option A: Jupyter**
```bash
jupyter notebook notebooks/05_training_stage1_adapter.ipynb
```

**Option B: Run as Python script**
```bash
cd /path/to/speechQwen2VL
python notebooks/05_training_stage1_adapter.py
```

**Option C: Convert and run**
```bash
jupyter nbconvert --to script notebooks/05_training_stage1_adapter.ipynb
python notebooks/05_training_stage1_adapter.py
```

### What to pay attention to (in order of when they appear)

1. **Section 1 — Environment**: Verify it shows 2x A6000 GPUs and prints `transformers: 4.56.0.dev0` from the forks path (not a pip-installed version).

2. **Section 2 — Dataset + filtering**: Watch the printed drop count after pre-filtering. If it's 0, the dataset is clean. If it drops many samples (>5% of train), investigate — the filter budget or template overhead might be miscalculated. If the `_processor.apply_chat_template` call fails on the dummy messages, the chat template may not handle the `"placeholder"` audio value — adjust the dummy message or measure overhead a different way.

3. **Section 4 — Model load**: Should show ~16.7 GB GPU memory. If OOM, check nothing else is using the GPU (`nvidia-smi`). `device_map="cuda:0"` pins to one GPU — this is intentional.

4. **Section 5 — Collator test**: **This is the most important sanity check.** Verify:
   - Decoded labels match the ground truth transcript + `<|im_end|>`
   - The token view around the assistant boundary shows MASKED → real label transition at the right place
   - If the collator asserts (assistant marker not found, multiple markers, zero labels, missing `<|im_end|>`), something is wrong with the chat template or tokenization — debug before continuing.

5. **Section 6 — SFTConfig**: If `max_seq_length` is not a valid parameter, see Step 5 above. If `dataset_kwargs` or `dataset_text_field` causes an error, check the TRL version's API. If SFTTrainer conflicts persist, fall back to plain `Trainer` from transformers (same API).

6. **Section 7 — wandb**: If you don't want wandb logging, change `report_to="wandb"` to `report_to="none"` in the SFTConfig.

7. **Section 8 — Training**: Watch the wandb dashboard (or console logs) for loss decreasing. First few steps will have high loss (random projector). Should drop significantly within the first epoch. If training is very slow (low GPU utilization), try increasing `dataloader_num_workers`. If you hit pickle errors in workers, set `dataloader_num_workers=0`.

8. **Section 9 — Post-training inference**: Output should be somewhat coherent — less refusal-like than Session 4's "I'm sorry, but I can't assist with that." Even partial/noisy transcription counts as success. If output is still garbage after 3 epochs, check that training loss actually decreased.

9. **Section 10 — Push**: Only push if inference looks reasonable. Only 1 of 4 safetensor shards should change on HuggingFace (the one containing audio_projector weights).

### Key project details for reference

- **GitHub**: `github.com/ZhuoyuanJiang/speechQwen2VL` (private)
- **HuggingFace model**: `DanJZY/Qwen2-VL-7B-Speech`
- **Transformers fork**: `ZhuoyuanJiang/transformers` branch `speech-qwen2vl`
- **Qwen3-VL fork**: `ZhuoyuanJiang/Qwen3-VL` branch `speech-qwen2vl`
- **Audio special tokens**: `<|audio_start|>` 151657, `<|audio_pad|>` 151658, `<|audio_end|>` 151659
- **Trainable params**: ~17M (audio_projector only), ~0.19% of total ~8.3B
- **Expected GPU memory**: ~22-25 GB on single A6000 (48GB available)
