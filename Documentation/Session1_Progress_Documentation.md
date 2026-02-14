# Session 1 — Progress Documentation

**Date**: February 13, 2026
**Objective**: Set up the project foundation for fine-tuning Qwen2-VL-7B for speech understanding (ASR)

---

## 1. Project Understanding & Exploration

### 1.1 Analyzed the Skeleton Notebook
- Read through the entire `fine_tuning_vlm_for_speech_understanding_trl.ipynb` (91 cells)
- This is a guided project notebook that defines the full architecture and task flow but contains no implementations — all code sections are marked "TASK"
- Identified the key components:
  - Dataset: `speechbrain/LargeScaleASR` (25,000 hours of English speech)
  - Audio encoder: Whisper-large-v3-turbo (encoder only, not decoder)
  - Base model: Qwen2-VL-7B (vision-language model extended with audio)
  - Training: Two-stage approach (adapter-only → QLoRA full model)
  - Requires forking `transformers` and `qwen-vl-utils` to add audio support

### 1.2 Analyzed the Reference Project
- Explored the reference repository: [vlm_Qwen2VL_object_detection_LOCAL](https://github.com/ZhuoyuanJiang/vlm_Qwen2VL_object_detection_LOCAL)
- This project fine-tuned Qwen2-VL for object detection (nutrition table bounding boxes)
- Extracted key information:
  - Well-structured codebase: `src/` (data, models, training, utils), `scripts/`, `notebooks/`, `docker/`
  - Environment: Python 3.10.18, PyTorch 2.4.1+cu121, transformers 4.56.0.dev0 (git commit pin), trl 0.22.0.dev0, peft 0.17.1, flash-attn 2.6.3
  - Uses same core ML stack as our speech project
  - Has `environment.yml`, `environment.lock.yml`, and `requirements.txt` with exact version pins

---

## 2. Implementation Plan

### 2.1 Created the Plan
- Designed a 7-notebook decomposition of the 91-cell skeleton:
  1. Data Exploration (cells 4-21)
  2. Tokenizer & Processor Modifications (cells 22-43)
  3. Model Architecture (cells 44-50)
  4. Inference & Testing (cells 51-57)
  5. Training Stage 1 — Adapter Only (cells 58-73)
  6. Training Stage 2 — QLoRA (cells 74-83)
  7. Evaluation (cells 84-91)
- Plan covers: environment setup, fork strategy, project structure, notebook breakdown, fork modifications (4 files across 2 repos), GitHub-ready conversion, and verification steps
- Saved as `Documentation/Session1_Plan.md`

### 2.2 Key Decisions Made
- **transformers version**: Fork from the reference project's commit (`0f9c9088`, resolving to 4.56.0.dev0), NOT the skeleton notebook's pin of 4.47.0 — rationale: proven compatible with our full stack, more mature Qwen2-VL code, avoids v5 breaking changes
- **Fork strategy**: Actual GitHub forks with `speech-qwen2vl` branch (not monkey-patching or local-only edits)
- **Environment**: New conda env (`speech_qwen2vl`) based on reference project, NOT reusing the object detection env (to avoid polluting it)
- **Development workflow**: Write code on Mac → push via Git → run on server or Google Colab

### 2.3 Hardware Configuration Documented
- Dedicated server: 2x A6000 (48GB each) — primary training machine
- Google Colab: Free tier T4 (16GB) or paid A100 — good for testing notebooks 01-04
- WSL (Windows): RTX 4070 (12GB) — too small for training, usable for code editing and light inference
- MacBook: No NVIDIA GPU — code editing only

---

## 3. Project Structure Created

### 3.1 Directory Layout
```
speechQwen2VL/
├── .gitignore
├── environment.yml
├── requirements.txt
├── fine_tuning_vlm_for_speech_understanding_trl.ipynb          (original skeleton, working copy)
├── fine_tuning_vlm_for_speech_understanding_trl_original.ipynb  (preserved backup, never edited)
├── Documentation/
│   ├── Session1_Plan.md
│   ├── Session1_Progress_Documentation.md
│   └── Lessons/
│       └── session1_QA.md
├── notebooks/
│   └── 01_data_exploration.ipynb
├── scripts/
│   └── setup_forks.sh
├── src/
│   ├── __init__.py
│   ├── data/__init__.py
│   ├── models/__init__.py
│   ├── training/__init__.py
│   └── utils/__init__.py
└── assets/                         (empty, for diagrams and plots later)
```

### 3.2 Original Notebook Preserved
- Copied `fine_tuning_vlm_for_speech_understanding_trl.ipynb` to `fine_tuning_vlm_for_speech_understanding_trl_original.ipynb`
- The `_original` version is never to be edited — serves as a reference for tracing back to the instructor's original task definitions

---

## 4. Environment Configuration

### 4.1 environment.yml
- Created based on the reference project's environment
- Renamed env to `speech_qwen2vl`
- Channels: conda-forge, defaults, nvidia
- Conda dependencies: Python 3.10.18, cuda-toolkit 12.1.1, ffmpeg, libsndfile (audio system libs from conda-forge)
- Pip dependencies: references `requirements.txt`

### 4.2 requirements.txt
- All pip packages with exact version pins matching the reference project
- Core ML: torch==2.4.1+cu121, torchaudio==2.4.1+cu121, torchvision==0.19.1+cu121
- HuggingFace ecosystem: trl (git commit pin `21529487`), peft==0.17.1, datasets==4.0.0, accelerate==1.10.0
- Quantization & training: bitsandbytes==0.47.0, flash-attn==2.6.3
- Audio processing (new): librosa>=0.10.0, soundfile>=0.12.0
- ASR evaluation (new): jiwer>=3.0.0
- Tracking: wandb==0.21.1, tensorboard==2.20.0
- Jupyter: jupyter==1.1.1, jupyterlab==4.4.6
- NOTE: transformers and qwen-vl-utils are intentionally excluded — they are installed from forks via `scripts/setup_forks.sh`

### 4.3 scripts/setup_forks.sh
- Automates the fork setup process:
  1. Creates `forks/` directory
  2. Clones the user's fork of `huggingface/transformers` → `forks/transformers/`
  3. Checks out the `speech-qwen2vl` branch
  4. Runs `pip install -e forks/transformers` (editable install)
  5. Repeats for `QwenLM/Qwen2-VL` → `forks/Qwen2-VL/`
  6. Verifies both packages import correctly and prints file paths
- Handles both fresh clone and re-pull (if fork already cloned)
- Must be run AFTER `pip install -r requirements.txt` to prevent other packages from overwriting the forks

### 4.4 .gitignore
- Excludes: Python bytecode, Jupyter checkpoints, `forks/` directory, training outputs (`outputs/`, `checkpoints/`, `wandb/`), large files (`.wav`, `.mp3`, `.pt`, `.bin`, `.safetensors`), HF cache, OS files (`.DS_Store`), IDE configs

---

## 5. Notebook 01: Data Exploration

### 5.1 Purpose
- Understand the dataset structure, audio fundamentals, and Whisper encoder before building the pipeline
- Maps to skeleton notebook cells 4-21
- Can run on Google Colab (free tier), server, or local machine (CPU-only is fine)

### 5.2 Sections Implemented
1. **Environment Setup**: Import checks, Colab-compatible install with pinned transformers version (same git commit as server: `0f9c9088`)
2. **Load Dataset (Streaming Mode)**: Loads `speechbrain/LargeScaleASR` using `data_files="small/train-00000*"` in streaming mode (no full download). Inspects sample structure and prints all fields.
3. **Audio Decoding**: Helper function `decode_audio()` that decodes raw bytes from `sample['wav']['bytes']` into a float32 numpy array using `soundfile`. Handles stereo-to-mono conversion.
4. **Visualize Audio**: Waveform plot and mel spectrogram using librosa, plus in-notebook audio playback via `IPython.display.Audio`.
5. **Explore Multiple Samples**: Iterates 10 samples, prints duration, sampling rate, text length, and text preview in a formatted table.
6. **Whisper Encoder**: Loads `openai/whisper-large-v3-turbo` with explicit `torch_dtype=torch.float32`. Prints encoder config (d_model=1280, 32 layers, 20 heads, 128 mel bins). Processes audio through feature extractor → encoder, shows output shape `[1, 1500, 1280]`.
7. **Audio Token Count vs Length**: Tests different audio durations (1s, 5s, 10s, 20s, 30s) to show that Whisper always pads to 30 seconds → always outputs 1500 tokens.
8. **Compare Spectrograms**: Side-by-side plot of librosa mel spectrogram vs Whisper's log-mel spectrogram.

### 5.3 Dataset Discovery
- The dataset stores audio as raw bytes in `sample['wav']['bytes']`, NOT as a decoded `audio` field with `array` and `sampling_rate`
- Decoding requires `soundfile.read(BytesIO(wav_bytes))`
- The dataset is also known as `speechbrain/LoquaciousSet` (alternate name)
- Four configs available: small (250h), medium (2,500h), large (25,000h), clean (13,000h)

### 5.4 Bug Fix: Whisper dtype Mismatch on Colab
- **Problem**: Whisper-large-v3-turbo stores weights in float16 on HuggingFace Hub. On Colab (transformers v5+), the model loads in float16 by default, but the feature extractor always outputs float32. This causes `RuntimeError: Input type (float) and bias type (c10::Half) should be the same` when the float32 spectrogram meets the float16 Conv1d layer weights.
- **Fix 1**: Load model with `torch_dtype=torch.float32` to force all weights to float32
- **Fix 2**: Pin transformers on Colab to the same git commit as the server (`0f9c9088` = 4.56.0.dev0), so behavior is identical across environments
- Both fixes applied in the notebook
- Documented in `Documentation/Lessons/session1_QA.md`

---

## 6. Documentation Created

### 6.1 Session1_Plan.md
- Full implementation plan with phases, notebook breakdown, fork modifications, verification steps, and implementation order
- Updated multiple times based on reviewer feedback and user clarifications
- Includes hardware configuration, version decision table, and environment setup instructions

### 6.2 Lessons/session1_QA.md
- Q1: Detailed explanation of the float16/float32 dtype mismatch error — what each component produces, where they crash, and how to fix it

### 6.3 Session1_Progress_Documentation.md
- This file — comprehensive record of all work done in Session 1

---

## 7. Git & GitHub Setup

### 7.1 Local Git Repository
- Initialized git repo in the project directory
- Created initial commit with all 14 files
- Commit message follows the agreed-upon format: title, summary, changed files with bullet points, files modified/created list

### 7.2 GitHub Repository
- Created private repository: [github.com/ZhuoyuanJiang/speechQwen2VL](https://github.com/ZhuoyuanJiang/speechQwen2VL)
- Pushed main branch to remote
- Enables multi-device workflow: edit on Mac → push → pull on Colab/server

---

## 8. Cross-Platform Development Notes

### 8.1 Multi-Device Workflow Established
- **Mac (current)**: Code editing only. No NVIDIA GPU, cannot run training or GPU inference.
- **Google Colab**: Testing notebooks 01-04. Pin package versions to match server. Free T4 (16GB) sufficient for exploration and inference testing.
- **WSL (Windows, RTX 4070)**: Code editing and light inference. 12GB VRAM too small for training Qwen2-VL-7B.
- **Dedicated server (2x A6000)**: Primary training machine. Conda environment created here. Notebooks 05-07 must run here.

### 8.2 Key Principle
- Always pin package versions across all environments (Colab, local, server) to avoid "works here, breaks there" issues
- The conda environment (`speech_qwen2vl`) is created only on the Linux server
- On Mac, just edit code and push via git

---

## 9. What's Next (Session 2)

According to the implementation plan, the next steps are:

1. **Fork the repos**: Fork `huggingface/transformers` and `QwenLM/Qwen2-VL` on GitHub, create `speech-qwen2vl` branches
2. **Set up conda env on server**: Create `speech_qwen2vl` env using `environment.yml` + `requirements.txt` + `setup_forks.sh`
3. **Run Notebook 01 on server**: Verify everything works in the production environment
4. **Build Notebook 02**: Tokenizer & Processor Modifications — extend chat template with audio tokens, modify qwen-vl-utils and transformers processor
