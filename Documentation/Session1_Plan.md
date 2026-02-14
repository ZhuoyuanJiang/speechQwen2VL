# Plan: Fine-Tuning Qwen2-VL-7B for Speech Understanding (ASR)

## Context

The project extends Qwen2-VL-7B (a vision-language model) to also understand speech/audio input, enabling automatic speech recognition (ASR). The skeleton notebook (`fine_tuning_vlm_for_speech_understanding_trl.ipynb`) defines the full architecture and tasks but contains no implementations. The approach integrates OpenAI's Whisper encoder as an audio feature extractor into Qwen2-VL, with a learned projection layer bridging the two. Training uses a two-stage strategy: first train only the audio projection, then QLoRA fine-tune the full model.

The reference project (`vlm_Qwen2VL_object_detection_LOCAL`) provides a reusable environment and structural patterns.

---

## Hardware & Accounts

- **Lab server**: 2x A6000 (48GB each) — primary training machine
- **Google Colab**: Free tier T4 (16GB) or paid A100 — good for testing notebooks 01-04
- **WSL (Windows)**: RTX 4070 (12GB) — too small for training, usable for code editing and light inference
- **MacBook**: No NVIDIA GPU — code editing only, no training/inference
- **Development workflow**: Edit code on Mac/WSL → push via Git → run on lab server or Colab
- **HuggingFace**: Account exists; need to generate/configure HF_TOKEN (Settings > Access Tokens on huggingface.co)
- **Fork strategy**: GitHub forks with `speech-qwen2vl` branch

---

## Version Decision (Critical)

The skeleton notebook pins `transformers==4.47.0`, but the reference repo uses `transformers==4.56.0.dev0` (git commit `0f9c908`). The latest on PyPI is `v5.1.0` (breaking changes, avoid).

**Decision: Fork from the reference repo's commit (`0f9c908`, resolving to 4.56.0.dev0)**
- Proven compatible with torch==2.4.1+cu121, peft==0.17.1, trl==0.22.0.dev0, flash-attn==2.6.3
- More mature Qwen2-VL code than 4.47.0
- Avoids v5.0 breaking changes
- Notebook instructions will need minor adaptation but core architecture is unchanged

| Package | Version | Source |
|---------|---------|--------|
| transformers | 4.56.0.dev0 | Fork from commit `0f9c9088` on `huggingface/transformers` |
| trl | 0.22.0.dev0 | git commit `21529487` on `huggingface/trl` |
| qwen-vl-utils | 0.0.11 | Fork from `QwenLM/Qwen2-VL` (PyPI version baseline) |
| torch | 2.4.1+cu121 | PyTorch index |
| peft | 0.17.1 | PyPI |
| flash-attn | 2.6.3 | PyPI |

---

## Phase 0: Project Organization

- Create `Documentation/` folder with session plans (e.g., `Session1_Plan.md`)
- Rename original notebook to `fine_tuning_vlm_for_speech_understanding_trl_original.ipynb` (preserved as reference, never edited)
- Use the separate 7-notebook approach for development in `notebooks/`

---

## Phase 1: Environment & Project Setup

### 1.1 Create a NEW conda environment from reference project (do NOT reuse the object detection env)
- Copy `environment.yml` from the reference repo ([vlm_Qwen2VL_object_detection_LOCAL](https://github.com/ZhuoyuanJiang/vlm_Qwen2VL_object_detection_LOCAL)), rename env to `speech_qwen2vl`
- Add audio-specific deps: `librosa>=0.10.0`, `soundfile>=0.12.0`
- Add system audio libs via conda-forge: `ffmpeg`, `libsndfile`
- Add ASR evaluation deps: `jiwer`, `evaluate`
- `torchaudio 2.4.1+cu121` is already present in the reference env
- Replace `transformers` and `qwen-vl-utils` entries with fork URLs (see 1.2)
- **Important**: Install forks LAST to prevent other packages from overwriting them

### 1.2 Fork transformers and qwen-vl-utils
- Fork `huggingface/transformers` to user's GitHub, checkout commit `0f9c9088` (=4.56.0.dev0), create branch `speech-qwen2vl`
- Fork `QwenLM/Qwen2-VL` (qwen-vl-utils) to user's GitHub, create branch `speech-qwen2vl`
- Install both as editable (`pip install -e`) for development — **must be done after all other pip installs**
- Create `scripts/setup_forks.sh` to automate clone + install

### 1.3 Project directory structure
```
speechQwen2VL/
├── environment.yml
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_tokenizer_and_processor.ipynb
│   ├── 03_model_architecture.ipynb
│   ├── 04_inference_and_testing.ipynb
│   ├── 05_training_stage1_adapter.ipynb
│   ├── 06_training_stage2_qlora.ipynb
│   └── 07_evaluation.ipynb
├── src/
│   ├── data/       (dataset.py, collators.py, audio_utils.py)
│   ├── models/     (loader.py, inference.py, lora.py)
│   ├── training/   (config.py, freeze.py)
│   └── utils/      (memory.py, visualization.py)
├── scripts/
│   ├── setup_forks.sh
│   ├── train_stage1.py
│   ├── train_stage2.py
│   └── evaluate.py
├── forks/          (local clones, not committed)
└── assets/         (diagrams, loss plots)
```

---

## Phase 2: Notebook Development (7 notebooks)

### Notebook 01: Data Exploration
**Maps to**: skeleton cells 4-21
- Load `speechbrain/LargeScaleASR` in streaming mode
- Grab samples, inspect fields (`audio.array`, `audio.sampling_rate`, `text`)
- Visualize waveform and mel spectrogram with `librosa`
- Load Whisper-large-v3-turbo encoder via `transformers.WhisperModel`
- Process a sample through Whisper feature extractor -> encoder -> understand output shape `[batch, 1500, 1280]`

### Notebook 02: Tokenizer & Processor Modifications
**Maps to**: skeleton cells 22-43
- Create HF repository for project assets
- **`format_data()`**: Convert dataset samples to OpenAI conversation format (user=audio+prompt, assistant=transcription)
- **Chat template**: Extend Qwen2-VL's Jinja2 template to handle `audio` content type with `<|audio_start|><|audio_pad|><|audio_end|>` tokens
- **Extend tokenizer**: Add 3 special tokens (IDs 151657-151659, within existing vocab size — no embedding resize needed)
- Push processor to HF
- **Modify qwen-vl-utils** (`vision_process.py`): Add `fetch_audio()` to load audio bytes -> numpy array at 16kHz
- **Modify transformers** (`processing_qwen2_vl.py`): Add audio token expansion logic (repeat `<|audio_pad|>` dynamically based on each audio's actual length, not a fixed count), accept `audios` parameter, return `audio_features` and `audio_lengths`
- Test processor end-to-end with a formatted conversation

### Notebook 03: Model Architecture
**Maps to**: skeleton cells 44-50
- **Modify `configuration_qwen2_vl.py`**: Add `Qwen2VLAudioConfig` class with Whisper dimensions (d_model=1280, 32 layers, 20 heads, 128 mel bins)
- **Modify `modeling_qwen2_vl.py`**: Add to `Qwen2VLForConditionalGeneration`:
  - `self.audio_encoder` — WhisperEncoder (32 layers, d_model=1280)
  - `self.audio_projector` — 2-layer MLP: `Linear(1280->3584) -> GELU -> Linear(3584->3584)`
  - Extend `forward()` to process audio through encoder+projector, then scatter into input embeddings at `<|audio_pad|>` positions
  - Extend `get_rope_index()` for audio token positions (1D sequential, like text)
- **Initialize model**: Load Qwen2-VL-7B base weights, then load Whisper-turbo encoder weights into `audio_encoder` (only `audio_projector` remains random)
- Push complete model to HF

### Notebook 04: Inference & Testing
**Maps to**: skeleton cells 51-57
- Build `run_inference()` function: conversation -> chat template -> process -> generate -> decode
- **VL test**: Red car bounding box detection — verify identical output to vanilla Qwen2-VL (confirms audio additions don't break VL)
- **Audio test**: Transcribe a dataset sample — expect garbage (random projector)

### Notebook 05: Training Stage 1 — Adapter Only
**Maps to**: skeleton cells 58-73
- Download dataset shards (small/train-0000*, small/train-0001*, test-00000*)
- Load model (bf16, no quantization)
- **Freeze everything except `audio_projector`** (~17M trainable params)
- Build `AudioTextCollator`: format conversations, process audio, create labels with masked audio/pad tokens
- SFTConfig: lr=1e-4, epochs=3, batch_size=2, gradient_accum=8, bf16
- wandb tracking, SFTTrainer, train
- Push to HF (only 1/4 shards changes — the audio_projector weights)

### Notebook 06: Training Stage 2 — QLoRA
**Maps to**: skeleton cells 74-83
- Load stage 1 model with 4-bit NF4 quantization (BitsAndBytesConfig)
- LoRA config: r=64, alpha=128, targets=q/k/v/o/gate/up/down_proj
- SFTConfig: lr=2e-5 (lower than stage 1)
- SFTTrainer with `peft_config`, train
- Push LoRA adapter weights to HF

### Notebook 07: Evaluation
**Maps to**: skeleton cells 84-91
- Load base model + LoRA adapters
- Prepare custom .wav audio: record -> convert -> resample to 16kHz -> BytesIO
- Run inference on custom audio and test dataset samples
- **Quantitative evaluation**: Compute WER (Word Error Rate) and CER (Character Error Rate) on test subset using `jiwer`
- Qualitative evaluation of transcription quality on custom recordings

---

## Phase 3: Fork Modifications (4 files across 2 repos)

### transformers fork (3 files in `src/transformers/models/qwen2_vl/`)

| File | Changes |
|------|---------|
| `configuration_qwen2_vl.py` | Add `Qwen2VLAudioConfig` class; add `audio_config`, `audio_token_id`, `audio_start_token_id`, `audio_end_token_id` to `Qwen2VLConfig` |
| `processing_qwen2_vl.py` | Add audio token expansion (parallel to image/video); accept `audios` param; process through WhisperFeatureExtractor; return `audio_features` + `audio_lengths` |
| `modeling_qwen2_vl.py` | Add `audio_encoder` (WhisperEncoder) + `audio_projector` (MLP); extend `forward()` for audio embedding injection; extend `get_rope_index()` for audio positions |

### qwen-vl-utils fork (1 file)

| File | Changes |
|------|---------|
| `vision_process.py` | Add `fetch_audio()` (bytes->numpy@16kHz); extend `process_vision_info()` to also return audio inputs |

---

## Phase 4: GitHub-Ready Conversion

After notebooks are working:
1. Extract reusable code from notebooks into `src/` modules
2. Create standalone training scripts (`scripts/train_stage1.py`, `scripts/train_stage2.py`)
3. Create `scripts/evaluate.py` for evaluation
4. Write README with architecture diagram, results, quick start, training reproduction
5. Add `.gitignore`, proper `requirements.txt`
6. Clean notebook outputs, keep notebooks as documentation/tutorials

---

## Verification Plan

1. **Processor test**: Format a sample, apply chat template, verify audio tokens expand to the correct number of positions based on audio length
2. **Model forward test**: Pass a batch through the model, verify output shape and no errors
3. **VL preservation test**: Run red car detection prompt, compare output with vanilla Qwen2-VL-7B
4. **Stage 1 training**: Confirm loss decreases, only audio_projector gradients are non-zero
5. **Stage 2 training**: Confirm loss decreases further with QLoRA
6. **End-to-end inference**: Transcribe a known audio sample, verify reasonable output

---

## Implementation Order

```
[1] Environment setup (copy env, add deps, fork repos)
[2] Notebook 01 — data exploration (no code deps)
[3] Notebook 02 — tokenizer/processor (fork modifications happen here)
[4] Notebook 03 — model architecture (depends on 3)
[5] Notebook 04 — inference testing (depends on 4)
[6] Notebook 05 — stage 1 training (depends on 5)
[7] Notebook 06 — stage 2 training (depends on 6)
[8] Notebook 07 — evaluation (depends on 7)
[9] GitHub conversion — extract to src/, scripts/, README
```
