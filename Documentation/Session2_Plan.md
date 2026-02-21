# Session 2 Plan: Fork Repos & Build Notebook 02

## Context

Session 1 is complete: project structure, environment config, Notebook 01 (data exploration), documentation. Session 2 has two goals:

1. Fork `huggingface/transformers` and `QwenLM/Qwen2-VL` on GitHub, create `speech-qwen2vl` branches
2. Build Notebook 02: Tokenizer & Processor Modifications — extend Qwen2-VL's chat template, tokenizer, and processor to support audio input

Development happens on Mac (code editing only). Testing happens on Google Colab.

## Summary

1. **Fork repos** — Fork `huggingface/transformers` and `QwenLM/Qwen2-VL` on GitHub using `gh` CLI, create `speech-qwen2vl` branches, fix a bug in `setup_forks.sh` (wrong pip install path)
2. **Fork modifications (2 files)** — Add `fetch_audio()` to `vision_process.py` in the Qwen2-VL fork, and add audio token expansion to `processing_qwen2_vl.py` in the transformers fork
3. **Build Notebook 02 (11 sections)** — Create HF repo, `format_data()` function, modify chat template for audio, add 3 special tokens (`audio_start/pad/end`), push processor to HF, test `fetch_audio` + `process_vision_info`, end-to-end processor test
4. **Commit & push** — Push fork changes to fork repos, push notebook + fixes to main repo

---

## Part 1: Fork Repos on GitHub

User has not forked yet. We will do it using `gh` CLI from Mac.

**What forking means**: We create copies of the original repos under your GitHub account. Then we create a `speech-qwen2vl` branch where our audio modifications will live. This keeps the original code intact and our changes isolated.

### 1.1 Fork huggingface/transformers
```bash
# Fork on GitHub (creates ZhuoyuanJiang/transformers)
gh repo fork huggingface/transformers --clone=false

# Clone locally (blobless clone — fast, ~1 min instead of ~10 min)
cd /path/to/speechQwen2VL
git clone --filter=blob:none https://github.com/ZhuoyuanJiang/transformers.git forks/transformers

# Create speech-qwen2vl branch from the exact commit we need (4.56.0.dev0)
cd forks/transformers
git checkout -b speech-qwen2vl 0f9c9088
git push -u origin speech-qwen2vl
```

**Why commit `0f9c9088`?** This is transformers 4.56.0.dev0, the same version used in the reference project. It has mature Qwen2-VL code and avoids v5.0 breaking changes.

### 1.2 Fork QwenLM/Qwen2-VL
```bash
# Fork on GitHub (creates ZhuoyuanJiang/Qwen2-VL)
gh repo fork QwenLM/Qwen2-VL --clone=false

# Clone locally
cd /path/to/speechQwen2VL
git clone https://github.com/ZhuoyuanJiang/Qwen2-VL.git forks/Qwen2-VL

# Create speech-qwen2vl branch from current main
cd forks/Qwen2-VL
git checkout -b speech-qwen2vl
git push -u origin speech-qwen2vl
```

### 1.3 Fix setup_forks.sh (bug found)
- **Bug**: Line 66 does `pip install -e "$FORKS_DIR/Qwen2-VL"` but the installable package is in the `qwen-vl-utils/` subdirectory
- **Fix**: Change to `pip install -e "$FORKS_DIR/Qwen2-VL/qwen-vl-utils"`

---

## Part 2: Fork Modifications (2 files)

### 2.1 qwen-vl-utils: `vision_process.py`
**File**: `forks/Qwen2-VL/qwen-vl-utils/src/qwen_vl_utils/vision_process.py`

**Add `fetch_audio()` function** (parallel to existing `fetch_image()`):
- Input: `ele: dict` with `audio` key (bytes, file path, or numpy array)
- Output: `(audio_array, sample_rate)` — float32 mono numpy array, resampled to 16kHz
- Uses `soundfile.read(BytesIO(...))` for bytes, file path handling for strings
- Resamples to 16kHz if needed (Whisper's expected sample rate)

**Extend `extract_vision_info()`** to also detect `audio` content type in conversations.

**Extend `process_vision_info()`** to return audio inputs:
- Current return: `(image_inputs, video_inputs)` or 3-tuple with video_kwargs
- New return: add `audio_inputs` (list of `(audio_array, sample_rate)` tuples)
- Must be backward-compatible (audio_inputs=None when no audio present)

### 2.2 transformers: `processing_qwen2_vl.py`
**File**: `forks/transformers/src/transformers/models/qwen2_vl/processing_qwen2_vl.py`

**Modify `__init__`**:
- Add `self.audio_token = "<|audio_pad|>"`
- Add `self.audio_token_id` lookup (same pattern as image_token_id)

**Modify `__call__`**:
- Add `audios` parameter (list of numpy arrays at 16kHz)
- Add audio token expansion logic (parallel to image/video expansion):
  - For each audio, compute `num_audio_tokens` based on audio duration
  - Formula: `min(ceil(duration_seconds * 50), 1500)` where 50 = 1500 tokens / 30 seconds
  - Replace single `<|audio_pad|>` with `num_audio_tokens` copies (same pattern as image_pad)
- Process audios through `WhisperFeatureExtractor` → mel spectrograms
- Return `audio_features` (stacked mel spectrograms) and `audio_lengths` (token counts) in BatchFeature output

### 2.3 Commit & push fork changes
- Commit changes to `speech-qwen2vl` branch in each fork
- Push to GitHub (so Colab can install from them)

---

## Part 3: Build Notebook 02

**File**: `notebooks/02_tokenizer_and_processor.ipynb`

Maps to skeleton cells 22-43. Follows same style as Notebook 01 (markdown headers, docstrings, print output, Colab-compatible).

### Section 1: Setup & Imports
- Colab install cell: `pip install` from our fork branches + pinned deps
- Imports: transformers, qwen_vl_utils, datasets, huggingface_hub, soundfile, etc.
- Version/device checks

### Section 2: Create HF Repository
- Use `HfApi().create_repo()` to create `ZhuoyuanJiang/Qwen2-VL-7B-Speech` (or user's preferred name)
- This repo will store the modified processor/tokenizer

### Section 3: Load Dataset
- Stream `speechbrain/LargeScaleASR` (same pattern as Notebook 01)
- Grab a few samples for testing

### Section 4: `format_data()` Function
- Convert dataset sample to OpenAI conversation format:
  ```python
  {"messages": [
      {"role": "user", "content": [
          {"type": "audio", "audio": audio_bytes},
          {"type": "text", "text": "Transcribe this audio."}
      ]},
      {"role": "assistant", "content": [
          {"type": "text", "text": transcription_text}
      ]}
  ]}
  ```
- Test on first sample, print formatted output

### Section 5: Modify Chat Template
- Download `Qwen2VLProcessor` from `Qwen/Qwen2-VL-7B-Instruct`
- Extract existing Jinja2 chat template
- Add `audio` content type handling (parallel to image/video):
  ```jinja2
  {% elif content['type'] == 'audio' %}
  <|audio_start|><|audio_pad|><|audio_end|>
  ```
- Assign modified template to `processor.chat_template`
- Test: apply template to a formatted sample, verify audio tokens appear

### Section 6: Add Audio Special Tokens
- Add 3 special tokens to tokenizer:
  - `<|audio_start|>` → ID 151657
  - `<|audio_pad|>` → ID 151658
  - `<|audio_end|>` → ID 151659
- These are within existing vocab size (152,064) — **no embedding resize needed**
- Explanation: The model's embedding matrix was initialized for 152,064 tokens but only ~151,657 are defined in the tokenizer. New tokens fill unused slots.

### Section 7: Save & Push Processor
- `processor.save_pretrained("./speech_processor")` (saves as folder)
- `HfApi().upload_folder()` to push to HF repo

### Section 8: Load & Test Processor from HF
- Load processor from HF repo
- Apply chat template to a formatted conversation
- Tokenize and verify audio tokens in input_ids
- Decode back to text to confirm token placement

### Section 9: Test `fetch_audio` & `process_vision_info`
- Import `process_vision_info` from `qwen_vl_utils`
- Call with a formatted conversation containing audio content
- Verify audio_inputs are returned correctly (numpy array, 16kHz, mono)

### Section 10: End-to-end Processor Test
- Full pipeline: format_data → process_vision_info → processor.__call__
- Verify BatchFeature contains: `input_ids`, `attention_mask`, `audio_features`, `audio_lengths`
- Print shapes, verify correctness

### Section 11: Cleanup
- Delete models/processors from memory
- Clear CUDA cache if available

---

## Part 4: Commit & Push Main Repo

- Commit Notebook 02 + setup_forks.sh fix to main repo
- Create `Documentation/Session2_Progress_20260220.md` (at end of session)
- Push to GitHub

---

## Files Modified/Created

| File | Action | Description |
|------|--------|-------------|
| `forks/Qwen2-VL/.../vision_process.py` | Modify (fork) | Add fetch_audio, extend process_vision_info |
| `forks/transformers/.../processing_qwen2_vl.py` | Modify (fork) | Add audio token expansion, audios parameter |
| `scripts/setup_forks.sh` | Fix | Correct pip install path for qwen-vl-utils |
| `notebooks/02_tokenizer_and_processor.ipynb` | Create | New notebook |
| `Documentation/Session2_Progress_20260220.md` | Create | Session record (at end) |

## Key References

- Existing chat template: `Qwen/Qwen2-VL-7B-Instruct` on HF (token IDs 151643-151656)
- `fetch_image()` pattern in `vision_process.py` (line ~120-170)
- Image token expansion pattern in `processing_qwen2_vl.py` (the `<|placeholder|>` replacement loop)
- Notebook 01 style: `notebooks/01_data_exploration.ipynb`

## Verification

1. **Fork setup**: `gh repo view ZhuoyuanJiang/transformers` and `ZhuoyuanJiang/Qwen2-VL` — confirm forks exist
2. **Branch check**: Verify `speech-qwen2vl` branches exist with our modifications
3. **On Colab**: Clone main repo, open Notebook 02, run all cells
4. **Chat template**: Verify `<|audio_start|><|audio_pad|><|audio_end|>` appears for audio content
5. **Token expansion**: Verify audio_pad is repeated correctly based on audio duration
6. **process_vision_info**: Verify audio_inputs returned alongside image/video inputs
7. **Processor end-to-end**: Verify BatchFeature contains audio_features, audio_lengths
