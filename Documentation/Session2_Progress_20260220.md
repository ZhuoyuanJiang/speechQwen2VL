# Session 2 — Progress Documentation

**Date**: February 20, 2026
**Objective**: Fork repos and build Notebook 02 (Tokenizer & Processor Modifications)
**Plan**: See `Documentation/Session2_Plan.md`

---

## 1. Fork Repos on GitHub

### 1.1 Forked huggingface/transformers

```bash
# Fork on GitHub (creates ZhuoyuanJiang/transformers)
gh repo fork huggingface/transformers --clone=false
# Output: https://github.com/ZhuoyuanJiang/transformers
```

- The fork copies the entire repo, including the latest v5.x code on main
- We don't use main — we create our own branch from an older commit (see 1.3)

### 1.2 Forked QwenLM/Qwen2-VL

```bash
# Fork on GitHub
gh repo fork QwenLM/Qwen2-VL --clone=false
# Output: https://github.com/ZhuoyuanJiang/Qwen3-VL
```

**Discovery: Repo was renamed from Qwen2-VL to Qwen3-VL**
- The upstream repo `QwenLM/Qwen2-VL` was renamed to `QwenLM/Qwen3-VL` on GitHub
- Our fork inherited the new name: `ZhuoyuanJiang/Qwen3-VL`
- We verified that the `qwen-vl-utils` code inside is still Qwen2-VL compatible:
  - `pyproject.toml` references "Qwen2-VL" throughout (repo URL, homepage)
  - Zero references to "Qwen3" in `vision_process.py`
  - The API is unchanged: `fetch_image()`, `fetch_video()`, `process_vision_info()`
- The version went from 0.0.11 (in our original plan) to 0.0.14, but since we fork and modify the code ourselves, this doesn't matter

### 1.3 Cloned Forks Locally

```bash
# Clone transformers (blobless clone for speed — downloads tree structure only, ~50-100 MB)
# Full repo is ~449 MB, Qwen2-VL is ~138 MB
mkdir -p forks/
git clone --filter=blob:none https://github.com/ZhuoyuanJiang/transformers.git forks/transformers

# Clone Qwen2-VL (naming the local folder Qwen2-VL even though the repo is Qwen3-VL)
git clone https://github.com/ZhuoyuanJiang/Qwen3-VL.git forks/Qwen2-VL
```

### 1.4 Created speech-qwen2vl Branches

**How forking and branching works:**
1. `gh repo fork` copies the entire repo with all history (all commits, old and new) to your GitHub account
2. The fork's `main` branch is the same as the original repo's main (e.g., transformers v5.x)
3. We create a `speech-qwen2vl` branch from a specific older commit, so our branch stays at the version we want
4. Our branch is independent — changes to the original repo's main don't affect it

```bash
# transformers: branch from commit 0f9c9088 (= version 4.56.0.dev0)
# Why this commit: the original repo has moved to v5.x with breaking changes.
# Commit 0f9c9088 is proven compatible with our stack (torch 2.4.1, peft 0.17.1, trl 0.22.0.dev0).
cd forks/transformers
git checkout -b speech-qwen2vl 0f9c9088
git push -u origin speech-qwen2vl

# Qwen2-VL: branch from commit 9658872 (= qwen-vl-utils version 0.0.14)
# We pin to a specific commit (not main) for reproducibility — if the repo updates
# in the future, our branch stays at this exact snapshot.
cd forks/Qwen2-VL
git checkout -b speech-qwen2vl 9658872
git push -u origin speech-qwen2vl
```

**Verification:**
```
=== transformers ===
* speech-qwen2vl
  main
0f9c9088d [3/3] make docs device agnostic, all en docs for existing models done (#40298)

=== Qwen2-VL ===
* speech-qwen2vl
  main
9658872 Merge pull request #1971 from 2003jiahang/patch-1
```

### 1.5 Fork Summary

| Fork | GitHub URL | Branch | Base Commit | Version |
|------|-----------|--------|-------------|---------|
| transformers | ZhuoyuanJiang/transformers | speech-qwen2vl | 0f9c9088 | 4.56.0.dev0 |
| Qwen2-VL | ZhuoyuanJiang/Qwen3-VL | speech-qwen2vl | 9658872 | qwen-vl-utils 0.0.14 |

---

## 2. Fixed setup_forks.sh

Three fixes applied to `scripts/setup_forks.sh`:

### 2.1 Bug Fix: pip install path for qwen-vl-utils
- **Problem**: Line 66 had `pip install -e "$FORKS_DIR/Qwen2-VL"` but the installable package (`pyproject.toml`) is in the `qwen-vl-utils/` subdirectory, not at the repo root
- **Fix**: Changed to `pip install -e "$FORKS_DIR/Qwen2-VL/qwen-vl-utils"`

### 2.2 Fix: Repo URL updated for Qwen3-VL rename
- **Problem**: `QWEN_VL_REPO` pointed to `ZhuoyuanJiang/Qwen2-VL.git` which no longer exists (renamed to Qwen3-VL)
- **Fix**: Changed to `ZhuoyuanJiang/Qwen3-VL.git`

### 2.3 Fix: Clone folder naming
- **Problem**: `git clone` of the Qwen3-VL repo would create a `Qwen3-VL/` folder by default, but the rest of the script expects `Qwen2-VL/`
- **Fix**: Added explicit folder name: `git clone "$QWEN_VL_REPO" Qwen2-VL`
- Syntax: `git clone <repo_url> <local_folder_name>` — the second argument overrides the default folder name

### 2.4 Enhancement: Detailed inline comments
- Added detailed comments explaining both forks: base repo, commit hash, why that commit, which files we modified
- Added explanation of the fork-then-branch workflow (fork copies entire repo including v5.x, but we branch from older commit)
- Added inline documentation of git clone syntax

### 2.5 Clarification: What setup_forks.sh Is For

`setup_forks.sh` is a **reproduction script**, not the initial fork setup. The distinction:

- **One-time developer setup** (what we did in this session): fork repos on GitHub, clone locally, create `speech-qwen2vl` branches from specific commits, make code modifications, commit and push. This is documented in this file and does not need to be repeated.

- **`setup_forks.sh`** (for anyone reproducing our work): assumes the forked repos already exist on GitHub with all modifications committed. It clones them, checks out the correct branch, and installs the packages. This is what someone would run when setting up a new machine (server, Colab, WSL) or reproducing the project from scratch.

The typical reproduction workflow is:
```bash
conda env create -f environment.yml && conda activate speech_qwen2vl
pip install -r requirements.txt
bash scripts/setup_forks.sh
```

---

## 3. Fork Modification: vision_process.py (Qwen2-VL fork)

**File**: `forks/Qwen2-VL/qwen-vl-utils/src/qwen_vl_utils/vision_process.py`
**Fork repo**: `ZhuoyuanJiang/Qwen3-VL`, branch `speech-qwen2vl`

### 3.1 Where fork changes live

The `forks/` directory is in `.gitignore` of the main repo — fork changes are NOT tracked by the main repo. Instead, each fork is its own git repository:

- `forks/Qwen2-VL/` → commits push to `github.com/ZhuoyuanJiang/Qwen3-VL` (speech-qwen2vl branch)
- `forks/transformers/` → commits push to `github.com/ZhuoyuanJiang/transformers` (speech-qwen2vl branch)

To see fork changes, visit the fork repo on GitHub or check `git log` inside the fork directory.

### 3.2 Changes made

Three changes to `vision_process.py`, all following existing patterns in the file:

**1. Added imports and audio constant** (top of file)
```python
import soundfile as sf

# Audio constants (Whisper expects 16kHz mono audio)
AUDIO_SAMPLE_RATE = 16000
```

**2. Added `fetch_audio()` function** (before `extract_vision_info()`, ~line 487)

Parallel to the existing `fetch_image()` function. Handles three input formats:
- `bytes`: raw audio bytes from dataset (e.g., `sample['wav']['bytes']`) — decoded with `soundfile.read(BytesIO(...))`
- `str`: file path to an audio file — decoded with `soundfile.read()`
- `np.ndarray`: pre-loaded audio array — used directly

All inputs are converted to float32 mono and resampled to 16kHz (Whisper's expected sample rate). `librosa.resample()` is used for resampling, imported lazily (only when resampling is actually needed).

Returns: `(audio_array, sample_rate)` tuple.

**3. Extended `extract_vision_info()`** (~line 539)

Added `"audio"` to the content type detection condition, so audio elements in conversations are extracted alongside images and videos:
```python
or "audio" in ele
or ele.get("type", "text") in ("image", "image_url", "video", "audio")
```

**4. Extended `process_vision_info()`** (~line 558)

- Added `audio_inputs = []` list alongside existing `image_inputs` and `video_inputs`
- Added `elif "audio" in vision_info:` branch that calls `fetch_audio()`
- Added `audio_inputs = None` when empty (same pattern as images/videos)
- Changed return from 2-tuple to **3-tuple**: `(image_inputs, video_inputs, audio_inputs)`
- With `return_video_kwargs=True`: returns 4-tuple `(images, videos, audios, video_kwargs)`

---

## 4. Fork Modification: processing_qwen2_vl.py (transformers fork)

**File**: `forks/transformers/src/transformers/models/qwen2_vl/processing_qwen2_vl.py`
**Fork repo**: `ZhuoyuanJiang/transformers`, branch `speech-qwen2vl`

### 4.1 Changes made

Four changes to `processing_qwen2_vl.py`, all following the existing image/video patterns:

**1. Added imports** (top of file)
```python
import math
from typing import List, Optional, Tuple, Union
```

**2. Added `self.audio_token` and `self.audio_token_id` to `__init__`** (~line 92)

Same pattern as existing `image_token`/`video_token` setup:
```python
self.audio_token = "<|audio_pad|>" if not hasattr(tokenizer, "audio_token") else tokenizer.audio_token
self.audio_token_id = (
    tokenizer.audio_token_id
    if getattr(tokenizer, "audio_token_id", None)
    else tokenizer.convert_tokens_to_ids(self.audio_token)
)
```

**3. Added `audios` parameter and audio processing to `__call__`** (~line 105, 160-176, 203-210)

- New parameter: `audios: Optional[List[Tuple[np.ndarray, int]]]` — list of `(audio_array, sample_rate)` tuples from `fetch_audio()`
- Audio feature extraction using `WhisperFeatureExtractor` (imported lazily inside the method):
  - Computes `num_audio_tokens = min(ceil(duration_seconds * 50), 1500)` per audio
  - Extracts mel spectrograms via `whisper_fe(audio_array, sampling_rate=sr, return_tensors="np")`
  - Stacks into `audio_features` (numpy array) and `audio_lengths` (list of token counts)
- Audio token expansion (same `<|placeholder|>` pattern as image/video):
  - Replaces each single `<|audio_pad|>` with `num_audio_tokens` copies

**4. Extended `BatchFeature` return** (~line 223-225)

Added `**audio_inputs` to the returned `BatchFeature`, so it now contains:
- `audio_features`: stacked mel spectrograms, shape `(num_audios, 128, 3000)` — 128 mel bins, 3000 time frames (Whisper's fixed output size)
- `audio_lengths`: list of token counts per audio (used by the model to know how many `<|audio_pad|>` tokens correspond to each audio)

### 4.2 Token count formula

`min(ceil(duration_seconds * 50), 1500)` where:
- 50 = tokens per second (Whisper produces 1500 tokens for 30 seconds of audio)
- 1500 = maximum tokens (Whisper's fixed output length)
- This dynamically sizes the audio representation in the token sequence based on actual audio duration

---

## 5. Bug Fix: WhisperFeatureExtractor mel bins (from code review)

**Problem**: `WhisperFeatureExtractor()` with no arguments defaults to `feature_size=80` (older Whisper models). But whisper-large-v3-turbo uses 128 mel bins. Our processor was producing `(num_audios, 80, 3000)` instead of `(num_audios, 128, 3000)`. This would crash when fed to the Whisper encoder in Session 3.

**Fix**: Changed `WhisperFeatureExtractor()` to `WhisperFeatureExtractor(feature_size=128)` in `processing_qwen2_vl.py` line 167.

**How it was caught**: External code review flagged the mismatch between documentation (128 mel bins) and the default constructor behavior (80 mel bins).

---

## 6. Notebook 02: Tokenizer & Processor Modifications

**File**: `notebooks/02_tokenizer_and_processor.ipynb`
**HuggingFace repo**: `DanJZY/Qwen2-VL-7B-Speech`

Built and tested on Google Colab. 11 sections covering:
1. Environment setup and HF login
2. Create HF repository
3. Load dataset (streaming)
4. `format_data()` function
5. Modify chat template (add audio branch)
6. Add 3 audio special tokens (IDs 151657-151659)
7. Save & push processor to HF
8. Load & test processor from HF
9. Test `fetch_audio` & `process_vision_info`
10. End-to-end processor test
11. Cleanup

### 6.1 Issues encountered during Colab testing

- **HF login on Colab IDE**: `google.colab.userdata` times out in IDE extension. Fixed by checking for cached token first (`HfFolder.get_token()`), falling back to interactive `login()`.
- **HF username mismatch**: GitHub username is `ZhuoyuanJiang`, HF username is `DanJZY`. `create_repo` failed with 403 until REPO_ID was corrected.
- **Chat template format**: The actual Jinja2 template format differs from HF documentation examples. The assert caught this. Fixed by using `{% elif 'text' in content %}` as the stable insertion point.
- **Pip caching git installs**: After fixing the mel bins bug in the fork and pushing to GitHub, restarting the Colab runtime and re-running `pip install` still used the old code. Pip caches packages installed from git URLs and won't re-download on restart. Fixed by adding `--force-reinstall --no-deps` to fork install lines.
- **Reproducibility**: Pinned fork install lines to exact commit hashes instead of branch names (`@e6f7d83ef...` and `@56b0756a7...`), so the notebook stays locked to the code it was tested against even if we push more commits to `speech-qwen2vl` in later sessions.

### 6.2 Final verified outputs (Colab)

- `audio_features` shape: `(1, 128, 3000)` — 128 mel bins confirmed after fix
- Token expansion: 856 tokens for 17.12s audio (`ceil(17.12 * 50) = 856`), assertion passed
- All 11 sections executed without errors

---

## 7. Session 2 Complete

All tasks from the plan are done:
1. Forked repos and created `speech-qwen2vl` branches
2. Modified `vision_process.py` (Qwen2-VL fork) — `fetch_audio`, `process_vision_info`
3. Modified `processing_qwen2_vl.py` (transformers fork) — audio token expansion, mel spectrogram extraction
4. Fixed `WhisperFeatureExtractor` mel bins bug (80 → 128)
5. Built and tested Notebook 02 on Google Colab — all cells pass
6. Committed and pushed all changes (forks + main repo)

**Next**: Session 3 — Model Architecture (add Whisper encoder + audio projector to Qwen2-VL)

---

## Commands Log

All commands executed in this session, in order:

```bash
# 1. Check gh CLI authentication
gh auth status

# 2. Fork repos on GitHub
gh repo fork huggingface/transformers --clone=false
gh repo fork QwenLM/Qwen2-VL --clone=false

# 3. Verify Qwen3-VL rename and check qwen-vl-utils compatibility
gh repo view QwenLM/Qwen2-VL --json name,url
gh repo view ZhuoyuanJiang/Qwen3-VL --json name,url,parent
gh api repos/ZhuoyuanJiang/Qwen3-VL/contents --jq '.[].name'

# 4. Check repo sizes before cloning
gh api repos/ZhuoyuanJiang/transformers --jq '.size'    # 448771 KB (~449 MB)
gh api repos/ZhuoyuanJiang/Qwen3-VL --jq '.size'        # 138076 KB (~138 MB)

# 5. Clone forks locally
mkdir -p forks/
git clone --filter=blob:none https://github.com/ZhuoyuanJiang/transformers.git forks/transformers
git clone https://github.com/ZhuoyuanJiang/Qwen3-VL.git forks/Qwen2-VL

# 6. Get Qwen2-VL HEAD commit for pinning
cd forks/Qwen2-VL && git log --oneline -1    # 9658872

# 7. Create speech-qwen2vl branches
cd forks/transformers && git checkout -b speech-qwen2vl 0f9c9088
cd forks/Qwen2-VL && git checkout -b speech-qwen2vl 9658872

# 8. Push branches to GitHub
cd forks/transformers && git push -u origin speech-qwen2vl
cd forks/Qwen2-VL && git push -u origin speech-qwen2vl

# 9. Verify branches
cd forks/transformers && git branch && git log --oneline -1
cd forks/Qwen2-VL && git branch && git log --oneline -1
```
