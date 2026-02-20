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

## 3. What's Next

According to the plan, the remaining tasks are:

1. **Modify `vision_process.py`** in the Qwen2-VL fork — add `fetch_audio()`, extend `extract_vision_info()` and `process_vision_info()` for audio
2. **Modify `processing_qwen2_vl.py`** in the transformers fork — add audio token expansion, `audios` parameter, WhisperFeatureExtractor processing
3. **Build Notebook 02** — 11 sections covering format_data, chat template, special tokens, processor push to HF, end-to-end testing
4. **Commit and push** — fork changes to fork repos, notebook + setup_forks.sh fix to main repo

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
