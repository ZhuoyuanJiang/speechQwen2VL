# Session 8 Progress — 2026-05-05

## Goal

Build a local Gradio demo backend for the trained model so the user can record
audio in their browser, send it to either Stage 1 or Stage 2, and verify the
transcription qualitatively. Also produce a conda env backup so the env can be
deleted later to free home-directory quota.

---

## What was built

### 1. Demo backend — `scripts/serve.py`

Single-file Gradio app (~280 lines). Key choices:

- **Stack:** Gradio 6.14.0 (chosen over vanilla HTML+FastAPI for built-in mic
  widget, examples panel, and loading-spinner UX — see
  `Documentation/Session8_Plan.md` for the alternative we considered).
- **Default port: 7870** (not Gradio's default 7860) so it doesn't clash with
  any other Gradio project the user is running on the same host.
- **`GRADIO_TEMP_DIR`** is set to a per-user path (`/tmp/gradio_<USER>`) at
  startup because `/tmp/gradio` on `vision-s006` is owned by another user
  and not writable for us.
- **Stage 1 / Stage 2 hot-swap** via a `ModelHolder` class — only one model in
  VRAM at a time (RTX 3090 has 24 GB, model is ~17 GB bf16 so Stage 1 + Stage 2
  together would not fit). Switching reloads in ~15-20 s.
- **Decoding defaults:** `num_beams=2`, `repetition_penalty=1.1`,
  `max_new_tokens=256` — the same configuration that produced 7.90% WER on the
  full LargeScaleASR test set in Session 7b. All three exposed as Gradio
  sliders so the user can experiment live.
- **Audio path:** Gradio yields `(sample_rate, np.ndarray)`; we cast int16 →
  float32 in `[-1, 1]`, resample to 16 kHz with `torchaudio.transforms.Resample`,
  collapse stereo to mono, re-encode to in-memory WAV bytes via
  `torchaudio.save(BytesIO(...), format="wav")`, then feed to the model with the
  same `process_vision_info` + chat-template message structure used in
  `notebooks/07_evaluation.py`.
- **Code reuse:** `run_inference()`, the Stage 1/Stage 2 model-loading
  patterns, and the resample/mono/encode pipeline are all adapted directly from
  `notebooks/07_evaluation.py` and `scripts/evaluate.py`.

### 2. Demo-example extractor — `scripts/extract_demo_samples.py`

One-shot helper that streams a few short clips from the HuggingFace
`speechbrain/LargeScaleASR` test split and writes them to
`scripts/demo_examples/*.wav` plus a `manifest.json` with reference transcripts.
Streams from HF rather than reading `./data` so it works even on servers where
the `./data` symlink is broken (which is the case on `vision-s006`).

Five samples extracted (all from `small/test`, 2.9–7.8 s each, ~1 MB total):

| File | Duration | Reference snippet |
|---|---|---|
| `01_*.wav` | 7.84 s | "WHILE WE CANNOT COMPROMISE ON OUR VALUES..." |
| `02_*.wav` | 2.92 s | "THIS REGULATION IS A CRUCIAL PART OF THAT" |
| `03_*.wav` | 7.48 s | "WE DEEPLY REGRET THAT THE COUNCIL IS PROPOSING..." |
| `04_*.wav` | 6.90 s | "A DAY WILL COME WHEN THE PEOPLES OF ASIA..." |
| `05_*.wav` | 7.10 s | "I LET EVERYBODY SPEAK A LITTLE BIT LONGER..." |

The Gradio `Examples` panel reads these on startup; if the directory is empty,
the UI prints an instruction to run the extractor.

### 3. Dependency pin

Added `gradio==6.14.0` to `requirements.txt` (under a new `# --- Demo backend
(Session 8) ---` section). Installed into the existing `speech_qwen2vl` env;
gradio brings in fastapi, uvicorn, starlette, python-multipart, etc. as
transitive deps.

---

## How to run

We are on **`vision-s006`** (8× RTX 3090, 24 GB) — distinct from the training
server `vllab15`.

From a remote machine:

```bash
ssh -L 7870:localhost:7870 vision-s006
```

On `vision-s006`:

```bash
cd ~/projects/speechQwen2VL
conda activate speech_qwen2vl
python scripts/serve.py --gpu 0
# Loaded stage2 in ~15-20s, then Gradio binds 127.0.0.1:7870
```

Open `http://localhost:7870` in the local browser.

CLI flags worth knowing:

```bash
python scripts/serve.py --gpu 0                 # default model: stage2
python scripts/serve.py --gpu 0 --default_model stage1
python scripts/serve.py --gpu 0 --port 7871     # if 7870 also taken
python scripts/serve.py --gpu 0 --share         # Gradio public share link
```

---

## What was tested

End-to-end smoke test on `vision-s006`, GPU 0:

1. `python scripts/serve.py --gpu 0` started cleanly. Stage 2 loaded in
   **15.3 s**, occupied **17.30 GB** VRAM. Gradio bound `127.0.0.1:7870`.
2. Called the `/transcribe` API via `gradio_client` with
   `02_20170314-0900-PLENARY-3-en_20170314-11_39_37_1.wav` (2.92 s clip):
   - Wall time: **10.32 s** end-to-end (first call includes CUDA kernel warmup;
     the info panel reports the model-only inference time at **8.46 s**).
   - Reference: `THIS REGULATION IS A CRUCIAL PART OF THAT`
   - Prediction: `BASE REGULATION IS A CRUCIAL PART OF THAT`
   - Single-word substitution at the start (1/7 word error). Within the
     model's typical noise floor — Session 7b reports a 4–8% WER on similar
     short clips with the decoding fixes enabled.

Browser-side mic recording is for the user to test interactively over the SSH
tunnel.

---

## Issues hit and resolved

1. **`PermissionError: /tmp/gradio/vibe_edit_history`** — Gradio 6.x writes
   under `/tmp/gradio`, but on `vision-s006` that path is owned by another
   user. Fixed by setting `GRADIO_TEMP_DIR=/tmp/gradio_<USER>` at startup.
2. **`TypeError: Textbox.__init__() got an unexpected keyword argument
   'show_copy_button'`** — Gradio 6.x dropped that kwarg. Removed.
3. **Default port collision risk** — the user has another Gradio project on
   the same host. Switched our default from 7860 (Gradio default) to 7870.

---

## Reproducibility: env backup + fork pinning (NOT deleted yet)

The user wants someone else to be able to clone this repo and 100%-reproduce
the working env. To get there cleanly, we made three changes (in addition to
the demo backend itself):

### 1. Pinned fork commit hashes — biggest single fix

`scripts/setup_forks.sh` previously did `git checkout speech-qwen2vl` then
`git pull`. Branch names are moving pointers — any future push to
`speech-qwen2vl` would have silently changed what got installed. **All other
deps were `==` pinned, but the forks (which contain our audio modifications)
were not.** Classic short-bucket reproducibility hole.

Fixed by adding `TRANSFORMERS_COMMIT` and `QWEN_VL_COMMIT` constants and
doing `git checkout --detach $COMMIT`:

| Fork | Pinned to | What's there |
|---|---|---|
| `ZhuoyuanJiang/transformers` | `934129b7701e7607facb39f286afc6bc4cc657df` | "Add guard for missing audio modules in forward pass" |
| `ZhuoyuanJiang/Qwen3-VL` | `56b0756a768cc3b01cba45b01c1bc3c8cb74ea3f` | "Add fetch_audio and extend process_vision_info for audio support" |

To bump a fork in the future: edit those constants. Nothing else needs to change.

### 2. Pip lockfile via `pip freeze` (not `conda-lock` / `uv pip compile`)

We tried both `conda-lock` and `uv pip compile --generate-hashes` to get a
hash-protected lockfile. Both **fail** on this stack because:

- `flash-attn==2.6.3` has **no prebuilt wheels** for our torch+CUDA combo —
  it must be built from source. Hash-based lockfiles physically cannot lock
  packages without wheels.
- `torch==2.4.1+cu121` and friends use a local-version qualifier (`+cu121`)
  that conda-lock's vendored Poetry solver does not recognize.

This is a known constraint for ML/CUDA projects, not a flaw in our setup.
The realistic professional standard for this kind of project is a curated
`requirements.txt` plus a `pip freeze` snapshot — which is what we now have:

| File | Generator | Pins |
|---|---|---|
| `requirements.txt` | hand-edited | top-level pip deps with `==` |
| `requirements.lock.txt` | `python scripts/freeze_lockfile.py` (filtered `pip freeze`) | every transitive pip dep at exact version |

`scripts/freeze_lockfile.py` runs `pip freeze`, drops editable git installs
(those go through `setup_forks.sh`), drops the one-time reproducibility tools
(pip-tools, conda-lock, uv) we tried so they don't pollute the lockfile, and
writes 207 packages with a long header explaining what the file is and isn't.

### 3. README documents two paths

Path A (exact reproduction): `environment.yml` → `requirements.lock.txt` →
`setup_forks.sh`. Locks every transitive pip version, locks fork commits.

Path B (development): `environment.yml` → `requirements.txt` (resolves
transitives fresh) → `setup_forks.sh`. Use this if you'll be editing deps.

Both paths use the same `environment.yml` for the conda layer and the same
`setup_forks.sh` for the forks. Only the pip side differs.

### Files touched for reproducibility

| File | Status |
|---|---|
| `scripts/setup_forks.sh` | modified — pinned fork commits |
| `requirements.lock.txt` | **new** — 207-package pip freeze snapshot with header |
| `scripts/freeze_lockfile.py` | **new** — regenerator for the above |
| `requirements.txt` | modified earlier (added `gradio==6.14.0`) |
| `environment.yml` | unchanged |
| `README.md` | modified — added "Path A vs Path B" setup section |

### What we tried and abandoned

- `conda env export` → wrote `environment.full.yml` and `environment.lock.yml`,
  then realized they're inferior to a real lockfile (no URLs/hashes, channel
  state assumptions) and redundant given conda-lock can't work on our stack.
  These two files were removed during cleanup.
- `conda-lock lock --file environment.yml -p linux-64` → fails on
  `torch==2.4.1+cu121` local-version qualifier and on `-r requirements.txt`.
- `uv pip compile --generate-hashes` → fails on `flash-attn` (no wheels) and
  on the existing editable git URLs.

The pivot to plain `pip freeze` is a deliberate engineering choice given
ML/CUDA constraints, documented in `requirements.lock.txt`'s header and in
the README's "Reproducibility caveats" subsection.

### To rebuild the env later

```bash
conda env create -f environment.yml
conda activate speech_qwen2vl
pip install -r requirements.lock.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121
bash scripts/setup_forks.sh
```

### Disk freed (planned, not yet done)

| Item | Location | Size | Affects home quota? |
|---|---|---|---|
| Conda env | `/home/zhuoyuan/miniconda3/envs/speech_qwen2vl` | **13 GB** | Yes |

`./data` symlink is already broken on `vision-s006` (no `/ssd1/zhuoyuan/speechQwen2VL/`),
HF cache lives at `/ssd1/zhuoyuan/hf_cache/hub` (local SSD, not in quota). Only
the conda env consumes home-quota space.

**The env has NOT been deleted.** Per the user's instruction, deletion waits
for explicit confirmation after they verify the demo backend works end-to-end
through the browser.

---

## Files added / changed

| File | Status |
|---|---|
| `scripts/serve.py` | **new** — Gradio demo backend |
| `scripts/extract_demo_samples.py` | **new** — one-shot HF-streaming demo-example extractor |
| `scripts/demo_examples/*.wav` (5) + `manifest.json` | **new** — preloaded test-set examples |
| `scripts/freeze_lockfile.py` | **new** — regenerator for `requirements.lock.txt` |
| `scripts/setup_forks.sh` | modified — pinned fork commit hashes |
| `requirements.txt` | modified — added `gradio==6.14.0` |
| `requirements.lock.txt` | **new** — full pip-freeze snapshot, 207 packages |
| `README.md` | modified — added Path A / Path B setup section |
| `Documentation/Session8_Plan.md` | **new** — copy of plan + considered alternative |
| `Documentation/Session8_Progress_20260505.md` | **new** — this file |

---

## Next steps

1. **(User)** Open `http://localhost:7870` after SSH tunnel and test:
   - mic record → transcribe
   - upload a `.wav` → transcribe
   - one of the preloaded examples → transcribe (verify against `manifest.json`)
   - flip the model radio Stage 2 → Stage 1, transcribe again, verify the swap
     reloads cleanly (~15–20 s)
2. **(User)** If the demo behaves as expected, give the explicit go-ahead and
   we'll commit Session 8 work.
3. **(After commit, with explicit user OK)** Delete the conda env to free
   ~13 GB of home quota:
   ```bash
   conda deactivate
   conda env remove -n speech_qwen2vl
   ```
