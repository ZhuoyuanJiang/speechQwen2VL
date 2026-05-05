# Plan: Local Demo Backend + Conda Env Backup/Cleanup

## Context

Training is finished (Stage 2 + decoding fixes → **7.90% WER** on the full LargeScaleASR test set, 8,087 samples). Models live at `DanJZY/Qwen2-VL-7B-Speech` (Stage 1) and `DanJZY/Qwen2-VL-7B-Speech-LoRA` (Stage 2). Training was done on **vllab15** (8× RTX 6000 Ada, confirmed by `Documentation/Session5_Server_Plan.md`, `Session5/6/7_Progress_*.md`, and the MEMORY note).

The user wants:
1. A **local backend** so they can record audio in their browser, send it to the model, and see the transcription — to qualitatively sanity-check the model on their own voice. **No deployment**, just a dev server they hit themselves.
2. After the backend is committed and tested, **back up the conda environment** (compare against any existing backup), then **delete the env** (~13 GB) to free home-directory quota for other repos.

This plan covers (1) in detail and gives a brief follow-up outline for (2).

---

## Part 1: Demo Backend

### Stack

**Gradio** (single-file Python). Decided after weighing Gradio vs vanilla HTML+MediaRecorder:
- Built-in audio recording widget handles browser/mic permissions and sample-rate quirks (no JS to write).
- Built-in `Examples` list lets us preload a few LargeScaleASR test samples for one-click "see this exact case" demos — fits the "show the process, look solid" goal.
- Built-in loading spinner during model swap is free.
- Gradio's only real downside (heavy dependency, ~200 MB + ~20 transitive packages) doesn't matter here because the env will be deleted afterwards.

### Where it runs

The model is ~17 GB in bf16 and the user wants to inference on a **24 GB RTX 3090** (whichever server has one) — fits one model at a time, **cannot hold both Stage 1 and Stage 2 simultaneously**. Access via SSH tunnel from a local browser:

```bash
ssh -L 7860:localhost:7860 <server>
# on the server:
conda activate speech_qwen2vl
python scripts/serve.py --gpu 0 --port 7860
# open http://localhost:7860 in local browser
```

(7860 is Gradio's default port. If the user lands on vllab15's RTX 6000 Ada instead of a 3090, everything still works — we just won't take advantage of the extra VRAM since switching-with-reload is the design.)

### Files to create

| File | Purpose |
|---|---|
| `scripts/serve.py` | Gradio app: loads a model, exposes `transcribe(audio, model_choice)` callback, supports model swap |
| `scripts/demo_examples/` | A few short `.wav` files extracted from the test set for the Gradio Examples panel |
| `scripts/extract_demo_samples.py` | Tiny helper that pulls 3-5 samples from `data/test/` and writes them as `.wav` (one-shot, run once) |
| `Documentation/Session8_Plan.md` | Copy of this plan |
| `Documentation/Session8_Progress_<DATE>.md` | Written after the backend works — how to run, what was tested |

New Python dependencies: `gradio` (one line in `environment.yml` under the pip section). No fastapi/uvicorn needed — Gradio brings its own server.

### Backend design (`scripts/serve.py`)

Single-file Gradio app, ~200 lines:

1. **Argparse**: `--gpu`, `--port` (default 7860), `--host` (default `127.0.0.1`), `--default_model {stage1,stage2}` (default `stage2`), `--repetition_penalty 1.1`, `--num_beams 2`, `--max_new_tokens 256`, `--share` (default false; if true, Gradio's tunneled share link).
2. **`ModelHolder` class** (encapsulates the swap):
   - Holds `model`, `processor`, `current_kind` ("stage1" | "stage2"), `device`.
   - `load(kind)`: if `current_kind == kind`, no-op. Otherwise: `del self.model; gc.collect(); torch.cuda.empty_cache()`, then load the requested kind (Stage 1 = base only; Stage 2 = base + LoRA). Pattern copied verbatim from `notebooks/07_evaluation.py:153-166` (Stage 2) and the analogous Stage 1 block in `scripts/evaluate.py:270-275`. Set `base_model.config.use_cache = True`.
   - Initial load is the `--default_model`.
3. **`transcribe(audio_input, model_choice)` callback** — Gradio passes `audio_input` as a tuple `(sample_rate, np.ndarray)` for both mic recordings and file uploads:
   - If `model_choice != holder.current_kind`: `gr.Info("Loading {model_choice}, please wait ~30-60s...")` then `holder.load(model_choice)`.
   - Resample to 16 kHz mono using `torchaudio` resampler — same logic as `transcribe_audio_file()` at `notebooks/07_evaluation.py:548-572`, just adapted for ndarray input instead of file path.
   - Encode to in-memory WAV bytes via `torchaudio.save(BytesIO(...), format="wav")`.
   - Build messages and call a slimmed-down `run_inference()` mirroring `notebooks/07_evaluation.py:191-214`, but with `repetition_penalty` and `num_beams` from CLI args.
   - Return: `(transcription_text, info_markdown)` where `info_markdown` shows model used, decoding params, audio duration, inference latency, and a note like "Loaded Stage 2 LoRA in {N}s" if a swap happened.
4. **`gr.Blocks` UI**:
   - Title + short description (model summary, link to HF repos, headline 7.90% WER on LargeScaleASR test).
   - `gr.Audio(sources=["microphone", "upload"], type="numpy")` — single component handles both record and upload.
   - `gr.Radio(["stage2", "stage1"], value="stage2", label="Model")` — choose which to run.
   - "Transcribe" button → calls `transcribe`.
   - `gr.Textbox` for the transcription, `gr.Markdown` for the info panel.
   - `gr.Examples([...])` populated from `scripts/demo_examples/*.wav` so the user (and any viewer) can replay deterministic samples.
5. **Concurrency**: `demo.queue(concurrency_count=1)` — model isn't thread-safe; one request at a time is fine for a personal demo.
6. **Health check**: a tiny `gr.Markdown` at the bottom showing current model, GPU, GPU memory used (refreshed on each transcribe call).

### Code reuse references (from existing files)

| What | Source location | How to reuse |
|---|---|---|
| Model loading (Stage 2 base + LoRA) | `notebooks/07_evaluation.py:153-166` | `ModelHolder.load("stage2")` body |
| Model loading (Stage 1 base only) | `scripts/evaluate.py:270-275` | `ModelHolder.load("stage1")` body |
| `run_inference()` | `notebooks/07_evaluation.py:191-214` | Copy, parameterize `num_beams`/`repetition_penalty`/`do_sample` |
| `transcribe_audio_file()` (resample/mono/encode) | `notebooks/07_evaluation.py:548-572` | Adapt to take a `(sr, np.ndarray)` from Gradio instead of a file path |
| GPU auto-pick `find_idle_gpus()` | `scripts/evaluate.py:59-78` | Use if `--gpu` not provided |
| `process_vision_info` import + usage | `notebooks/07_evaluation.py` Section 1 + 4 | Same import |

### Verification

1. `python scripts/serve.py --gpu 0` starts cleanly, prints `Loaded stage2 in N.Ns on cuda:0`, then prints the Gradio URL.
2. SSH tunnel from local: `ssh -L 7860:localhost:7860 <server>`, open `http://localhost:7860` in browser. Page loads, mic and upload widgets visible.
3. **Mic path**: click record, say a short phrase, click stop, click Transcribe → text appears in ~1-2 s for short clips. Info panel shows model + timing.
4. **Upload path**: drag-drop a `.wav` file → Transcribe → text appears.
5. **Examples path**: click one of the preloaded `demo_examples/*.wav` → loads into the audio component → Transcribe → output matches the dataset's reference transcript closely.
6. **Model swap**: switch the Radio from `stage2` to `stage1`, click Transcribe → "Loading stage1..." toast appears, after ~30-60 s the result for Stage 1 appears. Switch back, get Stage 2 again. GPU memory stays near 17 GB (no leak).
7. **Reproducibility check**: pick one sample from `checkpoints/eval_stage2_lora_*_cli.jsonl`, find the matching `.wav` in `demo_examples/`, run through Gradio, verify output matches the `prediction_raw` field within decoding noise.

### Commit

After the backend works end-to-end:
```
Session 8: Add local Gradio demo for live audio transcription
```
Files: `scripts/serve.py`, `scripts/extract_demo_samples.py`, `scripts/demo_examples/*.wav` (small files, OK in git), `environment.yml` (add `gradio` to pip section), `Documentation/Session8_Plan.md`, `Documentation/Session8_Progress_*.md`.

---

## Part 2: Conda Env Backup + Cleanup (after backend is done)

### Current state of "backups"

I searched the repo: **the only environment artifacts that exist are `environment.yml` (25 lines) and `requirements.txt` (59 lines)**, both pinned in commit `87f830f Update environment`. There is **no separate `environment_backup.yml`** anywhere in `/home/zhuoyuan/projects/speechQwen2VL/`. The "backup" the user is remembering is most likely commit `87f830f` itself, which pinned ffmpeg/libsndfile/librosa/soundfile/jiwer.

### What "back up" should mean here

`environment.yml` is the human-curated short list. For full reproducibility (every transitive dependency at exact versions), generate a complete export and a pip freeze, store them in the repo (or alongside it), and commit. Steps:

```bash
conda activate speech_qwen2vl
conda env export --no-builds > environment.full.yml          # all conda deps, no build hashes
conda env export > environment.lock.yml                       # exact reproduce, build-hash pinned
pip freeze > requirements.lock.txt                            # all pip deps incl. transitive
```

Then compare `environment.full.yml` against the curated `environment.yml` so we know what's missing from the curated file (anything important like fastapi, jiwer, peft versions). Commit the lockfiles.

### Deletion (after backup committed and pushed)

```bash
conda env list                                # confirm path
conda deactivate
conda env remove -n speech_qwen2vl            # frees ~13 GB at /home/zhuoyuan/miniconda3/envs/speech_qwen2vl
```

The HF cache at `/ssd1/zhuoyuan/hf_cache/hub` and the project data/checkpoints at `/ssd1/zhuoyuan/speechQwen2VL/` are on local SSD, not in the home quota — they don't need to be touched. To rebuild later: `conda env create -f environment.lock.yml && bash scripts/setup_forks.sh`.

### Disk that will actually be freed

| Item | Location | Size | Affects home quota? |
|---|---|---|---|
| Conda env | `/home/zhuoyuan/miniconda3/envs/speech_qwen2vl` | **13 GB** | Yes — frees quota |
| HF cache | `/ssd1/zhuoyuan/hf_cache/hub` | (large) | No (on SSD) |
| Data/checkpoints | `/ssd1/zhuoyuan/speechQwen2VL/` | (large) | No (on SSD) |

So deleting the env recovers ~13 GB of the home-directory NAS quota.

---

## Out of scope (deliberately)

- Public deployment (Cloudflare tunnel, Hugging Face Spaces, etc.) — user said no deploy.
- Authentication — local-only, behind SSH tunnel.
- Batched / streaming inference — single user, single request is enough for a demo.
- Repo-level packaging/archiving for cold storage — user said "let's talk about packaging later".

---

## Considered Alternative: Vanilla HTML + MediaRecorder + FastAPI

Recorded for completeness — this is the path we considered and decided against. Keeping it documented in case we want to swap later, or if the env deletion eventually makes Gradio's heavy dependency footprint matter again.

### Stack

- **FastAPI + uvicorn** as the Python backend (lightweight, ~10 MB of deps).
- A single `scripts/static/index.html` file (~150 lines, no build step) using the browser's native `MediaRecorder` API for mic capture.

### File layout

| File | Purpose |
|---|---|
| `scripts/serve.py` | FastAPI app: loads model on startup, exposes `POST /transcribe` and `GET /` (serves the HTML page) |
| `scripts/static/index.html` | Self-contained HTML/CSS/JS UI: record button, file upload, result area |

### Backend sketch

```python
@app.post("/transcribe")
async def transcribe(audio: UploadFile, model_choice: str = Form("stage2")):
    raw = await audio.read()
    wav_bytes = decode_to_16k_mono(raw)            # torchaudio (ffmpeg backend) or librosa fallback
    if model_choice != holder.current_kind:
        holder.load(model_choice)                  # same ModelHolder pattern as the Gradio version
    text = run_inference(holder.model, holder.processor, build_messages(wav_bytes), ...)
    return {"transcription": text, "model": model_choice, ...}
```

`GET /` serves the static HTML via `FileResponse`. CORS is locked to `http://localhost:*` since this is local-only behind an SSH tunnel.

### Frontend sketch

```javascript
const stream = await navigator.mediaDevices.getUserMedia({audio: true});
const recorder = new MediaRecorder(stream);   // browser default: WebM/Opus
recorder.ondataavailable = e => chunks.push(e.data);
recorder.onstop = async () => {
    const blob = new Blob(chunks, {type: "audio/webm"});
    const fd = new FormData();
    fd.append("audio", blob, "recording.webm");
    fd.append("model_choice", document.querySelector("#model").value);
    const res = await fetch("/transcribe", {method: "POST", body: fd}).then(r => r.json());
    document.querySelector("#out").textContent = res.transcription;
};
```

Plus a `<input type="file" accept="audio/*">` for the upload path and a `<select>` for model choice.

### Pros vs the Gradio choice

- **Lighter dependencies**: only fastapi + uvicorn + python-multipart (~10 MB total) instead of Gradio's ~200 MB and ~20 transitive packages. Matters more if the env stays around.
- **Custom UI**: full control over look-and-feel — can match a personal brand or be made very minimal/clean. No "Gradio template" feel.
- **No layer of magic**: the request/response shape is right there in plain code, easier to debug or wire into a future iOS/CLI client.
- **One less library**: avoids Gradio's queueing layer, `gr.Info` toast quirks, and any future API changes.

### Cons vs the Gradio choice

- **More frontend code to write and maintain** (~150 lines of HTML/CSS/JS).
- **Manual handling of browser quirks**: WebM/Opus decoding on the server, mic permissions, sample-rate variability across browsers.
- **No free Examples panel** — would have to hand-build a list of preloaded `.wav` files with click-to-load behavior.
- **No free loading spinner** during model swap — would need to wire up a JS progress indicator.

### When to switch to this path later

If we ever want to ship the demo more broadly (HF Space, public link, embedded in a portfolio site), the HTML+FastAPI path is more portable and lighter. Switching is straightforward: `ModelHolder`, `run_inference()`, and the audio decoding helper carry over verbatim — only the UI surface and request marshaling change.
