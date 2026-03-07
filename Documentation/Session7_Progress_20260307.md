# Session 7 Progress — 2026-03-07

## Goal

Quantitative evaluation of Stage 1 and Stage 2 models on the full test set using WER (Word Error Rate) and CER (Character Error Rate).

---

## Evaluation Results

### Full test set: 8,087 samples (all 6 test shards)

| Metric | Stage 1 (projector only) | Stage 2 (LoRA) | Relative Improvement |
|--------|--------------------------|----------------|----------------------|
| **WER** | 12.89% | 8.67% | **-32.7%** |
| **CER** | 7.43% | 4.75% | **-36.1%** |

- **Stage 2 LoRA reduced WER by ~33%** relative to Stage 1 (12.89% → 8.67%)
- **CER dropped by ~36%** (7.43% → 4.75%) — consistent improvement across both metrics
- 8.67% WER is solid for a fine-tuned 7B model on this dataset
- CLI and notebook runs for Stage 2 produced identical numbers (`corpus_wer: 0.08665821854637343`), confirming reproducibility

### Raw summary files

```json
// checkpoints/eval_stage2_lora_20260307_185549_cli_summary.json
{
  "model_name": "stage2_lora",
  "n_samples": 8087,
  "corpus_wer": 0.08665821854637343,
  "corpus_cer": 0.04747746936078782,
  "timestamp": "2026-03-07T18:55:49.043471"
}

// checkpoints/eval_stage1_projector_20260307_165831_cli_summary.json
{
  "model_name": "stage1_projector",
  "n_samples": 8087,
  "corpus_wer": 0.1288588289105674,
  "corpus_cer": 0.0743422363826482,
  "timestamp": "2026-03-07T16:58:31.207827"
}

// checkpoints/eval_stage2_lora_20260307_181030_notebook_summary.json
// (identical WER/CER to CLI — reproducibility confirmed)
{
  "model_name": "stage2_lora",
  "n_samples": 8087,
  "corpus_wer": 0.08665821854637343,
  "corpus_cer": 0.04747746936078782,
  "timestamp": "2026-03-07T18:10:30.031489"
}
```

### Runtime

| Model | Method | GPU | Time |
|-------|--------|-----|------|
| Stage 1 | CLI (`scripts/evaluate.py --model stage1 --gpu 1`) | GPU 1 | ~2h |
| Stage 2 | CLI (`scripts/evaluate.py --model stage2 --gpu 2`) | GPU 2 | ~2.5h |
| Stage 2 | Notebook (Option A, Sections 5A) | GPU 0 | ~2.5h |

All three ran simultaneously on separate GPUs. ~1s per sample inference time.

### GPU issue during evaluation

Initially the notebook and CLI Stage 2 both landed on GPU 0 (two processes competing on one card, slowing both down). Fixed by killing the CLI process (`kill <PID>`) and restarting it on GPU 2:

```bash
kill 2716158
python scripts/evaluate.py --model stage2 --gpu 2
```

After that, all three processes ran on separate GPUs at full speed.

---

## Step-by-step Reproduction

### Prerequisites

- Stage 1 model pushed to `DanJZY/Qwen2-VL-7B-Speech`
- Stage 2 LoRA adapters pushed to `DanJZY/Qwen2-VL-7B-Speech-LoRA`
- Conda env `speech_qwen2vl` set up (see `Documentation/Session5_Progress_20260306.md`)
- Forked libraries installed via `bash scripts/setup_forks.sh`
- `jiwer` installed (`pip install jiwer>=3.0.0` — already in environment)
- `./data` and `./checkpoints` directories exist (symlinks to local SSD on our server)

### Option A: Run evaluation in the notebook (single GPU, ~4.5h)

1. Open the notebook:
   ```bash
   conda activate speech_qwen2vl
   jupyter lab notebooks/07_evaluation.ipynb
   ```

2. Set `N_SAMPLES` at the top of Section 1:
   ```python
   N_SAMPLES = None       # None = full test set (~8,087), or 100/1000 for quicker runs
   ```

3. Run Sections 1-4 (setup, dataset, model load, helpers)

4. Run Section 5A (Stage 2 evaluation) — takes ~2.25h for full test set

5. Run Section 6A (frees Stage 2, loads Stage 1, runs Stage 1 evaluation) — takes ~2.25h

6. Run Sections 7-10 for analysis (comparison table, duration breakdown, cleanup)

Results are saved automatically as `_notebook.jsonl` and `_notebook_summary.json`.

### Option B: Run evaluation via CLI (multi-GPU, ~2.25h)

1. Run both models in parallel on separate GPUs:
   ```bash
   # Terminal 1
   python scripts/evaluate.py --model stage2 --gpu 0

   # Terminal 2
   python scripts/evaluate.py --model stage1 --gpu 1
   ```

2. After both finish, open the notebook and run Sections 1-4

3. **Skip Sections 5A and 6A** entirely

4. **Uncomment Section 5B** to load CLI results:
   ```python
   import glob

   s2_files = sorted(glob.glob("./checkpoints/eval_stage2_lora_*_cli.jsonl"))
   s1_files = sorted(glob.glob("./checkpoints/eval_stage1_projector_*_cli.jsonl"))
   # ... (auto-loads the latest CLI results)
   ```

5. Run Sections 7-10 for analysis

### Option A+B: Run both simultaneously

You can also run all three at once if you have enough GPUs:
- Notebook (Option A) on one GPU — gives you inline outputs and plots
- CLI Stage 1 + Stage 2 on two other GPUs — gives you `_cli.jsonl` backups

The notebook uses its own in-memory results for Sections 7-10, so there's no conflict.

### Quick sanity check

For a fast validation before committing to a full run:
```bash
python scripts/evaluate.py --model both --n_samples 100 --gpu 0
```
This takes ~3 minutes and prints a comparison table.

---

## What Was Built

### Notebook: `notebooks/07_evaluation.py` (10 sections)

| Section | What it does |
|---------|-------------|
| 1. Environment Setup | Same boilerplate as 05/06: `os.chdir`, `HF_DATASETS_CACHE`, `get_free_gpu()`. New imports: `jiwer`, `tqdm`, `json`, `datetime` |
| 2. Load Full Test Dataset | All 6 test shards (`data_files=["test/test-*"]`), deterministic shuffle (`seed=42`), optional `N_SAMPLES` subset, print duration stats |
| 3. Load Stage 2 Model | Base model from `DanJZY/Qwen2-VL-7B-Speech` + LoRA from `DanJZY/Qwen2-VL-7B-Speech-LoRA`. Sanity check: single-sample inference on sample 0 |
| 4. Helper Functions | `run_inference()`, `normalize_text()`, `evaluate_model()` — see details below |
| 5A. Evaluate Stage 2 (Option A) | Run `evaluate_model()` in notebook, save `_notebook.jsonl` + `_notebook_summary.json` |
| 6A. Evaluate Stage 1 (Option A) | Free Stage 2 model (`del` + `gc.collect`), load Stage 1 (no LoRA), run same evaluation |
| 5B. Load CLI Results (Option B) | Commented out by default. Uncomment to load `_cli.jsonl` files via `glob`. Includes sample-count mismatch validation |
| 7. Comparison Table + Analysis | Side-by-side WER/CER with improvement %, top 10 worst predictions (highest WER), WER distribution histogram (matplotlib) |
| 8. Duration Breakdown | WER/CER per duration bucket (0-5s, 5-10s, 10-15s, 15-20s, 20-30s, 30s+) using pandas |
| 9. Custom Audio Inference | `transcribe_audio_file(model, processor, audio_path)` — loads `.wav` via torchaudio, resamples to 16kHz, converts stereo→mono |
| 10. Cleanup | Guarded `del` (checks if variables exist before deleting), `gc.collect()`, `torch.cuda.empty_cache()` |

### Helper functions (Section 4)

**`run_inference(model, processor, messages, max_new_tokens=256)`**
- Reused from Notebooks 05/06
- Greedy decoding (`num_beams=1, do_sample=False`), single sample
- Returns decoded text string

**`normalize_text(text)`**
- ASR-standard normalization applied before WER/CER computation
- Steps: uppercase → strip → remove punctuation → collapse whitespace
- ~15% of test transcripts contain punctuation/apostrophes — removing them prevents punctuation differences from inflating metrics
```python
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

def normalize_text(text):
    text = text.upper().strip()
    text = _PUNCT_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

**`evaluate_model(model, processor, dataset, model_name, max_new_tokens=256)`**
- Main evaluation loop with tqdm progress bar
- Saves per-sample: index, ID, duration, speaker, sex, raw reference, raw prediction, normalized reference, normalized prediction, per-sample WER, per-sample CER
- Computes corpus-level WER/CER via `jiwer.wer(references, predictions)` (total errors / total reference words)
- Returns `(summary_dict, results_list)`

### CLI Script: `scripts/evaluate.py`

Standalone single-GPU evaluation script. Follows `scripts/train_stage2.py` structure.

**CLI arguments** (all optional, showing defaults):
```bash
python scripts/evaluate.py \
    --model stage2 \           # {stage1, stage2, both}
    --n_samples None \         # None = full test set
    --output_dir ./checkpoints \
    --data_dir ./data \
    --max_new_tokens 256 \
    --gpu None                 # None = auto-select idle GPU
```

**Structure**:
1. GPU selection (auto or `--gpu`)
2. Dataset loading (all 6 test shards, optional subset)
3. Model loading (based on `--model` flag)
4. Evaluation loop (same `evaluate_model()` function as notebook)
5. Save JSONL + summary JSON with `_cli` suffix
6. Print comparison table (if `--model both`)

No wandb — evaluation is a one-shot operation.

### File naming convention

Output files include a source suffix (`_notebook` or `_cli`) to distinguish results:

| Source | JSONL | Summary |
|--------|-------|---------|
| Notebook | `eval_{model}_{timestamp}_notebook.jsonl` | `eval_{model}_{timestamp}_notebook_summary.json` |
| CLI | `eval_{model}_{timestamp}_cli.jsonl` | `eval_{model}_{timestamp}_cli_summary.json` |

Each run gets a unique timestamp (`YYYYMMDD_HHMMSS`), so no files are ever overwritten.

---

## Key Design Decisions

1. **Single-sample inference loop** (not batched): Variable-length audio makes batching impractical with `model.generate()`. Padding to max length would waste compute; dynamic batching adds complexity for minimal gain. Each sample takes ~1s, so full test set takes ~2.25h per model.

2. **Corpus-level WER as primary metric**: `jiwer.wer(reference_list, hypothesis_list)` computes corpus-level WER (total errors / total reference words), which is the standard ASR metric. Per-sample WERs are saved for analysis but the headline number is corpus-level.

3. **ASR-standard text normalization**: Uppercase + strip whitespace + remove punctuation. The training data (`small/` split) is mostly clean ALL CAPS, but the full test set has ~15% of transcripts containing punctuation or apostrophes. Removing punctuation before scoring is standard ASR practice.

4. **Save both raw and normalized text** in JSONL: Inference is the expensive part (~2.5h). By saving raw model outputs alongside normalized versions, any later change to normalization or qualitative inspection can be done without re-running inference.

5. **Deterministic shuffle** (`dataset.shuffle(seed=42)`) before subsetting: So smaller `N_SAMPLES` runs produce a representative cross-section, not just the first contiguous slice from shard 0.

6. **Evaluate Stage 2 first, then Stage 1**: If the user only wants Stage 2 results, they can stop after Section 5A without needing to run the comparison.

7. **Single GPU only**: No DDP needed — inference has no gradients, model fits in ~17 GB VRAM. Both notebook and CLI use `get_free_gpu()` for auto-selection.

---

## Dataset Details

### Test set

| What | Shards | Samples |
|------|--------|---------|
| `test/` split total | 6 shards | **8,087** |
| What we evaluate | All 6 shards | **8,087** |

Duration distribution across the 8,087 test samples:
- Min: ~1s
- Max: ~39.7s
- Mean: ~10s (estimated)

Previous notebooks (05/06) only used 100 test samples (first 100 from 1 shard) for quick spot-checks during training. This notebook evaluates on the **full test set** for proper metrics.

---

## Files Created

| File | Description |
|------|-------------|
| `notebooks/07_evaluation.py` | Evaluation notebook (10 sections, jupytext percent-format) |
| `notebooks/07_evaluation.ipynb` | Generated from .py via jupytext |
| `scripts/evaluate.py` | Standalone CLI evaluation script (single GPU) |
| `Documentation/Session7_Plan.md` | Evaluation plan |
| `Documentation/Session7_Progress_20260307.md` | This file |
| `Documentation/Lessons/session7_QA.md` | Error analysis Q&A: failure categories, repetition degeneration, decoding fixes |
| `scripts/extract_worst_samples.py` | Extracts audio .wav files for the 10 worst-performing samples |

---

## Error Analysis

After inspecting the 10 worst predictions and listening to extracted audio samples, we identified 4 failure categories:

| Category | Samples | Cause | Model at fault? |
|----------|---------|-------|-----------------|
| Repetition loops | 414, 7512 | Model fails to emit `<\|im_end\|>`, greedy decoding loops | Yes — decoding pathology |
| Incomplete references | 8025, 6376, 540, 4216, 2352 | Audio contains more speech than the reference captures | No — model is more correct |
| Non-English audio | 4317, 5216 | French/German audio mislabeled as English | No — dataset quality issue |
| Word boundary differences | 3228 | "GREENHORNS" → "GREEN HORNS" | No — WER metric artifact |

**Key finding**: Most "worst" predictions are actually dataset quality issues, not model failures. Only 2 out of 10 are genuine model errors (repetition loops), and those are a decoding pathology, not a model quality issue.

### WER Distribution

| Metric | Stage 1 | Stage 2 |
|--------|---------|---------|
| Exact-match rate (WER=0%) | 35.1% | 45.6% |
| Median WER | 7.7% | 4.0% |
| Samples with WER > 100% | 72 | 34 |
| Samples hitting 256-token cap | 14 | 3 |

### Repetition penalty test

Tested `repetition_penalty=1.2` on the 10 worst samples:

| Sample | Old WER | New WER | Notes |
|--------|---------|---------|-------|
| 414 | 713% | 40% | "AH" loop eliminated |
| 7512 | 513% | 97% | Loop broken, still wrong content |
| 4216 | 175% | 150% | Slight improvement |
| Others | — | — | No change (not repetition issues) |

See `Documentation/Lessons/session7_QA.md` for full analysis including root cause (exposure bias), decoding mitigations, and training recommendations.

---

## Review Fixes Applied

Five issues were identified during code review and fixed before running evaluation:

1. **Save raw model outputs** (Medium) — Only normalized text was saved in JSONL, defeating the "re-analyze without rerunning" goal. Fixed: added `reference_raw` and `prediction_raw` fields before the normalized versions. Applied to both `07_evaluation.py` and `scripts/evaluate.py`.

2. **Guard cleanup for Option B** (Low) — Section 10 did `del model_s1` unconditionally, but `model_s1` only exists in the Option A path (Section 6A). Option B users would hit `NameError`. Fixed: Section 10 now checks if each variable exists before deleting.

3. **Guard division by zero** (Low) — Improvement percentage calculation (`(s1 - s2) / s1 * 100`) crashes if Stage 1 WER/CER is 0.0 (possible with tiny `N_SAMPLES` debug runs). Fixed: defaults to 0.0 when denominator is 0. Applied to both notebook and CLI script.

4. **Remove `matplotlib.use("Agg")`** (Low) — The `Agg` backend is correct for scripts (save-only) but prevents inline plot rendering in Jupyter notebooks. Fixed: removed the line. Plots now render inline via Jupyter's default backend.

5. **Validate sample count in Option B** (Low) — The Option B loader independently picks the latest Stage 1 and Stage 2 JSONL files. If one is from `N_SAMPLES=100` and the other from the full set, the comparison is misleading. Fixed: prints a warning if `n_samples` differs between the two loaded summaries.

---

## Next Steps

1. ~~Commit evaluation notebook, CLI script, and documentation~~ ✅ Done
2. ~~Analyze worst predictions and error patterns~~ ✅ Done — see Error Analysis section above
3. (Optional) Re-evaluate full test set with `repetition_penalty=1.1` + `num_beams=2` to get updated corpus WER/CER
4. (Optional) Analyze duration breakdown for per-bucket WER/CER patterns
5. (Optional) Run custom audio inference on external `.wav` files
6. (Optional, next training run) Use `load_best_model_at_end=True`, early stopping, and 1.8-2.0 epochs to preserve best checkpoint
