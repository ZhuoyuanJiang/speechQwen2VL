# Plan: Notebook 07 — Evaluation

> Save this plan as `Documentation/Session7_Plan.md` during implementation.

## Context

Stages 1 and 2 are complete. Stage 1 trained the audio projector (model at `DanJZY/Qwen2-VL-7B-Speech`). Stage 2 LoRA fine-tuned the LLM decoder layers (adapters at `DanJZY/Qwen2-VL-7B-Speech-LoRA`). Both models produce good transcriptions on spot-checks (3 test samples, all exact matches). But we have no quantitative metrics yet — Notebook 07 adds proper WER/CER evaluation on the full test set.

---

## What Notebook 07 Does

- Load trained models (Stage 1 and Stage 2) from HuggingFace
- Run inference on the **full test set** (~8,087 samples, all 6 shards)
- Compute **WER** (Word Error Rate) and **CER** (Character Error Rate) using `jiwer`
- Compare Stage 1 vs Stage 2 quantitatively
- Analyze errors by duration bucket and show worst predictions
- Support custom `.wav` file inference
- Save predictions to JSONL for later analysis without re-running inference

---

## Files to Create

| File | Description |
|------|-------------|
| `notebooks/07_evaluation.py` | Evaluation notebook (10 sections, jupytext percent-format) |
| `notebooks/07_evaluation.ipynb` | Generated from .py via jupytext |
| `scripts/evaluate.py` | Standalone CLI evaluation script (single GPU) |

---

## Key Design Decisions

1. **Single-sample inference loop** (not batched): Variable-length audio makes batching impractical with `model.generate()`. Each sample takes ~1s, so full test set (~8,087 samples) takes ~2.25 hours per model (~4.5 hours for `--model both`). Use `N_SAMPLES` for quicker runs.

2. **Corpus-level WER as primary metric**: `jiwer.wer(reference_list, hypothesis_list)` computes corpus-level WER (total errors / total reference words), which is the standard ASR metric. Per-sample WERs are saved for analysis but not the headline number.

3. **Save predictions to JSONL**: Inference is the expensive part. Save all predictions + per-sample metrics to JSONL so we can re-analyze without re-running inference.

4. **Single GPU only**: No DDP needed — inference has no gradients, model fits in ~17 GB. Both notebook and script use `get_free_gpu()`.

5. **ASR-standard text normalization**: Uppercase + strip whitespace + **remove punctuation**. The training data (`small/` split) is mostly clean ALL CAPS, but the full test set has ~15% of transcripts containing punctuation or apostrophes. Removing punctuation before scoring is standard ASR practice — punctuation differences shouldn't inflate WER/CER.

6. **Evaluate Stage 2 first, then Stage 1**: If user only wants Stage 2 results, they can skip the comparison section.

---

## Notebook Structure (10 Sections)

### Section 1: Environment Setup
Same boilerplate as Notebooks 05/06: `os.chdir`, `HF_DATASETS_CACHE`, `get_free_gpu()`, imports.

New imports: `from jiwer import wer, cer`, `from tqdm import tqdm`, `json`, `datetime`.

Configuration at top:
```python
N_SAMPLES = None  # None = full test set (~8,087), or 100/1000 for quicker runs
SAVE_PREDICTIONS = True
```

### Section 2: Load Full Test Dataset
Load **all 6 test shards** (not just 1 shard like training notebooks):
```python
test_dataset = load_dataset(
    "speechbrain/LargeScaleASR",
    data_files=["test/test-*"],  # all 6 shards
    num_proc=12,
)
test_dataset = test_dataset["train"]
test_dataset = test_dataset.cast_column("wav", Audio(decode=False))
```
Optionally subset with `N_SAMPLES`. Use a **deterministic shuffle** (`dataset.shuffle(seed=42)`) before subsetting so smaller samples are representative, not just the first contiguous slice. Print stats: total samples, duration distribution (min/max/mean).

### Section 3: Load Stage 2 Model (LoRA)
```python
STAGE1_REPO = "DanJZY/Qwen2-VL-7B-Speech"
LORA_REPO = "DanJZY/Qwen2-VL-7B-Speech-LoRA"

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    STAGE1_REPO, torch_dtype=torch.bfloat16, device_map=DEVICE,
)
model = PeftModel.from_pretrained(base_model, LORA_REPO)
processor = Qwen2VLProcessor.from_pretrained(STAGE1_REPO)
```
Sanity check: single-sample inference on sample 0 to verify model works.

### Section 4: Helper Functions
- **`run_inference()`** — reused from Notebooks 05/06 (greedy decoding, single sample)
- **`normalize_text(text)`** — uppercase, strip whitespace, remove punctuation (ASR standard)
- **`evaluate_model(model, processor, dataset, model_name)`** — main loop:
  - Iterates with tqdm progress bar
  - Computes per-sample WER/CER
  - Returns corpus-level summary dict + per-sample results list

### Section 5: Evaluate Stage 2 Model
Run `evaluate_model()`, print corpus WER/CER, save predictions to JSONL:
```
./checkpoints/eval_stage2_lora_{timestamp}.jsonl      (per-sample results)
./checkpoints/eval_stage2_lora_{timestamp}_summary.json  (aggregate metrics)
```

### Section 6: Evaluate Stage 1 Model (Comparison)
Free Stage 2 model, load Stage 1 base model (no LoRA), run same evaluation. Save results.

### Section 7: Comparison Table + Analysis
- Side-by-side WER/CER comparison with improvement percentage
- Histogram of per-sample WER distribution (matplotlib)
- Top 10 worst predictions from Stage 2 for qualitative inspection

### Section 8: Breakdown by Duration
Bin samples by duration (0-5s, 5-10s, 10-15s, 15-20s, 20-30s, 30s+), compute WER/CER per bucket. Reveals whether model struggles with short or long audio.

### Section 9: Custom Audio Inference
```python
def transcribe_audio_file(model, processor, audio_path, target_sr=16000):
    """Load a .wav file, resample to 16kHz, and transcribe."""
```
Uses `torchaudio` for loading and resampling. Example usage cell commented out with placeholder path.

### Section 10: Cleanup
`gc.collect()`, `torch.cuda.empty_cache()`.

---

## `scripts/evaluate.py`

Standalone CLI script (single GPU, no DDP). Follows `scripts/train_stage2.py` structure.

### CLI Arguments
```bash
python scripts/evaluate.py                              # Stage 2, full test set
python scripts/evaluate.py --model both --n_samples 100 # Compare both, quick
python scripts/evaluate.py --model stage1               # Stage 1 only
```

Arguments: `--model {stage1,stage2,both}`, `--n_samples`, `--output_dir`, `--data_dir`, `--max_new_tokens`, `--gpu`.

### Structure
1. GPU selection (auto or `--gpu`)
2. Dataset loading (all 6 test shards, optional subset)
3. Model loading (based on `--model`)
4. Evaluation loop (same `evaluate_model()` function)
5. Save JSONL + summary JSON
6. Print comparison table (if `--model both`)

No wandb — evaluation is a one-shot operation.

---

## Reusable Code from Existing Files

| What | Source | Reuse in |
|------|--------|----------|
| `run_inference()` | `notebooks/06_training_stage2_lora.py:538-561` | Notebook Section 4, script |
| `get_free_gpu()` | `notebooks/06_training_stage2_lora.py:62-81` | Notebook Section 1, script |
| Environment boilerplate | `notebooks/06_training_stage2_lora.py:38-106` | Notebook Section 1 |
| LoRA loading pattern | `notebooks/06_training_stage2_lora.py:504-528` | Notebook Section 3 |
| `find_idle_gpus()` | `scripts/train_stage2.py:59-78` | Script GPU selection |

---

## Implementation Order

1. Save this plan as `Documentation/Session7_Plan.md`
2. Create `notebooks/07_evaluation.py` (all 10 sections)
3. Generate `notebooks/07_evaluation.ipynb` via `jupytext --to notebook`
4. Create `scripts/evaluate.py`
5. Test Sections 1-4 in notebook (setup, dataset, model load, helpers)
6. Run Stage 2 evaluation (Section 5)
7. Run Stage 1 evaluation (Section 6)
8. Run analysis (Sections 7-8)

---

## Verification

1. Notebook runs end-to-end on single GPU without OOM
2. `jiwer` computes WER/CER without errors on full test set
3. JSONL results file is written and readable
4. Stage 2 WER < Stage 1 WER (confirming LoRA improvement)
5. Duration breakdown shows no catastrophic failure on any bucket
6. Custom audio inference works on a `.wav` file
7. `scripts/evaluate.py --model both --n_samples 100` completes and prints comparison table
