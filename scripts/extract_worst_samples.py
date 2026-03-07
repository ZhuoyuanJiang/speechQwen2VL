#!/usr/bin/env python3
"""Extract audio files for the worst-performing evaluation samples."""

import os
import json
import glob

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ["HF_DATASETS_CACHE"] = os.path.abspath("./data")

from datasets import load_dataset, Audio

# Worst sample indices from Stage 2 evaluation (by WER)
WORST_INDICES = [414, 7512, 8025, 6376, 2352, 3228, 4317, 540, 4216, 5216]

OUTPUT_DIR = "./checkpoints/worst_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load test dataset with same shuffle as evaluation
print("Loading test dataset...")
test_dataset = load_dataset(
    "speechbrain/LargeScaleASR",
    data_files=["test/test-*"],
    num_proc=12,
)
test_dataset = test_dataset["train"]
test_dataset = test_dataset.cast_column("wav", Audio(decode=False))
test_dataset = test_dataset.shuffle(seed=42)

# Load results for metadata
s2_files = sorted(glob.glob("./checkpoints/eval_stage2_lora_*_cli.jsonl")) + \
           sorted(glob.glob("./checkpoints/eval_stage2_lora_*_notebook.jsonl"))
results_by_index = {}
if s2_files:
    with open(s2_files[-1]) as f:
        for line in f:
            r = json.loads(line)
            results_by_index[r["index"]] = r

print(f"\nExtracting {len(WORST_INDICES)} audio samples to {OUTPUT_DIR}/\n")

for idx in WORST_INDICES:
    sample = test_dataset[idx]
    audio_bytes = sample["wav"]["bytes"]
    ref = sample["text"]

    r = results_by_index.get(idx, {})
    pred = r.get("prediction_raw", r.get("prediction", "N/A"))
    sample_wer = r.get("wer", -1)

    filename = f"sample_{idx}_wer{sample_wer:.0%}.wav".replace("%", "pct")
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(audio_bytes)

    print(f"Sample {idx} (WER={sample_wer:.0%}, duration={sample.get('duration', 0):.1f}s)")
    print(f"  REF: {ref}")
    print(f"  HYP: {pred}")
    print(f"  Saved: {filepath}")
    print()

print(f"Done. All files in {OUTPUT_DIR}/")
