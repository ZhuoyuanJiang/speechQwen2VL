#!/usr/bin/env python3
"""
Standalone evaluation script for speechQwen2VL models.

Computes WER/CER on the full test set using jiwer. Supports evaluating
Stage 1 (projector only), Stage 2 (LoRA), or both for comparison.

Usage:
    python scripts/evaluate.py                              # Stage 2, full test set
    python scripts/evaluate.py --model both --n_samples 100 # Compare both, quick
    python scripts/evaluate.py --model stage1               # Stage 1 only
    python scripts/evaluate.py --repetition_penalty 1.1 --num_beams 2  # With decoding fixes
"""

import os
import sys
import subprocess
import argparse
import json
import re
import gc
import time
import math
from datetime import datetime

# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate speechQwen2VL ASR models")
parser.add_argument("--model", type=str, choices=["stage1", "stage2", "both"], default="stage2",
                    help="Which model to evaluate (default: stage2)")
parser.add_argument("--n_samples", type=int, default=None,
                    help="Number of test samples (default: all ~8,087)")
parser.add_argument("--output_dir", type=str, default="./checkpoints",
                    help="Directory for saving results (default: ./checkpoints)")
parser.add_argument("--data_dir", type=str, default="./data",
                    help="Dataset cache directory (default: ./data)")
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--repetition_penalty", type=float, default=1.0,
                    help="Repetition penalty for decoding (default: 1.0 = off, try 1.1-1.2 to reduce loops)")
parser.add_argument("--num_beams", type=int, default=1,
                    help="Beam search width (default: 1 = greedy, try 2 for better stopping)")
parser.add_argument("--gpu", type=int, default=None,
                    help="GPU index (default: auto-select)")

args = parser.parse_args()

# ---------------------------------------------------------------------------
# GPU Selection
# ---------------------------------------------------------------------------
def find_idle_gpus():
    """Return list of (gpu_id, free_mb) for GPUs with < 500 MiB used."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    gpus = []
    for line in result.stdout.strip().split("\n"):
        idx, used, free = [int(x.strip()) for x in line.split(",")]
        if used < 500:
            gpus.append((idx, free))
    return sorted(gpus, key=lambda x: -x[1])  # most free first


if args.gpu is not None:
    gpu_id = args.gpu
else:
    idle = find_idle_gpus()
    if not idle:
        print("No idle GPUs found. Use --gpu to specify one.")
        sys.exit(1)
    gpu_id = idle[0][0]

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
os.environ["HF_DATASETS_CACHE"] = os.path.abspath(args.data_dir)

print(f"Using GPU {gpu_id}")

# ---------------------------------------------------------------------------
# Imports (after CUDA_VISIBLE_DEVICES is set)
# ---------------------------------------------------------------------------
import torch
from datasets import load_dataset, Audio
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from jiwer import wer, cer
from tqdm import tqdm

DEVICE = "cuda:0"
STAGE1_REPO = "DanJZY/Qwen2-VL-7B-Speech"
LORA_REPO = "DanJZY/Qwen2-VL-7B-Speech-LoRA"

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def run_inference(model, processor, messages, max_new_tokens=256,
                  repetition_penalty=1.0, num_beams=1):
    """Run inference on a single conversation."""
    image_inputs, video_inputs, audio_inputs = process_vision_info(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        return_tensors="pt",
        padding=True,
    )
    batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    model.eval()
    gen_kwargs = dict(max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False)
    if repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    with torch.inference_mode():
        output_ids = model.generate(**batch, **gen_kwargs)
    prompt_len = batch["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_len:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

def normalize_text(text):
    """Normalize text for WER/CER: uppercase, strip, remove punctuation."""
    text = text.upper().strip()
    text = _PUNCT_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evaluate_model(model, processor, dataset, model_name, max_new_tokens=256,
                   repetition_penalty=1.0, num_beams=1):
    """Run inference on dataset, compute WER/CER, return summary and per-sample results."""
    predictions = []
    references = []
    results = []

    for i in tqdm(range(len(dataset)), desc=f"Evaluating {model_name}"):
        sample = dataset[i]
        messages = [
            {"role": "user", "content": [
                {"type": "audio", "audio": sample["wav"]["bytes"]},
                {"type": "text", "text": "Transcribe this audio."},
            ]},
        ]

        pred = run_inference(model, processor, messages, max_new_tokens=max_new_tokens,
                            repetition_penalty=repetition_penalty, num_beams=num_beams)
        ref = sample["text"]

        pred_norm = normalize_text(pred)
        ref_norm = normalize_text(ref)

        predictions.append(pred_norm)
        references.append(ref_norm)

        sample_wer = wer(ref_norm, pred_norm) if ref_norm else float("inf")
        sample_cer = cer(ref_norm, pred_norm) if ref_norm else float("inf")

        results.append({
            "index": i,
            "id": sample.get("ID", ""),
            "duration": sample.get("duration", 0),
            "spk_id": sample.get("spk_id", ""),
            "sex": sample.get("sex", ""),
            "reference_raw": ref,
            "prediction_raw": pred,
            "reference": ref_norm,
            "prediction": pred_norm,
            "wer": sample_wer,
            "cer": sample_cer,
        })

    corpus_wer = wer(references, predictions)
    corpus_cer = cer(references, predictions)

    summary = {
        "model_name": model_name,
        "n_samples": len(dataset),
        "corpus_wer": corpus_wer,
        "corpus_cer": corpus_cer,
        "timestamp": datetime.now().isoformat(),
    }

    return summary, results


def save_results(summary, results, output_dir, model_name):
    """Save per-sample results (JSONL) and summary (JSON)."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path = os.path.join(output_dir, f"eval_{model_name}_{timestamp}_cli.jsonl")
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary_path = os.path.join(output_dir, f"eval_{model_name}_{timestamp}_cli_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {results_path}")
    print(f"Summary saved to {summary_path}")


def print_summary(summary):
    """Print evaluation summary."""
    print(f"\n=== {summary['model_name']} Results ===")
    print(f"Samples: {summary['n_samples']}")
    print(f"WER:     {summary['corpus_wer']:.4f} ({summary['corpus_wer']*100:.2f}%)")
    print(f"CER:     {summary['corpus_cer']:.4f} ({summary['corpus_cer']*100:.2f}%)")


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
print("\nLoading test dataset...")
test_dataset = load_dataset(
    "speechbrain/LargeScaleASR",
    data_files=["test/test-*"],
    num_proc=12,
)
test_dataset = test_dataset["train"]
test_dataset = test_dataset.cast_column("wav", Audio(decode=False))
print(f"Full test set: {len(test_dataset)} samples")

# Deterministic shuffle for representative subsets
test_dataset = test_dataset.shuffle(seed=42)
if args.n_samples is not None:
    test_dataset = test_dataset.select(range(min(args.n_samples, len(test_dataset))))
    print(f"Subsetted to {len(test_dataset)} samples")

# ---------------------------------------------------------------------------
# 2. Load processor
# ---------------------------------------------------------------------------
processor = Qwen2VLProcessor.from_pretrained(STAGE1_REPO)

# ---------------------------------------------------------------------------
# 3. Evaluate
# ---------------------------------------------------------------------------
summaries = {}

if args.model in ("stage2", "both"):
    print("\nLoading Stage 2 model (LoRA)...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        STAGE1_REPO, torch_dtype=torch.bfloat16, device_map=DEVICE,
    )
    base_model.config.use_cache = True
    model = PeftModel.from_pretrained(base_model, LORA_REPO)
    model.eval()
    print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    summary_s2, results_s2 = evaluate_model(
        model, processor, test_dataset, "stage2_lora", args.max_new_tokens,
        args.repetition_penalty, args.num_beams
    )
    print_summary(summary_s2)
    save_results(summary_s2, results_s2, args.output_dir, "stage2_lora")
    summaries["stage2"] = summary_s2

    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

if args.model in ("stage1", "both"):
    print("\nLoading Stage 1 model (projector only)...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        STAGE1_REPO, torch_dtype=torch.bfloat16, device_map=DEVICE,
    )
    model.config.use_cache = True
    model.eval()
    print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    summary_s1, results_s1 = evaluate_model(
        model, processor, test_dataset, "stage1_projector", args.max_new_tokens,
        args.repetition_penalty, args.num_beams
    )
    print_summary(summary_s1)
    save_results(summary_s1, results_s1, args.output_dir, "stage1_projector")
    summaries["stage1"] = summary_s1

    del model
    gc.collect()
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 4. Comparison table (if both models evaluated)
# ---------------------------------------------------------------------------
if "stage1" in summaries and "stage2" in summaries:
    s1 = summaries["stage1"]
    s2 = summaries["stage2"]

    wer_improve = (s1["corpus_wer"] - s2["corpus_wer"]) / s1["corpus_wer"] * 100 if s1["corpus_wer"] > 0 else 0.0
    cer_improve = (s1["corpus_cer"] - s2["corpus_cer"]) / s1["corpus_cer"] * 100 if s1["corpus_cer"] > 0 else 0.0

    print(f"\n{'='*56}")
    print(f"{'Metric':<12} {'Stage 1':>12} {'Stage 2':>12} {'Improvement':>16}")
    print(f"{'-'*56}")
    print(f"{'WER':<12} {s1['corpus_wer']:>11.2%} {s2['corpus_wer']:>11.2%} {wer_improve:>+15.1f}%")
    print(f"{'CER':<12} {s1['corpus_cer']:>11.2%} {s2['corpus_cer']:>11.2%} {cer_improve:>+15.1f}%")
    print(f"{'='*56}")

print("\nEvaluation complete.")
