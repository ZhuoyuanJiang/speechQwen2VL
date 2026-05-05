#!/usr/bin/env python3
"""
Extract a few short LargeScaleASR test samples as .wav files for the Gradio demo.

Streams from the Hugging Face dataset (no need for the local ./data symlink to
exist), filters to short clips so the demo loads fast, and writes one .wav per
sample plus a manifest.json with the reference transcripts.

Usage:
    python scripts/extract_demo_samples.py            # default 5 samples, ≤8s each
    python scripts/extract_demo_samples.py --n 8 --max_seconds 12
"""

import argparse
import json
import os
from io import BytesIO

import torch
import torchaudio
import torchaudio.transforms as T
from datasets import load_dataset, Audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="number of samples to extract")
    parser.add_argument("--max_seconds", type=float, default=8.0,
                        help="reject samples longer than this")
    parser.add_argument("--min_seconds", type=float, default=2.0,
                        help="reject samples shorter than this")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "demo_examples"),
                        help="output directory for .wav files + manifest.json")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for the deterministic shuffle (matches eval pipeline)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output dir: {args.out_dir}")

    # Stream the test split — avoids a full download / cache hit
    print("Streaming test split from Hugging Face (no local ./data needed)...")
    ds = load_dataset(
        "speechbrain/LargeScaleASR", "small",
        split="test",
        streaming=True,
    )

    # Decode bytes ourselves so we can keep streaming behavior simple
    manifest = []
    written = 0
    seen = 0
    target_sr = 16000

    for sample in ds:
        seen += 1
        duration = float(sample.get("duration") or 0.0)
        if duration < args.min_seconds or duration > args.max_seconds:
            continue

        # `wav` may be a dict with "bytes"/"path" or a decoded array depending on dataset config.
        wav_field = sample["wav"]
        if isinstance(wav_field, dict) and wav_field.get("bytes") is not None:
            waveform, sr = torchaudio.load(BytesIO(wav_field["bytes"]))
        elif isinstance(wav_field, dict) and "array" in wav_field:
            arr = torch.tensor(wav_field["array"], dtype=torch.float32).unsqueeze(0)
            sr = int(wav_field["sampling_rate"])
            waveform = arr
        else:
            print(f"  skipping sample {seen}: unrecognized wav field shape")
            continue

        if sr != target_sr:
            waveform = T.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        sample_id = sample.get("ID") or sample.get("id") or f"sample_{seen:06d}"
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(sample_id))
        out_name = f"{written + 1:02d}_{safe_id}.wav"
        out_path = os.path.join(args.out_dir, out_name)
        torchaudio.save(out_path, waveform, target_sr, format="wav")

        ref_text = sample.get("text", "")
        manifest.append({
            "file": out_name,
            "id": str(sample_id),
            "duration_s": round(duration, 2),
            "reference": ref_text,
        })
        print(f"  wrote {out_name}  ({duration:.1f}s)  ref: {ref_text[:60]}...")
        written += 1

        if written >= args.n:
            break

    if written == 0:
        print(f"WARNING: no samples written (scanned {seen}). "
              f"Try widening --min_seconds / --max_seconds.")
        return

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {written} samples (scanned {seen}). Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
