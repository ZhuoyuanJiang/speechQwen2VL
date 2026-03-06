# Session 5 — Q&A

Questions that came up during Stage 1 training setup and their answers.

---

## Q: We're only training a 17M audio projector. Why did we OOM on a 49GB GPU?

17M is only the **trainable parameter** count. But the GPU holds the **entire 8.3B model** because the forward pass runs through every layer:

| What | Size | Why |
|---|---|---|
| Model weights (bf16) | ~16.7 GB | All 8.3B params loaded — forward pass needs every layer |
| Activations (training) | ~25+ GB | Intermediate results from every layer during forward pass |
| Optimizer states | ~0.07 GB | Only for 17M trainable params — tiny |
| Logits tensor | ~0.6 GB/sample | `[seq_len, vocab_size]` = `[2048, 151K]` × 2 bytes |

The OOM happened during **evaluation**, not training. The difference:

- **Training** uses `gradient_checkpointing`: doesn't store all layer activations, recomputes them during backward pass. Trades time for memory.
- **Evaluation** does a full forward pass without gradient checkpointing: all intermediate activations stay in memory, plus the full logits tensor for computing eval loss.

With `eval_batch_size=2`, the logits tensor alone is ~1.2 GB, and the uncompressed activations push total usage over 49 GB.

**Fix**: Set `per_device_eval_batch_size=1`. This doesn't affect any metrics — eval loss is per-sample averaged regardless of batch size.

---

## Q: Can we use 2 GPUs for evaluation to avoid eval_batch_size=1?

Technically possible, but not worth it. It would require copying the model to a second GPU (~16.7 GB extra), and the eval set is only 100 samples — each eval round takes about 1-2 minutes regardless. The batch size difference between train and eval has zero effect on results.

---

## Q: Why are all 8 GPUs being used when the plan is single-GPU?

HuggingFace `Trainer` automatically uses all visible GPUs via DataParallel. Even though `device_map=DEVICE` loads the model onto one GPU, the Trainer still tries to distribute across all 8.

**Fix**: Set `os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)` right after GPU selection, before any CUDA initialization. After this, PyTorch sees exactly 1 GPU (as `cuda:0`).

---

## Q: Is evaluating every 100 steps reasonable?

For our run (~5,592 total steps), **every 100 steps is too frequent**. That's 55 evaluations, each running 100 samples through the full 8.3B model. Rough estimate: each eval takes ~1-2 minutes, so **55-110 minutes total spent on eval alone** — about 15-25% of the ~7 hour training run.

**Standard practice**:
- Eval every 0.5-1 epoch, or a few times per epoch for longer runs
- For a ~5.6K step run, every 500 steps (matching `save_steps`) is more typical — that gives ~11 evals

**Recommendation**: Change `eval_steps` from 100 to 500. This saves significant wall-clock time while still giving enough eval checkpoints to track progress.

---

## Q: 17M params taking ~8 hours — isn't that too slow? What about QLoRA on 8B later?

The training time has almost nothing to do with the 17M trainable params. The bottleneck is the **full forward pass through 8.3B frozen params** — every training step pushes the input through the entire Whisper encoder + Qwen2-VL LLM. The 17M projector is a tiny MLP in the middle; the other 8.3B params do most of the compute.

Additional overhead:
- **Gradient checkpointing** effectively runs the forward pass ~2x (recomputing activations during backward)
- **Single GPU** — no parallelism
- **Small batch size** (2) — lower GPU utilization

**Ways to speed up training (most impactful first)**:

1. **Multi-GPU data parallel (DDP)**: Use 2-4 GPUs. Near-linear speedup. 4 GPUs → ~4x faster → ~2 hours instead of 8. This is the single biggest improvement.
2. **Increase eval_steps**: 500 instead of 100. Saves ~15-25% wall time.
3. **Increase batch size**: If memory allows, try `per_device_train_batch_size=4` (reduce `gradient_accumulation_steps` to 4 to keep effective batch=16). Larger batches = better GPU woutilization.
4. **Fewer epochs**: 2 epochs might be sufficient for Stage 1. The projector just needs to learn a basic linear-ish mapping.

**For QLoRA Stage 2**: The model gets quantized to 4-bit (~4 GB instead of ~16 GB for weights), freeing memory. But the forward pass still goes through the full model, so per-step time is similar. The key speedup for Stage 2 will also be multi-GPU DDP.

**Why trainable param count doesn't determine training time**: Intuitively you'd think 17M → fast, 8B → slow. But the forward pass always runs through the **entire** model regardless of how many params are trainable. The only thing trainable param count affects is optimizer memory and the backward pass scope — both minor compared to the forward pass through 8B params.

---

## Q: What should I monitor during training?

- **Loss curve** on wandb (project: `speechQwen2VL`, run: `stage1-audio-projector`). Should drop quickly in the first few hundred steps, then gradually flatten.
- **Checkpoints** save every 500 steps, keeping only the last 3 (`save_total_limit=3`), at `./checkpoints/stage1_audio_projector/`.
- **Total time**: ~4.7s/step × 5,592 steps ≈ 7.3 hours (single GPU).
- **After training**: Run Section 9 (inference test) and Section 10 (push to HuggingFace) in the notebook.

---

## Q: What's the dataset? Why only 20 shards out of 72?

Dataset: `speechbrain/LargeScaleASR`, `small` config.

| What | Shards | Samples |
|---|---|---|
| Full `small/` split | 72 shards | ~107K |
| What we use for train | 20 shards | ~29,820 |
| What we use for eval | 1 shard, first 100 | 100 |

Stage 1 only trains a 17M projector to learn the basic audio-to-text mapping. 30K samples over 3 epochs is sufficient — this is a relatively simple task (align two embedding spaces). More data matters more for Stage 2 (QLoRA fine-tuning the LLM itself).

To scale up later: change the data_files glob from `"small/train-0000*", "small/train-0001*"` to `"small/train-*"` for all 72 shards.

---

## Q: How does multi-GPU DDP actually speed things up?

DDP (Distributed Data Parallel) runs a **copy of the full model on each GPU**. Each step:

1. Each GPU gets its own mini-batch of `per_device_train_batch_size` samples
2. All GPUs do forward + backward in parallel
3. Gradients are averaged across GPUs (all-reduce)
4. Each GPU applies the same optimizer step

So with 4 GPUs, you process 4× more samples per step in the same wall time. The speedup is near-linear (with some communication overhead for gradient sync).

| GPUs | Estimated time (Stage 1) |
|---|---|
| 1 | ~7.3 hours |
| 4 | ~2 hours |
| 8 | ~1 hour |

**Important**: DDP requires launching with `torchrun` (multi-process), so it can't run inside a Jupyter notebook. The notebook is for learning and debugging; actual training runs as a Python script:

```bash
torchrun --nproc_per_node=4 scripts/train_stage1.py
```

This is standard practice — notebooks for prototyping, scripts for training. The script is converted from the notebook with the same logic.

---

## Q: Notebook vs script — what's the workflow?

- **Notebook** (`notebooks/05_training_stage1_adapter.ipynb`): For interactive development, debugging, and understanding the pipeline step by step. Single-GPU only.
- **Training script** (`scripts/train_stage1.py`): Converted from the notebook. Supports multi-GPU DDP via `torchrun`. This is what goes in the README for others to reproduce training.

The notebook is the "source of truth" for the training logic. The script is derived from it for production runs.
