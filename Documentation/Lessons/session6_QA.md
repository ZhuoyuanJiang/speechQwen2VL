# Session 6 — Q&A / Lessons Learned

---

## Q: Which layers does Stage 2 fine-tune? LLM only, or also the vision/audio encoders?

**Context**: Stage 2 applies LoRA adapters for end-to-end fine-tuning. The model has four major components: LLM decoder, vision encoder, audio encoder (Whisper), and audio projector. Which ones get trained?

**Answer**:

| Component | Trainable? | Method |
|---|---|---|
| LLM decoder layers (28 layers) | Yes | LoRA on q/k/v/o/gate/up/down_proj |
| Audio projector | Yes | Full params (via `modules_to_save`) |
| Audio encoder (Whisper) | No | Frozen — already pretrained |
| Vision encoder | No | Frozen — not changing vision capabilities |

**Critical gotcha**: PEFT's `target_modules` matches by suffix when given a list of strings. Using `["q_proj", "k_proj", "v_proj"]` would also hit Whisper's 96 attention layers (`audio_encoder.layers.*.self_attn.{q,k,v}_proj`), not just the LLM's 196 modules. To scope LoRA to the LLM only, use a regex pattern:
```python
target_modules=r"model\.language_model\.layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
```
The `model.` prefix is required because `Qwen2VLForConditionalGeneration` wraps the LLM as `self.model.language_model`, and PEFT uses `re.fullmatch` against the full dotted module path. This matches exactly 196 modules (28 layers × 7 projections) and excludes all audio_encoder and visual modules.

The audio projector stays trainable as full parameters (not LoRA) via PEFT's `modules_to_save` mechanism — it's only 17M params, too small for LoRA to make sense.

**Lesson**: Never blindly use suffix-based `target_modules` on a model with multiple encoders. Always verify with `model.print_trainable_parameters()` and inspect `model.named_parameters()` to confirm LoRA was applied only where intended. Use regex patterns to scope LoRA to specific submodules.

---

## Q: Do we need QLoRA (4-bit quantization) or can we use plain LoRA?

**Context**: The skeleton notebook specifies QLoRA (4-bit NF4 quantization). But our GPUs are 49 GB each. Do we actually need quantization?

**What's the difference**:
- **LoRA** = load model in bf16 (~16.7 GB) + add LoRA adapters on top
- **QLoRA** = load model in 4-bit (~4.5 GB) + add LoRA adapters on top
- The LoRA adapters and training procedure are identical. The only difference is whether the frozen base weights are stored in bf16 or 4-bit.

**VRAM comparison**:

| | LoRA (bf16 base) | QLoRA (4-bit base) |
|---|---|---|
| Model weights | ~16.7 GB | ~4.5 GB |
| LoRA adapters + optimizer | ~1.5 GB | ~1.5 GB |
| Activations + grad ckpt | ~3-6 GB | ~3-6 GB |
| **Total** | **~22-25 GB** | **~12-16 GB** |
| Fits on 49 GB GPU? | Yes | Yes, easily |

**Trade-offs**:
- **LoRA (no quantization)**: Full-precision gradients flowing through the base model. No risk of the audio projector getting accidentally quantized to 4-bit. Simpler code (no `BitsAndBytesConfig`, no `prepare_model_for_kbit_training`).
- **QLoRA**: Uses less VRAM (useful for smaller GPUs or larger batch sizes). Slight quantization noise in the base model, though research shows minimal quality impact in practice.

**Speed comparison**: LoRA is generally **faster** than QLoRA per step. 4-bit quantized weights must be dequantized to bf16 on-the-fly for every matrix multiplication during forward/backward. This dequantization overhead adds ~10-20% to step time. With bf16 LoRA, the weights are already in compute-ready format.

**Recommendation**: On our 49 GB GPUs, plain LoRA is preferred — faster training, simpler code, no quantization risks. QLoRA is the fallback if we need more VRAM (e.g., larger batches, longer sequences, or running on smaller GPUs).

**Lesson**: QLoRA exists to make fine-tuning fit on smaller GPUs. If you have enough VRAM for bf16, plain LoRA is strictly better — same adapters, faster training, higher precision.

---

## Q: Why do GPUs have very different VRAM usage during DDP training?

**Context**: During Stage 2 training with 6 GPUs, we observed highly uneven memory usage — some GPUs at ~30 GB while others hit ~45 GB. All GPUs run the same model with the same batch size. Why the difference?

**Observed VRAM** (6 DDP ranks, same step):
```
GPU 0: 29.8 GB   GPU 1: 30.4 GB   GPU 2: 35.3 GB
GPU 3: 36.1 GB   GPU 6: 42.0 GB   GPU 7: 45.0 GB
```

**Answer**: Three factors combine to cause this:

1. **DDP splits by index, not by length**: `DistributedSampler` assigns samples to GPUs by index (GPU 0 gets samples 0, 6, 12...; GPU 1 gets 1, 7, 13...). It does **not** balance by audio duration. Some GPUs end up with more long clips than others.

2. **Padding to max length within each GPU's batch**: Each GPU independently pads its batch to the length of its longest sample. If GPU 6 gets two 25-second clips while GPU 0 gets two 5-second clips, GPU 6's input tensors are ~5x larger.

3. **Activations scale with sequence length**: The forward pass through 8.3B params produces intermediate activations proportional to sequence length. Even with gradient checkpointing (which only stores activations at checkpoint boundaries), longer sequences = more memory for the currently-computed layers.

**Why some GPUs stay high**: Two possible reasons:
1. **Sampling luck**: `DistributedSampler` shuffles once per epoch, so a GPU assigned more long-duration samples will hit higher peaks more often throughout the epoch.
2. **CUDA caching allocator**: PyTorch's memory allocator reserves memory at peak usage and does **not** release it back to the OS. A GPU that processes one long batch will show high `nvidia-smi` usage even after subsequent shorter batches — the allocator holds the memory for future allocations. So a persistent 45 GB reading may reflect a past spike's high-water mark, not ongoing high usage.

In practice, both factors likely contribute. `nvidia-smi` reports the allocator's reserved memory, not necessarily what's actively in use.

**Why we also see transient spikes**: Within a single GPU's assigned samples, some batches happen to contain longer clips than others. GPU 3 spiked from ~30 GB to 44 GB on one step, then dropped back — it hit a batch with unusually long samples, then the next batch was shorter.

**Is this a problem?**: Not usually. The worst case is if the unluckiest GPU hits a long batch during evaluation (when gradient checkpointing is off). With `eval_batch_size=2` and only 100 eval samples, this hasn't caused OOM in practice. But it's worth monitoring the first eval to confirm.

**Lesson**: Variable-length data (audio, text) causes uneven GPU memory in DDP. The imbalance is structural (per-epoch sampling) plus stochastic (per-batch variation). If VRAM is tight, consider length-based batch sampling or reducing max sequence length to cap the worst case.
