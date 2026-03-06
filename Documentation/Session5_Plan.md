# Session 5 Plan: Training Stage 1 — Audio Projector Only (Notebook 05)

## Context

Sessions 1-4 are complete. The model at `DanJZY/Qwen2-VL-7B-Speech` has all components pretrained **except** `model.model.audio_projector` (~17M params), which is randomly initialized. Session 5 trains this projector so it learns to map Whisper audio embeddings into the LLM's text embedding space.

**Where to run**: Server (2x A6000, 48GB each). This is the transition from Colab — training needs persistent storage and long runtimes.

**Maps to**: Skeleton notebook cells 58-73.

---

## Notebook Structure (10 sections)

### Section 1: Environment Setup
- No pip installs on server — conda env `speech_qwen2vl` already has editable fork installs
- Imports: `Qwen2VLForConditionalGeneration`, `Qwen2VLProcessor`, `process_vision_info`, `SFTConfig`, `SFTTrainer`
- HF login (same pattern as Notebook 04 — `get_token()` → env var → manual)
- Print versions, GPU info (should show 2x A6000)

### Section 2: Download Dataset
- From skeleton cell 59:
  ```python
  train_dataset = load_dataset("speechbrain/LargeScaleASR",
      data_files=["small/train-0000*", "small/train-0001*"], num_proc=12)
  test_dataset = load_dataset("speechbrain/LargeScaleASR",
      data_files=["test/test-00000*"], num_proc=12)
  train_dataset = train_dataset["train"]
  test_dataset = test_dataset["train"]
  test_dataset = test_dataset.select(range(100))
  ```
- `num_proc=12` — adjust to server's CPU core count
- **Pre-filter over-budget samples**: After loading, compute actual token counts and drop samples that would exceed `max_length=2048`. Implementation: use the tokenizer to get the exact transcript token count per sample (`len(tokenizer.encode(sample["text"], add_special_tokens=False))` — must use `add_special_tokens=False` to avoid overcounting from BOS tokens), compute audio pad count from duration (`min(ceil(duration * 50), 1500)`), add fixed template overhead (system header + user header + audio delimiters + prompt text + assistant header + `<|im_end|>` — measure once by tokenizing a dummy template that uses the exact same chat template, prompt text `"Transcribe this audio."`, and message structure as the collator; if the prompt changes later, this overhead must be re-measured), and filter out any sample where `audio_pads + transcript_tokens + template_overhead > 2048`. This is better than collator-level skipping because a batch where all labels are -100 still counts as a training step — the Trainer advances the LR scheduler, logs a (zero/NaN) loss value, and dilutes gradient accumulation, introducing noise in training dynamics.
- Note on audio longer than ~30s: Whisper's mel spectrogram extraction uses a fixed 30-second window (`processing_qwen2_vl.py:167`). For a 37s audio clip, the last 7 seconds are silently truncated at the feature level — the audio features only capture the first 30s. However, the transcript still contains all words including those from the last 7 seconds, creating a training mismatch (model hears 30s but is expected to transcribe 37s). The audio pad count is capped at 1500 (`min(ceil(duration*50), 1500)`) so both 30s and 37s produce the same number of pads. For Stage 1 this mismatch is unlikely to matter much (dataset is mostly short clips, projector just needs to learn the general mapping), but true long-form ASR would need sliding-window chunking — a separate feature, not Stage 1's concern.

### Section 3: Memory Cleanup Utility
- `clear_memory()` function from skeleton cell 63
- Cleans GPU memory, deletes globals, runs `gc.collect()` + `torch.cuda.empty_cache()`

### Section 4: Load Model & Processor, Freeze Parameters
- Load from `DanJZY/Qwen2-VL-7B-Speech` in bf16 with `device_map="cuda:0"` (single GPU)
  - Memory estimate is ~22-25 GB, well within one A6000's 48 GB. `device_map="auto"` would split model layers across 2 GPUs (model parallelism), meaning activations must transfer between devices at every split point during both forward and backward passes. But we're only training 17M params — the other ~8.3B frozen params just do forward passes. We'd pay cross-device communication overhead on every batch while getting no training speedup. Data parallelism (same model on each GPU, different batches) would help, but `device_map="auto"` does model parallelism, not data parallelism. Since everything fits on one card, single GPU is simpler and faster.
- Set `model.config.use_cache = False` — KV cache is for inference; it conflicts with gradient checkpointing during training
- **No quantization** for Stage 1 (skeleton says "Don't use NF4 quantization")
- Token ID consistency check (processor vs model config)
- **Freeze all → unfreeze only `audio_projector`**:
  ```python
  for param in model.parameters():
      param.requires_grad = False
  for param in model.model.audio_projector.parameters():
      param.requires_grad = True
  ```
- Assert: `trainable_params == audio_projector_params` (~17M, 0.19% of total)

### Section 5: Data Collator — `AudioTextCollator`

The most complex piece. Takes raw dataset samples → model-ready batches with masked labels.

**Step-by-step flow**:
1. Format each sample as a conversation:
   - User: `{"type": "audio", "audio": sample["wav"]["bytes"]}` + `"Transcribe this audio."`
   - Assistant: `{"type": "text", "text": sample["text"]}`
2. `process_vision_info(messages)` → extract audio inputs
3. `processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)` → text
4. `processor(text=texts, audios=all_audios, return_tensors="pt", padding=True, truncation=True, max_length=2048)` → batch
5. **Truncation safety check (backup)**: Most over-budget samples are already removed by dataset-level filtering in Section 2. As a safety net, after label masking, verify each sample has at least some real labels. If a sample somehow slipped through filtering and truncation ate its transcript entirely, log a warning and assert — this should never happen after proper filtering, so a hard failure is better than silently passing through a broken batch.
6. Create labels by masking non-assistant tokens to -100

**Label masking strategy**:
1. In `__init__`, tokenize `<|im_start|>assistant\n` once → cache as a token ID list (e.g., `[151644, ...]`)
2. For each sequence in the batch, search `input_ids` for this exact token subsequence
3. Assert exactly one match per sequence (multiple matches = broken template)
4. Mask everything up to and including the match → -100
5. Keep everything after (transcript + `<|im_end|>`) as real labels
6. Mask padding tokens → -100

```
<|im_start|>system\n...<|im_end|>\n         → -100 (masked)
<|im_start|>user\n<|audio_start|>...<|im_end|>\n  → -100 (masked)
<|im_start|>assistant\n                      → -100 (masked)
TRANSCRIPT TEXT                              → REAL LABELS (train on these)
<|im_end|>                                   → REAL LABEL (model learns to stop)
[padding]                                    → -100 (masked)
```

**Key decisions**:
- `add_generation_prompt=False` for training (include full conversation with response)
- Include `<|im_end|>` in labels so model learns to produce stop token
- `max_length=2048` training sequence cap — this is NOT the model's context limit (Qwen2-VL supports 32K+), but a practical training choice to manage GPU memory. Longer sequences consume more activation memory per sample. Most ASR clips in this dataset are short (5-30s, ~500-1500 total tokens), so 2048 is generous. Could increase to 4096 but would need to halve batch size. Note: even at inference, audio is effectively capped at ~30s by Whisper's fixed mel extraction — true long-form ASR would require sliding-window chunking, which is a separate feature beyond Stage 1.

### Section 6: SFTConfig

```python
SFTConfig(
    output_dir="./checkpoints/stage1_audio_projector",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,      # effective batch = 16
    learning_rate=1e-4,                 # higher than typical LLM fine-tune (projector is random)
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=10,
    report_to="wandb",
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    remove_unused_columns=False,        # collator needs raw dataset columns
    dataset_text_field=None,            # bypass SFTTrainer's built-in processing
    dataset_kwargs={"skip_prepare_dataset": True},  # prevent SFTTrainer from preprocessing
    max_seq_length=2048,                # NOTE: verify param name against server's trl version
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)
```

### Section 7: wandb Init
- Project: `speechQwen2VL`, run name: `stage1-audio-projector`
- Log config: trainable params, lr, epochs, effective batch size

### Section 8: Create Trainer & Train

```python
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=AudioTextCollator(processor),
    processing_class=processor.tokenizer,
)
trainer.train()
```

### Section 9: Push to HuggingFace
- `model.push_to_hub(REPO_ID)`
- Note: only 1 of 4 safetensor shards changes (audio_projector weights)

### Section 10: Verify & Cleanup
- Quick inference test on a test sample using `run_inference()` pattern from Notebook 04
- Compare output vs ground truth — should be somewhat coherent (not garbage like pre-training). Specifically: less refusal-like than Session 4, partial/noisy transcription = success
- `clear_memory()`

---

## Memory Estimation (Single A6000, 48GB)

| Component | Memory |
|-----------|--------|
| Model weights (bf16) | ~16.7 GB |
| Forward activations (batch=2, seq~1000) | ~2-4 GB |
| Projector gradients + optimizer states (17M params, AdamW fp32) | ~0.2 GB |
| Gradient checkpointing overhead | ~1-2 GB |
| Temporary tensors | ~1-2 GB |
| **Total estimate** | **~22-25 GB** |

Fits comfortably on one A6000 (48GB). Fallback: reduce `batch_size` to 1 + increase `grad_accum` to 16.

---

## Key Design Decisions

1. **Server, not Colab**: Training needs persistent storage (dataset downloads, checkpoints) and long runtimes. Colab has idle timeouts and session limits.

2. **SFTTrainer with custom collator**: Skeleton notebook uses SFTTrainer. Custom collator handles all data processing — we bypass SFTTrainer's built-in processing with three settings: `dataset_text_field=None` (don't look for a text column), `remove_unused_columns=False` (keep raw dataset columns for the collator), and `dataset_kwargs={"skip_prepare_dataset": True}` (prevent SFTTrainer from running its own dataset preprocessing). Also set `model.config.use_cache = False` since KV cache conflicts with gradient checkpointing.

3. **lr=1e-4**: Higher than typical LLM fine-tuning (2e-5) because the projector is randomly initialized and small (17M params).

4. **Gradient checkpointing**: Even though only 17M params are trainable, the forward pass through the frozen 7B model still consumes activation memory. Gradient checkpointing trades compute for memory.

5. **No pre-computed features**: Audio features are computed on-the-fly in the collator via `WhisperFeatureExtractor`. With `dataloader_num_workers=4`, CPU workers run in parallel to keep the GPU fed.

---

## Potential Issues & Mitigations

1. **Slow data loading**: If GPU utilization is low, increase `dataloader_num_workers`. Pre-computing audio features is a fallback.
2. **Multi-GPU if needed**: Default is single GPU (`cuda:0`). If memory is tight (e.g., larger batch size or longer sequences), switch to `device_map="auto"` to split across 2 A6000s — but expect to debug Trainer/DDP interactions.
3. **SFTTrainer conflicts**: If SFTTrainer's VLM detection causes issues with the custom collator, fall back to plain `Trainer` from transformers (same API).
4. **"assistant\n" tokenization**: Cache and verify the token encoding in collator `__init__`. Assert exactly one match per sequence. Adjust matching logic if tokenizer splits it unexpectedly.
5. **Over-budget training sequences**: The total token count (system + user header + audio pads + prompt + assistant transcript + `<|im_end|>`) can exceed `max_length=2048` even though audio pads alone are capped at ~1500. Long transcripts combined with long audio push over budget. Primary defense: dataset-level filtering (Section 2) removes these before training. Backup: collator asserts no empty-label samples. We prefer dataset-level filtering over collator-level skipping because: when a collator zeros out all labels in a sample, it still sits in the batch — the Trainer computes a forward pass (wasting GPU time), the loss returns 0 or NaN, the LR scheduler still advances one step, and with gradient accumulation (8 steps), the effective gradient is diluted (averaging over 8 mini-batches but only 7 contributed real gradients). None of this crashes, but it introduces noise in training dynamics. Removing bad samples upfront avoids all of this. Note: this is a training-only constraint — at inference there are no labels, so there's no "label-space" competition; the model processes the input and generates output tokens freely beyond it.

---

## Verification

1. Freeze verification: exactly 17M trainable params
2. Collator output: print one batch, verify label masking visually (decoded labels should match ground truth transcript + `<|im_end|>`)
3. Training loss decreases over 3 epochs
4. Post-training inference on a few fixed test samples: output should be clearly less refusal-like than Session 4's "I'm sorry, but I can't assist with that" — even partial/noisy transcription counts as success for Stage 1
5. HF push: only 1 of 4 shards changed

---

## Files

| File | Action |
|------|--------|
| `notebooks/05_training_stage1_adapter.ipynb` | Created |
| `Documentation/Session5_Plan.md` | Created (this file) |
| `Documentation/Session5_Progress_*.md` | Created (after training) |
| No fork changes needed | — |
