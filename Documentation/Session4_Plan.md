# Session 4 Plan: Inference & Testing (Notebook 04)

## Context

Sessions 1-3 are complete. We have:
- Processor with audio tokens pushed to `DanJZY/Qwen2-VL-7B-Speech` (Session 2)
- Model with WhisperEncoder + audio projector pushed to same HF repo (Session 3)
- Audio projector weights are **random** (not yet trained)

Session 4 builds Notebook 04 to test inference end-to-end. Two goals:
1. **VL test**: Confirm vision-language still works (audio additions didn't break anything)
2. **Audio test**: Run ASR inference — expect garbage output (random projector), but this validates the full generation pipeline including `model.generate()`, which was flagged as untested by the Session 3 reviewer

**Where to run**: Colab Pro (L4 24GB) or server (A6000 48GB).

---

## Notebook Structure

### Section 1: Environment Setup
- Same install pattern as Notebook 03
- Pin `tokenizers>=0.21,<0.22`
- Fork installs with `--force-reinstall --no-deps` at pinned commit hashes:
  - transformers: `934129b7701e7607facb39f286afc6bc4cc657df`
  - Qwen3-VL: `56b0756a768cc3b01cba45b01c1bc3c8cb74ea3f`
- Imports: `Qwen2VLForConditionalGeneration`, `Qwen2VLProcessor`, `process_vision_info`, `torch`, `datasets`
- Print versions, GPU info

### Section 2: Load Model & Processor
- Load from `DanJZY/Qwen2-VL-7B-Speech` (has audio config + weights from Notebook 03)
- `Qwen2VLForConditionalGeneration.from_pretrained(..., torch_dtype=torch.bfloat16, device_map="auto")`
- `Qwen2VLProcessor.from_pretrained(...)`
- Print model type, device, audio_encoder existence check
- Consistency check: `assert processor.tokenizer.convert_tokens_to_ids('<|audio_pad|>') == model.config.audio_token_id` — catches processor/config mismatch early

### Section 3: Build `run_inference()` Function
```python
def run_inference(model, processor, messages, max_new_tokens=256):
    """
    Full inference pipeline:
    1. Extract image/video/audio inputs via process_vision_info()
    2. Apply chat template
    3. Process through Qwen2VLProcessor (tokenize + featurize)
    4. model.generate()
    5. Decode output tokens (only the generated part, not the prompt)
    """
```

Key details:
- `process_vision_info(messages)` → `image_inputs, video_inputs, audio_inputs`
- `processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` → text
- `processor(text=[text], images=image_inputs, videos=video_inputs, audios=audio_inputs, return_tensors="pt", padding=True)` → batch
- Move batch to device
- Run under `model.eval()` + `torch.inference_mode()` for deterministic inference and lower memory
- `model.generate(**batch, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)` → output_ids
- **`num_beams=1` is required**: Audio beam expansion is intentionally TODO (Session 3 Q3). Using beam search would hit the unimplemented `_expand_inputs_for_generation` path and break. `do_sample=False` gives deterministic greedy decoding.
- Trim prompt tokens: `output_ids[:, batch["input_ids"].shape[1]:]`
- `processor.batch_decode(trimmed, skip_special_tokens=True)` → text

### Section 4: VL Test — Red Car Bounding Box
- Image URL: `https://t4.ftcdn.net/jpg/01/57/82/05/360_F_157820583_agejYX5XeczPZuWRSCDF2YYeCGwJqUdG.jpg` (from skeleton notebook)
- Full message format:
  ```python
  messages = [
      {
          "role": "user",
          "content": [
              {"type": "image", "image": IMAGE_URL},
              {"type": "text", "text": "Detect the bounding box of the red car."},
          ],
      },
  ]
  ```
- Run `run_inference()`, print output
- Expected: bounding box coordinates (same as vanilla Qwen2-VL since audio additions don't affect VL path)
- This confirms our model modifications are backward-compatible

### Section 5: Audio Test — ASR (Expect Garbage)
- Load a sample from `speechbrain/LargeScaleASR` (streaming, same as Notebook 03)
- Full message format:
  ```python
  messages = [
      {
          "role": "user",
          "content": [
              {"type": "audio", "audio": sample["wav"]["bytes"]},
              {"type": "text", "text": "Transcribe this audio."},
          ],
      },
  ]
  ```
- Run `run_inference()`, print output
- Print ground truth for comparison
- Expected: **garbage/random text** — audio_projector is untrained, so the projected audio embeddings are meaningless
- This is the key test: validates `model.generate()` works with audio (prefill + decode + clearing)

### Section 6: Summary
- Markdown cell summarizing results
- VL: works correctly, backward compatible
- Audio: garbage as expected, generation pipeline works
- Note: audio_projector will be trained in Session 5

### Section 7: Cleanup
- `del model, processor`
- `gc.collect()`, `torch.cuda.empty_cache()`

---

## Key Design Decisions

1. **Load from HuggingFace, not local**: Model was pushed in Notebook 03. Loading from HF is cleaner and tests that the push worked correctly.

2. **No vanilla Qwen2-VL comparison**: Loading two 7B models would exceed L4's 24GB. Instead, we just verify our model produces reasonable VL output (bounding box coordinates). The VL path is completely unchanged — if it works, it's identical.

3. **`model.generate()` is the main test**: Session 3 reviewer flagged that generation was untested. This notebook exercises the full generation path including `prepare_inputs_for_generation()` and prefill clearing.

4. **No fork modifications needed**: All code changes are done. This is a pure notebook session.

---

## Verification

1. VL test produces bounding box coordinates for the red car
2. Audio test completes without runtime errors (proves generation pipeline works — output may be empty or garbage since projector is random)
3. Processor/model token ID consistency check passes
4. GPU memory stays within L4's 24GB

---

## Files

| File | Action |
|------|--------|
| `notebooks/04_inference_and_testing.ipynb` | Created |
| `Documentation/Session4_Progress_*.md` | Created (after notebook tested) |
| No fork changes needed | — |
