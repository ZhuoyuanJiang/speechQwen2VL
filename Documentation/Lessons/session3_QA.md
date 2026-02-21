# Session 3 — Q&A and Lessons Learned

## Q1: Why must audio_features be cleared after prefill in generation?

**Context**: During `model.generate()`, the model runs in two phases: (1) prefill — process the full input sequence including audio, (2) decode — generate tokens one at a time using KV cache. The `prepare_inputs_for_generation()` method prepares inputs for each step.

**Problem**: If `audio_features` is not set to `None` after prefill, the WhisperEncoder (32 transformer layers, ~635M params) re-runs on every single decode token. For a 100-token generation, that's 100 redundant encoder forward passes instead of 1.

**How images/videos handle it** (line 1490-1492 in `modeling_qwen2_vl.py`):
```python
if model_inputs["cache_position"][0] != 0:
    model_inputs["pixel_values"] = None
    model_inputs["pixel_values_videos"] = None
```
After the first step (prefill), `cache_position[0] != 0`, so pixel_values are cleared. The vision embeddings are already baked into the KV cache from prefill — no need to re-compute them.

**Fix**: Add the same clearing for audio:
```python
model_inputs["audio_features"] = None
model_inputs["audio_lengths"] = None
```

**Lesson**: When adding a new encoder to a generation-capable model, always check the `prepare_inputs_for_generation` cache path. Encoders should run once during prefill, not on every decode step. The signal is `cache_position[0] != 0` (meaning we're past the first step).

## Q2: dtype/device casting for audio encoder input

**Context**: `audio_features` arrives from the processor as numpy arrays → tensors, typically float32 on CPU. But the WhisperEncoder may be in bf16 on GPU.

**Pattern from vision** (line 1131): `pixel_values = pixel_values.type(self.visual.dtype)`

**Fix for audio**: Cast before encoder forward:
```python
audio_features = audio_features.to(device=self.audio_encoder.device, dtype=self.audio_encoder.dtype)
```

**Also**: When indexing with `audio_lengths` (which may be a tensor), use `int(length.item())` to convert to Python int before slicing. This avoids CUDA scalar indexing warnings.

## Q3: Beam search expansion for audio — why we skip it for now

**Context**: `_expand_inputs_for_generation()` (line 1547) handles beam search by repeating visual features per-sample. It has custom split/repeat logic because visual tensors don't have a simple batch dimension (they're concatenated across samples).

**Decision**: We do NOT add `audio_features`/`audio_lengths` to the `visual_keys` list without implementing the full split/repeat branch. Adding them to the list without the logic would break beam expansion. Instead, we add a TODO comment in the code.

**Why it's safe to skip**: ASR uses greedy or sampling decoding, not beam search. If beam search is needed later, implement the per-sample split logic at that time.
