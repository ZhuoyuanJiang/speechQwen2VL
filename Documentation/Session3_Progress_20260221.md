# Session 3 — Progress Documentation

**Date**: February 21, 2026
**Objective**: Add Whisper audio encoder + MLP projector to Qwen2-VL model architecture
**Plan**: See `Documentation/Session3_Plan.md`
**Lessons**: See `Documentation/Lessons/session3_QA.md`

---

## 1. Fork Modification: configuration_qwen2_vl.py (Part 1)

**File**: `forks/transformers/src/transformers/models/qwen2_vl/configuration_qwen2_vl.py`
**Fork repo**: `ZhuoyuanJiang/transformers`, branch `speech-qwen2vl`

### 1.1 Added `Qwen2VLAudioConfig` class

New config class added after `Qwen2VLTextConfig`, before `Qwen2VLConfig`. Mirrors the `Qwen2VLVisionConfig` pattern. Stores Whisper encoder dimensions with defaults matching whisper-large-v3-turbo:

```python
class Qwen2VLAudioConfig(PretrainedConfig):
    model_type = "qwen2_vl"
    base_config_key = "audio_config"

    def __init__(
        self,
        d_model=1280,              # Whisper hidden dimension
        encoder_layers=32,          # 32 transformer layers
        encoder_attention_heads=20, # 20 attention heads
        encoder_ffn_dim=5120,       # FFN intermediate size
        num_mel_bins=128,           # 128 mel bins (NOT 80 — see Session 2 Q&A)
        max_source_positions=1500,  # 1500 output time steps (3000 mel frames / stride 2)
        encoder_layerdrop=0.0,
        activation_function="gelu",
        scale_embedding=False,
        **kwargs,
    ):
```

Includes `to_whisper_config()` method that converts to a `WhisperConfig` object for the `WhisperEncoder` constructor. Uses lazy import (`from ..whisper.configuration_whisper import WhisperConfig`) to avoid circular dependencies.

### 1.2 Modified `Qwen2VLConfig`

Three changes to the top-level config class:

**1. Added `audio_config` to `sub_configs` dict:**
```python
sub_configs = {
    "vision_config": Qwen2VLVisionConfig,
    "text_config": Qwen2VLTextConfig,
    "audio_config": Qwen2VLAudioConfig,    # NEW
}
```

**2. Added `audio_config` and `audio_token_id` parameters to `__init__`:**
```python
def __init__(
    self,
    text_config=None,
    vision_config=None,
    audio_config=None,           # NEW — None means audio disabled (backward compat)
    image_token_id=151655,
    video_token_id=151656,
    audio_token_id=None,         # NEW — None, not 151658 (see lesson Q4)
    **kwargs,
):
```

**Why `audio_token_id=None` instead of `151658`**: `image_token_id` and `video_token_id` have numeric defaults because they're part of the original Qwen2-VL design — every checkpoint uses them. Audio is our addition, so it defaults to `None` for backward compatibility. Old configs load cleanly without silently claiming token 151658. Audio is explicitly opted into in Notebook 03 by setting `config.audio_token_id = 151658`. See `session3_QA.md` Q4 for full explanation.

**3. Added deserialization for `audio_config`:**
```python
if isinstance(audio_config, dict):
    self.audio_config = self.sub_configs["audio_config"](**audio_config)
else:
    self.audio_config = audio_config
```

When loading from JSON, `audio_config` arrives as a dict and needs to be converted to a `Qwen2VLAudioConfig` object. When `None` (old configs), it stays `None`. This follows the same pattern as `vision_config` and `text_config` deserialization above it.

### 1.3 Updated `__all__`

```python
__all__ = ["Qwen2VLAudioConfig", "Qwen2VLConfig", "Qwen2VLTextConfig"]
```

---

## 2. Fork Modification: modeling_qwen2_vl.py (Part 2)

**File**: `forks/transformers/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py`
**Fork repo**: `ZhuoyuanJiang/transformers`, branch `speech-qwen2vl`

Nine changes made to the model file, organized by class:

### 2.1 Updated config import (line 49)

Added `Qwen2VLAudioConfig` to the import from `configuration_qwen2_vl`:

```python
from .configuration_qwen2_vl import Qwen2VLAudioConfig, Qwen2VLConfig, Qwen2VLTextConfig, Qwen2VLVisionConfig
```

### 2.2 `Qwen2VLModel.__init__` — conditional audio components

Added after `self.rope_deltas = None`, before `self.post_init()`:

```python
if getattr(config, "audio_config", None) is not None:
    from ..whisper.modeling_whisper import WhisperEncoder

    whisper_config = config.audio_config.to_whisper_config()
    self.audio_encoder = WhisperEncoder(whisper_config)
    text_hidden_size = config.text_config.hidden_size  # 3584 for 7B
    self.audio_projector = nn.Sequential(
        nn.Linear(config.audio_config.d_model, text_hidden_size),
        nn.GELU(),
        nn.Linear(text_hidden_size, text_hidden_size),
    )
```

**Key design decisions:**
- `getattr(config, "audio_config", None)` with fallback — safe for old configs that don't have `audio_config` attribute at all
- `WhisperEncoder` imported lazily inside the `if` block — avoids importing the entire Whisper module when audio is disabled
- `audio_projector` is a 2-layer MLP: `Linear(1280→3584) → GELU → Linear(3584→3584)`. The 3584 matches Qwen2-VL-7B's text `hidden_size`, making the projected audio embeddings compatible with `masked_scatter` into the text embedding sequence

**Resulting model hierarchy:**
```
Qwen2VLForConditionalGeneration
├── model: Qwen2VLModel
│   ├── visual: Qwen2VisionTransformerPretrainedModel   (existing)
│   ├── language_model: Qwen2VLTextModel                (existing)
│   ├── audio_encoder: WhisperEncoder                   ← NEW (~635M params)
│   └── audio_projector: nn.Sequential                  ← NEW (~17M params)
└── lm_head: nn.Linear                                  (existing)
```

### 2.3 Added `get_audio_features` method

New method on `Qwen2VLModel`, added after `get_image_features`. Parallel structure to `get_image_features` and `get_video_features`:

```python
def get_audio_features(self, audio_features, audio_lengths):
    audio_features = audio_features.to(device=self.audio_encoder.device, dtype=self.audio_encoder.dtype)
    encoder_output = self.audio_encoder(audio_features)
    audio_hidden = encoder_output.last_hidden_state  # (num_audios, 1500, 1280)
    audio_embeds = []
    for i, length in enumerate(audio_lengths):
        length = int(length.item()) if hasattr(length, "item") else int(length)
        trimmed = audio_hidden[i, :length, :]  # (length, 1280)
        projected = self.audio_projector(trimmed)  # (length, 3584)
        audio_embeds.append(projected)
    return audio_embeds
```

**Three important details:**

1. **dtype/device casting** (`audio_features.to(device=..., dtype=...)`): Input arrives as float32 on CPU from the processor. The WhisperEncoder may be in bf16 on GPU. This matches the vision pattern at line 1143: `pixel_values = pixel_values.type(self.visual.dtype)`. See `session3_QA.md` Q2.

2. **`int(length.item())`**: When `audio_lengths` is a CUDA tensor, using it directly as a slice index triggers a CUDA-to-CPU scalar transfer warning. Converting to Python int with `.item()` avoids this. The `hasattr` guard handles the case where length is already a plain int.

3. **Trimming to `audio_lengths[i]`**: WhisperEncoder always outputs 1500 time steps (fixed for 30s max audio), regardless of actual audio duration. A 10-second clip only has ~500 meaningful steps; the rest is encoder output from zero-padded input. We trim to discard these. See `session3_QA.md` Q5 for the full dimension breakdown.

### 2.4 Modified `Qwen2VLModel.forward()` signature

Added two parameters after `video_grid_thw`:

```python
audio_features: Optional[torch.FloatTensor] = None,
audio_lengths: Optional[torch.LongTensor] = None,
```

### 2.5 Added audio merge block in `Qwen2VLModel.forward()`

Inserted after the video merge block (after `inputs_embeds.masked_scatter(video_mask, video_embeds)`), before the `position_ids` calculation. Follows the same `masked_scatter` pattern as image and video:

```python
if audio_features is not None:
    if audio_lengths is None:
        raise ValueError("audio_features provided but audio_lengths is None")
    if self.config.audio_token_id is None:
        raise ValueError("audio_features provided but audio_token_id is not set in config")
    audio_embeds = self.get_audio_features(audio_features, audio_lengths)
    audio_embeds = torch.cat(audio_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    if input_ids is None:
        special_audio_mask = (
            inputs_embeds
            == self.get_input_embeddings()(
                torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
        ).all(-1)
    else:
        special_audio_mask = input_ids == self.config.audio_token_id
    n_audio_tokens = special_audio_mask.sum()
    special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
    if inputs_embeds[special_audio_mask].numel() != audio_embeds.numel():
        raise ValueError(
            f"Audio features and audio tokens do not match: tokens: {n_audio_tokens}, features {audio_embeds.shape[0]}"
        )
    inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_embeds)
```

**How the merge works** (see `session3_QA.md` Q7 for detailed walkthrough):

1. `get_audio_features` runs WhisperEncoder → trim → project → returns list of per-audio tensors
2. `torch.cat` flattens them into one stream (same as images on line 1250)
3. Two-path mask creation: integer comparison on `input_ids` (normal), or embedding comparison with `.all(-1)` (when `input_ids` is unavailable)
4. Validation: checks that the number of `<|audio_pad|>` tokens matches the number of audio embedding vectors
5. `masked_scatter` replaces placeholder embeddings with real audio features sequentially

**Two guard `ValueError`s added:**
- `audio_lengths is None` — prevents silent failure if caller forgets lengths
- `audio_token_id is None` — prevents attempting audio masking on a config that never opted into audio

### 2.6 No changes to `get_rope_index`

Audio tokens use `<|audio_start|>` (151657), not `<|vision_start|>` (151652). The `get_rope_index` function only scans for `vision_start_token_id`, so `<|audio_pad|>` tokens are naturally treated as regular text and get 1D sequential positions. This is correct — audio is temporal-only with no spatial dimensions (no height/width grid).

### 2.7 Updated `_checkpoint_conversion_mapping` regex

**Before:**
```python
_checkpoint_conversion_mapping = {
    "^visual": "model.visual",
    r"^model(?!\.(language_model|visual))": "model.language_model",
}
```

**After:**
```python
_checkpoint_conversion_mapping = {
    "^visual": "model.visual",
    r"^model(?!\.(language_model|visual|audio_encoder|audio_projector))": "model.language_model",
}
```

**Why this is critical**: The regex remaps checkpoint keys. Without the negative lookahead update, `model.audio_encoder.*` would match the `^model` pattern and get remapped to `model.language_model.audio_encoder.*` during save/load. The weights would silently move to the wrong location — the audio encoder would load with random weights while the saved weights sit under `language_model` where nothing reads them. This would be a **silent weight loss bug** — no error, just a broken model.

### 2.8 Modified `Qwen2VLForConditionalGeneration.forward()`

Added `audio_features` and `audio_lengths` to the method signature, and passed both through to `self.model(...)`:

```python
outputs = self.model(
    input_ids=input_ids,
    pixel_values=pixel_values,
    pixel_values_videos=pixel_values_videos,
    image_grid_thw=image_grid_thw,
    video_grid_thw=video_grid_thw,
    audio_features=audio_features,      # NEW
    audio_lengths=audio_lengths,         # NEW
    position_ids=position_ids,
    ...
)
```

This is a pure passthrough — `Qwen2VLForConditionalGeneration` doesn't process audio itself, it delegates to `Qwen2VLModel.forward()` where the actual merge happens.

### 2.9 Modified `prepare_inputs_for_generation()`

Three changes:

**1. Added params to signature:**
```python
audio_features=None,
audio_lengths=None,
```

**2. Passed to `super().prepare_inputs_for_generation()`:**
```python
model_inputs = super().prepare_inputs_for_generation(
    input_ids,
    ...
    audio_features=audio_features,
    audio_lengths=audio_lengths,
    use_cache=use_cache,
    **kwargs,
)
```

**3. Clear after prefill** (added to existing `cache_position[0] != 0` block):
```python
if model_inputs["cache_position"][0] != 0:
    model_inputs["pixel_values"] = None
    model_inputs["pixel_values_videos"] = None
    model_inputs["audio_features"] = None      # NEW
    model_inputs["audio_lengths"] = None        # NEW
```

**Why clearing is essential**: During `model.generate()`, there are two phases: prefill (process full input including audio) and decode (generate tokens one at a time using KV cache). Without clearing, the WhisperEncoder (32 layers, ~635M params) would re-run on every decode token — 100 redundant forward passes for a 100-token generation. The audio embeddings are already baked into the KV cache from prefill. See `session3_QA.md` Q1.

### 2.10 Added TODO comment in `_expand_inputs_for_generation()`

```python
visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]
# TODO: Add audio_features/audio_lengths expansion logic here if beam search is needed for ASR.
# Currently safe to skip — ASR uses greedy/sampling decoding, not beam search.
```

**Why NOT add audio to `visual_keys`**: The `_expand_inputs_for_generation` method handles beam search by splitting visual tensors per-sample and repeating them. Adding `audio_features`/`audio_lengths` to `visual_keys` without implementing the corresponding split/repeat branch would break beam expansion. ASR uses greedy/sampling decoding, not beam search, so this is safe to skip. See `session3_QA.md` Q3.

---

## 3. Documentation: session3_QA.md Updates

Nine Q&A entries total — Q1-Q3 written during plan review, Q4-Q9 added during implementation:

- **Q1**: Why `audio_features` must be cleared after prefill in generation — encoder re-run waste
- **Q2**: dtype/device casting for audio encoder input — float32 CPU → bf16 GPU, `int(length.item())`
- **Q3**: Beam search expansion for audio — why we skip it (greedy/sampling only for ASR)
- **Q4**: Why `audio_token_id` defaults to `None` (not `151658`) — backward compatibility lesson
- **Q5**: WhisperEncoder output shape `(num_audios, 1500, 1280)` — what 1500 and 1280 mean, why we trim
- **Q6**: Why we need both `transformers` and `Qwen2-VL` forks — processor vs utility layer split
- **Q7** (user-contributed): How `masked_scatter` finds audio placeholder tokens — two-path mask creation, `unsqueeze`/`expand_as`, `torch.cat` before scatter
- **Q8** (user-contributed): What `masked_scatter` is and how it works — sequential consumption of source tensor
- **Q9** (user-contributed): What `_checkpoint_conversion_mapping` is and why we updated it — regex negative lookahead for audio weights

---

## 4. Session 3 Status (Parts 1-2 Complete)

### Completed

| Step | Description | Status |
|------|-------------|--------|
| Part 1.1 | Add `Qwen2VLAudioConfig` class | Done |
| Part 1.2 | Modify `Qwen2VLConfig` (audio_config, audio_token_id) | Done |
| Part 1.3 | Update `__all__` | Done |
| Part 2.1 | Update import | Done |
| Part 2.2 | `Qwen2VLModel.__init__` — audio_encoder + audio_projector | Done |
| Part 2.3 | `get_audio_features` method | Done |
| Part 2.4 | `Qwen2VLModel.forward()` signature | Done |
| Part 2.5 | Audio merge block (masked_scatter) | Done |
| Part 2.6 | `_checkpoint_conversion_mapping` regex fix | Done |
| Part 2.7 | `Qwen2VLForConditionalGeneration.forward()` passthrough | Done |
| Part 2.8 | `prepare_inputs_for_generation()` — params + prefill clearing | Done |
| Part 2.9 | `_expand_inputs_for_generation()` — TODO comment | Done |

### Remaining (not started yet)

| Step | Description |
|------|-------------|
| Part 3 | Build Notebook 03 (model init, weight loading, forward pass test, push to HF) |
| Part 4 | Commit fork changes + notebook + docs, push to GitHub |

### Files Modified

| File | Action | Description |
|------|--------|-------------|
| `forks/transformers/.../configuration_qwen2_vl.py` | Modified | Added `Qwen2VLAudioConfig`, extended `Qwen2VLConfig` |
| `forks/transformers/.../modeling_qwen2_vl.py` | Modified | Added audio_encoder, audio_projector, forward() merge, generation support |
| `Documentation/Lessons/session3_QA.md` | Modified | Added Q4, Q5, Q7 |
| `Documentation/Session3_Progress_20260221.md` | Created | This file |
