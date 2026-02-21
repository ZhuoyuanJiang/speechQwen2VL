# Session 3 Plan: Model Architecture Modifications + Notebook 03

## Context

Sessions 1-2 complete: project setup, data exploration, tokenizer/processor modifications, fork repos with audio support in `processing_qwen2_vl.py` and `vision_process.py`. Session 3 adds the audio encoder and projector to the Qwen2-VL model so it can process audio embeddings, then builds Notebook 03 to initialize and push the complete model.

**Goal**: Extend Qwen2-VL's model to accept audio input by adding a Whisper encoder + MLP projector, initialize with pretrained weights, push to HuggingFace.

Development happens on Mac (code editing only). Testing happens on server (A6000, 48GB).

## Summary

1. **Modify config** (`configuration_qwen2_vl.py`) — Add `Qwen2VLAudioConfig` class with Whisper dimensions, extend `Qwen2VLConfig` with `audio_config` and `audio_token_id`
2. **Modify model** (`modeling_qwen2_vl.py`) — Add `audio_encoder` (WhisperEncoder) and `audio_projector` (2-layer MLP) to `Qwen2VLModel`, extend `forward()` to scatter audio embeddings at `<|audio_pad|>` positions, extend `Qwen2VLForConditionalGeneration` to pass audio params through
3. **Build Notebook 03 (8 sections)** — Load Qwen2-VL-7B base weights + Whisper encoder weights, verify forward pass with audio, push complete model to HF (`DanJZY/Qwen2-VL-7B-Speech`)
4. **Commit & push** — Fork changes to transformers fork, notebook + docs to main repo

---

## Architecture Context: How Qwen2-VL Processes Vision (and how audio will follow the same pattern)

Understanding the existing vision pipeline is essential for implementing audio support. Here's how the model is structured and how vision embeddings flow through it.

### Model class hierarchy

```
Qwen2VLForConditionalGeneration       (has lm_head, calls self.model)
└── model: Qwen2VLModel               (multimodal fusion happens here)
    ├── visual: Qwen2VisionTransformerPretrainedModel   (ViT encoder)
    ├── language_model: Qwen2VLTextModel                (transformer LLM)
    ├── audio_encoder: WhisperEncoder                   ← NEW (Session 3)
    └── audio_projector: nn.Sequential                  ← NEW (Session 3)
```

`Qwen2VLForConditionalGeneration` (line 1273) is the top-level class used for inference/training. It wraps `Qwen2VLModel` (line 929) and adds `lm_head`. The multimodal fusion logic (embedding merge) lives in `Qwen2VLModel.forward()`.

### How vision embeddings get merged into the token sequence

The forward pass in `Qwen2VLModel.forward()` (line 1178) works as follows:

1. **Text embedding**: `inputs_embeds = self.get_input_embeddings()(input_ids)` — converts token IDs to embeddings. At this point, `<|image_pad|>` and `<|video_pad|>` positions have placeholder embeddings.

2. **Image embedding merge** (lines 1215-1221):
   ```python
   image_embeds = self.get_image_features(pixel_values, image_grid_thw)
   image_embeds = torch.cat(image_embeds, dim=0)
   image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_embeds)
   inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
   ```
   - `get_image_features` runs the ViT encoder on pixel_values → embeddings of shape `(total_patches, 3584)`
   - `get_placeholder_mask` creates a boolean mask where `input_ids == image_token_id` (151655)
   - `masked_scatter` replaces the placeholder embeddings with the actual image embeddings

3. **Video embedding merge** (lines 1223-1229): Same pattern with `video_token_id` (151656).

4. **RoPE position calculation** (line 1231-1247): `get_rope_index()` computes 3D position IDs (temporal, height, width). Vision tokens get 3D spatial positions; text tokens get 1D sequential positions (same value across all 3 dimensions).

5. **Language model forward**: The merged `inputs_embeds` (with vision embeddings injected) goes through the transformer layers.

### get_placeholder_mask (line 1137)

Creates boolean masks for image and video token positions. Takes `input_ids` (or `inputs_embeds` if input_ids is None) and returns `(special_image_mask, special_video_mask)`. Validates that the number of placeholder tokens matches the number of feature vectors. We will add audio masking using the same pattern directly in `forward()`.

### get_rope_index (line 954)

Computes 3D rotary position embeddings. Key behavior:
- Scans for `vision_start_token_id` (151652) to find where vision patches are
- Assigns 3D positions to image/video tokens (temporal, height, width grids)
- Text tokens get 1D positions (same value for all 3 dimensions)
- Returns `position_ids` shape `(3, batch_size, seq_len)` and `mrope_position_deltas`

**Audio tokens do NOT need changes here**: `<|audio_pad|>` (151658) is not preceded by `<|vision_start|>` (151652), so the function treats audio tokens as regular text with 1D sequential positions. This is correct — audio is temporal-only with no spatial dimensions.

### WhisperEncoder architecture

- **Input**: mel spectrograms `(batch, 128, 3000)` — 128 mel bins, 3000 time frames
- **Processing**: conv1 (128→1280, kernel=3) → conv2 (1280→1280, kernel=3, stride=2) → 32 transformer layers
- **Output**: `(batch, 1500, 1280)` — 1500 time steps (3000/2 from stride), 1280-dim embeddings
- The encoder always outputs 1500 steps regardless of actual audio length (Whisper pads/truncates to 30s). We trim to `audio_lengths[i]` steps per audio before projecting.

### Audio projector

- **Input**: `(length, 1280)` from WhisperEncoder
- **Architecture**: `Linear(1280→3584) → GELU → Linear(3584→3584)`
- **Output**: `(length, 3584)` — matches text hidden size, ready for `masked_scatter`
- 3584 is the text model's hidden_size for Qwen2-VL-7B (same as vision_config.hidden_size)

### _checkpoint_conversion_mapping (critical detail)

`Qwen2VLForConditionalGeneration` has a regex mapping (line 1274-1277):
```python
r"^model(?!\.(language_model|visual))": "model.language_model"
```
This remaps checkpoint keys: anything starting with `model.` (except `model.language_model` and `model.visual`) gets remapped to `model.language_model.*`. Without updating this regex, `model.audio_encoder.*` weights would be incorrectly remapped to `model.language_model.audio_encoder.*` on save/load, causing silent weight loss. We must add `|audio_encoder|audio_projector` to the negative lookahead.

---

## Part 1: Fork Modification — `configuration_qwen2_vl.py`

**File**: `forks/transformers/src/transformers/models/qwen2_vl/configuration_qwen2_vl.py`

### 1.1 Add `Qwen2VLAudioConfig` class (new, before `Qwen2VLConfig`)

Stores Whisper encoder dimensions. Mirrors `Qwen2VLVisionConfig` pattern. Defaults match whisper-large-v3-turbo.

```python
class Qwen2VLAudioConfig(PretrainedConfig):
    model_type = "qwen2_vl"
    base_config_key = "audio_config"

    def __init__(self, d_model=1280, encoder_layers=32, encoder_attention_heads=20,
                 encoder_ffn_dim=5120, num_mel_bins=128, max_source_positions=1500,
                 encoder_layerdrop=0.0, activation_function="gelu",
                 scale_embedding=False, **kwargs):
```

Includes `to_whisper_config()` method to convert to `WhisperConfig` for the `WhisperEncoder` constructor.

### 1.2 Modify `Qwen2VLConfig` class

- ~~Add `"audio_config": Qwen2VLAudioConfig` to `sub_configs`~~ **Revised during implementation**: audio_config must NOT be in `sub_configs` because it can be `None`, and the framework's `to_diff_dict()` crashes on None sub-configs. Deserialization is handled manually in `__init__` instead. See session3_QA.md Q10.
- Add `audio_config=None` and `audio_token_id=None` parameters to `__init__`
- Handle deserialization: dict → `Qwen2VLAudioConfig`, None stays None (backward compat)
- Update `__all__` to include `Qwen2VLAudioConfig`

---

## Part 2: Fork Modification — `modeling_qwen2_vl.py`

**File**: `forks/transformers/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py`

### 2.1 `Qwen2VLModel.__init__` (~line 933)

Conditionally add audio components after `self.language_model`:

```python
if getattr(config, 'audio_config', None) is not None:
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

### 2.2 Add `get_audio_features` method to `Qwen2VLModel`

Parallel to `get_image_features` (line 1126). Runs encoder + projector, trims to actual lengths. Must cast `audio_features` to encoder dtype/device before forward (same pattern as `pixel_values.type(self.visual.dtype)` on line 1131), and use `int(length.item())` for tensor indexing to avoid CUDA scalar warnings:

```python
def get_audio_features(self, audio_features, audio_lengths):
    audio_features = audio_features.to(device=self.audio_encoder.device, dtype=self.audio_encoder.dtype)
    encoder_output = self.audio_encoder(audio_features)
    audio_hidden = encoder_output.last_hidden_state  # (num_audios, 1500, 1280)
    audio_embeds = []
    for i, length in enumerate(audio_lengths):
        length = int(length.item()) if hasattr(length, 'item') else int(length)
        trimmed = audio_hidden[i, :length, :]      # (length, 1280)
        projected = self.audio_projector(trimmed)   # (length, 3584)
        audio_embeds.append(projected)
    return audio_embeds
```

### 2.3 Modify `Qwen2VLModel.forward()` (~line 1178)

- Add `audio_features` and `audio_lengths` parameters
- Add audio embedding merge block after video merge (~line 1229), using same `masked_scatter` pattern
- Include placeholder-count validation before scatter (same pattern as `get_placeholder_mask` lines 1163-1166):

```python
if audio_features is not None:
    if audio_lengths is None:
        raise ValueError("audio_features provided but audio_lengths is None")
    audio_embeds = self.get_audio_features(audio_features, audio_lengths)
    audio_embeds = torch.cat(audio_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    # Create mask
    if input_ids is None:
        special_audio_mask = (inputs_embeds == self.get_input_embeddings()(
            torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
        )).all(-1)
    else:
        special_audio_mask = input_ids == self.config.audio_token_id
    # Validate token count matches feature count
    n_audio_tokens = special_audio_mask.sum()
    special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
    if inputs_embeds[special_audio_mask].numel() != audio_embeds.numel():
        raise ValueError(
            f"Audio features and audio tokens do not match: tokens: {n_audio_tokens}, features {audio_embeds.shape[0]}"
        )
    inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_embeds)
```

### 2.4 No changes to `get_rope_index`

Audio tokens use `<|audio_start|>` (ID 151657), not `<|vision_start|>` (ID 151652). The function scans for `vision_start_token_id` only, so `<|audio_pad|>` tokens are naturally treated as text and get 1D sequential positions — which is correct for temporal-only audio (no spatial dimensions).

### 2.5 Modify `Qwen2VLForConditionalGeneration`

**`_checkpoint_conversion_mapping`** (~line 1274): Update regex to exclude audio components:
```python
r"^model(?!\.(language_model|visual|audio_encoder|audio_projector))": "model.language_model"
```
Without this, saved audio weights would be incorrectly remapped to `model.language_model.audio_encoder.*` on loading.

**`forward()`** (~line 1318): Add `audio_features` and `audio_lengths` params, pass to `self.model(...)`.

**`prepare_inputs_for_generation()`** (~line 1421): Add `audio_features` and `audio_lengths` params, pass to `super()`. Also clear audio after prefill (line ~1490), same pattern as images/videos:
```python
if model_inputs["cache_position"][0] != 0:
    model_inputs["pixel_values"] = None
    model_inputs["pixel_values_videos"] = None
    model_inputs["audio_features"] = None      # NEW
    model_inputs["audio_lengths"] = None        # NEW
```
Without this, the WhisperEncoder (32 layers) would re-run on every decode token during generation.

**`_expand_inputs_for_generation()`** (~line 1547): Do NOT add `audio_features`/`audio_lengths` to `visual_keys` — adding them without implementing the per-sample split/repeat branch would break beam expansion. Instead, add a `TODO` comment in the code noting that audio expansion logic is needed if beam search is used later. Safe to skip for now since ASR uses greedy/sampling decoding.

---

## Part 3: Build Notebook 03

**File**: `notebooks/03_model_architecture.ipynb`
**Maps to**: skeleton cells 44-50
**Where to run**: Server (A6000, 48GB) — peak ~18.5GB GPU memory

### Sections:

1. **Setup & Imports** — pip install from fork commit hashes, imports
2. **Load & modify config** — Load `Qwen/Qwen2-VL-7B-Instruct` config, add `Qwen2VLAudioConfig()` + `audio_token_id=151658`
3. **Create model & load base weights** — Create model from modified config (audio components random), load Qwen2-VL-7B state dict with `strict=False`
4. **Load Whisper encoder weights** — Load whisper-large-v3-turbo, copy encoder state_dict into `model.model.audio_encoder`, delete Whisper
5. **Verify model structure** — Print architecture, count params (total ~7.6B, audio_encoder ~635M, audio_projector ~17M)
6. **Test forward pass** — Use processor from NB02 with an audio sample, run `model(**batch)`, verify output shape
7. **Save & push to HuggingFace** — `model.save_pretrained()` + upload to `DanJZY/Qwen2-VL-7B-Speech`
8. **Cleanup**

### Weight initialization:

| Component | Source | Status |
|-----------|--------|--------|
| visual (ViT) | Qwen2-VL-7B-Instruct | Pretrained |
| language_model (LLM) | Qwen2-VL-7B-Instruct | Pretrained |
| lm_head | Qwen2-VL-7B-Instruct | Pretrained |
| audio_encoder | whisper-large-v3-turbo | Pretrained |
| audio_projector | Random init | Random (trained in Session 5) |

### Memory estimate:
- Qwen2-VL-7B in bf16: ~14GB
- Audio encoder (635M params) in bf16: ~1.3GB
- Audio projector (~17M params): ~35MB
- Whisper-turbo temporary load (float32): ~3.2GB
- **Peak**: ~18.5GB, **after cleanup**: ~15.3GB

---

## Part 4: Commit & Push

1. Commit fork changes (`configuration_qwen2_vl.py` + `modeling_qwen2_vl.py`) to transformers fork, push
2. Commit notebook + docs to main repo, push
3. Create `Documentation/Session3_Progress.md`

---

## Files Modified/Created

| File | Action | Description |
|------|--------|-------------|
| `forks/transformers/.../configuration_qwen2_vl.py` | Modify (fork) | Add `Qwen2VLAudioConfig`, extend `Qwen2VLConfig` |
| `forks/transformers/.../modeling_qwen2_vl.py` | Modify (fork) | Add audio_encoder, audio_projector, extend forward() |
| `notebooks/03_model_architecture.ipynb` | Create | Model init notebook |
| `Documentation/Session3_Progress.md` | Create | Session record |

## Verification

1. **Config round-trip**: Save config with `audio_config`, reload, verify it deserializes correctly
2. **Model structure**: `model.model.audio_encoder` and `model.model.audio_projector` exist
3. **Weight loading**: Base Qwen2-VL weights load (only audio components as missing keys), Whisper encoder weights transfer with 0 missing/unexpected keys
4. **Forward pass**: `model(input_ids=..., audio_features=..., audio_lengths=...)` → logits tensor, no errors
5. **Save/load round-trip**: Save model, reload from saved path, verify audio components persist with correct weights
