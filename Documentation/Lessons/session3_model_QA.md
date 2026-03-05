# Session 3 — Model Architecture Q&A (modeling_qwen2_vl.py)

## Q1: How many classes and how are they nested?

**File**: `modeling_qwen2_vl.py` — 18 classes total, but the architecture is a **2-layer nesting**:

### The nesting (from actual code)

```
Qwen2VLForConditionalGeneration          ← the ONLY class users touch
│
├── self.model  (Qwen2VLModel)           ← fusion layer: combines vision + audio + text
│   │
│   ├── self.visual (ViT)                ← vision encoder (image/video → embeddings)
│   ├── self.audio_encoder (Whisper)     ← audio encoder (mel → embeddings)
│   ├── self.audio_projector (Linear)    ← dimension alignment for audio
│   └── self.language_model (Qwen2VLTextModel)  ← the LLM (text decoder)
│
└── self.lm_head  (Linear)              ← hidden states → vocabulary logits
```

### Just 2 layers of nesting

**Layer 1: `Qwen2VLForConditionalGeneration`** (line 1341)
```python
def __init__(self, config):
    self.model = Qwen2VLModel(config)          # the fusion model
    self.lm_head = nn.Linear(3584, 152064)     # hidden_size → vocab_size
```
This is a thin wrapper. It holds `Qwen2VLModel` + `lm_head`. The `lm_head` converts the LLM's hidden states into a probability over all tokens in the vocabulary (so the model can predict the next word).

**Layer 2: `Qwen2VLModel`** (line 929)
```python
def __init__(self, config):
    self.visual = ViT(config.vision_config)            # vision encoder
    self.language_model = Qwen2VLTextModel(config.text_config)  # LLM
    self.audio_encoder = WhisperEncoder(whisper_config) # audio encoder
    self.audio_projector = nn.Sequential(...)           # audio projection
```
This is where the 4 components live side-by-side. It handles the **fusion** — calling each encoder, then merging everything into one sequence for the LLM.

### What does "fusion" mean concretely?

`Qwen2VLModel.forward()` does this:

```
Step 1: Encode vision    →  image_embeds = self.visual(pixel_values)
Step 2: Encode audio     →  audio_embeds = self.audio_encoder(audio_features)
Step 3: Project audio    →  audio_embeds = self.audio_projector(audio_embeds)
Step 4: Text embedding   →  text_embeds = self.language_model.embed_tokens(input_ids)
Step 5: Fuse (masked_scatter) →  replace <|image_pad|> with image_embeds
                                  replace <|audio_pad|> with audio_embeds
Step 6: Run LLM          →  output = self.language_model(fused_embeds)
```

After Step 5, the LLM just sees one long sequence of embeddings — it doesn't know which ones came from images, audio, or text. They're all 3584-dim vectors.

### The 4 components inside Qwen2VLModel

| Component | What it processes | Input | Output dim | Params |
|-----------|------------------|-------|------------|--------|
| `self.visual` (ViT) | Images/videos | pixel patches (1176-dim) | 3584-dim | ~675M |
| `self.audio_encoder` (Whisper) | Audio | mel spectrogram (128 bins) | 1280-dim | ~635M |
| `self.audio_projector` | Audio (dim alignment) | 1280-dim | 3584-dim | ~25M |
| `self.language_model` (LLM) | Fused sequence | 3584-dim embeddings | 3584-dim | ~7B |

Note: ViT outputs 3584-dim directly (PatchMerger handles this). Audio outputs 1280-dim, so it needs the projector to match the LLM's 3584-dim. That's why there's a projector for audio but not for vision.

### Where do the other 14 classes fit?

They're **internal building blocks** of the 4 main components:

```
self.visual (ViT) is built from:
  ├── PatchEmbed           — Conv3D, pixels → vectors
  ├── VisionRotaryEmbedding — 2D RoPE for ViT
  ├── VisionBlock × 32     — ViT transformer layers
  │   ├── VisionAttention   — attention inside ViT
  │   └── VisionMlp         — FFN inside ViT
  └── PatchMerger          — 4 patches → 1 token

self.language_model (LLM) is built from:
  ├── Embedding            — token IDs → vectors
  ├── Qwen2VLDecoderLayer × 28  — LLM transformer layers
  │   ├── Qwen2VLAttention  — GQA attention (with 3D RoPE)
  │   └── Qwen2MLP          — FFN inside LLM
  └── Qwen2RMSNorm         — final normalization

Other utility classes:
  ├── Qwen2VLRotaryEmbedding     — 3D RoPE (used by LLM attention)
  ├── Qwen2VLPreTrainedModel     — base class (weight init, from_pretrained)
  ├── Qwen2VLModelOutputWithPast — output container for Qwen2VLModel
  └── Qwen2VLCausalLMOutputWithPast — output container for ForConditionalGeneration
```

### What are output containers?

Not part of the model — just **data classes** that package multiple return values into one object:

```python
# Without output container — messy tuple:
return logits, loss, hidden_states, attentions, past_key_values

# With output container — clean named fields:
return Qwen2VLCausalLMOutputWithPast(
    loss=loss,
    logits=logits,
    past_key_values=past_key_values,
)

# Usage:
output = model(**inputs)
output.logits   # predicted token probabilities
output.loss     # training loss (if labels provided)
```

Two containers because two levels of nesting:
- `Qwen2VLModelOutputWithPast` — returned by `Qwen2VLModel` (has hidden_states, no logits)
- `Qwen2VLCausalLMOutputWithPast` — returned by `ForConditionalGeneration` (has logits + loss)

## Q2: What are `output_attentions`, `output_hidden_states`, and `return_dict`?

**File**: `modeling_qwen2_vl.py` — top of `forward()` method

These are **boolean switches** (True/False), not tensors. The forward method resolves them with a common pattern: use the user's value if provided, otherwise fall back to config defaults.

```python
output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
```

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `output_attentions` | False | Return attention weights from every layer |
| `output_hidden_states` | False | Return hidden states from every layer |
| `return_dict` | True | Return `OutputWithPast` object vs raw tuple |

All three default to off. Only turn on for debugging or visualization:

```python
# Normal usage — don't pass these
output = model(**inputs)

# Debug — want to see attention patterns
output = model(**inputs, output_attentions=True)
output.attentions  # now contains 28 layers of attention weights
```

### What does `output.attentions` actually look like?

When `output_attentions=True`, the result is a **tuple of 28 tensors** (one per layer), each 4D:

```python
output.attentions[0].shape  # (batch, num_heads, seq_len, seq_len)
                            # e.g., (1, 28, 500, 500)
```

The core concept is a 2D matrix (seq_len × seq_len) — "who attends to who":

```
         token_0  token_1  token_2  ...  token_499
token_0  [0.02    0.01     0.15     ...  0.003   ]  ← how much token 0 looks at each other token
token_1  [0.01    0.03     0.08     ...  0.012   ]
...
```

It's 4D because there are multiple copies of this 2D matrix:
- 28 attention heads, each with its own attention pattern
- batch_size samples, each with its own attention pattern

Stacked together: `(batch, 28 heads, seq_len, seq_len)` = 4D.

**Why default off**: 28 layers × `(1, 28, 500, 500)` = GBs of memory. Not needed for normal training/inference — only for analyzing model behavior.

## Q3: What are `input_ids`, `inputs_embeds`, and how does `masked_scatter` fuse vision into text?

**File**: `modeling_qwen2_vl.py` — `Qwen2VLModel.forward()`, the image processing block

### What are input_ids?

The tokenizer converts text into a sequence of integer IDs:

```python
text = "Describe this image: <|image_pad|><|image_pad|><|image_pad|>..."
                                    ↓ tokenizer
input_ids = [8826, 419, 1674, 25, 151658, 151658, 151658, ...]
#           "Describe" "this" "image" ":"  pad     pad     pad
```

`get_input_embeddings()(input_ids)` looks up each ID in an embedding table → 3584-dim vector:

```
input_ids:    [8826,        419,         151658,      151658,      ...]
                ↓            ↓             ↓            ↓
inputs_embeds: [vec_describe, vec_this,    vec_pad,     vec_pad,    ...]
               (3584-dim)    (3584-dim)   (3584-dim)   (3584-dim)
```

At this point the `<|image_pad|>` positions have **meaningless** vectors — they just looked up a random entry in the embedding table, with no image information.

### The goal: replace placeholder vectors with real image embeddings

```
Before (placeholders have no image info):
[vec_describe, vec_this, vec_colon, vec_pad, vec_pad, vec_pad, vec_pad, ...]

After (placeholders replaced with ViT output):
[vec_describe, vec_this, vec_colon, img_tok0, img_tok1, img_tok2, img_tok3, ...]
```

### The 3 steps in code

**Step 1: Run ViT** to get real image embeddings
```python
image_embeds = self.get_image_features(pixel_values, image_grid_thw)
# e.g., one 448×448 image → 256 tokens, each 3584-dim
image_embeds = torch.cat(image_embeds, dim=0)  # stack multiple images
# shape: (total_image_tokens, 3584)
```

**Step 2: Find where `<|image_pad|>` tokens are** in the input sequence
```python
image_mask = self.get_placeholder_mask(input_ids, ...)
# input_ids = [8826, 419, 151658, 151658, 151658, 151658, 102]
# image_mask = [F,    F,   T,      T,      T,      T,      F]
#                          these 4 positions are True
```

**Step 3: `masked_scatter`** — replace True positions with image embeddings, in order
```python
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

How `masked_scatter` works — scan the mask, every time it hits True, consume the next embedding from `image_embeds`:

```
mask:          [F,     F,     T,      T,      T,      T,      F     ]
image_embeds:                 img[0]  img[1]  img[2]  img[3]
                               ↓       ↓       ↓       ↓
inputs_embeds: [vec_d, vec_t, img[0], img[1], img[2], img[3], vec_end]
```

After this, the LLM sees one mixed sequence of text + image embeddings. It doesn't know which came from where — they're all 3584-dim vectors.

Audio uses the exact same pattern: find `<|audio_pad|>` positions → replace with audio embeddings from Whisper + projector.

## Q4: 3D position IDs and rope_deltas — how Qwen2-VL tracks position

**File**: `modeling_qwen2_vl.py` — `Qwen2VLModel.forward()`, Step 4

### Why 3 coordinates per token?

Standard LLMs give each token 1 position number (1D). Qwen2-VL gives each token **3 coordinates (t, h, w)** because images have 2D spatial structure that 1D positions can't express.

Different token types use the 3 axes differently:

```
Text tokens:   only t increments (like standard LLM, h=0, w=0)
  "Hello"  → (t=0, h=0, w=0)
  "world"  → (t=1, h=0, w=0)

Image tokens:  t is shared (same moment), h/w = patch spatial position
  patch(0,0) → (t=2, h=0, w=0)
  patch(0,1) → (t=2, h=0, w=1)
  patch(1,0) → (t=2, h=1, w=0)
  patch(1,1) → (t=2, h=1, w=1)

Video tokens:  all 3 axes vary (t = which frame, h/w = spatial)
  frame 0, patch(0,0) → (t=2, h=0, w=0)
  frame 0, patch(0,1) → (t=2, h=0, w=1)
  frame 1, patch(0,0) → (t=3, h=0, w=0)   ← t increments per frame
  frame 1, patch(0,1) → (t=3, h=0, w=1)
```

Image is essentially 2D (h, w), video is 3D (t, h, w). But all use a unified 3-coordinate system — text and images just set unused axes to 0 or constant.

`position_ids` shape: `(3, batch, seq_len)` — 3 tracks instead of 1.

### What is rope_deltas?

**Problem**: An image with 256 patches takes 256 positions in the sequence, but temporally it's just **one moment** (1 step, not 256 steps).

Example input: `"Hi <image with 256 patches> OK"`

```
Sequence:       [Hi,  img0, img1, ... img255, OK]
Sequence index:   0    1     2    ...  256    257   ← 258 positions in total

Temporal (t):    [0,   1,    1,   ...  1,     2  ]
                       ↑ all 256 patches share t=1    ↑ "OK" is t=2
```

The image occupies 256 sequence positions but only advances t by 1 step:
```
Sequence positions consumed by image: 256
Temporal steps consumed by image:     1
Gap: 256 - 1 = 255 "inflated" positions

rope_deltas = actual_t_advance - sequence_positions = 1 - 256 = -255
```

`rope_deltas` remembers this offset: "the image inflated the sequence by 255 extra positions that don't count as time advancement."

### Two cases in the code

**Case 1: Prefill (first forward pass)** — compute everything from scratch

```python
position_ids, rope_deltas = self.get_rope_index(
    input_ids, image_grid_thw, video_grid_thw, attention_mask
)
self.rope_deltas = rope_deltas  # cache for later
```

`get_rope_index` scans the input, identifies text/image/video tokens, and assigns correct (t, h, w) to each. This is the most complex function — we'll read it later.

**Case 2: Generation (subsequent tokens)** — use cached delta

When generating the 258th token (first new word after "OK"):

```python
cache_position[0] = 258           # new token's absolute sequence index
rope_deltas = -255                 # cached from prefill
position_t = 258 + (-255) = 3     # correct t coordinate for new token
```

Without rope_deltas, the model would think t=258 (as if 258 time steps passed). With rope_deltas, it knows t=3 (because the image's 256 positions only counted as 1 time step).

**One-liner**: `rope_deltas` = "how much the image inflated the sequence without advancing time." Cache it once at prefill, reuse during generation.

## Q5: What does the language model receive in Step 5?

**File**: `modeling_qwen2_vl.py` — `Qwen2VLModel.forward()`, the `self.language_model(...)` call

By Step 5, all fusion is done. The language model just sees a sequence of vectors — it doesn't know which came from text, images, or audio.

### Each argument with concrete example

Input: `"Hi <256 image patches> OK"` (259 tokens total, batch_size=1)

**`input_ids=None`** — intentionally None. We already fused vision/audio into `inputs_embeds`. If we pass `input_ids`, the LLM would re-lookup the embedding table and overwrite our fused embeddings.

**`inputs_embeds`** — the fused sequence, shape `(1, 259, 3584)`:
```
[vec_Hi, img[0], img[1], ... img[255], vec_OK]
 ↑ from      ↑ from ViT output          ↑ from
 embed table                              embed table
All 3584-dim. LLM can't tell which is text vs image.
```

**`position_ids`** — 3D coordinates, shape `(3, 1, 259)`:
```
t: [0, 1, 1, 1, ..., 1, 2]     ← image patches share t=1
h: [0, 0, 0, 0, ..., 15, 0]    ← spatial row of each patch
w: [0, 0, 1, 0, ..., 15, 0]    ← spatial col of each patch
```

**`attention_mask`** — which positions exist, shape `(1, 259)`:
```
[1, 1, 1, ..., 1]    ← all 1s (no padding in this example)
```
Only needs 1 axis (not 3) because "does this token exist" doesn't depend on t/h/w.

**`past_key_values`** — KV cache:
```
Prefill:    None (first time, no cache yet)
Generation: contains K/V from all previous tokens
```

**`cache_position`** — absolute index of current token(s):
```
Prefill:    [0, 1, 2, ..., 258]   ← all positions
Generation: [259]                  ← just the new token
```

### Why batch is the 2nd dimension in position_ids?

```
position_ids shape: (3, batch, seq_len)
```

The 3 axes (t, h, w) are first so you can index by axis:
- `position_ids[0]` → all t values for all samples
- `position_ids[1]` → all h values for all samples
- `position_ids[2]` → all w values for all samples

This is a design choice for Qwen2-VL's 3D RoPE — it needs to process each axis separately when computing rotary embeddings. Standard LLMs with 1D positions use shape `(batch, seq_len)` with batch first.

### KV cache shape explained

Each layer stores one (K, V) pair. 28 layers total:

```python
K_layer0.shape = (batch, num_kv_heads, seq_len, head_dim)
#                  1       4            259      128
```

What each dimension means — think of it as 259 tokens, each with a 128-dim key vector, repeated across 4 heads:

```
                     head 0          head 1          head 2          head 3
token "Hi"      [128-dim key]   [128-dim key]   [128-dim key]   [128-dim key]
token "img[0]"  [128-dim key]   [128-dim key]   [128-dim key]   [128-dim key]
token "img[1]"  [128-dim key]   [128-dim key]   [128-dim key]   [128-dim key]
...
token "OK"      [128-dim key]   [128-dim key]   [128-dim key]   [128-dim key]

259 rows × 4 heads × 128 dims = shape (1, 4, 259, 128)
```

**Where does 128 come from?** The LLM's hidden_size (3584) is split across attention heads. Qwen2-VL has 28 query heads:
```
head_dim = hidden_size / num_attention_heads = 3584 / 28 = 128
```

Each key is a 128-dim vector — a compressed representation of one token that other tokens use to compute attention scores.

**Why 4 heads, not 28?** This is GQA (Grouped Query Attention): 28 query heads but only 4 KV heads. Every 7 query heads share the same K/V. Saves memory (4 instead of 28 sets of cached K/V) with minimal quality loss.

**How the cache grows during generation:**
```
After prefill (259 tokens):     K.shape = (1, 4, 259, 128)
After generating 1st new token: K.shape = (1, 4, 260, 128)
After generating 2nd new token: K.shape = (1, 4, 261, 128)
```

Each new token only computes its own K/V (1 row), then appends to the cache. Without caching, every new token would recompute K/V for all previous tokens.

## Q6: `Qwen2VLForConditionalGeneration.__init__` — lm_head, weight tying, post_init

**File**: `modeling_qwen2_vl.py` line 1341

This is the outermost class. It's very thin — just two things:

```python
def __init__(self, config):
    self.model = Qwen2VLModel(config)   # the fusion model (ViT + Whisper + LLM)
    self.lm_head = nn.Linear(3584, 152064, bias=False)  # predict next token
```

### What is lm_head?

The LLM's final step — converts a hidden state vector into "which word comes next":

```
LLM output: hidden_state = [0.12, -0.34, ..., 0.78]       ← 3584-dim
                           ↓ lm_head (Linear)
logits:     [0.01, 0.002, ..., 5.23, ..., 0.003]           ← 152064-dim
                                ↑
                           "hello" gets highest score → predict "hello"
```

152064 = vocabulary size (how many tokens Qwen2-VL knows).

### Weight tying — why embedding and lm_head share the same matrix

Both have the exact same shape: `(152064, 3584)` = `(vocab_size, hidden_dim)`.

They do opposite operations with the same matrix:

```
Embedding (forward lookup): token ID → pick row → vector
   "hello" = ID 8826
   → take row 8826 → [0.5, 0.3, -0.1, ..., 0.8]  (3584-dim)

lm_head (reverse dot product): vector → dot product with every row → scores
   hidden = [0.12, -0.34, ..., 0.78]  (3584-dim)
   → dot with row 0     = 0.01    ← "the" score
   → dot with row 8826  = 5.23    ← "hello" score (highest!)
   → ...152064 scores total
```

**Why sharing makes sense**: A word's "input representation" and "output representation" should be the same thing. If `[0.5, 0.3, ...]` means "hello" when reading, it should also mean "hello" when predicting. Two separate matrices would learn two inconsistent representations — wasteful and less coherent.

```
Without tying: embedding says "hello" = [0.5, 0.3, ...]
               lm_head says "hello"   = [0.9, -0.1, ...]   ← inconsistent!

With tying:    both use [0.5, 0.3, ...] for "hello"         ← consistent
```

Saves ~2GB memory (one copy of 152064 × 3584 instead of two).

### What is post_init()?

A standard HuggingFace `PreTrainedModel` method called at the end of every `__init__`. Every model class has it:

```
line 791:  Qwen2VLTextModel.__init__()              → self.post_init()
line 952:  Qwen2VLModel.__init__()                  → self.post_init()
line 1353: Qwen2VLForConditionalGeneration.__init__() → self.post_init()
```

It does:
1. Initialize weights (random normal with `initializer_range=0.02`) — only matters when training from scratch; `from_pretrained()` overwrites these with checkpoint values
2. Set up gradient checkpointing and other training config

Not Qwen2-VL specific — every HuggingFace model has this. Framework boilerplate.

### What is `_checkpoint_conversion_mapping`?

Renames parameter keys when loading old-format checkpoints:

```
Old checkpoint key:              New code expects:
"visual.blocks.0.weight"    →   "model.visual.blocks.0.weight"
"model.layers.0.weight"     →   "model.language_model.layers.0.weight"
```

Backward compatibility only. Not important for understanding the architecture.

## Q7: Why do the one-liner methods exist in ForConditionalGeneration?

**File**: `modeling_qwen2_vl.py` — `get_image_features`, `get_input_embeddings`, `visual` property, etc.

These are all **forwarding methods** — they just pass the call to the inner `self.model` (Qwen2VLModel). No logic, pure syntactic sugar.

**Why they exist**: The nesting creates an awkward double `.model`:

```python
model = Qwen2VLForConditionalGeneration.from_pretrained(...)

# Without forwarding — have to write .model.model:
model.model.visual                           # ViT
model.model.language_model                   # TextModel
model.model.get_image_features(pixels, grid) # image encoding

# With forwarding — cleaner:
model.visual                                 # same ViT
model.language_model                         # same TextModel
model.get_image_features(pixels, grid)       # same result
```

The double `.model` happens because both layers use `model` as the attribute name:

```
model                          = ForConditionalGeneration instance
model.model                    = Qwen2VLModel instance (self.model in ForCondGen)
model.model.visual             = ViT instance
model.model.language_model     = TextModel instance
```

The `@property` and one-liner methods just save users from writing `model.model.X` everywhere. Skip these when reading — no architectural significance.

## Q8: How does the loss function work in ForConditionalGeneration?

**File**: `modeling_qwen2_vl.py` — `Qwen2VLForConditionalGeneration.forward()`, Step 3

### Next-token prediction

The model is trained to predict the next token at every position:

```
Input:   [Hi,    img0, img1, ..., img255, OK,     how  ]
Labels:  [img0,  img1, img2, ..., OK,     how,    are  ]
          ↑ each position's "correct answer" is the next token
```

### Cross-entropy loss

`logits` has 152064 scores (one per word in vocabulary). The correct answer is one specific word. Loss measures "how high did the correct answer score?":

```
logits = [0.01, 0.003, ..., 5.23, ..., 0.002]
                              ↑
                         "OK" scored 5.23 (high) → loss small ✓

logits = [0.01, 0.003, ..., 0.12, ..., 0.002]
                              ↑
                         "OK" scored 0.12 (low)  → loss large ✗
```

### Why pass vocab_size?

Cross-entropy needs to know how many classes there are — like a multiple-choice test needs to know how many options:
```
152064-way classification: pick 1 correct token out of 152064 candidates
```

In practice PyTorch can infer this from logits shape. The parameter is mainly for safety checks and edge cases (e.g., truncated logits).

### What does -100 mean in labels?

PyTorch convention: positions with label = -100 are **excluded from loss**:

```
Labels:  [img0, img1, ..., OK,   -100, -100]
                             ↑     ↑     ↑
                          counted  skip  skip
```

Use cases for -100:
- Image/audio token positions (model shouldn't predict these)
- Padding positions (don't exist, shouldn't count)
- Prompt tokens (only want loss on the model's response, not the question)

## Q9: What does `prepare_inputs_for_generation` do?

**File**: `modeling_qwen2_vl.py` — `Qwen2VLForConditionalGeneration.prepare_inputs_for_generation()`

### Where is `.generate()`?

Not in this file. It's inherited from `GenerationMixin`:

```python
class Qwen2VLForConditionalGeneration(Qwen2VLPreTrainedModel, GenerationMixin):
#                                                              ↑ generate() lives here
```

`generate()` is a loop that calls `prepare_inputs_for_generation()` + `forward()` for each new token:

```python
# Simplified generate() pseudocode:
def generate(self, input_ids, max_new_tokens=100):
    for step in range(max_new_tokens):
        model_inputs = self.prepare_inputs_for_generation(input_ids, ...)
        outputs = self.forward(**model_inputs)
        next_token = select_token(outputs.logits)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    return input_ids
```

### What `prepare_inputs_for_generation` decides

It prepares **what to pass to forward() at each step**. Different steps need different inputs:

**Step 1 (prefill) — process everything:**
```python
{
    "input_ids": [Hi, pad, pad, ..., pad],      # all 258 tokens
    "pixel_values": image pixel data,            # for ViT
    "image_grid_thw": [[1, 32, 32]],             # for ViT
    "position_ids": full 3D positions,            # computed from scratch
    "cache_position": [0, 1, 2, ..., 257],       # all positions
}
→ forward() runs ViT, processes everything, predicts "a"
→ KV cache now stores K/V for all 258 tokens
```

**Step 2 — only the new token "a":**
```python
{
    "input_ids": [a],                            # just 1 new token
    "pixel_values": None,                        # cleared! no need to rerun ViT
    "position_ids": computed from rope_deltas,    # quick, no full recompute
    "cache_position": [258],                     # just the new position
}
→ forward() processes only 1 token, uses KV cache for previous 258
→ predicts "cute"
```

**Step 3 — only the new token "cute":**
```python
{
    "input_ids": [cute],
    "pixel_values": None,
    "cache_position": [259],
}
→ predicts "cat"
```

### Why set pixel_values to None?

Because `Qwen2VLModel.forward()` has an `if` check:

```python
if pixel_values is not None:                    # ← this check
    image_embeds = self.get_image_features(...)  # runs ViT (expensive!)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

- `pixel_values = image data` → `if` is True → ViT runs (slow)
- `pixel_values = None` → `if` is False → entire block skipped (fast)

```
Step 1: pixel_values = image data  → ViT runs → embeddings enter KV cache
Step 2: pixel_values = None        → ViT skipped → uses KV cache instead
Step 3: pixel_values = None        → ViT skipped → uses KV cache instead
```

Without setting None, ViT (32 transformer layers) would re-run on every single generated token — wasting compute for identical results. Same applies to audio: Whisper encoder would re-run if audio_features isn't cleared.

This is exactly what Q1 in session3_QA.md describes: we added `audio_features = None` clearing for our speech modification, following the same pattern as images/videos.

## Q10: Summary — the two outer model classes

### Qwen2VLForConditionalGeneration (outermost)

Two things:
- `self.model` = Qwen2VLModel (the fusion model)
- `self.lm_head` = Linear(3584 → 152064), converts hidden states to vocabulary probabilities

Forward does 3 steps: call self.model → lm_head for logits → compute loss

Extra responsibility: `prepare_inputs_for_generation` controls what `generate()` passes at each step (prefill passes image data, subsequent steps pass None).

### Qwen2VLModel (fusion layer)

Four components:
```
self.visual          — ViT (images/videos → 3584-dim embeddings)
self.audio_encoder   — Whisper (audio → 1280-dim)
self.audio_projector — Linear (1280 → 3584-dim alignment)
self.language_model  — LLM (standard transformer decoder)
```

Forward does 5 steps:
```
1. input_ids → inputs_embeds               lookup embedding table
2. pixel_values → ViT → masked_scatter     replace <|image_pad|>
3. audio_features → Whisper → masked_scatter replace <|audio_pad|>
4. compute 3D position_ids (t, h, w)       prefill: full compute / generation: use rope_deltas
5. pass to language_model                   LLM sees one sequence of 3584-dim vectors
```

One-liner: ForConditionalGeneration = shell (lm_head + generate), Qwen2VLModel = core (encode + fuse + LLM).

### Reading progress

**Read in detail:**
- `Qwen2VLModel.forward()` — all 5 steps ✓
- `Qwen2VLForConditionalGeneration.__init__`, `forward()`, `prepare_inputs_for_generation` ✓

**Not yet read (will cover later):**
- `get_rope_index` — the most complex function, save for RoPE section
- `get_image_features` / `get_audio_features` — short (3-5 lines), will see when reading ViT
- `get_placeholder_mask` — helper for masked_scatter
