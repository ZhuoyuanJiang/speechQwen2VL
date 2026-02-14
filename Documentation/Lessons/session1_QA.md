# Session 1 — Questions & Lessons Learned

---

## Q1: What is loaded in fp16 and what is in float32, and why does this mismatch occur?

### Context

While running Notebook 01 on Google Colab, we loaded the Whisper-large-v3-turbo model and tried to pass audio through its encoder:

```python
whisper_model = WhisperModel.from_pretrained("openai/whisper-large-v3-turbo")
whisper_encoder = whisper_model.encoder

whisper_features = whisper_processor.feature_extractor(audio_array, sampling_rate=sr, return_tensors="pt")

with torch.no_grad():
    encoder_output = whisper_encoder(whisper_features.input_features)  # ← CRASH
```

**Error**:
```
RuntimeError: Input type (float) and bias type (c10::Half) should be the same
```

### Answer: Two separate components, two separate dtypes

**Component 1: Feature Extractor** (signal processing, always float32)
```
Raw audio waveform (float32 numpy array)
    → WhisperFeatureExtractor
    → Log-mel spectrogram tensor: shape [1, 128, 3000], dtype=float32
```
This is pure math (FFT, mel filterbank, logarithm). It always outputs float32. No model weights involved — just signal processing.

**Component 2: Encoder model** (neural network, dtype depends on how you loaded it)
```
Conv1d layer (conv1.weight, conv1.bias)  ← these are MODEL WEIGHTS
    → Transformer layers
    → Output hidden states
```
When you call `WhisperModel.from_pretrained(...)`, transformers downloads the weight files and loads them into the model. The dtype of these weights depends on:
- How they're **stored** on HuggingFace Hub (Whisper-turbo: float16)
- What `torch_dtype` you specify (override)

### Where the crash happens

```python
# Inside WhisperEncoder.forward():
inputs_embeds = nn.functional.gelu(self.conv1(input_features))
#                                       ↑
#                              This is F.conv1d(input, weight, bias)
```

PyTorch's `F.conv1d` requires all three to be the same dtype:

| Argument | What it is | Dtype |
|----------|-----------|-------|
| `input` | The mel spectrogram from feature extractor | **float32** |
| `weight` | `conv1.weight` — a model parameter | **float16** (loaded from Hub) |
| `bias` | `conv1.bias` — a model parameter | **float16** (loaded from Hub) |

```
float32 input + float16 weight/bias → RuntimeError: Input type (float) and bias type (c10::Half) should be the same
```

### The fix

`torch_dtype=torch.float32` tells `from_pretrained`: "after downloading the float16 weights, cast them all to float32 before putting them into the model." Now everything is float32 and `F.conv1d` is happy.

```
Feature extractor output: float32  ✓
conv1.weight:             float32  ✓  (was float16, cast to float32)
conv1.bias:               float32  ✓  (was float16, cast to float32)
```

That's the whole story. Two independent components producing different dtypes, crashing when they meet inside a PyTorch operation that demands matching types.
