# Session 2 — Q&A and Lessons Learned

## Q1: Should I read every line of modeling_qwen2_vl.py (1,635 lines)?

**Context**: After modifying `processing_qwen2_vl.py` (the processor), I wondered whether I should also read every line of `modeling_qwen2_vl.py` (the actual model) to fully understand what tensors the model expects and how our changes fit in.

**Answer**: You don't need to read all 1,635 lines to understand the full pipeline from raw audio to model output. Here's why:

`modeling_qwen2_vl.py` is roughly:
- ~400 lines of attention/transformer block code (standard, same as any LLM)
- ~300 lines of visual encoder (ViT that processes pixel_values into embeddings)
- ~200 lines of RoPE / position encoding
- ~500 lines of the main model class (embedding layer, forward pass, generation)
- ~200 lines of boilerplate (docstrings, config, etc.)

Most of this is standard transformer architecture that every VLM uses. What actually matters for interviews is a much smaller set of questions:

1. **How does `pixel_values` become part of the token sequence?** — The visual encoder converts pixel_values into embeddings, then those embeddings replace the `<|image_pad|>` positions in the input embedding sequence. That's maybe ~50 lines of code.

2. **How will `audio_features` work the same way?** — We'll add a Whisper encoder that converts mel spectrograms into embeddings, then inject them at `<|audio_pad|>` positions. Same pattern. This is Session 3-4 work.

3. **What's the forward pass flow?** — text tokens → embedding lookup → merge in vision/audio embeddings at pad positions → transformer layers → output logits. Maybe ~30 lines of the `forward()` method.

**Recommendation**: When we get to Session 3 (model modifications), walk through the specific sections of `modeling_qwen2_vl.py` that matter — the embedding merge logic and the forward pass. That's when it'll make sense, because you'll see exactly where audio gets plugged in. Reading it now in isolation, without that context, would be 1,635 lines of "I see code but I don't know what's relevant."

**However**, the individual components (attention, transformer blocks, ViT, RoPE) are all foundational concepts that are commonly asked in interviews. Understanding them deeply is valuable — just better to study them in context (when we modify the model) rather than reading raw code in isolation.

## Q2: WhisperFeatureExtractor defaults to 80 mel bins, not 128

**Context**: A code reviewer found that `WhisperFeatureExtractor()` with no arguments creates a feature extractor with `feature_size=80` (the default for older Whisper models). But whisper-large-v3-turbo uses 128 mel bins. In Notebook 01, we loaded via `WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")` which correctly used 128, but in `processing_qwen2_vl.py` we used the bare constructor.

**Lesson**: Always check constructor defaults against the specific model variant you're using. The default parameters in HuggingFace classes often correspond to the original/smallest model, not the variant you want. When in doubt, either:
- Use `from_pretrained()` to load the exact config
- Or explicitly pass the parameters you need (e.g., `feature_size=128`)

**Fix**: Changed `WhisperFeatureExtractor()` to `WhisperFeatureExtractor(feature_size=128)`.

## Q3: process_vision_info backward compatibility tradeoff

**Context**: Our change to `process_vision_info()` in `vision_process.py` returns a 3-tuple `(image_inputs, video_inputs, audio_inputs)` instead of the original 2-tuple. A reviewer noted this breaks existing scripts in the Qwen2-VL repo that unpack 2 values (e.g., `web_demo_mm.py`, `run_realworldqa.py`).

**Decision**: We chose Option A (always return 3-tuple) because:
- We're on our own fork branch (`speech-qwen2vl`), isolated from the original repo
- Those scripts aren't part of our project
- Adding a 3rd return value is cleaner than conditional logic
- If we ever need to run those scripts, we can update them on our branch

**Lesson**: When modifying a library's API, consider whether existing consumers of that API exist on your branch. If you're on an isolated fork branch, backward compatibility with upstream scripts is less important than clean code.

## Q4: Pip caches git installs — restarting the Colab runtime isn't enough

**Context**: After fixing the mel bins bug in the transformers fork (80 → 128) and pushing to GitHub, we restarted the Colab runtime and re-ran all cells. The install cell ran `pip install git+...@speech-qwen2vl`, but the output still showed `mel_bins=80`.

**Root cause**: Pip caches packages installed from git URLs. When you restart a Colab runtime, installed packages persist (they're on the VM's disk, not in memory). Running `pip install` again sees the package is already installed and skips the download, even though the remote branch has new commits.

**Fix**: Two changes to the install cell:
1. `--force-reinstall --no-deps` — forces pip to re-download and reinstall, but skips dependencies (fast)
2. Pin to exact commit hashes instead of branch names (`@e6f7d83ef...` instead of `@speech-qwen2vl`) — ensures reproducibility even if the branch moves forward in later sessions

**Lesson**: When installing from git URLs during development, always use `--force-reinstall --no-deps` to ensure you get the latest code. For production/reproducibility, pin to exact commit hashes.
