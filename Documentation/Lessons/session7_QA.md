# Session 7 — Q&A / Lessons Learned

---

## Q: What causes the worst WER predictions, and what do they tell us?

**Context**: After evaluating Stage 2 (LoRA) on the full test set (8,087 samples, corpus WER 8.67%), we inspected the 10 worst predictions by per-sample WER. The failure modes fall into distinct categories.

**Key stat**: Only 3 out of 8,087 Stage 2 predictions hit the full 256-token cap, versus 14 out of 8,087 in Stage 1. So the repetition failure mode is rarer after LoRA fine-tuning, not worse.

### Category 1: Repetition loops (Samples 414, 7512)

**What happens**: The model correctly transcribes the actual speech, then fails to emit the stop token (`<|im_end|>`) and gets stuck in a repetition loop until `max_new_tokens=256` cuts it off. This inflates per-sample WER to 500-700%.

**Examples**:
- Sample 414 (WER=713%, 21.4s): Transcribes speech correctly, then appends "AH AH AH AH..." 200+ times. The audio has a ~1 second "ah" at the end — the model correctly detects it but instead of outputting one "AH" and stopping, it loops.
- Sample 7512 (WER=513%, 10.0s): Generates "WE'RE GOING TO DO THIS OR THAT" on repeat — completely wrong content, looping.

**Root cause — why the model doesn't stop**:

The underlying issue is a **train/inference mismatch** (exposure bias):
- During training (teacher forcing), the model always sees the correct previous token. It learns to predict the next token given perfect history.
- During inference (free-running generation), the model conditions on its own previous outputs. If it makes a small mistake (e.g., generates "AH" instead of `<|im_end|>`), that mistake feeds back in and can compound.

Once greedy decoding falls into a loop, it keeps reinforcing itself — the model sees "AH AH AH" in context and assigns high probability to another "AH". There's no randomness to escape.

**Contributing factors**:
- Greedy decoding (`do_sample=False, num_beams=1`) is the most brittle strategy — no randomness to break loops
- LoRA changed the decoder's language prior, which improved most samples but may have made some individual samples worse at knowing when to stop
- The final Stage 2 checkpoint is not the best-eval checkpoint — the best one (around epoch 1.8) was deleted due to `save_total_limit=3`. The final checkpoint (epoch 3.0) is slightly overfit.

**Possible fixes**:
- `repetition_penalty=1.2` in `model.generate()` — penalizes tokens that already appeared (simplest)
- `no_repeat_ngram_size=3` — hard-blocks repeating any 3-gram
- Lower `max_new_tokens` (e.g., 128) — limits how much damage a loop can do
- Post-processing to detect and truncate repetitions

**Impact on corpus WER**: Minimal. Corpus-level WER weights by reference length (total errors / total reference words). These samples have short references (few reference words), so even 700% per-sample WER contributes very little to the corpus metric. The 5 repetition-loop samples together likely inflate corpus WER by < 0.1%.

---

### Category 2: Incomplete references / model is more correct (Samples 8025, 6376, 540, 4216, 2352)

**What happens**: The model transcribes more speech than the reference captures, or the audio genuinely repeats content that the reference only lists once.

**Examples**:
- Sample 8025 (WER=400%, 8.1s): Reference is "DID YOU KNOW THAT" but the audio actually says it multiple times. The model is correct — the reference is truncated.
- Sample 6376 (WER=300%, 8.2s): Reference is "FOUR" in 8.2s of audio. The audio likely repeats "four" multiple times.
- Sample 540 (WER=175%, 1.3s): Reference is "LONG MAY THIS CONTINUE", model outputs "DO NOT LET US HAVE IT SO LONG MAY THIS CONTINUE" — the audio contains the full phrase.
- Sample 4216 (WER=175%, 13.5s): Model transcribes additional game commentary that's audibly present in the audio but missing from the reference.

**Lesson**: High per-sample WER doesn't always mean the model is wrong — sometimes the reference is incomplete. This is a known limitation of WER as a metric and a dataset quality issue.

---

### Category 3: Non-English or mislabeled audio (Samples 4317, 5216)

**What happens**: The model produces entirely wrong text because the audio is in a different language.

**Examples**:
- Sample 4317 (WER=200%, 3.6s): Reference is "BEAUMONT LES RANDAN" (French place name). Model outputs "THE ONE WHO PAYS HIM DOWN".
- Sample 5216 (WER=150%, 4.9s): Reference is "CENTER FOR SUSTAINABLE DEVELOPMENT" but model outputs German ("SINDA FORSA STEINBOCK VOM MÜTTERBACH MEHR"). Likely a dataset labeling error.

**Lesson**: These are dataset quality issues, not model failures. A production pipeline should include language detection to filter non-English samples.

---

### Category 4: Word boundary differences (Sample 3228)

**Example**:
- Sample 3228 (WER=200%, 2.6s): Reference is "GREENHORNS FLATHEADS" (2 words), model outputs "GREEN HORNS FLAT HEADS" (4 words). Semantically identical — WER metric artifact.

**Lesson**: WER is sensitive to tokenization. For compound words, the model is correct.

---

## Q: How much do the worst samples affect corpus-level WER?

**Answer**: Very little. Corpus-level WER = total word errors / total reference words. The 10 worst samples have short references (1-38 words each), while the full test set has tens of thousands of reference words total. Removing all 10 worst samples would change corpus WER by < 0.2% absolute.

**Lesson**: Corpus-level WER is robust to outliers because it weights by reference length.

---

## Q: What is repetition degeneration and how do you fix it?

**Context**: The model fails to emit `<|im_end|>` and gets stuck generating repeated tokens/phrases until `max_new_tokens` cuts it off.

**What happens at the token level**:
1. Model generates token "AH" (correct — matches audio)
2. "AH" enters the context window
3. Given "...AH", the model assigns high probability to "AH" again
4. "AH AH" enters context → even stronger signal to generate "AH"
5. This positive feedback loop continues indefinitely

**Why greedy decoding is especially vulnerable**: With `do_sample=False`, the model always picks the highest-probability next token. Once a repetition starts, there's no randomness to escape it.

### Decoding mitigations tested

We tested `repetition_penalty=1.2` on the 10 worst samples:

| Sample | Old WER | New WER | Notes |
|--------|---------|---------|-------|
| 414 | 713% | 40% | "AH" loop eliminated |
| 7512 | 513% | 97% | Loop broken, still wrong content |
| 4216 | 175% | 150% | Slight improvement |
| Others | — | — | No change (not repetition issues) |

**What works**:
- `repetition_penalty=1.1-1.2` — soft penalty on already-seen tokens. Fixes actual loops without hurting legitimate repeated speech. Best first fix.
- `num_beams=2` — beam search explores alternative continuations, making it easier to find the stop token. Combined with `repetition_penalty=1.1`, this is effective.

**What to avoid**:
- `no_repeat_ngram_size` globally — hard-blocks repeating any n-gram, which distorts legitimate repeated speech (e.g., "JJ JJ JJ" in sample 414's reference, or "DID YOU KNOW THAT" repeated in sample 8025's audio). Don't use this for ASR.

### Possible fixes

**For inference (no retraining needed)**:
1. Use `repetition_penalty=1.1` + `num_beams=2` as default decoding parameters
2. Keep `do_sample=False` for deterministic results
3. If a sample hits `max_new_tokens`, could retry with stricter decoding as a fallback

**For next training run**:
1. Preserve the best checkpoint — use `load_best_model_at_end=True` or increase `save_total_limit`
2. Add early stopping — the best eval loss was at epoch ~1.8, but training continued to epoch 3.0
3. Stop around 1.8-2.0 epochs instead of 3 to avoid the slight overfitting
4. The current final checkpoint (epoch 3.0, eval loss 0.170) is still good — 8.67% WER — but the epoch 1.8 checkpoint (eval loss 0.164) would likely score even better

---

## Q: WER can be above 100%?

**Context**: Several worst samples show WER of 200%, 400%, even 713%. How is that possible?

**Answer**: WER is computed as:

```
WER = (Substitutions + Insertions + Deletions) / Reference_words
```

There's no upper bound. If the reference is 4 words and the model generates 20 words (all wrong), that's ~20 insertions / 4 reference words = 500% WER. Insertion errors (extra words) are what push WER above 100%.

In practice, WER > 100% almost always means the model generated far more text than the reference — either from repetition loops or from transcribing more audio content than the reference captured.
