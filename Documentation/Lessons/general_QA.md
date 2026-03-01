# General Q&A — Project-Level Lessons

Cross-session questions about workflow, tooling, and common practices that aren't tied to a specific session's technical work.

## Q1: Fork repos vs vendoring code — why are commits spread across 3 repos?

**Confusion**: Our project has 3 repos: the main repo (`speechQwen2VL`), a transformers fork (`ZhuoyuanJiang/transformers`), and a Qwen2-VL fork (`ZhuoyuanJiang/Qwen3-VL`). Commit history is scattered — someone would need to check 3 repos (and specific branches) to see all our changes. Is this normal? Should we just copy the modified files into the main repo instead?

**Our setup**:
```
speechQwen2VL/                          ← main repo (GitHub: ZhuoyuanJiang/speechQwen2VL)
├── Documentation/                       commits: plans, progress docs, lessons
├── notebooks/                           commits: Jupyter notebooks
├── scripts/                             commits: setup scripts
└── forks/                               ← in .gitignore, NOT tracked by main repo
    ├── transformers/                    ← separate git repo (fork, branch: speech-qwen2vl)
    │   └── .../modeling_qwen2_vl.py     commits: model architecture changes
    └── Qwen2-VL/                        ← separate git repo (fork, branch: speech-qwen2vl)
        └── .../vision_process.py        commits: fetch_audio, process_vision_info
```

### Two approaches people use

**Approach 1: Fork mode (what we do)**

Keep modified library code in forked repos, install with `pip install -e` (editable mode).

```bash
# In setup_forks.sh — editable install means Python uses the fork's code directly
pip install -e forks/transformers
pip install -e forks/Qwen2-VL/qwen-vl-utils
```

| Pros | Cons |
|------|------|
| `pip install -e` — edit code, changes take effect immediately | Commit history spread across 3 repos |
| Preserves git relationship with upstream (can rebase/merge updates) | Fork branches not easily discoverable |
| Standard practice in HuggingFace ecosystem for model modifications | Reviewer must visit fork to see actual code diffs |

**Approach 2: Vendor mode (copy files into main repo)**

Copy the modified files directly into the main repo (e.g., `src/qwen2_vl/modeling_qwen2_vl.py`).

| Pros | Cons |
|------|------|
| All commits in one repo — easy to review | Lose git relationship with upstream transformers |
| `git log` shows complete project history | Must manually manage import paths |
| Single clone to see everything | Can't easily pull upstream bug fixes |

### What most research projects do

Most HuggingFace-based research projects use **fork mode** (Approach 1), because:
1. Editable install is essential for rapid iteration on model code
2. The upstream relationship matters — transformers updates frequently with bug fixes
3. Fork branches can be pinned to exact commits for reproducibility

The downside (scattered commits) is mitigated by **documentation in the main repo** — detailed progress docs that record every fork change with code snippets. This is exactly what our `Documentation/SessionN_Progress_*.md` files do.

### Our mitigation strategy

The main repo's `Documentation/` folder serves as the **unified view** of all work:

```
Documentation/
├── Session1_Progress_Documentation.md   ← records what happened (no fork changes in S1)
├── Session2_Progress_20260220.md        ← records every change to both forks + notebook
├── Session3_Progress_20260221.md        ← records every change to transformers fork + notebook
└── Lessons/
    ├── session1_QA.md                   ← technical lessons per session
    ├── session2_QA.md
    ├── session3_QA.md
    └── general_QA.md                    ← this file (cross-session lessons)
```

Someone reading the main repo can understand **all** changes without visiting the forks. The progress docs include full code snippets, design rationale, and before/after comparisons. The forks are there for `pip install -e` and upstream tracking, not as the primary record of work.

**Lesson**: For research projects modifying HuggingFace libraries, fork mode is standard. Compensate for scattered commits by keeping thorough documentation in the main repo that records all fork changes. The main repo's commit history documents *what was done*; the fork repos contain *the actual code*.

## Q2: What is a "Processor" in HuggingFace Transformers?

**Key concept**: A Processor is **NOT a model** — it is a **preprocessing pipeline** that converts raw inputs (images, text, video, audio) into the tensor format the model expects.

```
Raw inputs                    Processor                         Model
─────────────                ──────────                        ───────
PIL images    ─┐
text strings  ─┼──→  processor(images=..., text=...)  ──→  model(**inputs)
audio arrays  ─┘         │                                     │
                         ▼                                     ▼
                   {"input_ids": ...,                    logits, generated text
                    "pixel_values": ...,
                    "attention_mask": ...}
```

The Processor lives in `processing_*.py`. The Model lives in `modeling_*.py`. They are separate files with separate responsibilities:
- **Processor**: tokenization, image resizing/patching, audio mel extraction → tensors
- **Model**: attention, RoPE, feedforward, generation → predictions

Users almost always interact with the Processor, not the individual sub-components (tokenizer, image_processor) directly.

## Q3: HuggingFace ProcessorMixin design pattern

**Context**: Every multimodal model's Processor class inherits from `ProcessorMixin` and follows the same pattern.

### The pattern has 4 parts:

**1. Declare sub-component names** via `attributes`:
```python
class Qwen2VLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
```
This tells ProcessorMixin: "this processor has two parts, stored as `self.image_processor` and `self.tokenizer`."

**2. Declare sub-component classes** via `*_class` attributes (for auto-loading):
```python
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
```
This tells `from_pretrained()` what class to instantiate for each sub-component (see Q20 in session3_QA.md for full explanation).

**3. Provide a `__call__` method** that delegates to each sub-component:
```python
    def __call__(self, images=None, text=None, ...):
        # Step 1: use self.image_processor to convert images → pixel_values
        # Step 2: use self.tokenizer to convert text → input_ids
        # Step 3: combine into one BatchFeature dict
```

**4. Inherit `save_pretrained` / `from_pretrained`** from ProcessorMixin:
- `save_pretrained()` saves both the image_processor config and tokenizer to disk
- `from_pretrained()` auto-loads both using the `*_class` attributes

### Why this design?

Without ProcessorMixin, users would have to manually load and coordinate multiple components:
```python
# Without ProcessorMixin (manual, error-prone):
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
image_processor = AutoImageProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
text_inputs = tokenizer(text, return_tensors="pt")
image_inputs = image_processor(images, return_tensors="pt")
inputs = {**text_inputs, **image_inputs}  # manually merge

# With ProcessorMixin (one-liner):
processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
inputs = processor(images=images, text=text, return_tensors="pt")  # handles everything
```

The Processor is the **user-facing API** — it hides the complexity of coordinating multiple preprocessing components behind a single `__call__` method.

## Q4: Notebook shows unexpected git diff after disconnecting Colab kernel — why?

**Situation**: Opened `.ipynb` notebooks in Cursor with a remote Colab kernel connected. After disconnecting the kernel and closing the files, Cursor prompted to save. After saving, `git diff` showed changes even though no code was edited.

**What changed** (all cosmetic, no functional impact):
- Cell `source` format: single string `"source": "line1\nline2"` → array of lines `"source": ["line1\n", "line2\n"]`
- `language_info` metadata block removed (Python version, mimetype, etc.)
- Trailing newline added at end of file

**Why it happens**: Different editors serialize `.ipynb` JSON differently when saving.
- **With Colab kernel connected**: Cursor saves with `language_info` (kernel provides this metadata)
- **After kernel disconnects**: Cursor saves without `language_info` (no kernel to provide it)
- Cursor also prefers array-of-lines format for cell sources, while Colab uses single strings

Both formats are valid per the Jupyter notebook spec — functionally identical, but git sees different JSON and flags it as a change.

**How to avoid**: When the Colab kernel disconnects, if Cursor prompts you to save the notebook, **choose "Don't Save"**. You didn't change any code, so there's nothing to save. The save prompt is just Cursor re-serializing the file in its own format.

**If it happens anyway**: These changes are safe to `git restore`:
```bash
git restore notebooks/the_affected_notebook.ipynb
```
