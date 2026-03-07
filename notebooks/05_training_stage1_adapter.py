# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 05 - Training Stage 1: Audio Projector Only
#
# **Goal**: Train the randomly initialized `audio_projector` (~17M params) to map Whisper audio embeddings into the LLM's text embedding space. All other model weights are frozen.
#
# **What we train**:
# - `model.model.audio_projector` — 2-layer MLP (Whisper hidden dim → LLM hidden dim)
# - Everything else is frozen (~8.3B params)
#
# **Prerequisites**:
# - Notebook 04 completed (inference pipeline verified)
# - Model at `DanJZY/Qwen2-VL-7B-Speech` has audio encoder but random projector
#
# **Where to run**: Server (2x A6000, 48GB each). Conda env `speech_qwen2vl` with editable fork installs.

# %% [markdown]
# ## 1. Environment Setup

# %%
# Change to project root so all relative paths (./data, ./checkpoints) work correctly.
# This also ensures HF_DATASETS_CACHE points to the right place.
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") 
         if "__file__" in dir() 
         else os.path.join(os.getcwd(), "..") if os.path.basename(os.getcwd()) == "notebooks" 
         else os.getcwd())
print(f"Working directory: {os.getcwd()}")

# Dataset cache must be set BEFORE importing datasets library,
# otherwise HF reads the default cache path at import time.
os.environ["HF_DATASETS_CACHE"] = os.path.abspath("./data")

import subprocess
import torch
import gc
import time
import math
from datasets import load_dataset, Audio
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, Trainer, TrainingArguments
from qwen_vl_utils import process_vision_info
import transformers

# Auto-select the GPU with the most free memory, then restrict visibility to that GPU only.
# This must happen before any torch CUDA call so the Trainer sees exactly 1 GPU.
def get_free_gpu():
    """Return the GPU index with the most free memory."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    gpus = []
    for line in result.stdout.strip().split("\n"):
        idx, used, free = [int(x.strip()) for x in line.split(",")]
        gpus.append((idx, used, free))

    available = [(idx, free) for idx, used, free in gpus if used < 500]  # <500 MB used = idle
    if not available:
        available = [(idx, free) for idx, _, free in gpus]  # fallback: pick least-busy
    best_idx, best_free = max(available, key=lambda x: x[1])

    idle_ids = [str(idx) for idx, used, _ in gpus if used < 500]
    print(f"Available (idle) GPUs: [{', '.join(idle_ids) if idle_ids else 'none'}]")
    return best_idx

GPU_ID = get_free_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
# After this, torch sees only 1 GPU. It becomes cuda:0 regardless of the physical index.
DEVICE = "cuda:0"

props = torch.cuda.get_device_properties(0)
print(f"Using GPU {GPU_ID}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
print(f"Visible devices: {os.environ['CUDA_VISIBLE_DEVICES']} (torch sees {torch.cuda.device_count()} GPU)")
print(f"Dataset cache:   {os.environ['HF_DATASETS_CACHE']}")

print(f"\ntransformers: {transformers.__version__} ({transformers.__file__})")
print(f"torch: {torch.__version__}")

# HuggingFace login
from huggingface_hub import get_token
HF_TOKEN = get_token()
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    print("HF token loaded.")
else:
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        print("HF token loaded from environment.")
    else:
        print("No HF token found. Set HF_TOKEN or run `huggingface-cli login`.")

# %% [markdown]
# ## 2. Download Dataset
#
# Downloads a subset of `speechbrain/LargeScaleASR`:
# - **Train**: ~20 shards from `small/` split
# - **Test**: 100 samples from `test/` split
#
# Audio column is cast to `decode=False` to keep raw bytes (needed by our `process_vision_info` pipeline).
#
# After loading, we pre-filter over-budget samples whose total token count would exceed `max_length=2048`.

# %%
# HF_DATASETS_CACHE was set in cell-2 to os.path.abspath("./data").
# Since we os.chdir'd to the project root, this resolves to <project_root>/data.
# On our server, ./data is a symlink to /ssd1, so data goes to the local SSD.
# On other machines without the symlink, data goes to a local ./data folder.

# The small/ split has 72 shards total (~107K samples). We use 20 shards (~30K samples)
# following the skeleton notebook — sufficient for Stage 1 projector training.
# To scale up: change glob to "small/train-*" for all 72 shards.

train_dataset = load_dataset(
    "speechbrain/LargeScaleASR",
    data_files=["small/train-0000*", "small/train-0001*"],
    num_proc=12,
)
test_dataset = load_dataset(
    "speechbrain/LargeScaleASR",
    data_files=["test/test-00000*"],
    num_proc=12,
)

train_dataset = train_dataset["train"]
test_dataset = test_dataset["train"]
test_dataset = test_dataset.select(range(100))

# Keep audio as raw bytes (our process_vision_info pipeline expects bytes, not decoded arrays)
train_dataset = train_dataset.cast_column("wav", Audio(decode=False))
test_dataset = test_dataset.cast_column("wav", Audio(decode=False))

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples:  {len(test_dataset)}")
print(f"Dataset cache: {os.environ['HF_DATASETS_CACHE']}")
print(f"Columns:       {train_dataset.column_names}")

# %%
# Pre-filter over-budget samples.
# Total token budget per sample: audio_pads + transcript_tokens + template_overhead <= MAX_LENGTH
#
# We measure template overhead once from a dummy template that uses the exact same
# chat template, prompt text, and message structure as the collator.
#
# BPE token counts aren't strictly additive across message boundaries, so we subtract
# a small safety margin to avoid edge cases where separately-counted pieces undercount.

MAX_LENGTH = 2048
FILTER_SAFETY_MARGIN = 10  # accounts for BPE boundary effects

# Load processor just for filtering (will reload properly in Section 4)
REPO_ID = "DanJZY/Qwen2-VL-7B-Speech"
_processor = Qwen2VLProcessor.from_pretrained(REPO_ID)
_tokenizer = _processor.tokenizer

# Measure fixed template overhead: tokenize a dummy conversation with empty audio and empty transcript
dummy_messages = [
    {"role": "user", "content": [
        {"type": "audio", "audio": "placeholder"},
        {"type": "text", "text": "Transcribe this audio."},
    ]},
    {"role": "assistant", "content": [
        {"type": "text", "text": ""},
    ]},
]
dummy_text = _processor.apply_chat_template(dummy_messages, tokenize=False, add_generation_prompt=False)
# Remove the audio placeholder to isolate template-only tokens
# The chat template inserts <|audio_start|><|audio_pad|><|audio_end|> for the audio block
dummy_text_no_audio = dummy_text.replace("<|audio_start|>", "").replace("<|audio_pad|>", "").replace("<|audio_end|>", "")
template_overhead = len(_tokenizer.encode(dummy_text_no_audio, add_special_tokens=False))
effective_budget = MAX_LENGTH - template_overhead - FILTER_SAFETY_MARGIN
print(f"Template overhead: {template_overhead} tokens")
print(f"Safety margin:     {FILTER_SAFETY_MARGIN} tokens")
print(f"Effective budget for audio_pads + transcript: {effective_budget} tokens")

def is_within_budget(sample):
    """Check if a sample's total token count fits within MAX_LENGTH (with safety margin)."""
    audio_pads = min(math.ceil(sample["duration"] * 50), 1500)
    transcript_tokens = len(_tokenizer.encode(sample["text"], add_special_tokens=False))
    total = audio_pads + transcript_tokens
    return total <= effective_budget

train_before = len(train_dataset)
test_before = len(test_dataset)

train_dataset = train_dataset.filter(is_within_budget, num_proc=12)
test_dataset = test_dataset.filter(is_within_budget, num_proc=12)

print(f"\nTrain: {train_before} → {len(train_dataset)} (dropped {train_before - len(train_dataset)})")
print(f"Test:  {test_before} → {len(test_dataset)} (dropped {test_before - len(test_dataset)})")

del _processor, _tokenizer


# %% [markdown]
# ## 3. Memory Cleanup Utility

# %%
def clear_memory():
    """Clean up GPU memory by deleting common global variables and clearing CUDA cache."""
    for var_name in ['inputs', 'model', 'processor', 'trainer', 'peft_model', 'bnb_config']:
        if var_name in globals():
            del globals()[var_name]
    time.sleep(2)

    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

print("clear_memory() defined.")

# %% [markdown]
# ## 4. Load Model & Processor, Freeze Parameters
#
# Load from HuggingFace in bf16, single GPU (no quantization for Stage 1).
# Freeze everything, then unfreeze only `audio_projector`.

# %%
model = Qwen2VLForConditionalGeneration.from_pretrained(
    REPO_ID,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,  # auto-selected GPU with most free memory
)
processor = Qwen2VLProcessor.from_pretrained(REPO_ID)

# KV cache conflicts with gradient checkpointing — disable for training
model.config.use_cache = False

# Token ID consistency check
assert processor.tokenizer.convert_tokens_to_ids('<|audio_pad|>') == model.config.audio_token_id, \
    "Processor/model audio_token_id mismatch!"
print("Processor/model token ID consistency check passed.")

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze only audio_projector
for param in model.model.audio_projector.parameters():
    param.requires_grad = True

# Verify freeze
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
projector_params = sum(p.numel() for p in model.model.audio_projector.parameters())

print(f"\nTotal parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Projector parameters: {projector_params:,}")
print(f"Trainable %:          {trainable_params / total_params * 100:.2f}%")

assert trainable_params == projector_params, \
    f"Mismatch! trainable={trainable_params}, projector={projector_params}"
print("\nFreeze verification passed: only audio_projector is trainable.")

if torch.cuda.is_available():
    print(f"\nGPU memory used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


# %% [markdown]
# ## 5. Data Collator — `AudioTextCollator`
#
# Converts raw dataset samples into model-ready batches with masked labels.
#
# **Pipeline**: sample → conversation messages → `process_vision_info()` → chat template → processor → label masking
#
# **Label masking strategy**:
# 1. Tokenize `<|im_start|>assistant\n` once in `__init__`, cache as token ID list
# 2. Search each sequence's `input_ids` for this exact token subsequence
# 3. Mask everything up to and including the match → -100
# 4. Keep everything after (transcript + `<|im_end|>`) as real labels
# 5. Mask padding tokens → -100
#
# ```
# <|im_start|>system\n...<|im_end|>\n          → -100 (masked)
# <|im_start|>user\n<|audio_start|>...<|im_end|> → -100 (masked)
# <|im_start|>assistant\n                        → -100 (masked)
# TRANSCRIPT TEXT                                → REAL LABELS
# <|im_end|>                                     → REAL LABEL (stop token)
# [padding]                                      → -100 (masked)
# ```

# %%
class AudioTextCollator:
    """Collator that converts raw dataset samples into training batches with masked labels."""

    def __init__(self, processor):
        self.processor = processor

        # Cache the token sequence for "<|im_start|>assistant\n" to find label boundaries
        im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_tokens = processor.tokenizer.encode("assistant\n", add_special_tokens=False)
        self.assistant_start_tokens = torch.tensor([im_start_id] + assistant_tokens)

        self.pad_token_id = processor.tokenizer.pad_token_id
        self.im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

        print(f"AudioTextCollator initialized.")
        print(f"  assistant marker tokens: {self.assistant_start_tokens.tolist()}")
        print(f"  decoded: {processor.tokenizer.decode(self.assistant_start_tokens)}")
        print(f"  pad_token_id: {self.pad_token_id}")
        print(f"  im_end_id:    {self.im_end_id}")

    def _find_subsequence(self, seq, subseq, start=0):
        """Find the starting index of subseq in seq, searching from start. Returns None if not found."""
        seq_len = len(seq)
        sub_len = len(subseq)
        for i in range(start, seq_len - sub_len + 1):
            if torch.equal(seq[i:i + sub_len], subseq):
                return i
        return None

    def __call__(self, samples):
        all_texts = []
        all_audios = []

        for sample in samples:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": sample["wav"]["bytes"]},
                        {"type": "text", "text": "Transcribe this audio."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": sample["text"]},
                    ],
                },
            ]

            # Extract audio inputs
            _, _, audio_inputs = process_vision_info(messages)
            if audio_inputs:
                all_audios.extend(audio_inputs)

            # Apply chat template (add_generation_prompt=False for training)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            all_texts.append(text)

        # Process batch through Qwen2VLProcessor
        batch = self.processor(
            text=all_texts,
            audios=all_audios if all_audios else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        # Create labels: clone input_ids, then mask non-target tokens
        labels = batch["input_ids"].clone()

        for i in range(labels.shape[0]):
            seq = batch["input_ids"][i]

            # Find the assistant boundary — assert exactly one match
            pos = self._find_subsequence(seq, self.assistant_start_tokens)
            assert pos is not None, (
                f"Assistant marker not found in sample {i}. "
                f"This indicates a broken chat template. "
                f"First 20 token IDs: {seq[:20].tolist()}"
            )

            # Verify no second assistant marker exists
            second_pos = self._find_subsequence(seq, self.assistant_start_tokens, start=pos + 1)
            assert second_pos is None, (
                f"Multiple assistant markers found in sample {i} (at positions {pos} and {second_pos}). "
                f"This indicates a malformed chat template."
            )

            # Mask everything up to and including the assistant marker
            mask_end = pos + len(self.assistant_start_tokens)
            labels[i, :mask_end] = -100

            # Find <|im_end|> after the assistant content and mask everything after it.
            # The chat template adds a trailing \n after <|im_end|> — we want the model
            # to learn to stop at <|im_end|>, not predict the trailing \n.
            im_end_positions = (seq == self.im_end_id).nonzero(as_tuple=True)[0]
            # Get the last <|im_end|> that's after the assistant marker
            im_end_after_assistant = im_end_positions[im_end_positions > mask_end]
            assert len(im_end_after_assistant) > 0, (
                f"Sample {i}: no <|im_end|> found after assistant marker. "
                f"The transcript was likely truncated."
            )
            last_im_end = im_end_after_assistant[0].item()  # first (and should be only) <|im_end|> after assistant
            labels[i, last_im_end + 1:] = -100  # mask everything after <|im_end|>

            # Mask padding tokens
            labels[i, seq == self.pad_token_id] = -100

        # Safety net: verify every sample has real labels and ends with <|im_end|>.
        for i in range(labels.shape[0]):
            real_label_mask = labels[i] != -100
            real_label_count = real_label_mask.sum().item()
            assert real_label_count > 0, (
                f"Sample {i} has zero real labels after masking — "
                f"truncation likely ate the transcript. "
                f"This sample should have been removed by dataset pre-filtering."
            )

            # Verify last real label is <|im_end|>
            real_label_ids = labels[i][real_label_mask]
            assert real_label_ids[-1].item() == self.im_end_id, (
                f"Sample {i}: last real label is {real_label_ids[-1].item()}, "
                f"expected <|im_end|> ({self.im_end_id}). "
                f"The transcript was likely partially truncated."
            )

        batch["labels"] = labels
        return batch

collator = AudioTextCollator(processor)

# %%
# Test collator on a small batch
test_batch = collator([train_dataset[0], train_dataset[1]])

print("Batch keys:", list(test_batch.keys()))
print(f"input_ids shape:      {test_batch['input_ids'].shape}")
print(f"attention_mask shape: {test_batch['attention_mask'].shape}")
print(f"labels shape:         {test_batch['labels'].shape}")

# Verify label masking on first sample
sample_ids = test_batch["input_ids"][0]
sample_labels = test_batch["labels"][0]

print(f"\n--- Label masking verification (sample 0) ---")
print(f"Total tokens:   {len(sample_ids)}")
print(f"Masked (-100):  {(sample_labels == -100).sum().item()}")
print(f"Real labels:    {(sample_labels != -100).sum().item()}")

# Decode the real label portion to verify it matches the transcript
real_label_mask = sample_labels != -100
real_label_ids = sample_labels[real_label_mask]
decoded_labels = processor.tokenizer.decode(real_label_ids, skip_special_tokens=False)
print(f"\nDecoded labels: {decoded_labels}")
print(f"Ground truth:   {train_dataset[0]['text']}")

# Show token-by-token view around the assistant boundary
print(f"\n--- Token view around assistant boundary ---")
for j in range(len(sample_labels)):
    if sample_labels[j] != -100:
        start = max(0, j - 3)
        for k in range(start, min(j + 5, len(sample_labels))):
            token_str = processor.tokenizer.decode([sample_ids[k]])
            label_str = "MASKED" if sample_labels[k] == -100 else processor.tokenizer.decode([sample_labels[k]])
            print(f"  [{k:4d}] id={sample_ids[k]:6d}  token={token_str!r:20s}  label={label_str}")
        break

del test_batch

# %% [markdown]
# ## 6. Training Config
#
# Using plain `Trainer` + `TrainingArguments` instead of SFTTrainer + SFTConfig.
# SFTTrainer adds automatic dataset preprocessing (which we bypass — our AudioTextCollator handles it)
# and token accuracy logging (which has a shape mismatch bug with gradient accumulation).
# The core training (loss, backprop, optimizer) is identical.

# %%
training_args = TrainingArguments(
    output_dir="./checkpoints/stage1_audio_projector",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,        # explicit — HF defaults to 8 which can OOM
    gradient_accumulation_steps=8,      # effective batch = 2 * 8 = 16
    learning_rate=1e-4,                 # higher than typical fine-tune (projector is random)
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=10,
    report_to="wandb",
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    remove_unused_columns=False,        # collator needs raw dataset columns
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)

print(f"Output dir:      {training_args.output_dir}")
print(f"Epochs:          {training_args.num_train_epochs}")
print(f"Batch size:      {training_args.per_device_train_batch_size}")
print(f"Eval batch size: {training_args.per_device_eval_batch_size}")
print(f"Grad accum:      {training_args.gradient_accumulation_steps}")
print(f"Effective batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Learning rate:   {training_args.learning_rate}")
print(f"Gradient ckpt:   {training_args.gradient_checkpointing}")
print(f"Eval steps:      {training_args.eval_steps}")

# %% [markdown]
# ## 7. wandb Init

# %%
import wandb

wandb.init(
    project="speechQwen2VL",
    name="stage1-audio-projector",
    config={
        "trainable_params": trainable_params,
        "total_params": total_params,
        "learning_rate": training_args.learning_rate,
        "num_epochs": training_args.num_train_epochs,
        "per_device_batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "effective_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
    },
)
print("wandb initialized.")

# %% [markdown]
# ## 8. Train or Load Checkpoint
#
# **Option A**: Run `trainer.train()` below for single-GPU training (~7 hours).
#
# **Option B** (recommended): Run the multi-GPU DDP script instead:
# ```bash
# cd /home/zhuoyuan/projects/speechQwen2VL
# python scripts/train_stage1.py              # auto-detect idle GPUs
# python scripts/train_stage1.py --nproc 4    # limit to 4 GPUs
# ```
# Then skip the training cell and load the checkpoint in the next cell.

# %%
# Option A: Train from scratch (single GPU, ~7 hours)
# Uncomment the lines below to train. Skip this cell if using DDP script.

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     data_collator=collator,
#     processing_class=processor.tokenizer,
# )
# trainer.train()

# %%
# Option B: Load from a trained checkpoint (after DDP script finishes).
# Trainer.save_model() saves the full 8.3B model, which HF shards into multiple
# safetensor files. We use from_pretrained() to handle sharding automatically.

import glob

checkpoint_dir = "./checkpoints/stage1_audio_projector"

# Find the latest checkpoint (highest step number), or use final saved model
checkpoint_folders = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")))
load_path = checkpoint_folders[-1] if checkpoint_folders else checkpoint_dir

if os.path.exists(os.path.join(load_path, "config.json")):
    print(f"Loading trained model from: {load_path}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    )
    model.config.use_cache = False
    print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
else:
    print(f"No checkpoint found in {checkpoint_dir}. Train first!")


# %% [markdown]
# ## 9. Verify — Post-Training Inference
#
# Quick inference test to check that the trained projector produces coherent transcriptions
# (should be much better than the garbage output from Notebook 04).
# Run this **before** pushing to HuggingFace so we don't publish a bad checkpoint.

# %%
def run_inference(model, processor, messages, max_new_tokens=256):
    """Run inference on a single conversation (same as Notebook 04)."""
    image_inputs, video_inputs, audio_inputs = process_vision_info(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        return_tensors="pt",
        padding=True,
    )
    batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
    prompt_len = batch["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_len:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("run_inference() defined.")

# %% [markdown]
# ## 10. Push to HuggingFace & Cleanup
#
# Only 1 of 4 safetensor shards should change (the one containing `audio_projector` weights).

# %%
# Test on a few samples from the test set
for idx in [0, 1, 2]:
    test_sample = test_dataset[idx]

    audio_messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": test_sample["wav"]["bytes"]},
                {"type": "text", "text": "Transcribe this audio."},
            ],
        },
    ]

    print(f"--- Sample {idx} ---")
    output = run_inference(model, processor, audio_messages)
    print(f"Model output:  {output}")
    print(f"Ground truth:  {test_sample['text']}")
    print()

# %%
model.push_to_hub(REPO_ID)
processor.push_to_hub(REPO_ID)
print(f"Model and processor pushed to {REPO_ID}")

# %%
wandb.finish()
clear_memory()
print("Training complete. Cleanup done.")
