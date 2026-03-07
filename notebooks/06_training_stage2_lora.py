# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 06 - Training Stage 2: LoRA Fine-Tuning
#
# **Goal**: Fine-tune the LLM decoder layers with LoRA adapters, building on the Stage 1
# trained audio projector. This teaches the LLM to better utilize the audio embeddings
# for transcription.
#
# **What we train**:
# - LoRA adapters on all LLM attention + MLP projections (28 layers × 7 modules = 196 targets, ~161M params)
# - `audio_projector` continues training as full params via `modules_to_save` (~17M params)
# - Total trainable: ~178M (2.1% of 8.3B)
#
# **What stays frozen**:
# - Audio encoder (Whisper) — already pretrained
# - Vision encoder — not changing vision capabilities
# - LLM base weights — only LoRA adapters are trainable
#
# **Prerequisites**:
# - Stage 1 complete: trained projector pushed to `DanJZY/Qwen2-VL-7B-Speech`
# - Conda env `speech_qwen2vl` with PEFT installed

# %% [markdown]
# ## 1. Environment Setup

# %%
# Change to project root so all relative paths (./data, ./checkpoints) work correctly.
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
         if "__file__" in dir()
         else os.path.join(os.getcwd(), "..") if os.path.basename(os.getcwd()) == "notebooks"
         else os.getcwd())
print(f"Working directory: {os.getcwd()}")

# Dataset cache must be set BEFORE importing datasets library.
os.environ["HF_DATASETS_CACHE"] = os.path.abspath("./data")

import subprocess
import torch
import gc
import time
import math
from datasets import load_dataset, Audio
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from qwen_vl_utils import process_vision_info
import transformers

# Auto-select the GPU with the most free memory, then restrict visibility to that GPU only.
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

    available = [(idx, free) for idx, used, free in gpus if used < 500]
    if not available:
        available = [(idx, free) for idx, _, free in gpus]
    best_idx, best_free = max(available, key=lambda x: x[1])

    idle_ids = [str(idx) for idx, used, _ in gpus if used < 500]
    print(f"Available (idle) GPUs: [{', '.join(idle_ids) if idle_ids else 'none'}]")
    return best_idx

GPU_ID = get_free_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
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
# ## 2. Load Dataset (from cache)
#
# Same dataset as Stage 1 — already cached in `./data` from the previous run.
# `load_dataset()` reads from cache in seconds, no re-download.
#
# **Data strategy**: The `small/` split has 72 shards (~107K samples). We start with
# 20 shards (~30K samples) to validate the pipeline, then scale to the full split
# for the actual training run. To use all 72 shards, change `data_files` to
# `["small/train-*"]`.

# %%
# Option A: 20 shards for pipeline validation (~30K samples)
train_dataset = load_dataset(
    "speechbrain/LargeScaleASR",
    data_files=["small/train-0000*", "small/train-0001*"],
    num_proc=12,
)

# Option B: Full small split for actual training (~107K samples)
# train_dataset = load_dataset(
#     "speechbrain/LargeScaleASR",
#     data_files=["small/train-*"],
#     num_proc=12,
# )
test_dataset = load_dataset(
    "speechbrain/LargeScaleASR",
    data_files=["test/test-00000*"],
    num_proc=12,
)

train_dataset = train_dataset["train"]
test_dataset = test_dataset["train"]
test_dataset = test_dataset.select(range(100))

train_dataset = train_dataset.cast_column("wav", Audio(decode=False))
test_dataset = test_dataset.cast_column("wav", Audio(decode=False))

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples:  {len(test_dataset)}")
print(f"Dataset cache: {os.environ['HF_DATASETS_CACHE']}")
print(f"Columns:       {train_dataset.column_names}")

# %%
# Pre-filter over-budget samples.
MAX_LENGTH = 2048
FILTER_SAFETY_MARGIN = 10

REPO_ID = "DanJZY/Qwen2-VL-7B-Speech"
_processor = Qwen2VLProcessor.from_pretrained(REPO_ID)
_tokenizer = _processor.tokenizer

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
dummy_text_no_audio = dummy_text.replace("<|audio_start|>", "").replace("<|audio_pad|>", "").replace("<|audio_end|>", "")
template_overhead = len(_tokenizer.encode(dummy_text_no_audio, add_special_tokens=False))
effective_budget = MAX_LENGTH - template_overhead - FILTER_SAFETY_MARGIN
print(f"Template overhead: {template_overhead} tokens")
print(f"Effective budget for audio_pads + transcript: {effective_budget} tokens")

def is_within_budget(sample):
    audio_pads = min(math.ceil(sample["duration"] * 50), 1500)
    transcript_tokens = len(_tokenizer.encode(sample["text"], add_special_tokens=False))
    return audio_pads + transcript_tokens <= effective_budget

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
# ## 4. Load Model + Apply LoRA
#
# Load the Stage 1 model (with trained audio projector) in bf16.
# Apply LoRA adapters to LLM decoder layers only (regex-scoped to avoid Whisper).
# Keep `audio_projector` trainable as full params via `modules_to_save`.

# %%
# Step 4a: Load model in bf16
model = Qwen2VLForConditionalGeneration.from_pretrained(
    REPO_ID,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
)
processor = Qwen2VLProcessor.from_pretrained(REPO_ID)

model.config.use_cache = False

print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# %%
# Step 4b: Apply LoRA
# Regex scoped to LLM only — plain suffix matching would also hit Whisper's attention layers.
# PEFT uses re.fullmatch, and the module path starts with model.language_model...
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=r"model\.language_model\.layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["audio_projector"],  # keep full-param training for the projector
)

model = get_peft_model(model, lora_config)

# Step 4c: Verification
model.print_trainable_parameters()

# Manual verification
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTrainable: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.2f}%)")

# Verify audio_projector is trainable and in bf16
for name, param in model.named_parameters():
    if "audio_projector" in name and param.requires_grad:
        print(f"  {name}: {param.shape}, dtype={param.dtype}")

# Verify no audio_encoder params are trainable
audio_encoder_trainable = sum(
    p.numel() for n, p in model.named_parameters()
    if "audio_encoder" in n and p.requires_grad
)
assert audio_encoder_trainable == 0, f"Bug: {audio_encoder_trainable} audio_encoder params are trainable!"
print(f"\nAudio encoder trainable params: {audio_encoder_trainable} (expected 0)")

if torch.cuda.is_available():
    print(f"GPU memory used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# %% [markdown]
# ## 5. Data Collator — `AudioTextCollator`
#
# Identical to Notebook 05. Same label masking strategy.

# %%
class AudioTextCollator:
    """Collator that converts raw dataset samples into training batches with masked labels."""

    def __init__(self, processor):
        self.processor = processor

        im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_tokens = processor.tokenizer.encode("assistant\n", add_special_tokens=False)
        self.assistant_start_tokens = torch.tensor([im_start_id] + assistant_tokens)

        self.pad_token_id = processor.tokenizer.pad_token_id
        self.im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

        print(f"AudioTextCollator initialized.")
        print(f"  assistant marker tokens: {self.assistant_start_tokens.tolist()}")
        print(f"  pad_token_id: {self.pad_token_id}")
        print(f"  im_end_id:    {self.im_end_id}")

    def _find_subsequence(self, seq, subseq, start=0):
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

            _, _, audio_inputs = process_vision_info(messages)
            if audio_inputs:
                all_audios.extend(audio_inputs)

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            all_texts.append(text)

        batch = self.processor(
            text=all_texts,
            audios=all_audios if all_audios else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        labels = batch["input_ids"].clone()

        for i in range(labels.shape[0]):
            seq = batch["input_ids"][i]

            pos = self._find_subsequence(seq, self.assistant_start_tokens)
            assert pos is not None, f"Assistant marker not found in sample {i}"

            second_pos = self._find_subsequence(seq, self.assistant_start_tokens, start=pos + 1)
            assert second_pos is None, f"Multiple assistant markers in sample {i}"

            mask_end = pos + len(self.assistant_start_tokens)
            labels[i, :mask_end] = -100

            im_end_positions = (seq == self.im_end_id).nonzero(as_tuple=True)[0]
            im_end_after_assistant = im_end_positions[im_end_positions > mask_end]
            assert len(im_end_after_assistant) > 0, f"Sample {i}: no <|im_end|> after assistant"
            labels[i, im_end_after_assistant[0].item() + 1:] = -100

            labels[i, seq == self.pad_token_id] = -100

        for i in range(labels.shape[0]):
            real_label_mask = labels[i] != -100
            real_label_count = real_label_mask.sum().item()
            assert real_label_count > 0, f"Sample {i} has zero real labels"

            real_label_ids = labels[i][real_label_mask]
            assert real_label_ids[-1].item() == self.im_end_id, \
                f"Sample {i}: last real label is not <|im_end|>"

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

sample_ids = test_batch["input_ids"][0]
sample_labels = test_batch["labels"][0]

print(f"\n--- Label masking verification (sample 0) ---")
print(f"Total tokens:   {len(sample_ids)}")
print(f"Masked (-100):  {(sample_labels == -100).sum().item()}")
print(f"Real labels:    {(sample_labels != -100).sum().item()}")

real_label_mask = sample_labels != -100
real_label_ids = sample_labels[real_label_mask]
decoded_labels = processor.tokenizer.decode(real_label_ids, skip_special_tokens=False)
print(f"\nDecoded labels: {decoded_labels}")
print(f"Ground truth:   {train_dataset[0]['text']}")

del test_batch

# %% [markdown]
# ## 6. Training Config
#
# Same Trainer setup as Stage 1, but with lower learning rate (2e-5 vs 1e-4)
# since we're fine-tuning pretrained LLM layers, not training from random init.

# %%
training_args = TrainingArguments(
    output_dir="./checkpoints/stage2_lora",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,        # explicit — HF defaults to 8 which can OOM
    gradient_accumulation_steps=8,      # effective batch = 2 * 8 = 16
    learning_rate=2e-5,                 # lower than Stage 1 (fine-tuning, not random init)
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
    remove_unused_columns=False,
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
    name="stage2-lora",
    config={
        "stage": 2,
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_targets": str(lora_config.target_modules),
        "lora_dropout": lora_config.lora_dropout,
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
# **Option A**: Run `trainer.train()` below for single-GPU training.
#
# **Option B** (recommended): Run the multi-GPU DDP script instead:
# ```bash
# cd /home/zhuoyuan/projects/speechQwen2VL
# python scripts/train_stage2.py              # auto-detect idle GPUs
# python scripts/train_stage2.py --nproc 4    # limit to 4 GPUs
# ```
# Then skip the training cell and load the checkpoint in the next cell.

# %%
# Option A: Train from scratch (single GPU)
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
# PEFT's save_pretrained() saves only the LoRA adapters + modules_to_save (~700MB).

import glob

checkpoint_dir = "./checkpoints/stage2_lora"

checkpoint_folders = sorted(
    glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")),
    key=lambda x: int(x.rsplit("-", 1)[-1]),
)
load_path = checkpoint_folders[-1] if checkpoint_folders else checkpoint_dir

if os.path.exists(os.path.join(load_path, "adapter_config.json")):
    print(f"Loading LoRA adapter from: {load_path}")
    # Load base model first
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        REPO_ID,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    )
    base_model.config.use_cache = False
    # Load LoRA adapters on top
    model = PeftModel.from_pretrained(base_model, load_path)
    print(f"LoRA adapter loaded. GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
else:
    print(f"No checkpoint found in {checkpoint_dir}. Train first!")

# %% [markdown]
# ## 9. Verify — Post-Training Inference
#
# Compare transcription quality with Stage 1. Should be more accurate.

# %%
def run_inference(model, processor, messages, max_new_tokens=256):
    """Run inference on a single conversation."""
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

# %% [markdown]
# ## 10. Save & Push to HuggingFace
#
# PEFT's `save_pretrained()` saves only the LoRA adapter weights + `modules_to_save`
# (~700MB), not the full base model. Push to a separate repo.

# %%
ADAPTER_REPO = "DanJZY/Qwen2-VL-7B-Speech-LoRA"

model.save_pretrained("./checkpoints/stage2_lora")
processor.save_pretrained("./checkpoints/stage2_lora")
print(f"LoRA adapter saved to ./checkpoints/stage2_lora")

model.push_to_hub(ADAPTER_REPO)
processor.push_to_hub(ADAPTER_REPO)
print(f"LoRA adapter pushed to {ADAPTER_REPO}")

# %% [markdown]
# ## 11. Cleanup

# %%
wandb.finish()
clear_memory()
print("Stage 2 training complete. Cleanup done.")
