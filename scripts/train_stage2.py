"""
Stage 2 Training: LoRA Fine-Tuning (Multi-GPU DDP)

Fine-tunes the LLM decoder layers with LoRA adapters on top of the Stage 1
trained audio projector. Audio encoder (Whisper) and vision encoder stay frozen.

Usage:
    python scripts/train_stage2.py              # auto-detect idle GPUs, launch DDP
    python scripts/train_stage2.py --nproc 4    # use at most 4 GPUs
    CUDA_VISIBLE_DEVICES=0,1 python scripts/train_stage2.py  # manual GPU selection
"""

import os
import sys
import subprocess
import argparse

# ---------------------------------------------------------------------------
# 0. Auto-detect idle GPUs and re-launch with torchrun if needed
# ---------------------------------------------------------------------------

def find_idle_gpus(max_used_mb=500):
    """Return list of GPU indices with < max_used_mb memory used."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    idle = []
    for line in result.stdout.strip().split("\n"):
        idx, used = [int(x.strip()) for x in line.split(",")]
        if used < max_used_mb:
            idle.append(idx)
    return idle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc", type=int, default=None,
                        help="Max number of GPUs to use (default: all idle GPUs)")
    parser.add_argument("--output_dir", type=str,
                        default="./checkpoints/stage2_lora")
    parser.add_argument("--data_dir", type=str,
                        default="./data",
                        help="HF dataset cache directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eval_batch_size", type=int, default=2,
                        help="Per-device eval batch size (default: 2)")
    parser.add_argument("--num_evals", type=int, default=5,
                        help="Number of evenly-spaced evals during training (default: 5)")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser.parse_args()


if "LOCAL_RANK" not in os.environ:
    # We're the launcher — find idle GPUs and re-launch with torchrun
    args = parse_args()

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    else:
        gpu_ids = [str(g) for g in find_idle_gpus()]
        if not gpu_ids:
            print("No idle GPUs found!")
            sys.exit(1)

    if args.nproc is not None:
        gpu_ids = gpu_ids[:args.nproc]

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    print(f"Using {len(gpu_ids)} GPUs: [{', '.join(gpu_ids)}]")

    cmd = [
        "torchrun",
        f"--nproc_per_node={len(gpu_ids)}",
        sys.argv[0],
    ] + sys.argv[1:]

    sys.exit(subprocess.call(cmd))

# ---------------------------------------------------------------------------
# Below here: we're inside a torchrun worker process
# ---------------------------------------------------------------------------

args = parse_args()
os.environ["HF_DATASETS_CACHE"] = args.data_dir

import math
import torch
from datasets import load_dataset, Audio
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info

local_rank = int(os.environ["LOCAL_RANK"])
is_main = local_rank == 0

if is_main:
    print(f"Dataset cache: {args.data_dir}")
    print(f"Output dir:    {args.output_dir}")

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

REPO_ID = "DanJZY/Qwen2-VL-7B-Speech"
MAX_LENGTH = 2048

# Option A: 20 shards for pipeline validation (~30K samples)
# train_dataset = load_dataset(
#     "speechbrain/LargeScaleASR",
#     data_files=["small/train-0000*", "small/train-0001*"],
#     num_proc=12,
# )

# Option B: Full small split for actual training (~107K samples)
train_dataset = load_dataset(
    "speechbrain/LargeScaleASR",
    data_files=["small/train-*"],
    num_proc=12,
)
test_dataset = load_dataset(
    "speechbrain/LargeScaleASR",
    data_files=["test/test-00000*"],
    num_proc=12,
)

train_dataset = train_dataset["train"]
test_dataset = test_dataset["train"].select(range(100))

train_dataset = train_dataset.cast_column("wav", Audio(decode=False))
test_dataset = test_dataset.cast_column("wav", Audio(decode=False))

# Pre-filter over-budget samples
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
FILTER_SAFETY_MARGIN = 10
effective_budget = MAX_LENGTH - template_overhead - FILTER_SAFETY_MARGIN


def is_within_budget(sample):
    audio_pads = min(math.ceil(sample["duration"] * 50), 1500)
    transcript_tokens = len(_tokenizer.encode(sample["text"], add_special_tokens=False))
    return audio_pads + transcript_tokens <= effective_budget


train_before = len(train_dataset)
train_dataset = train_dataset.filter(is_within_budget, num_proc=12)
test_dataset = test_dataset.filter(is_within_budget, num_proc=12)

if is_main:
    print(f"Train: {train_before} → {len(train_dataset)} samples")
    print(f"Test:  {len(test_dataset)} samples")

del _processor, _tokenizer

# ---------------------------------------------------------------------------
# 2. Model + LoRA
# ---------------------------------------------------------------------------

model = Qwen2VLForConditionalGeneration.from_pretrained(
    REPO_ID,
    torch_dtype=torch.bfloat16,
    device_map={"": local_rank},
)
processor = Qwen2VLProcessor.from_pretrained(REPO_ID)

model.config.use_cache = False

# Apply LoRA — regex scoped to LLM decoder layers only
lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=r"model\.language_model\.layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["audio_projector"],
)

model = get_peft_model(model, lora_config)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
if is_main:
    model.print_trainable_parameters()
    print(f"Trainable parameters: {trainable_params:,}")

# ---------------------------------------------------------------------------
# 3. Collator
# ---------------------------------------------------------------------------

class AudioTextCollator:
    def __init__(self, processor):
        self.processor = processor
        im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_tokens = processor.tokenizer.encode("assistant\n", add_special_tokens=False)
        self.assistant_start_tokens = torch.tensor([im_start_id] + assistant_tokens)
        self.pad_token_id = processor.tokenizer.pad_token_id
        self.im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def _find_subsequence(self, seq, subseq, start=0):
        seq_len, sub_len = len(seq), len(subseq)
        for i in range(start, seq_len - sub_len + 1):
            if torch.equal(seq[i:i + sub_len], subseq):
                return i
        return None

    def __call__(self, samples):
        all_texts, all_audios = [], []

        for sample in samples:
            messages = [
                {"role": "user", "content": [
                    {"type": "audio", "audio": sample["wav"]["bytes"]},
                    {"type": "text", "text": "Transcribe this audio."},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": sample["text"]},
                ]},
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

            mask_end = pos + len(self.assistant_start_tokens)
            labels[i, :mask_end] = -100

            im_end_positions = (seq == self.im_end_id).nonzero(as_tuple=True)[0]
            im_end_after_assistant = im_end_positions[im_end_positions > mask_end]
            assert len(im_end_after_assistant) > 0, f"Sample {i}: no <|im_end|> after assistant"
            labels[i, im_end_after_assistant[0].item() + 1:] = -100

            labels[i, seq == self.pad_token_id] = -100

        batch["labels"] = labels
        return batch


collator = AudioTextCollator(processor)

# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------

n_gpus = int(os.environ.get("WORLD_SIZE", 1))
total_steps = math.ceil(
    len(train_dataset) / (args.batch_size * n_gpus * args.grad_accum)
) * args.epochs
eval_steps = max(total_steps // args.num_evals, 1)
save_steps = eval_steps

if is_main:
    print(f"Total steps: {total_steps}, eval every {eval_steps} steps ({args.num_evals} evals)")

training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=10,
    report_to="wandb" if is_main else "none",
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_strategy="steps",
    save_steps=save_steps,
    save_total_limit=3,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    ddp_find_unused_parameters=True,  # precaution for PEFT modules_to_save
)

if is_main:
    import wandb
    wandb.init(
        project="speechQwen2VL",
        name="stage2-lora",
        config={
            "stage": 2,
            "trainable_params": trainable_params,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "learning_rate": args.lr,
            "num_epochs": args.epochs,
            "per_device_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "num_gpus": n_gpus,
        },
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collator,
    processing_class=processor.tokenizer,
)

if is_main:
    effective_batch = args.batch_size * args.grad_accum * n_gpus
    print(f"Training: {len(train_dataset)} samples, {args.epochs} epochs")
    print(f"Batch: {args.batch_size}/gpu × {args.grad_accum} accum × {n_gpus} GPUs = {effective_batch} effective")

trainer.train()

# Final eval — guarantees an eval at the end regardless of step alignment
trainer.evaluate()

if is_main:
    # PEFT save_pretrained saves only adapters + modules_to_save (~700MB)
    trainer.model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    wandb.finish()
    print(f"Training complete. LoRA adapter saved to {args.output_dir}")
