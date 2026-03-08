# speechQwen2VL

Fine-tuning [Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) for speech understanding (ASR) by adding an audio encoder and training in two stages.

## Results

Evaluated on the full [speechbrain/LargeScaleASR](https://huggingface.co/datasets/speechbrain/LargeScaleASR) test set (8,087 samples):

| Model | WER | CER |
|-------|-----|-----|
| Stage 1 — Projector only | 12.89% | 7.43% |
| Stage 2 — LoRA fine-tuned | 8.67% | 4.75% |
| Stage 2 + decoding fixes | **7.90%** | **4.25%** |

Decoding fixes: `repetition_penalty=1.1`, `num_beams=2` (no retraining needed).

## Architecture

The model extends Qwen2-VL-7B with audio capabilities:

- **Audio encoder**: Whisper encoder (frozen) — extracts audio features
- **Audio projector**: Learned MLP (~17M params) — maps Whisper features to Qwen2-VL's embedding space
- **LLM decoder**: Qwen2-VL-7B — generates text from audio + text inputs

### Training stages

**Stage 1** — Train audio projector only (0.19% of params). Teaches the model to map audio representations to text. ~1.5h on 6 GPUs.

**Stage 2** — LoRA fine-tune LLM decoder + continue training projector (2.1% of params). Adapts the language model for ASR. ~5.5h on 6 GPUs.

## HuggingFace Models

| Repository | What it contains | Size |
|------------|------------------|------|
| [`DanJZY/Qwen2-VL-7B-Speech`](https://huggingface.co/DanJZY/Qwen2-VL-7B-Speech) | Full model with trained audio projector (Stage 1) | ~17 GB |
| [`DanJZY/Qwen2-VL-7B-Speech-LoRA`](https://huggingface.co/DanJZY/Qwen2-VL-7B-Speech-LoRA) | LoRA adapters only (Stage 2). **Requires the base model above.** | ~700 MB |

## Quick Start

### Inference

```python
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# Load Stage 2 model (base + LoRA adapters)
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "DanJZY/Qwen2-VL-7B-Speech", torch_dtype=torch.bfloat16, device_map="cuda",
)
model = PeftModel.from_pretrained(base_model, "DanJZY/Qwen2-VL-7B-Speech-LoRA")
model.eval()
processor = Qwen2VLProcessor.from_pretrained("DanJZY/Qwen2-VL-7B-Speech")

# Transcribe audio
messages = [
    {"role": "user", "content": [
        {"type": "audio", "audio": "path/to/audio.wav"},
        {"type": "text", "text": "Transcribe this audio."},
    ]},
]

image_inputs, video_inputs, audio_inputs = process_vision_info(messages)
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
batch = processor(text=[text], audios=audio_inputs, return_tensors="pt", padding=True)
batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

with torch.inference_mode():
    output_ids = model.generate(
        **batch, max_new_tokens=256, num_beams=2,
        do_sample=False, repetition_penalty=1.1,
    )
prompt_len = batch["input_ids"].shape[1]
transcription = processor.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)[0]
print(transcription)
```

### Stage 1 only (no LoRA)

```python
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "DanJZY/Qwen2-VL-7B-Speech", torch_dtype=torch.bfloat16, device_map="cuda",
)
```

## Setup

### 1. Create conda environment

```bash
conda env create -f environment.yml
conda activate speech_qwen2vl
```

If the pip install step fails (flash-attn build issue), see `Documentation/Session5_Progress_20260306.md` for the manual 3-stage install.

### 2. Install forked libraries

```bash
bash scripts/setup_forks.sh
```

This installs our forked `transformers` (with audio support) and `qwen-vl-utils` in editable mode.

### 3. Set up data directories

```bash
# On a server with local SSDs (recommended):
mkdir -p /path/to/local/ssd/data /path/to/local/ssd/checkpoints
ln -s /path/to/local/ssd/data ./data
ln -s /path/to/local/ssd/checkpoints ./checkpoints

# Or just use local directories:
mkdir -p data checkpoints
```

## Training

### Stage 1 — Audio projector

```bash
python scripts/train_stage1.py --num_evals 5
```

### Stage 2 — LoRA fine-tuning

```bash
python scripts/train_stage2.py --num_evals 10
```

Both scripts auto-detect idle GPUs and launch DDP training. See `--help` for all arguments.

## Evaluation

```bash
# Full test set, greedy decoding
python scripts/evaluate.py --model stage2 --gpu 0

# With decoding fixes (recommended)
python scripts/evaluate.py --model stage2 --gpu 0 --repetition_penalty 1.1 --num_beams 2

# Compare Stage 1 vs Stage 2
python scripts/evaluate.py --model both --n_samples 100 --gpu 0
```

Or use `notebooks/07_evaluation.ipynb` for interactive analysis with plots and error inspection.

## Repository Structure

```
notebooks/
  01-04  Exploration notebooks (data, tokenizer, model architecture, inference)
  05     Stage 1 training notebook
  06     Stage 2 LoRA training notebook
  07     Evaluation notebook (WER/CER metrics, error analysis)

scripts/
  train_stage1.py       Multi-GPU DDP training (Stage 1)
  train_stage2.py       Multi-GPU DDP training (Stage 2)
  evaluate.py           CLI evaluation script
  setup_forks.sh        Install forked transformers & qwen-vl-utils

Documentation/
  Session*_Progress_*   Detailed session logs with reproduction steps
  Session*_Plan.md      Planning documents
  Lessons/              Q&A and lessons learned per session
```

## Dataset

[speechbrain/LargeScaleASR](https://huggingface.co/datasets/speechbrain/LargeScaleASR) — `small` split:
- Training: 72 shards (~107K samples)
- Test: 6 shards (8,087 samples)

## Dependencies

Key dependencies (see `requirements.txt` for full list):
- PyTorch 2.4.1 + CUDA 12.1
- Transformers (forked, with audio support)
- PEFT 0.17.1
- Flash Attention 2.6.3
- jiwer (WER/CER evaluation)
