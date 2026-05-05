#!/usr/bin/env python3
"""
Local Gradio demo server for speechQwen2VL.

Loads either Stage 1 (projector only) or Stage 2 (Stage 1 + LoRA) and exposes a
browser UI for live mic recording + file upload + preloaded examples. Switching
between Stage 1 and Stage 2 reloads the model on demand (24 GB GPUs only fit one).

Usage:
    python scripts/serve.py                        # default: stage2, auto-pick GPU, port 7860
    python scripts/serve.py --gpu 0                # pin a GPU
    python scripts/serve.py --default_model stage1 # start on Stage 1
    python scripts/serve.py --port 7860 --host 127.0.0.1
"""

import argparse
import gc
import os
import subprocess
import sys
import time
from io import BytesIO


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Gradio demo for speechQwen2VL")
parser.add_argument("--gpu", type=int, default=None, help="GPU index (default: auto-pick idle)")
parser.add_argument("--port", type=int, default=7870,
                    help="Port (default 7870 — picked to avoid clash with other Gradio "
                         "projects that use Gradio's default 7860)")
parser.add_argument("--host", type=str, default="127.0.0.1",
                    help='Bind address. Use "0.0.0.0" to listen on all interfaces (only with SSH tunnel)')
parser.add_argument("--default_model", type=str, choices=["stage1", "stage2"], default="stage2")
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--repetition_penalty", type=float, default=1.1,
                    help="Default decoding repetition_penalty (1.0 = off)")
parser.add_argument("--num_beams", type=int, default=2,
                    help="Default decoding num_beams (1 = greedy)")
parser.add_argument("--share", action="store_true",
                    help="Generate a Gradio public share link (off by default)")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# GPU selection (must run before importing torch)
# ---------------------------------------------------------------------------
def _find_idle_gpus():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    gpus = []
    for line in out.stdout.strip().splitlines():
        idx, used, free = [int(x.strip()) for x in line.split(",")]
        if used < 500:
            gpus.append((idx, free))
    return sorted(gpus, key=lambda x: -x[1])


if args.gpu is None:
    idle = _find_idle_gpus()
    if not idle:
        print("No idle GPUs found. Use --gpu to pin one explicitly.")
        sys.exit(1)
    gpu_id = idle[0][0]
else:
    gpu_id = args.gpu

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
DEVICE = "cuda:0"
print(f"Using GPU {gpu_id}")

# Gradio's default temp dir is /tmp/gradio, which on shared servers is often
# owned by another user and not writable by us. Point it at a per-user path.
if "GRADIO_TEMP_DIR" not in os.environ:
    user_tmp = os.path.join("/tmp", f"gradio_{os.environ.get('USER', 'user')}")
    os.makedirs(user_tmp, exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = user_tmp
print(f"GRADIO_TEMP_DIR = {os.environ['GRADIO_TEMP_DIR']}")


# ---------------------------------------------------------------------------
# Imports (after CUDA_VISIBLE_DEVICES is set)
# ---------------------------------------------------------------------------
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info


STAGE1_REPO = "DanJZY/Qwen2-VL-7B-Speech"
LORA_REPO = "DanJZY/Qwen2-VL-7B-Speech-LoRA"
TARGET_SR = 16000
TRANSCRIBE_PROMPT = "Transcribe this audio."


# ---------------------------------------------------------------------------
# Model holder (Stage 1 / Stage 2 hot swap)
# ---------------------------------------------------------------------------
class ModelHolder:
    def __init__(self):
        self.model = None
        self.processor = None
        self.current_kind = None
        self.last_load_seconds = None

    def load(self, kind: str) -> bool:
        """Load Stage 1 or Stage 2. Returns True if a swap happened."""
        if kind == self.current_kind and self.model is not None:
            return False

        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()

        t0 = time.time()
        if kind == "stage2":
            base = Qwen2VLForConditionalGeneration.from_pretrained(
                STAGE1_REPO, torch_dtype=torch.bfloat16, device_map=DEVICE,
            )
            base.config.use_cache = True
            self.model = PeftModel.from_pretrained(base, LORA_REPO)
        elif kind == "stage1":
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                STAGE1_REPO, torch_dtype=torch.bfloat16, device_map=DEVICE,
            )
            self.model.config.use_cache = True
        else:
            raise ValueError(f"Unknown model kind: {kind!r}")

        self.model.eval()
        if self.processor is None:
            self.processor = Qwen2VLProcessor.from_pretrained(STAGE1_REPO)

        self.current_kind = kind
        self.last_load_seconds = time.time() - t0
        mem_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"Loaded {kind} in {self.last_load_seconds:.1f}s on {DEVICE} ({mem_gb:.2f} GB)")
        return True


HOLDER = ModelHolder()


# ---------------------------------------------------------------------------
# Audio preprocessing (gradio gives (sr, np.ndarray))
# ---------------------------------------------------------------------------
def numpy_to_wav_bytes(audio: tuple) -> tuple:
    """Take (sample_rate, np.ndarray) from gradio. Return (wav_bytes, duration_s)."""
    sr, arr = audio
    if arr.ndim == 1:
        arr = arr[None, :]            # (samples,) -> (1, samples)
    else:
        arr = arr.T                    # gradio gives (samples, channels) for stereo

    # Convert to float32 in [-1, 1]. Gradio mic recordings are int16 by default.
    if np.issubdtype(arr.dtype, np.integer):
        max_val = float(np.iinfo(arr.dtype).max)
        arr = arr.astype(np.float32) / max_val
    else:
        arr = arr.astype(np.float32)

    waveform = torch.from_numpy(arr.copy())  # (channels, samples), float32

    if sr != TARGET_SR:
        waveform = T.Resample(orig_freq=sr, new_freq=TARGET_SR)(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    duration_s = waveform.shape[1] / TARGET_SR

    buf = BytesIO()
    torchaudio.save(buf, waveform, TARGET_SR, format="wav")
    return buf.getvalue(), duration_s


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(model, processor, wav_bytes: bytes,
                  max_new_tokens: int, repetition_penalty: float, num_beams: int) -> str:
    messages = [
        {"role": "user", "content": [
            {"type": "audio", "audio": wav_bytes},
            {"type": "text", "text": TRANSCRIBE_PROMPT},
        ]},
    ]
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

    gen_kwargs = dict(max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False)
    if repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty

    with torch.inference_mode():
        output_ids = model.generate(**batch, **gen_kwargs)
    prompt_len = batch["input_ids"].shape[1]
    return processor.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)[0]


# ---------------------------------------------------------------------------
# Gradio callback
# ---------------------------------------------------------------------------
def transcribe(audio, model_choice, repetition_penalty, num_beams, max_new_tokens):
    if audio is None:
        return "", "*No audio provided. Record from your mic, upload a file, or pick an example.*"

    swap_note = ""
    if model_choice != HOLDER.current_kind:
        gr.Info(f"Loading {model_choice} (~30-60 s)...")
        HOLDER.load(model_choice)
        swap_note = f" — loaded {model_choice} in {HOLDER.last_load_seconds:.1f}s"

    wav_bytes, duration_s = numpy_to_wav_bytes(audio)

    t0 = time.time()
    text = run_inference(
        HOLDER.model, HOLDER.processor, wav_bytes,
        max_new_tokens=int(max_new_tokens),
        repetition_penalty=float(repetition_penalty),
        num_beams=int(num_beams),
    )
    inference_s = time.time() - t0

    mem_gb = torch.cuda.memory_allocated() / 1024**3
    info = (
        f"**Model:** `{HOLDER.current_kind}`{swap_note}  \n"
        f"**Audio duration:** {duration_s:.2f} s  \n"
        f"**Inference time:** {inference_s*1000:.0f} ms  \n"
        f"**Decoding:** num_beams={int(num_beams)}, "
        f"repetition_penalty={float(repetition_penalty):.2f}, "
        f"max_new_tokens={int(max_new_tokens)}  \n"
        f"**GPU:** {DEVICE} (vision-s006 RTX 3090), "
        f"VRAM allocated {mem_gb:.2f} GB"
    )
    return text.strip(), info


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui():
    examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_examples")
    example_files = []
    if os.path.isdir(examples_dir):
        example_files = sorted(
            os.path.join(examples_dir, f)
            for f in os.listdir(examples_dir)
            if f.lower().endswith((".wav", ".mp3", ".flac", ".m4a"))
        )

    with gr.Blocks(title="speechQwen2VL — Live Demo") as demo:
        gr.Markdown(
            "# speechQwen2VL — Live ASR Demo\n"
            "Qwen2-VL-7B + Whisper encoder + LoRA fine-tuned for ASR. "
            "Models on Hugging Face: "
            "[`DanJZY/Qwen2-VL-7B-Speech`](https://huggingface.co/DanJZY/Qwen2-VL-7B-Speech) (Stage 1), "
            "[`DanJZY/Qwen2-VL-7B-Speech-LoRA`](https://huggingface.co/DanJZY/Qwen2-VL-7B-Speech-LoRA) (Stage 2).  \n"
            "**Headline result:** **7.90% WER / 4.25% CER** on the full LargeScaleASR test set "
            "(8,087 samples) with Stage 2 + decoding fixes.  \n"
            "Switching Stage 1 ↔ Stage 2 reloads the model (~30-60 s) — "
            "RTX 3090 fits one at a time."
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Audio (record from mic or upload a file)",
                )
                model_choice = gr.Radio(
                    choices=["stage2", "stage1"],
                    value=args.default_model,
                    label="Model",
                    info="stage2 = LoRA fine-tuned (best). stage1 = projector only (baseline).",
                )
                with gr.Accordion("Decoding settings", open=False):
                    rep_pen = gr.Slider(1.0, 1.5, value=args.repetition_penalty, step=0.05,
                                        label="repetition_penalty",
                                        info="1.0 = off. 1.1-1.2 reduces repetition loops.")
                    n_beams = gr.Slider(1, 5, value=args.num_beams, step=1,
                                        label="num_beams",
                                        info="1 = greedy. 2 helps the model find the stop token.")
                    max_tok = gr.Slider(32, 512, value=args.max_new_tokens, step=32,
                                        label="max_new_tokens")
                btn = gr.Button("Transcribe", variant="primary")

            with gr.Column(scale=1):
                out_text = gr.Textbox(label="Transcription", lines=4)
                out_info = gr.Markdown()

        if example_files:
            gr.Examples(
                examples=[[f] for f in example_files],
                inputs=[audio_in],
                label=f"Test-set examples ({len(example_files)})",
            )
        else:
            gr.Markdown(
                "_Examples panel hidden — run `python scripts/extract_demo_samples.py` to "
                "populate `scripts/demo_examples/` with a few test samples._"
            )

        btn.click(
            fn=transcribe,
            inputs=[audio_in, model_choice, rep_pen, n_beams, max_tok],
            outputs=[out_text, out_info],
        )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Default model: {args.default_model}")
    print(f"Default decoding: num_beams={args.num_beams}, "
          f"repetition_penalty={args.repetition_penalty}, "
          f"max_new_tokens={args.max_new_tokens}")
    HOLDER.load(args.default_model)

    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
