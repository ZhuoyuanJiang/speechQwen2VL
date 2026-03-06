# Plan: Session 5 — Set Up Environment & Train Audio Projector (Stage 1)

## Context
The project fine-tunes Qwen2-VL to understand speech by adding a Whisper audio encoder and training a projection layer. Sessions 1-4 built the model architecture; the `audio_projector` (~17M params) is still randomly initialized. This session sets up the server environment from scratch and runs Notebook 05 to train it.

**Server**: vllab15 — 8x RTX 6000 Ada (49GB each), CUDA driver 12.4, local SSDs at `/ssd1-4/`
**Caches**: All HF caches (models + datasets) already configured to `/ssd1/zhuoyuan/hf_cache`

**Download locations** (all on local SSD, not home dir):
- **Model** (`DanJZY/Qwen2-VL-7B-Speech`): `/ssd1/zhuoyuan/hf_cache/hub` (via `HF_HOME`)
- **Dataset** (`speechbrain/LargeScaleASR`): `/ssd1/zhuoyuan/hf_cache` (via `HF_DATASETS_CACHE`)

---

## Step 2: Create conda environment

1. Run `conda env create -f environment.yml` (installs Python 3.10, cuda-toolkit 12.1.1, ffmpeg, libsndfile, and all pip deps including PyTorch 2.4.1+cu121)
   - CUDA 12.1 runtime is compatible with server's 12.4 driver (driver ≥ runtime)
   - This environment.yml is tested on this server from another project
2. If `conda env create` fails, fall back to manual setup per the prompt (create env, conda install deps, pip install -r requirements.txt)
3. Activate: `conda activate speech_qwen2vl`

## Step 3: Install forked libraries

1. Run `bash scripts/setup_forks.sh`
   - Clones `ZhuoyuanJiang/transformers` (branch `speech-qwen2vl`) and `ZhuoyuanJiang/Qwen3-VL` (branch `speech-qwen2vl`) into `forks/`
   - Installs both in editable mode (`pip install -e`)
2. Verify forks are installed correctly:
   ```
   python -c "import transformers; print(transformers.__version__, transformers.__file__)"
   # Expect: 4.56.0.dev0, path inside forks/transformers/
   python -c "import qwen_vl_utils; print(qwen_vl_utils.__file__)"
   # Expect: path inside forks/Qwen2-VL/qwen-vl-utils/
   ```
   If paths point to site-packages, re-run `setup_forks.sh`.

## Step 4: HuggingFace & wandb login

- HF token is already in `.bashrc` as `HF_TOKEN` — verify with `huggingface-cli whoami`
- If not logged in: `huggingface-cli login` (account: DanJZY)
- `wandb login` — enter API key when prompted

## Step 5: Verify TRL compatibility

```
python -c "import trl; print(trl.__version__)"
python -c "from trl import SFTConfig; import inspect; sig = inspect.signature(SFTConfig); print('max_seq_length' in sig.parameters, 'max_length' in sig.parameters)"
```
- If `True False` → notebook is correct as-is
- If `False True` → rename `max_seq_length` to `max_length` in notebook's SFTConfig cell
- Also verify `dataset_text_field` and `dataset_kwargs` are valid SFTConfig params

## Step 6: Run the notebook

Convert and run as Python script:
```bash
cd /home/zhuoyuan/projects/speechQwen2VL
jupyter nbconvert --to script notebooks/05_training_stage1_adapter.ipynb
python notebooks/05_training_stage1_adapter.py
```

### What to watch at each section (per the prompt):
1. **Section 1 (Env)**: Should show GPUs and `transformers: 4.56.0.dev0` from forks path
2. **Section 2 (Dataset)**: Check drop count from pre-filtering. If >5% dropped, investigate. If `apply_chat_template` fails on placeholder, adjust dummy message.
3. **Section 4 (Model load)**: ~16.7 GB GPU memory. If OOM, check `nvidia-smi` for other processes.
4. **Section 5 (Collator test)**: **Most critical.** Decoded labels must match ground truth + `<|im_end|>`. Token view must show MASKED→real label transition at correct boundary.
5. **Section 6 (SFTConfig)**: Watch for param name issues (see Step 5)
6. **Section 8 (Training)**: Loss should decrease significantly within first epoch. If GPU utilization is low, try increasing `dataloader_num_workers`. If pickle errors, set to 0.
7. **Section 9 (Inference)**: Output should be less refusal-like than Session 4's garbage. Partial/noisy transcription = success.
8. **Section 10 (Push)**: Only push if inference looks reasonable. Only 1 of 4 safetensor shards should change.

### Handling errors
- If SFTTrainer conflicts with custom collator → fall back to plain `Trainer` + `TrainingArguments` from transformers
- If flash-attn compilation fails during env setup → install without flash-attn first, add it after (or use `--no-build-isolation`)
- If `dataloader_num_workers > 0` causes pickle errors → set to 0

## Verification
1. Conda env created and forks installed (transformers from forks/, not site-packages)
2. Freeze check: exactly ~17M trainable params
3. Collator sanity: decoded labels match transcript + `<|im_end|>`
4. Training loss decreases over 3 epochs
5. Post-training inference: coherent output (not garbage/refusal)
6. HF push: only 1 of 4 shards changed
