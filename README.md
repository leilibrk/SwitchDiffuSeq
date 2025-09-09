
# 🌀 SwitchDiffuSeq: Mixture‑of‑Experts in Text Diffusion Models

This project experiments with **replacing the BERT‑style Transformer backbone** in a text diffusion model with a **Switch Transformer** (Mixture‑of‑Experts).  
The goal is to study whether sparse routing can improve **efficiency**, **expert specialization**, and **generation quality/diversity** in diffusion‑based text generation.

---

## Project Structure

```
.
├── model_arch/
│   ├── gaussian_diffusion.py   # Core diffusion process (forward & reverse, training losses)
│   ├── run_train.py            # Factory: build model (Bert or Switch) + diffusion
│   ├── sampling.py             # Sampling utilities for diffusion models
│   ├── Switch_Transformer.py   # Switch Transformer (MoE) implementation
│   ├── tokenizer.py            # Tokenizer setup (BERT/custom)
│   ├── train.py                # Training loop (optimizer, scheduler, expert logging)
│   ├── transformer.py          # Baseline Transformer encoder + Switch wrapper
├── config/
│   └── config.yaml             # Training configuration
├── run.py                      # Main entry point for training (parses flags, calls model_arch/run_train)
├── multiple_sample.py          # Multi‑sample generation script for diffusion
├── gpt.py                      # GPT baseline training
├── gpt_sample.py               # GPT baseline multiple sampling
│ 
├── models/                     # Saved models and results
│ ├── final models/             # Main experiments (organized per dataset)
│ │ ├── greetings/              # Greeting dataset results
│ │ ├── math/                   # Math reasoning dataset results
│ │ ├── QQP/                    # Quora Question Pairs results
│ │ └── TruthQA/                # TruthfulQA results
│ └── old models/               # preliminary runs (kept for reference)
│
└── run.ipynb                   # Structured notebook to re-generate plots
```

To reproduce the figures in the report, open `run.ipynb` and run it cell-by-cell.

### Contents of Each Dataset Folder in `models/final models/`

Inside every dataset directory (e.g., `greetings/`, `math/`, `QQP/`, `TruthQA/`), the following results are stored:

- **Model Variants**
  - BERT baseline (Diffusion Model with BERT encoder)
  - Switch Transformer with 2 experts (Diffusion Model with Switch Transformer encoder)
  - Switch Transformer with 4 experts (Diffusion Model with Switch Transformer encoder)
  - Switch Transformer with 8 experts (Diffusion Model with Switch Transformer encoder)
  - GPT baseline

- **Training Settings**
  - Each model is trained both **with** and **without gradient clipping**

- **Artifacts**
  - Saved checkpoints (`.pt` files)  
  - CSV training logs  
  - Generated sample outputs  
  - Training plots (loss curves, negative log perplexity)
  - Expert utilization logs (per-expert load and importance statistics) 

---

## Installation

```bash
git clone https://github.com/yourusername/SwitchDiffuSeq.git
cd SwitchDiffuSeq

# (optional) create a venv/conda env
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Minimum deps: `torch`, `transformers`, `numpy`, `pandas`, `tqdm`, `matplotlib`, `fvcore`.

---

## Configuration

Global defaults live in **`config/config.yaml`**. Example (snippet):

```yaml
tokenizer: 'bert'                 # tokenizer choice
custom_vocab_fp: 'bert-base-uncased'

lr: 0.0001
batch_size: 32
microbatch: 4
epochs: 2000
eval_interval: 1000
ema_rate: '0.9999'

diffusion_steps: 1000
noise_schedule: 'sqrt'
predict_xstart: True
rescale_timesteps: True

seq_len: 256
hidden_t_dim: 128
hidden_dim: 128
dropout: 0.1


sampling_step: 1000
```

---

## Training

### Train a **Switch Transformer** diffusion model
```bash
python run.py   --model_type Switch  --num_experts 4   --model_name Switch_greet   --data_dir data/greetings
```
### Train a diffusion model with BERT Transformer
```bash
python run.py   --model_type Bert   --model_name Bert_greet   --data_dir data/greetings
```
#### Training Arguments

| Argument            | Description |
|---------------------|-------------|
| `--model_type`      | Backbone type: `Switch` (Switch Transformer) or `Bert` (standard Transformer). |
| `--num_experts`     | Number of experts in the SwitchFFN (only used when `--model_type Switch`). |
| `--model_name`      | Name for saving checkpoints, logs, and outputs. |
| `--data_dir`        | Path to the dataset directory (e.g. `data/greetings`). |

### Train a **GPT** baseline (for comparison)
```bash
python gpt.py   --model_name gpt_greet  --data_dir data/greetings 
```




---

## Sampling

### Multiple‑sample generation (diffusion)
```bash
python multiple_sample.py \
  --model_type Switch \
  --num_experts 4 \
  --model_name Switch_greet \
  --data_dir data/greetings \
  --ckpt_fp "./models/final models/greetings/final models no clip/Switch_4e_greet_2000_noclip_cap=1.75/final.pt"
```

### Multiple‑sample generation (GPT)
```bash
python gpt_sample.py   --checkpoint_path "models/final models/math/gpt2_scratch_gsm8k_2000/final.pt"
```
Note: `--ckpt_fp` and `--checkpoint_path` is the file path to the model checkpoint (`.pt`) used for sampling.

**Sample output format**
```
[Sample 1]
Source : ...
Output : ...
Target : ...
---------------
```

---

## Logging & Monitoring

- **Curves:** training loss & negative log perplexity are saved under `models/<run_name>_timestamp/`:
  - `loss_curve_*.png`
  - `neg_log_ppl_*.png`
  - CSV logs for progress (`ppl_progress_*.csv`) and pickled curves.

- **FLOPs:** every 100 steps the loop can emit estimates via `fvcore`:
  ```
  [Step 100] Estimated FLOPs: 100.08 GFLOPs
  ```
  which means ~0.1 TFLOPs per forward for the given microbatch/sequence length.

- **Expert Utilization (MoE):** printed every 100 steps (configurable) from the training loop in `model_arch/train.py`:
- Utilization (% experts above activity threshold)
- Drop rate (capacity overflow)
- Router entropy (soft importance spread)
- Router skew (max/min prob)
```
[Step 700] Expert Utilization: 81.25%
  Avg drop rate: 60.77%
  Avg router entropy: 1.959
  Avg router skew (max/min): 4.99
```

---

## Key Implementation Points

- `model_arch/transformer.py`
  - `TransformerNetModel.forward(...)` handles **both** Bert and Switch backbones.
  - In **Switch** mode it returns `MoEModelOutput(last_hidden_state=h, aux_loss=aux_loss)` so `gaussian_diffusion.training_losses(...)` can add auxiliary load‑balancing.
  - Positional/time embeddings are combined with inputs consistently across both modes.
- `model_arch/Switch_Transformer.py`
  - `SwitchGate`: top‑1 routing with capacity factor; logs `last_importance`, `last_load`, `last_drop_rate`, `last_kept_total` for monitoring.
  - `SwitchMoE`: dispatches tokens to experts; dropped tokens can be passed through (identity) to stabilize diffusion.
  - `SwitchTransformerBlock`: MQA attention + MoE FFN with residual + LayerNorm.
- `model_arch/run_train.py`
  - `create_model_and_diffusion(...)` wires backbone + diffusion (`SpacedDiffusion`) using the chosen noise schedule and step count.
- `model_arch/gaussian_diffusion.py`
  - Implements training losses for diffusion with optional inclusion of `aux_loss` when available from the backbone.
- `model_arch/train.py`
  - **TrainLoop:** training/eval, EMA, checkpoints, logging.
  - Optimizer: trunk/experts/router with scaled LRs (for Switch) or AdamW/LLRD.
  - Adds **aux MoE loss** to main loss.
  - Logs **expert stats** (utilization, drop, entropy, skew) + saves CSV/plots.
  - Supports FLOPs reporting + optional gradient clipping.
  - Saves checkpoints and training curves (loss, -logPPL).
- `run.py`
  - Loads global defaults from `config/config.yaml`, tokenizer & embeddings, builds `(model, diffusion)` via `create_model_and_diffusion(...)`, creates a schedule sampler, and starts training with `TrainLoop(...).run_loop()`.
  - After training, calls `model_arch/sampling.py::sampling(...)` to generate qualitative samples

---

## Configuration Notes

- **Capacity Factor (MoE):**  
  To change the routing **capacity factor** in the Switch Transformer, edit  
  `model_arch/Switch_Transformer.py` → inside the `SwitchGate` and `SwitchMoE` classes.  
  Set the argument `capacity_factor: float` to your desired value (recommended range: **1.0 – 2.0**).  
  This controls how many tokens per expert are processed before dropping.

- **Gradient Clipping:**  
  To enable **gradient clipping**, open `model_arch/train.py` → inside the `optimize_normal(self)` method.  
  Uncomment the line with `clip_grad_norm_`.  
  This will prevent exploding gradients during training.
---

## Research Notes

- All models here are **trained from scratch** by default (no pre‑trained BERT/Switch weights) to isolate the effect of the backbone architecture.
- MoE knobs you may want to sweep:
  - `--num_experts` (e.g., 2/4/8)
  - capacity factor (set inside the MoE gate; higher reduces drops but raises compute)
  - aux‑loss weight (scales contribution in the total objective)
- For long sequences or small microbatches, raise capacity factor to reduce drop rate during early training.

---

## Acknowledgements

This project builds upon existing open-source implementations:

- **Baseline Model (DiffuSeq):**  
  Original code from [xianyingkong/diffusion-text-generation](https://github.com/xianyingkong/diffusion-text-generation/).  
  Modifications have been made in:
  - `train.py`
  - `gaussian_diffusion.py`
  - `transformer.py`
---
### References

- **DiffuSeq (Baseline model):**  
  Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu, and Lingpeng Kong.  
  *DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models.*  
  *International Conference on Learning Representations (ICLR), 2023.*  
  [[Paper]](https://openreview.net/forum?id=K3dv1-rY94)

```bibtex
@inproceedings{gong2022diffuseq,
  author    = {Gong, Shansan and Li, Mukai and Feng, Jiangtao and Wu, Zhiyong and Kong, Lingpeng},
  title     = {{DiffuSeq}: Sequence to Sequence Text Generation with Diffusion Models},
  booktitle = {International Conference on Learning Representations, ICLR},
  year      = {2023}
}
