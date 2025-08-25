import os, json, math, pickle, time, sys
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, get_linear_schedule_with_warmup

CONFIG = {
    "tokenizer": "gpt2",
    "seq_len": 256,
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 512,
    "dropout": 0.1,
    "lr": 3e-4,
    "batch_size": 32,
    "max_steps": 2000,
    "warmup_steps": 100,
    "data_dir": "data/QQP",
    "seed": 102,
    "model_name": "gpt2_scratch_QQP_2000",
    "sample_max_new_tokens": 50,
}
torch.manual_seed(CONFIG["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"
LN2 = math.log(2.0)

class TextDataset(Dataset):
    """
    Expects jsonl with {"src": "...", "trg": "..."}.
    Trains causal LM to predict trg given src (+eos).
    """
    def __init__(self, path, tokenizer, seq_len):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.samples.append((obj.get("src",""), obj.get("trg","")))
        self.tok = tokenizer
        self.seq_len = seq_len
        self.eos = self.tok.eos_token

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        src, tgt = self.samples[idx]
        text = src + self.eos + tgt  # [src + <eos> + tgt]
        enc = self.tok(
            text, truncation=True, max_length=self.seq_len,
            padding="max_length", return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Boundary length for src+eos
        src_ids = self.tok(src + self.eos, truncation=True, max_length=self.seq_len, return_tensors="pt")["input_ids"].squeeze(0)
        src_len = (src_ids != self.tok.pad_token_id).sum().item()

        labels = input_ids.clone()
        labels[:src_len-1] = -100              # ignore prompt
        labels[attention_mask == 0] = -100     # ignore padding

        return input_ids, attention_mask, labels, src, tgt

def build_model_and_tokenizer():
    tok = GPT2TokenizerFast.from_pretrained(CONFIG["tokenizer"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = GPT2Config(
        vocab_size=tok.vocab_size,
        n_positions=CONFIG["seq_len"],
        n_ctx=CONFIG["seq_len"],
        n_embd=CONFIG["n_embd"],
        n_layer=CONFIG["n_layer"],
        n_head=CONFIG["n_head"],
        resid_pdrop=CONFIG["dropout"],
        embd_pdrop=CONFIG["dropout"],
        attn_pdrop=CONFIG["dropout"],
    )
    model = GPT2LMHeadModel(cfg)
    model.resize_token_embeddings(len(tok))
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M")

    return model.to(device), tok

def train():
    model, tok = build_model_and_tokenizer()

    train_ds = TextDataset(os.path.join(CONFIG["data_dir"], "train.jsonl"), tok, CONFIG["seq_len"])
    test_ds  = TextDataset(os.path.join(CONFIG["data_dir"], "test.jsonl"),  tok, CONFIG["seq_len"])
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True)

    optim = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=CONFIG["warmup_steps"],
        num_training_steps=CONFIG["max_steps"]
    )

    # === Tracking (to mirror your other model) ===
    train_loss_curve = []           # same as NLL in nats
    train_nll_curve = []            # duplicate of loss for compatibility
    train_ppl_curve = []
    training_timestamps = []

    timestamp = datetime.now().strftime("%m%d_%H%M")
    model_name = CONFIG['model_name']
    model_dir = f"models/{model_name}_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)

    model.train()
    step = 0
    start_time = time.time()

    with tqdm(total=CONFIG["max_steps"], desc="Training Steps", ascii=True, ncols=100, dynamic_ncols=False, mininterval=0.1, file=sys.stdout) as pbar:
        while step < CONFIG["max_steps"]:
            for input_ids, attention_mask, labels, _, _ in train_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss  # mean CE over non-ignored labels (nats)

                optim.zero_grad()
                loss.backward()
                optim.step()
                sched.step()

                ppl = math.exp(loss.item())

                # Record (match your other model’s fields)
                train_loss_curve.append(loss.item())
                train_nll_curve.append(loss.item())  # same values as loss
                train_ppl_curve.append(ppl)
                training_timestamps.append(time.time() - start_time)

                if step % 20 == 0:
                    print(f"Step {step} | loss {loss.item():.4f} | ppl {ppl:.2f} | -log(PPL) {-loss.item():.4f} nats")

                step += 1
                pbar.update(1)
                if step >= CONFIG["max_steps"]:
                    break

    # === Save CSV in your format ===
    df = pd.DataFrame({
        "step": list(range(1, step+1)),
        "time_sec": training_timestamps,
        "train_neg_log_ppl": [-n for n in train_nll_curve],  # -loss
        "val_neg_log_ppl": [None]*len(train_nll_curve),      # placeholder to match your format
    })
    df.to_csv(f"{model_dir}/ppl_progress_{model_name}_{timestamp}.csv", index=False)

    # === Save pickles with your names ===
    with open(f"{model_dir}/train_nll_curve.pkl", "wb") as f:
        pickle.dump(train_nll_curve, f)
    with open(f"{model_dir}/train_loss_curve.pkl", "wb") as f:
        pickle.dump(train_loss_curve, f)

    # === Plots matching your naming ===
    plt.figure(figsize=(8, 4))
    plt.plot(train_loss_curve, label="Training Loss")
    plt.xlabel("Training Step"); plt.ylabel("Loss"); plt.title("Training Loss Curve")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{model_dir}/loss_curve_{model_name}_{timestamp}.png")

    plt.figure(figsize=(8, 4))
    plt.plot([-n for n in train_nll_curve], label="Neg Log Perplexity")
    plt.xlabel("Training Step"); plt.ylabel("Neg Log Perplexity"); plt.title("Negative Log Perplexity")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{model_dir}/neg_log_ppl_{model_name}_{timestamp}.png")
    plt.close('all')

    # === Sampling ===
    model.eval()
    eos_id = tok.eos_token_id
    with open(f"{model_dir}/samples.txt", "w", encoding="utf-8") as out_fp:
        out_fp.write("===== SAMPLES =====\n\n")
        for i, (src, tgt) in enumerate(test_ds.samples[:20], start=1):
            prompt_ids = tok(src + tok.eos_token, return_tensors="pt").input_ids.to(device)
            gen = prompt_ids.clone()
            with torch.no_grad():
                for _ in range(CONFIG["sample_max_new_tokens"]):
                    logits = model(input_ids=gen).logits[:, -1, :]
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)
                    gen = torch.cat([gen, next_id], dim=1)
                    if next_id.item() == eos_id:
                        break
            cont = gen[0, prompt_ids.size(1):]
            out_fp.write(f"[Sample {i}]\n")
            out_fp.write(f"Source : {src}\n")
            out_fp.write(f"Output : {tok.decode(cont, skip_special_tokens=True)}\n")
            out_fp.write(f"Target : {tgt}\n\n")

    # Optional: save a final checkpoint
    torch.save({"model_state_dict": model.state_dict(),
                "tokenizer": tok.__class__.__name__,
                "config": CONFIG}, os.path.join(model_dir, "final.pt"))

if __name__ == "__main__":
    train()
