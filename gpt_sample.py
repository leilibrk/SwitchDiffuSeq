import os, json, math, time, sys, random
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

CHECKPOINT_PATH = "models/final models/math/gpt2_scratch_gsm8k_2000/final.pt"          # point this to saved final.pt
NUM_SAMPLES_PER_INPUT = 3                
MAX_EVAL_SAMPLES = 20                    
device = "cuda" if torch.cuda.is_available() else "cpu"

class TextDataset(Dataset):
    """Expects jsonl with {"src": "...", "trg": "..."}."""
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
    def __getitem__(self, idx): return self.samples[idx]

def build_from_checkpoint(ckpt_path):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

    ckpt = torch.load(ckpt_path, map_location="cpu")

    cfg_saved = ckpt.get("config", {})
    saved_tokenizer_field = ckpt.get("tokenizer", None) 
    tokenizer_id = cfg_saved.get("tokenizer", None)
    if tokenizer_id is None:
        # Fallback mapping for common cases
        if isinstance(saved_tokenizer_field, str) and saved_tokenizer_field.lower().startswith("gpt2tokenizer"):
            tokenizer_id = "gpt2"
        else:
            tokenizer_id = "gpt2"  # safe default

    seq_len   = cfg_saved.get("seq_len", 256)
    n_layer   = cfg_saved.get("n_layer", 4)
    n_head    = cfg_saved.get("n_head", 4)
    n_embd    = cfg_saved.get("n_embd", 512)
    dropout   = cfg_saved.get("dropout", 0.1)
    data_dir  = cfg_saved.get("data_dir", "data/gsm8k")
    model_name = cfg_saved.get("model_name", "gpt2_loaded")

    tok = GPT2TokenizerFast.from_pretrained(tokenizer_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = GPT2Config(
        vocab_size=tok.vocab_size,
        n_positions=seq_len,
        n_ctx=seq_len,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        resid_pdrop=dropout,
        embd_pdrop=dropout,
        attn_pdrop=dropout,
    )
    model = GPT2LMHeadModel(cfg)
    model.resize_token_embeddings(len(tok))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    model.to(device).eval()
    return model, tok, {"seq_len": seq_len, "data_dir": data_dir, "model_name": model_name}


@torch.no_grad()
def sample_once(model, tok, src, max_new=50, temperature=1.0, top_k=None, top_p=0.9):
    eos_id = tok.eos_token_id
    prompt_ids = tok(src + tok.eos_token, return_tensors="pt").input_ids.to(device)
    gen = prompt_ids.clone()
    for _ in range(max_new):
        logits = model(input_ids=gen).logits[:, -1, :] / max(temperature, 1e-8)
        # top-k
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits = logits.masked_fill(logits < v[..., [-1]], float("-inf"))
        # top-p (nucleus)
        if top_p is not None and 0 < top_p < 1:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            mask = cum > top_p
            mask[..., 0] = False
            filtered = torch.full_like(sorted_logits, float("-inf"))
            filtered[~mask] = sorted_logits[~mask]
            logits = torch.zeros_like(logits).scatter_(1, sorted_idx, filtered)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        gen = torch.cat([gen, next_id], dim=1)
        if next_id.item() == eos_id:
            break
    cont = gen[0, prompt_ids.size(1):]
    return tok.decode(cont, skip_special_tokens=True)

def main():
    model, tok, meta = build_from_checkpoint(CHECKPOINT_PATH)
    seq_len   = meta["seq_len"]
    data_dir  = meta["data_dir"]
    model_name= meta["model_name"]

    # Load test set pairs
    test_fp = os.path.join(data_dir, "test.jsonl")
    test_ds = TextDataset(test_fp, tok, seq_len)
    pairs = test_ds.samples[:MAX_EVAL_SAMPLES]

    timestamp = datetime.now().strftime("%m%d_%H%M")
    out_path = f"samples_{model_name}_{timestamp}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("===== SAMPLES =====\n\n")

        for k in range(NUM_SAMPLES_PER_INPUT):
            print(f"\n=== Generation round {k+1} ===")
            # make runs reproducible but different
            torch.manual_seed(1234 + k); random.seed(1234 + k)
            for i, (src, tgt) in enumerate(pairs, start=1):
                out = sample_once(
                    model, tok, src,
                    max_new=50, 
                    temperature=0.9 + 0.05*k,
                    top_k=None,
                    top_p=0.9
                )
                f.write(f"[Sample {i}-run{k+1}]\n")
                f.write(f"Source : {src}\n")
                f.write(f"Output : {out}\n")
                f.write(f"Target : {tgt}\n")
                f.write("---------------\n")

                # console log
                print(f"[Sample {i}-run{k+1}]")
                print(f"Source : {src}")
                print(f"Output : {out}")
                print(f"Target : {tgt}")
                print("---------------")

    print(f"\nAll samples (with {NUM_SAMPLES_PER_INPUT} per input) saved to: {out_path}")

if __name__ == "__main__":
    main()
