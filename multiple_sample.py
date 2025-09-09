from model_arch.run_train import create_model_and_diffusion
from utils import dist_util
from utils.data import load_data_text
from model_arch.tokenizer import load_tokenizer, load_model_emb
from model_arch.sampling import sampling
from transformers import set_seed
import yaml
from datetime import datetime
import torch
import time
import argparse

config_fp = './config/config.yaml'
NUM_SAMPLES_PER_INPUT = 3                        # how many generations per input

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_fp', type=str, required=True, help="Path to checkpoint file (.pt)")

    args = parser.parse_args()
    dist_util.clear_cache()
    config = yaml.load(open(config_fp, 'r'), Loader=yaml.SafeLoader)
    set_seed(config['seed'])

    # tokenizer + embeddings 
    tokenizer = load_tokenizer(config['tokenizer'], config['custom_vocab_fp'])
    model_weight, tokenizer = load_model_emb(config['hidden_dim'], tokenizer)
    vocab_size = tokenizer.vocab_size
    print('Vocab size: ', vocab_size)

    # model + diffusion 
    model, diffusion = create_model_and_diffusion(
        config['hidden_t_dim'],
        config['hidden_dim'],
        vocab_size,
        config['transformer'],
        config['use_plm_init'],
        config['dropout'],
        config['diffusion_steps'],
        config['noise_schedule'],
        config['predict_xstart'],
        config['rescale_timesteps'],
        config['model_type'],
        config['num_experts']
    )
    model.to(dist_util.dev())

    # load weights
    ckpt = torch.load(args.ckpt_fp, map_location=dist_util.dev())
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    print(f"Loaded checkpoint from {args.ckpt_fp}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # sampling loop
    model.eval()
    model_name = config['model_name']
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_path = f"samples_{model_name}_{timestamp}.txt"
    total_generated = 0
    start_time = time.time()  # start timing
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("===== SAMPLES =====\n\n")

        for k in range(NUM_SAMPLES_PER_INPUT):
            print(f"\n=== Generation round {k+1} ===")
            word_src, word_rec, word_ref, _ = sampling(
                model, diffusion, tokenizer,
                data_dir=config['data_dir'],
                batch_size=config['sampling_batch_size'],
                split='test',
                seq_len=config['seq_len'],
                clip_denoised=config['clip_denoised'],
                top_p=config['top_p'],
                clamp_step=config['clamp_step']
            )

            for i, (src, pred, ref) in enumerate(zip(word_src, word_rec, word_ref)):
                total_generated += 1  # count examples
                f.write(f"[Sample {i+1}-run{k+1}]\n")
                f.write(f"Source : {src}\n")
                f.write(f"Output : {pred}\n")
                f.write(f"Target : {ref}\n")
                f.write("---------------\n")

                # print to console
                print(f"[Sample {i+1}-run{k+1}]")
                print(f"Source : {src}")
                print(f"Output : {pred}")
                print(f"Target : {ref}")
                print("---------------")

    # print(f"\nAll samples (with {NUM_SAMPLES_PER_INPUT} per input) saved to: {output_path}")
    end_time = time.time()  # end timing
    elapsed = end_time - start_time
    examples_per_sec = total_generated / elapsed
    
    print(f"\nAll samples saved to: {output_path}")
    print(f"Total generated: {total_generated}")
    print(f"Elapsed time: {elapsed:.2f} sec")
    print(f"Inference speed: {examples_per_sec:.2f} examples/sec")
