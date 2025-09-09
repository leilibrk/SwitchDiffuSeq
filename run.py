from model_arch.run_train import create_model_and_diffusion
from utils.step_sample import create_named_schedule_sampler
from utils import dist_util
from model_arch.train import TrainLoop
from utils.data import load_data_text
from model_arch.tokenizer import load_tokenizer, load_model_emb
from model_arch.sampling import sampling
from transformers import set_seed
import yaml
from datetime import datetime
import argparse

config_fp = './config/config.yaml'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='Bert', help="Model type: Bert or Switch")
    parser.add_argument('--num_experts', type=int, default=1, help="Number of experts (used if Switch)")
    parser.add_argument('--model_name', type=str, required=True, help="Name for saving model outputs")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to dataset directory")
    args = parser.parse_args()
    
    dist_util.clear_cache()
    config = yaml.load(open(config_fp, 'r'), Loader=yaml.SafeLoader)
    set_seed(config['seed'])
    
    tokenizer = load_tokenizer(config['tokenizer'], config['custom_vocab_fp'])
    model_weight, tokenizer = load_model_emb(config['hidden_dim'], tokenizer)
    vocab_size = tokenizer.vocab_size
    print('Vocab size: ', vocab_size)
    
    data = load_data_text(
        batch_size=config['batch_size'],
        seq_len=config['seq_len'],
        data_dir=args.data_dir,
        loaded_vocab=tokenizer,
        model_emb=model_weight # use model's weights as init
    )
    
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
                        args.model_type,
                        args.num_experts
                    )
    
    model.to(dist_util.dev())
    print(model.input_transformers)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)

    TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=config['batch_size'],
            microbatch=config['microbatch'],
            lr=config['lr'],
            ema_rate=config['ema_rate'],
            schedule_sampler=schedule_sampler,
            weight_decay=config['weight_decay'],
            epochs=config['epochs'],
#             eval_data=data_valid,
            eval_interval=config['eval_interval'],
            model_name = args.model_name
        ).run_loop()
    
    word_lst_source, word_lst_recover, word_lst_ref, inter_lst_recover = sampling(model, 
                                                               diffusion, 
                                                               tokenizer, 
                                                               data_dir=args.data_dir, 
                                                               batch_size=config['sampling_batch_size'], 
                                                               split='test', 
                                                               seq_len=config['seq_len'],
                                                               clip_denoised=config['clip_denoised'],
                                                               top_p=config['top_p'],
                                                               clamp_step=config['clamp_step'])
    print("\n===== SAMPLES =====\n")
    # Get encoder class name
    model_name = args.model_name
    # Create timestamp
    timestamp = datetime.now().strftime("%m%d_%H%M")
    # Compose file name
    output_path = f"samples_{args.model_name}_{timestamp}.txt"
    with open(output_path, "w") as f:
        f.write("===== SAMPLES =====\n\n")
        for i, (src, pred, ref) in enumerate(zip(word_lst_source, word_lst_recover, word_lst_ref)):
            f.write(f"[Sample {i+1}]\n")
            f.write(f"Source : {src}\n")
            f.write(f"Output : {pred}\n")
            f.write(f"Target : {ref}\n")
            f.write("---------------\n")

            # Also print to console
            print(f"[Sample {i+1}]")
            print(f"Source : {src}")
            print(f"Output : {pred}")
            print(f"Target : {ref}")
            print("---------------")

    print(f"\nAll samples saved to: {output_path}")