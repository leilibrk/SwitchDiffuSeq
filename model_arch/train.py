import copy
import functools
import torch
import numpy as np
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import pickle
import os
import glob
from tqdm import tqdm
from utils import dist_util
from utils.fp16_util import (
    zero_grad
)
from utils.nn import update_ema
from utils.step_sample import LossAwareSampler, UniformSampler
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import time
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torch.nn.utils import clip_grad_norm_
from model_arch.Switch_Transformer import SwitchGate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def clear_dir(directory_path):
    try:
        files = glob.glob(os.path.join(directory_path, '*'))
        for file in files:
            if os.path.isfile(file):
                os.remove(file)
        print("Cleared directory to save new best model.")
    except OSError:
        print("Error occurred while deleting files.")

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        schedule_sampler=None,
        weight_decay=0.0,
        epochs=0,
        eval_data=None,
        eval_interval=-1,
        warm_up_steps=100,
        use_llrd=False,
        llrd_rate=0.9,
        model_name="default_model"
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.eval_interval = eval_interval
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.learning_steps = epochs
        self.llrd_rate = llrd_rate

        self.step = 1

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        def _build_moe_groups(model, base_lr, weight_decay):
            m = model.module if hasattr(model, "module") else model
            trunk, experts, router = [], [], []
            for name, p in m.named_parameters():
                if not p.requires_grad:
                    continue
                if "gate" in name:
                    router.append(p)
                elif "ffn.experts" in name or "experts" in name or "moe" in name:
                    experts.append(p)
                else:
                    trunk.append(p)

            print(f"[opt] trunk={len(trunk)} experts={len(experts)} router={len(router)}")
            return torch.optim.AdamW([
                {"params": trunk,   "lr": base_lr},
                {"params": experts, "lr": base_lr * 2.0},
                {"params": router,  "lr": base_lr * 3.0},
            ], weight_decay=weight_decay)

        if "Switch" in model_name:
            print('seperate optimizers for expert and router')
            self.opt = _build_moe_groups(self.model, self.lr, self.weight_decay)
        else:
            self.opt = self.AdamW_LLRD() if use_llrd else AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
 
        self.scheduler = get_cosine_schedule_with_warmup(self.opt, num_warmup_steps = warm_up_steps, num_training_steps=epochs)
        self.ema_params = [copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))]
        self.min_val_loss = float('inf')
        self.train_loss_curve = []
        self.val_loss_curve = []
        self.train_nll_curve = []
        self.val_nll_curve = []
        self.expert_usage_history = defaultdict(list)
        self.expert_drop_history  = defaultdict(list)  
        self.util_log_every = 100  
        self.model_dir = None
        self.model_name = model_name

    def log_expert_stats(self, aux_loss):
        # print every 100 steps;
        if (self.step % 100) != 0:
            return

        m = self.model.module if hasattr(self.model, "module") else self.model

        total_experts = 0
        active_experts = 0
        total_drop_rates, entropies, skews, details = [], [], [], []
        details = []
        for name, module in m.named_modules():
            # accept any module that cached MoE stats
            if not (hasattr(module, "last_load") or hasattr(module, "last_expert_usage")):
                continue

            load = getattr(module, "last_load", None)
            if load is None:
                load = getattr(module, "last_expert_usage", None)

            if load is None:            # still nothing to read
                continue
            if not torch.is_tensor(load):
                # just in case 
                load = torch.as_tensor(load)

            E = load.numel()
            if E == 0:
                continue
            load_np = load.detach().float().cpu().numpy()
            self.expert_usage_history[name].append(load_np)

            total_experts += E

            imp  = getattr(module, "last_importance", None)
            drop = getattr(module, "last_drop_rate", None)
            kept = getattr(module, "last_kept_total", None)

            # threshold: at least ~1 kept token or 1%
            if isinstance(kept, (int, float)) and kept > 0:
                thr = max(0.01, 1.0 / float(kept))
            else:
                thr = 0.01

            # count "active" experts
            layer_active = int((load > thr).sum().item())
            active_experts += layer_active

            # router entropy / skew from soft importance if present
            if imp is not None and torch.is_tensor(imp) and imp.numel() == E:
                p = (imp / imp.sum().clamp_min(1e-8)).clamp_min(1e-12)
                entropies.append(float((-(p * p.log()).sum()).item()))
                skews.append(float((p.max() / p.min().clamp_min(1e-8)).item()))

            if isinstance(drop, torch.Tensor):
                drop = float(drop.item())
            if isinstance(drop, (int, float)):
                total_drop_rates.append(drop)
                self.expert_drop_history[name].append(drop)

            details.append(f"{name}: active {layer_active}/{E} (thr≈{thr:.4f})")

        if total_experts == 0:
            return

        utilization = active_experts / total_experts
        msg = [f"\n[Step {self.step}] Expert Utilization: {utilization:.2%}"]
        if total_drop_rates:
            msg.append(f"  Avg drop rate: {float(sum(total_drop_rates)/len(total_drop_rates)):.2%}")
        if entropies:
            msg.append(f"  Avg router entropy: {float(sum(entropies)/len(entropies)):.3f}")
        if skews:
            msg.append(f"  Avg router skew (max/min): {float(sum(skews)/len(skews)):.2f}")
        from tqdm import tqdm as _tqdm
        _tqdm.write("\n".join(msg))

        if (self.step % 500) == 0:
            for d in details:
                _tqdm.write("  " + d)


    def save_checkpoint(self, directory, filename="checkpoint.pt"):
        os.makedirs(directory, exist_ok=True)
        ckpt = {
            "model_state":  self.model.state_dict(),
            "optimizer_state": self.opt.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "step": self.step,
        }
        torch.save(ckpt, os.path.join(directory, filename))
        print(f"Saved checkpoint to {directory}/{filename}")

    def _save_expert_utilization_artifacts(self):
        if not self.expert_usage_history:
            print("No expert utilization history recorded.")
            return

        # 1) CSV per layer + aggregated CSV
        aggregate_rows = []
        for layer, series in self.expert_usage_history.items():
            # series: list of arrays [num_experts], logged every util_log_every steps
            arr = np.stack(series, axis=0)  # [T, E]
            # Save per-layer CSV (rows=time index, cols=expert_k)
            df = pd.DataFrame(arr, columns=[f"expert_{i}" for i in range(arr.shape[1])])
            df["log_step_index"] = np.arange(len(series))
            df["train_step"] = df["log_step_index"] * self.util_log_every
            cols = ["train_step", "log_step_index"] + [c for c in df.columns if c.startswith("expert_")]
            df = df[cols]
            csv_path = os.path.join(self.model_dir, f"expert_usage_{self._clean(layer)}.csv")
            df.to_csv(csv_path, index=False)

            # For aggregate CSV (mean over time per expert)
            mean_over_time = arr.mean(axis=0)
            for e_idx, v in enumerate(mean_over_time):
                aggregate_rows.append({"layer": layer, "expert": e_idx, "mean_util": v})

            # 2) Per-layer final histogram (mean utilization across training)
            plt.figure(figsize=(7, 4))
            plt.bar(np.arange(arr.shape[1]), mean_over_time)
            plt.xlabel("Expert")
            plt.ylabel("Mean utilization")
            plt.title(f"Mean Expert Utilization — {layer}")
            plt.tight_layout()
            plt.grid(True, axis="y", alpha=0.3)
            plt.savefig(os.path.join(self.model_dir, f"hist_mean_util_{self._clean(layer)}.png"))
            plt.close()

            # 3) Per-layer heatmap over time (experts × time)
            # (transpose so x=time, y=expert)
            plt.figure(figsize=(8, 4))
            plt.imshow(arr.T, aspect="auto", origin="lower", interpolation="nearest")
            plt.colorbar(label="Utilization (fraction)")
            plt.xlabel(f"Log step index (every {self.util_log_every} steps)")
            plt.ylabel("Expert")
            plt.title(f"Expert Utilization Over Time — {layer}")
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_dir, f"heatmap_util_{self._clean(layer)}.png"))
            plt.close()

        if aggregate_rows:
            agg_df = pd.DataFrame(aggregate_rows)
            agg_df.to_csv(os.path.join(self.model_dir, "expert_usage_aggregate.csv"), index=False)

            # overall histogram across layers (mean of means)
            pivot = agg_df.pivot_table(index="expert", values="mean_util", aggfunc="mean")
            plt.figure(figsize=(7, 4))
            plt.bar(pivot.index.values, pivot["mean_util"].values)
            plt.xlabel("Expert (index)")
            plt.ylabel("Mean utilization (avg across layers)")
            plt.title("Global Mean Expert Utilization (averaged across layers)")
            plt.tight_layout()
            plt.grid(True, axis="y", alpha=0.3)
            plt.savefig(os.path.join(self.model_dir, "hist_mean_util_global.png"))
            plt.close()

    @staticmethod
    def _clean(name: str) -> str:
        # file-safe layer name
        return name.replace("/", "_").replace(".", "_").replace(":", "_")

    def AdamW_LLRD(self): 
        print("\n\n======== Using Layer-wise Learning Rate Decay with AdamW ========\n\n")
        lr = self.lr
        lr_decay = lr
        decay_rate = self.llrd_rate # decay from top to bottom layers
        
        new_model_params = []
        
        # ==== layers arrangement (from most bottom to most top): 
        # ==== word_embedding -> lm_head -> time_embed -> input_up_proj.0 -> input_up_proj.2 -> 0 to input_transformers.layer.11 -> position_embeddings -> LayerNorm -> output_down_proj.0 -> output_down_proj.2.
        hidden_layers = [f'input_transformers.layer.{i}.' for i in range(12)]
        before_hidden = ['word_embedding', 'lm_head', 'time_embed', 'input_up_proj'] 
        after_hidden = ['position_embeddings', 'LayerNorm', 'output_down_proj']
        
        for c in before_hidden:
            for name, param in self.model.named_parameters():
                if name.startswith(c):
                    new_model_params += [{'params': param, 'lr': lr_decay}]
                    print(f'name: {name}, lr: {lr_decay}') # for checking
                    
        lr_decay = lr_decay/decay_rate
        
        for c in hidden_layers:
            for name, param in self.model.named_parameters():
                if name.startswith(c):
                    new_model_params += [{'params': param, 'lr': lr_decay}]
                    print(f'name: {name}, lr: {lr_decay}') # for checking           
            lr_decay = lr_decay/decay_rate # lr increases as we move from bottom to top layers
            
        for c in after_hidden:
            for name, param in self.model.named_parameters():
                if name.startswith(c):
                    new_model_params += [{'params': param, 'lr': lr_decay}]
                    print(f'name: {name}, lr: {lr_decay}') # for checking
        
        assert len(new_model_params) == len(list(self.model.parameters()))
        
        return torch.optim.AdamW(new_model_params, weight_decay=self.weight_decay)
        
    def run_loop(self):
        print("\n\n======== Training starts now ========\n\n")
        # Create directory
        model_name = self.model_name
        timestamp = datetime.now().strftime("%m%d_%H%M")
        self.model_dir = f"models/{model_name}_{timestamp}"
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.training_timestamps = []
        start_time = time.time()
        with tqdm( total=self.learning_steps, desc="Training Steps", ascii=True, ncols=100, dynamic_ncols=False, mininterval=0.1, file=sys.stdout ) as pbar:
            while (
                not self.learning_steps or self.step < self.learning_steps
            ):
                batch, cond = next(self.data)
                self.run_step(batch, cond)
                if self.eval_data is not None and self.step % self.eval_interval == 0:
                    batch_eval, cond_eval = next(self.eval_data)
                    self.forward_only(batch_eval, cond_eval)
                
                elapsed_time = time.time() - start_time
                self.training_timestamps.append(elapsed_time)
                self.step += 1
                pbar.update(1)
        
        
        import pandas as pd
        df = pd.DataFrame({
            "step": list(range(1, self.step)),
            "time_sec": self.training_timestamps,
            "train_neg_log_ppl": [-n for n in self.train_nll_curve],
            "val_neg_log_ppl": [-n for n in self.val_nll_curve] if self.val_nll_curve else [None]*len(self.train_nll_curve),
        })
        df.to_csv(f"{self.model_dir}/ppl_progress_{model_name}_{timestamp}.csv", index=False)
        with open(f"{self.model_dir}/train_nll_curve.pkl", "wb") as f:
            pickle.dump(self.train_nll_curve, f)
        with open(f"{self.model_dir}/train_loss_curve.pkl", "wb") as f:
            pickle.dump(self.train_loss_curve, f)

        plt.figure(figsize=(8, 4))
        plt.plot(self.train_loss_curve, label="Training Loss")
        # plt.plot(self.val_loss_curve, label="Validation Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        loss_curve_path = f"{self.model_dir}/loss_curve_{model_name}_{timestamp}.png"
        plt.savefig(loss_curve_path)
        
        # Save separate plot for -log(PPL)
        plt.figure(figsize=(8, 4))
        plt.plot([-n for n in self.train_nll_curve], label="Neg Log Perplexity")
        # plt.plot([-n for n in self.val_nll_curve], label="-log(PPL) Val")
        plt.xlabel("Training Step")
        plt.ylabel("Neg Log Perplexity")
        plt.title("Negative Log Perplexity")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save
        log_ppl_curve_path = f"{self.model_dir}/neg_log_ppl_{model_name}_{timestamp}.png"
        plt.savefig(log_ppl_curve_path)
        plt.close()
        self._save_expert_utilization_artifacts()
        self.save_checkpoint(self.model_dir, filename="final.pt")
        plt.show()
    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.optimize_normal()

    def forward_only(self, batch, cond):
        val_losses = []
        nlls = []
        with torch.no_grad():
            zero_grad(self.model_params)
            for i in range(0, batch.shape[0], self.microbatch):
                micro = batch[i: i + self.microbatch].to(dist_util.dev())
                micro_cond = {
                    k: v[i: i + self.microbatch].to(dist_util.dev())
                    for k, v in cond.items()
                }
                last_batch = (i + self.microbatch) >= batch.shape[0]
                t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
                # print(micro_cond.keys())
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.model,
                    micro,
                    t,
                    model_kwargs=micro_cond,
                )

                losses = compute_losses()
                loss = (losses["loss"] * weights).mean()
                nll = losses["nll"].detach().cpu().mean()
                nlls.append(nll)
                val_losses.append(loss.detach().cpu())
            print(f'Epoch {self.step}/{self.learning_steps} Validation Loss: {np.mean(val_losses)}')
            val_loss = np.mean(val_losses)
            self.val_loss_curve.append(val_loss)
            mean_nll = np.mean(nlls)
            self.val_nll_curve.append(mean_nll)
        dt = datetime.now().strftime("%m%d")
        if not os.path.isdir(f'models/{dt}'):
            os.mkdir(f'models/{dt}')
            
        if self.min_val_loss > np.mean(val_losses):
            self.min_val_loss = np.mean(val_losses)
            clear_dir(f'models/{dt}')
            print(f'============>Saving current best model with min_val_loss={self.min_val_loss}<=============')
            pickle.dump(self.model, open(f"models/{dt}/model_best_epoch_{self.step}_min_val_loss_{np.round(self.min_val_loss, 4)}.pkl", 'wb'))

    def forward_backward(self, batch, cond):
        train_losses = []
        train_nlls = []
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            # print(micro_cond.keys())
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            losses = compute_losses()
            ##########
            # Add auxiliary MoE loss if available
            # model_output = losses.get("model_output", None)
            aux_loss = losses.get("aux_loss", None)

            main_loss = (losses["loss"] * weights).mean()
            if aux_loss is not None:
                total_loss = main_loss + 0.05 * aux_loss
            else:
                total_loss = main_loss 
            ###########
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            # loss = (losses["loss"] * weights).mean()
            nll = losses["nll"].detach().cpu().mean()  # average over batch
            # train_losses.append(loss.detach().cpu())
            train_losses.append(total_loss.detach().cpu())
            train_nlls.append(nll)

            # loss.backward()
            #### FLOP Count Logging Every 100 Steps
            if self.step % 100 == 0 and i == 0:
                try:
                    # Create a dummy timesteps tensor:
                    dummy_timesteps = torch.zeros(
                        micro.size(0), dtype=torch.long, device=micro.device
                    )
                    # Compute FLOPs with both inputs:
                    flop_analysis = FlopCountAnalysis(
                        self.model, (micro, dummy_timesteps)
                    )
                    total_flops = flop_analysis.total()
                    print(f"\n[Step {self.step}] Estimated FLOPs: {total_flops/1e9:.2f} GFLOPs")
                    print(flop_count_table(flop_analysis, max_depth=2))
                except Exception as e:
                    print(f"[Step {self.step}] FLOP analysis failed: {e}")
            ####
            total_loss.backward()
            if aux_loss is not None:
                tqdm.write(f"[Step {self.step}] Aux Loss: {aux_loss.item():.6f}")

        # Add expert utilization monitoring
        self.log_expert_stats(aux_loss)    
        mean_loss = np.mean(train_losses)
        mean_nll = np.mean(train_nlls)
        tqdm.write(f'Epoch {self.step}/{self.learning_steps} Training Loss: {mean_loss}')
        self.train_loss_curve.append(mean_loss)
        self.train_nll_curve.append(mean_nll)

    def optimize_normal(self):
#         self._anneal_lr()
        # 1) clip before stepping
        # print('grad clip')
        # clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.opt.step()
        self.scheduler.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.learning_steps:
            return
        frac_done = self.step / self.learning_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr