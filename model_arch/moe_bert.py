# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers.models.bert.modeling_bert import BertAttention
# from typing import Optional, Tuple
# from transformers.modeling_outputs import BaseModelOutput
# class SimpleMoEFFN(nn.Module):
#     def __init__(self, hidden_dim, expert_dim, num_experts=2, k=1):
#         super().__init__()
#         self.num_experts = num_experts
#         self.k = k

#         self.experts = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(hidden_dim, expert_dim),
#                 nn.ReLU(),
#                 nn.Linear(expert_dim, hidden_dim)
#             ) for _ in range(num_experts)
#         ])
#         self.gate = nn.Linear(hidden_dim, num_experts)
#         self.dropout = nn.Dropout(0.1)
#         self.layernorm = nn.LayerNorm(hidden_dim)

#     def forward(self, x):
#         B, T, D = x.shape
#         residual = x
#         x_flat = x.view(-1, D)

#         gate_logits = self.gate(x_flat)
#         topk_val, topk_idx = torch.topk(gate_logits, self.k, dim=-1)
#         topk_weights = F.softmax(topk_val, dim=-1)

#         outputs = []
#         for i in range(self.k):
#             expert_ids = topk_idx[:, i]
#             output_i = torch.zeros_like(x_flat)
#             for e in range(self.num_experts):
#                 mask = (expert_ids == e)
#                 if mask.any():
#                     output_i[mask] = self.experts[e](x_flat[mask])
#             outputs.append(output_i)

#         stacked = torch.stack(outputs, dim=1)
#         mixed = (stacked * topk_weights.unsqueeze(-1)).sum(1)

#         x = mixed.view(B, T, D)
#         x = self.dropout(x)
#         return self.layernorm(x + residual)


# class MoEBertLayer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         config._attn_implementation = "sdpa"
#         self.attention = BertAttention(config)
#         self.moe = SimpleMoEFFN(config.hidden_size, config.hidden_size * 4, num_experts=2, k=1)

#     def forward(self, hidden_states, attention_mask=None):
#         attention_output = self.attention(hidden_states, attention_mask=attention_mask)[0]
#         layer_output = self.moe(attention_output)
#         return layer_output, None  # to be compatible with `BertEncoder`


# class MoEBertEncoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.layer = nn.ModuleList([MoEBertLayer(config) for _ in range(config.num_hidden_layers)])

#     def forward(self, hidden_states, attention_mask=None):
#         for layer_module in self.layer:
#             hidden_states, _ = layer_module(hidden_states, attention_mask)
#         return BaseModelOutput(last_hidden_state=hidden_states)
import torch
import torch.nn.functional as F
from torch import Tensor, nn
# from zeta.nn import FeedForward, MultiQueryAttention
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

@dataclass
class MoEModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    aux_loss: Optional[torch.FloatTensor] = None
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)
class MultiQueryAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, self.head_dim * 2)  # One shared k & v for all heads
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, d = x.shape
        h = self.heads

        # Q: [b, n, h, d_head]
        q = self.q(x).view(b, n, h, self.head_dim)

        # k, v: [b, n, 1, d_head] -> broadcast to [b, n, h, d_head]
        kv = self.kv(x).view(b, n, 2, self.head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]
        k = k.unsqueeze(2).expand(-1, -1, h, -1)
        v = v.unsqueeze(2).expand(-1, -1, h, -1)

        # Attention: [b, h, n, n]
        attn_scores = torch.einsum('bnhd,bmhd->bhnm', q, k) * self.scale
        attn = attn_scores.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention: [b, n, h, d_head]
        out = torch.einsum('bhnm,bmhd->bnhd', attn, v)

        # Reshape: [b, n, d]
        out = out.reshape(b, n, d)
        return self.out(out), attn, None


class SwitchGate(nn.Module):
    """
    SwitchGate module for MoE (Mixture of Experts) model.

    Args:
        dim (int): Input dimension.
        num_experts (int): Number of experts.
        capacity_factor (float, optional): Capacity factor for sparsity. Defaults to 1.0.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        dim,
        num_experts: int,
        capacity_factor: float = 1.0,
        epsilon: float = 1e-6,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x: Tensor, use_aux_loss=False):
        """
        Forward pass of the SwitchGate module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Gate scores.
        """
        B, T, _ = x.shape
        N = B * T
        E = self.num_experts

        # 1) Soft gating
        gate_scores = F.softmax(self.w_gate(x), dim=-1)        # (B,T,E)
        flat_scores = gate_scores.view(N, E)                  # (N,E)

        # 2) Top-1 assignment
        top1_idx = flat_scores.argmax(dim=-1)             # (N,)
        capacity = int(self.capacity_factor * N / E)

        # Step 1: Count how many times each expert is selected (histogram)
        mask = F.one_hot(top1_idx, E).float()             # (N, E)
        cum_count = mask.cumsum(dim=0)                    # running total over rows
        position_in_expert = cum_count.gather(1, top1_idx.unsqueeze(1)).squeeze(1) - 1  # (N,)
        keep = position_in_expert < capacity              # boolean mask (N,)

        # Step 2: Apply mask
        mask = mask * keep.unsqueeze(1).float()           # (N, E)

        # 4) Renormalize to sum=capacity
        flat_masked = flat_scores * mask
        denom       = flat_masked.sum(0, keepdim=True).clamp_min(self.epsilon)
        flat_norm   = flat_masked / denom * capacity

        # 5) reshape back
        gate_scores = flat_norm.view(B, T, E)

        # 6) Aux loss
        if use_aux_loss:
            importance = flat_norm.sum(0)
            load       = (flat_norm > 0).float().sum(0)
            importance = importance / (importance.sum() + self.epsilon)
            load       = load       / (load.sum()       + self.epsilon)
            loss = ((load - importance)**2).mean()
            return gate_scores, loss

        return gate_scores, None


class SwitchMoE(nn.Module):
    """
    A module that implements the Switched Mixture of Experts (MoE) architecture.

    Args:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float, optional): The capacity factor that controls the capacity of the MoE. Defaults to 1.0.
        mult (int, optional): The multiplier for the hidden dimension of the feedforward network. Defaults to 4.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float): The capacity factor that controls the capacity of the MoE.
        mult (int): The multiplier for the hidden dimension of the feedforward network.
        experts (nn.ModuleList): The list of feedforward networks representing the experts.
        gate (SwitchGate): The switch gate module.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        mult: int = 4,
        use_aux_loss: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.mult = mult
        self.use_aux_loss = use_aux_loss

        self.experts = nn.ModuleList(
            [
                FeedForward(dim, dim, mult, *args, **kwargs)
                for _ in range(num_experts)
            ]
        )

        self.gate = SwitchGate(
            dim,
            num_experts,
            capacity_factor,
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the SwitchMoE module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor of the MoE.

        """
        # (batch_size, seq_len, num_experts)
        gate_scores, loss = self.gate(
            x, use_aux_loss=self.use_aux_loss
        )

        # # Dispatch to experts
        # expert_outputs = [expert(x) for expert in self.experts]

        # # Check if any gate scores are nan and handle
        # if torch.isnan(gate_scores).any():
        #     print("NaN in gate scores")
        #     gate_scores[torch.isnan(gate_scores)] = 0

        # # Stack and weight outputs
        # stacked_expert_outputs = torch.stack(
        #     expert_outputs, dim=-1
        # )  # (batch_size, seq_len, output_dim, num_experts)
        # if torch.isnan(stacked_expert_outputs).any():
        #     stacked_expert_outputs[
        #         torch.isnan(stacked_expert_outputs)
        #     ] = 0

        # # Combine expert outputs and gating scores
        # moe_output = torch.sum(
        #     gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1
        # )
        B, T, D = x.shape
        flat_x        = x.view(-1, D)                  # (B·T, D)
        flat_idx      = gate_scores.argmax(-1).view(-1) # (B·T,)
        flat_out      = torch.zeros_like(flat_x)       # (B·T, D)

        for e, expert in enumerate(self.experts):
            mask = (flat_idx == e)
            if mask.sum() == 0:
                continue
            tokens_e = flat_x[mask]                    # (N_e, D)
            out_e    = expert(tokens_e)                # (N_e, D)
            flat_out[mask] = out_e

        moe_output = flat_out.view(B, T, D)

        return moe_output, loss


class SwitchTransformerBlock(nn.Module):
    """
    SwitchTransformerBlock is a module that represents a single block of the Switch Transformer model.

    Args:
        dim (int): The input dimension of the block.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        mult (int, optional): The multiplier for the hidden dimension in the feed-forward network. Defaults to 4.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        depth (int, optional): The number of layers in the block. Defaults to 12.
        num_experts (int, optional): The number of experts in the SwitchMoE layer. Defaults to 6.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension of the block.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        mult (int): The multiplier for the hidden dimension in the feed-forward network.
        dropout (float): The dropout rate.
        attn_layers (nn.ModuleList): List of MultiQueryAttention layers.
        ffn_layers (nn.ModuleList): List of SwitchMoE layers.

    Examples:
        >>> block = SwitchTransformerBlock(dim=512, heads=8, dim_head=64)
        >>> x = torch.randn(1, 10, 512)
        >>> out = block(x)
        >>> out.shape

    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mult: int = 4,
        dropout: float = 0.1,
        num_experts: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.mult = mult
        self.dropout = dropout

        self.attn = MultiQueryAttention(dim, heads, dropout)

        self.ffn = SwitchMoE(
            dim, dim * mult, dim, num_experts, use_aux_loss=True, *args, **kwargs
        )
        
        # self.add_norm = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: Tensor):
        """
        Forward pass of the SwitchTransformerBlock.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        # resi = x
        # x, _, _ = self.attn(x)
        # x = x + resi
        # x = self.add_norm(x)
        # add_normed = x
        
        # ##### MoE #####
        # # x, _ = self.ffn(x)
        # x, moe_loss = self.ffn(x)
        # x = x + add_normed
        # x = self.add_norm(x)
        # return x, moe_loss
        # 1) Attention sub-layer
        residual1 = x
        attn_out, _, _ = self.attn(x)
        x = self.norm1(residual1 + attn_out)

        # 2) MoE sub-layer
        residual2 = x
        moe_out, aux_loss = self.ffn(x)
        x = self.norm2(residual2 + moe_out)

        return x, aux_loss


class SwitchTransformer(nn.Module):
    """
    SwitchTransformer is a PyTorch module that implements a transformer model with switchable experts.

    Args:
        num_tokens (int): The number of tokens in the input vocabulary.
        dim (int): The dimensionality of the token embeddings and hidden states.
        heads (int): The number of attention heads.
        dim_head (int, optional): The dimensionality of each attention head. Defaults to 64.
        mult (int, optional): The multiplier for the hidden dimension in the feed-forward network. Defaults to 4.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        num_experts (int, optional): The number of experts in the switchable experts mechanism. Defaults to 3.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        heads: int,
        dim_head: int = 64,
        mult: int = 4,
        dropout: float = 0.1,
        num_experts: int = 2,
        depth: int = 4,
        max_len: int = 512,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.mult = mult
        self.dropout = dropout
        self.num_experts = num_experts
        self.depth = depth
        self.pos_emb = nn.Embedding(max_len, dim)
        self.embedding = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])
        
        for _ in range(depth):
            self.layers.append(
                SwitchTransformerBlock(
                    dim,
                    heads,
                    dim_head,
                    mult,
                    dropout,
                    num_experts,
                    *args,
                    **kwargs,
                )
            )

        # self.to_out = nn.Sequential(
        #     nn.Softmax(dim=-1),
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_tokens),
        # )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the SwitchTransformer.

        Args:
            x (Tensor): The input tensor of shape (batch_size, sequence_length).

        Returns:
            Tensor: The output tensor of shape (batch_size, sequence_length, num_tokens).
        """
        # Embed tokens through embedding layer
        # x = self.embedding(x)
        if x.dtype in [torch.int64, torch.int32]:  # Token IDs
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # (1, T)
            x = self.embedding(x) + self.pos_emb(positions)
        
        total_aux_loss = 0.0
        # Pass through the transformer block with MoE, it's in modulelist
        # for layer in self.layers:
        #     x = layer(x)
        for layer in self.layers:
            x, layer_aux_loss = layer(x)
            if layer_aux_loss is not None:
                total_aux_loss = total_aux_loss + layer_aux_loss

        # Project to output tokens
        # x = self.to_out(x)
        # return BaseModelOutput(last_hidden_state=x)
        return MoEModelOutput(last_hidden_state=x, aux_loss=total_aux_loss)