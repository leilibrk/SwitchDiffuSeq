import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertAttention
from typing import Optional, Tuple
from transformers.modeling_outputs import BaseModelOutput
class SimpleMoEFFN(nn.Module):
    def __init__(self, hidden_dim, expert_dim, num_experts=2, k=1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, hidden_dim)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(hidden_dim, num_experts)
        self.dropout = nn.Dropout(0.1)
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        B, T, D = x.shape
        residual = x
        x_flat = x.view(-1, D)

        gate_logits = self.gate(x_flat)
        topk_val, topk_idx = torch.topk(gate_logits, self.k, dim=-1)
        topk_weights = F.softmax(topk_val, dim=-1)

        outputs = []
        for i in range(self.k):
            expert_ids = topk_idx[:, i]
            output_i = torch.zeros_like(x_flat)
            for e in range(self.num_experts):
                mask = (expert_ids == e)
                if mask.any():
                    output_i[mask] = self.experts[e](x_flat[mask])
            outputs.append(output_i)

        stacked = torch.stack(outputs, dim=1)
        mixed = (stacked * topk_weights.unsqueeze(-1)).sum(1)

        x = mixed.view(B, T, D)
        x = self.dropout(x)
        return self.layernorm(x + residual)


class MoEBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        config._attn_implementation = "sdpa"
        self.attention = BertAttention(config)
        self.moe = SimpleMoEFFN(config.hidden_size, config.hidden_size * 4, num_experts=2, k=1)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask=attention_mask)[0]
        layer_output = self.moe(attention_output)
        return layer_output, None  # to be compatible with `BertEncoder`


class MoEBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([MoEBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states, _ = layer_module(hidden_states, attention_mask)
        return BaseModelOutput(last_hidden_state=hidden_states)