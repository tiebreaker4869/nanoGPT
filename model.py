"""
Implementation of the model components.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linear Layers and LayerNorms

class LayerNorm(nn.Module):
    """
    LayerNormalization: https://arxiv.org/abs/1607.06450
    """
    def __init__(self, n_embed: int, bias:bool, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(n_embed))
        self.bias = nn.Parameter(torch.zeros(n_embed)) if bias else None
        self.eps = eps
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mean = input.mean(-1, keepdim=True)
        std  = input.std(-1, keepdim=True)
        return self.weight * (input - mean) / (std + self.eps) + self.bias

class MLP(nn.Module):
    """
    The feedforward Layer
    """
    def __init__(self, config: GPTConfig):
        super(MLP, self).__init__()
        self.up_proj = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU()
        self.down_proj = nn.Linear(config.n_embed * 4, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(x)
        x = self.gelu(x)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x

class CausalSelfAttention(nn.Module):
    """
    Masked self-attention used in the decoder block of transformers.
    """
    def __init__(self, config: GPTConfig):
        super(CausalSelfAttention, self).__init__()
        assert config.n_embed % config.n_head == 0, "n_embed must be divisible by n_head"
        self.proj_qkv = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias) # project original input to q, k, v
        self.out_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)

        self.atten_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embed = config.n_embed

        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # project input to q, k, v
        # x: (B, T, C) -> (B, T, 3C)
        # calculate q, k, v for all batches and move head dimension
        B, T, C = x.size()
        q, k, v = self.proj_qkv(x).split(self.n_embed, dim=-1)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # calculate attention logits
        # (B, n_head, T, head_dim) x (B, n_head, head_dim, T) -> (B, n_head, T, T)
        d_k = C // self.n_head
        atten_logits = torch.matmul(q, k.transpose(-1, -2))
        atten_logits = atten_logits * d_k ** -0.5
        atten_logits = atten_logits.masked_fill(self.mask == 0, float("-inf"))
        # calculate attention scores
        atten_scores = F.softmax(atten_logits, dim=-1)
        atten_scores = self.atten_dropout(atten_scores)

        # (B, n_head, T, T) x (B, n_head, T, self.head_dim) -> (B, n_head, T, head_dim)
        out = torch.matmul(atten_scores, v)
        # resume to original shape
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.output_dropout(self.out_proj(out))

class Block(nn.Module):
    """
    The decoder-only transformer block
    """
    def __init__(self, config: GPTConfig):
        super(Block, self).__init__()
        self.layernorm_1 = LayerNorm(config.n_embed, config.bias)
        self.attention = CausalSelfAttention(config)
        self.layernorm_2 = LayerNorm(config.n_embed, config.bias)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.layernorm_1(x))
        x = x + self.mlp(self.layernorm_2(x))
        return x
    
class GPT(nn.Module):
    """
    The full GPT language model.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embed)
        self.positional_emb = nn.Embedding(config.block_size, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.layernorm_final = LayerNorm(config.n_embed, config.bias)
        self.final_proj = nn.Linear(config.n_embed, config.vocab_size)

        self.apply(self._init_weights)
    
    def forward(self, x: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        # x: (B, T)
        device = x.device
        B, T = x.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        temb = self.token_emb(x)
        pemb = self.positional_emb(torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0))
        x = temb + pemb
        x = self.dropout(x)
        #(B, T, C)

        for block in self.blocks:
            x = block(x)
        x = self.layernorm_final(x)

        if target is not None:
            logits = self.final_proj(x)
            # logits: (B, T, vocab_size)
            # target: (B, T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        else:
            # inference time optimization: only forward the logits in last position
            logits = self.final_proj(x[:, [-1], :])
            loss = None
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)