import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """One head of self-attention within a group"""

    def __init__(
        self,
        head_size: int,
        n_embd: int,
        dropout: float,
        block_size: int,
        shared_kv=False,
    ):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)

        if shared_kv:
            self.key = None
            self.value = None
        else:
            self.key = nn.Linear(n_embd, head_size, bias=False)
            self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, shared_k, shared_v):
        B, T, C = x.shape
        q = self.query(x)

        k = shared_k if shared_k is not None else self.key(x)
        v = shared_v if shared_v is not None else self.value(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention with grouped query attention"""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        dropout: float,
        n_embd: int,
        block_size: int,
        num_groups: int,
    ):
        super().__init__()

        assert (
            num_heads % num_groups == 0
        ), "Number of heads must be divisible by number of groups."

        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups

        self.groups = nn.ModuleList()
        for _ in range(num_groups):
            group_heads = nn.ModuleList(
                [
                    Head(head_size, n_embd, dropout, block_size, shared_kv=True)
                    for _ in range(self.heads_per_group)
                ]
            )
            self.groups.append(group_heads)

        self.shared_keys = nn.ModuleList(
            [nn.Linear(n_embd, head_size, bias=False) for _ in range(num_groups)]
        )
        self.shared_values = nn.ModuleList(
            [nn.Linear(n_embd, head_size, bias=False) for _ in range(num_groups)]
        )

        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        group_outputs = []
        for group_idx, group_heads in enumerate(self.groups):
            shared_k = self.shared_keys[group_idx](x)
            shared_v = self.shared_values[group_idx](x)

            group_out = [head(x, shared_k, shared_v) for head in group_heads]
            group_outputs.extend(group_out)

        out = torch.cat(group_outputs, dim=-1)
        out = self.dropout(self.proj(out))
        return out


class MLP(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        block_size: int,
        num_groups: int,
    ):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(
            n_head,
            head_size,
            dropout,
            n_embd,
            block_size,
            num_groups,
        )
        self.mlp = MLP(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        dropout: float,
        num_groups: int,
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd,
                    n_head=n_head,
                    dropout=dropout,
                    block_size=block_size,
                    num_groups=num_groups,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
