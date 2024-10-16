# %%
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size, maxlen, rpe):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        pe = torch.zeros(maxlen, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, maxlen).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        self.norm = nn.LayerNorm(d_model)
        self.rpe = rpe

    def forward(self, x):
        if self.rpe:
            embedding = self.tok_embed(x)
        else:
            embedding = self.tok_embed(x) + self.pe[:, : x.size(1)]
        return self.norm(embedding)


class NewGELU(nn.Module):
    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, drop, maxlen, rpe):
        super().__init__()
        assert d_model % nhead == 0
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(drop)
        self.resid_dropout = nn.Dropout(drop)
        self.register_buffer(
            "bias", torch.tril(torch.ones(maxlen, maxlen)).view(1, 1, maxlen, maxlen)
        )
        self.rpe = rpe
        rpe = torch.zeros(1, nhead, maxlen, maxlen)
        for i in range(1, maxlen):
            rpe = rpe - torch.tril(torch.ones(maxlen, maxlen), diagonal=-i).view(
                1, 1, maxlen, maxlen
            )
        for i in range(nhead):
            rpe[0, i] = rpe[0, i] * 2 ** (-8 / nhead * (i + 1))
        self.register_buffer("RPE", rpe)
        self.n_head = nhead
        self.dmodel = d_model

    def forward(self, x, mask=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.dmodel, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.rpe:
            att = att + self.RPE[:, :, :T, :T]
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        if mask is not None:
            att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    def __init__(self, d_model, nhead, drop, maxlen, rpe, causal=True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            nhead=nhead,
            drop=drop,
            maxlen=maxlen,
            rpe=rpe,
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(d_model, 4 * d_model),
                c_proj=nn.Linear(4 * d_model, d_model),
                act=NewGELU(),
                dropout=nn.Dropout(drop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x, ys=None):
        x = x + self.attn(self.ln_1(x), ys)
        x = x + self.mlpf(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.transformer = nn.ModuleDict(
            dict(
                embedding=Embedding(
                    d_model=args.dmodel,
                    vocab_size=args.vocab,
                    maxlen=args.maxlen,
                    rpe=args.rpe,
                ),
                drop=nn.Dropout(args.drop),
                h=nn.ModuleList(
                    [
                        Block(
                            d_model=args.dmodel,
                            nhead=args.head,
                            drop=args.drop,
                            maxlen=args.maxlen,
                            rpe=args.rpe,
                        )
                        for _ in range(args.num_layer)
                    ]
                ),
                ln_f=nn.LayerNorm(args.dmodel),
            )
        )
        self.lm_head = nn.Linear(args.dmodel, args.vocab, bias=True)

    def forward(self, idx):
        b, t = idx.size()
        emb = self.transformer.embedding(idx)
        x = self.transformer.drop(emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, idx, start):
        b, t = idx.size()
        tmp_start = start + 0
        while True:
            logits = self.forward(idx)
            idx_new = torch.argmax(logits, dim=2)
            idx[torch.arange(b), tmp_start + 1] = idx_new[torch.arange(b), tmp_start]
            if (torch.sum(idx_new[torch.arange(b), tmp_start] != 2) == 0) or (
                torch.sum(tmp_start == t - 2) != 0
            ):
                break
            tmp_start[idx_new[torch.arange(b), tmp_start] != 2] += 1
        return idx


class LoopedGPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_loop = args.num_loop
        self.transformer = nn.ModuleDict(
            dict(
                embedding=Embedding(
                    d_model=args.dmodel,
                    vocab_size=args.vocab,
                    maxlen=args.maxlen,
                    rpe=args.rpe,
                ),
                drop=nn.Dropout(args.drop),
                attn=CausalSelfAttention(
                    d_model=args.dmodel,
                    nhead=args.head,
                    drop=args.drop,
                    maxlen=args.maxlen,
                    rpe=args.rpe,
                ),
                mlp=nn.Sequential(
                    nn.Linear(args.dmodel, 4 * args.dmodel),
                    NewGELU(),
                    nn.Linear(4 * args.dmodel, args.dmodel),
                ),
                norm1=nn.RMSNorm(args.dmodel),
                norm2=nn.RMSNorm(args.dmodel),
                ln_f=nn.LayerNorm(args.dmodel),
            )
        )
        self.lm_head = nn.Linear(args.dmodel, args.vocab, bias=True)

    def f_attn(self, x, idx):
        norm1 = self.transformer.norm1
        attn = self.transformer.attn
        return attn(norm1(x))

    def f_mlp(self, x, idx):
        norm2 = self.transformer.norm2
        mlp = self.transformer.mlp
        return mlp(norm2(x))

    def f(self, x, idx):
        x = x + self.f_attn(x, idx)
        x = x + self.f_mlp(x, idx)
        return x

    def forward(self, idx, ys=None):
        b, t = idx.size()
        emb = self.transformer.embedding(idx)
        x = self.transformer.drop(emb)
        for l in range(self.num_loop):
            x = self.f(x, l)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # [b, t, vocab]
        return logits


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        # t: (N,) tensor of timesteps
        # t_emb: (N, D) tensor of timestep embeddings
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class HyperBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.dmodel, elementwise_affine=False)
        self.attn = CausalSelfAttention(
            config.dmodel, config.head, config.drop, config.maxlen, config.rpe
        )
        self.norm2 = nn.RMSNorm(config.dmodel, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(config.dmodel, 4 * config.dmodel),
            nn.SiLU(),
            nn.Linear(4 * config.dmodel, config.dmodel),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.dmodel, 4 * config.dmodel, bias=True)
        )

    def forward(self, x, t_emb):
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        # x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        scale_msa, scale_mlp, gate_msa, gate_mlp = self.adaLN_modulation(t_emb).chunk(
            4, dim=1
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            self.norm1(x) * (1 + scale_msa.unsqueeze(1))
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            self.norm2(x) * (1 + scale_mlp.unsqueeze(1))
        )
        return x


class HyperLoopedGPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_loop = args.num_loop
        self.transformer = nn.ModuleDict(
            dict(
                embedding=Embedding(
                    d_model=args.dmodel,
                    vocab_size=args.vocab,
                    maxlen=args.maxlen,
                    rpe=args.rpe,
                ),
                drop=nn.Dropout(args.drop),
                h=HyperBlock(args),
                ln_f=nn.LayerNorm(args.dmodel),
            )
        )
        self.lm_head = nn.Linear(args.dmodel, args.vocab, bias=True)

        self.timestep_embedder = TimestepEmbedder(args.dmodel)

        # zero init for adaLN_modulation
        nn.init.zeros_(self.transformer.h.adaLN_modulation[1].weight)
        nn.init.zeros_(self.transformer.h.adaLN_modulation[1].bias)

        n_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6))

    def forward(self, idx, ys=None):
        b, _ = idx.size()
        emb = self.transformer.embedding(idx)
        x = self.transformer.drop(emb)
        for t in range(self.num_loop):
            t_tensor = torch.full((b,), t, dtype=torch.long, device=idx.device)
            t_emb = self.timestep_embedder(t_tensor)
            x = self.transformer.h(x, t_emb)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # [b, t, vocab]
        return logits


# %%
if __name__ == "__main__":
    from argparse import Namespace

    args = Namespace(
        dmodel=256,
        vocab=21,
        maxlen=120,
        drop=0.1,
        num_layer=3,
        head=4,
        num_loop=100,
        rpe=True,
    )
    model = HyperLoopedGPT(args)
    print(model)
    idx = torch.randint(0, 21, (4, 120))
    print(model(idx).shape)
# %%
