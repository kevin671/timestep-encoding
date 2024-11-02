import math
from copy import deepcopy

import torch
from torch import nn

from seq2seq.tools.config import EOS, PAD

from .modules.state import State
from .modules.transformer_blocks import CharWordEmbedder, positional_embedding
from .seq2seq_base import Seq2Seq
from .transformer import (
    TransformerAttentionDecoder,
    TransformerAttentionEncoder,
    index_select_2d,
    permuted_order,
    repeat,
)


class LoopedTransformerEncoder(TransformerAttentionEncoder):
    def __init__(
        self,
        vocab_size,
        hidden_size=512,
        embedding_size=None,
        num_layers=6,
        num_loops=12,
        num_heads=8,
        inner_linear=2048,
        inner_groups=1,
        prenormalized=False,
        mask_symbol=PAD,
        batch_first=True,
        layer_norm=True,
        weight_norm=False,
        dropout=0,
        embedder=None,
        time_dependent=False,
    ):
        assert num_layers == 1, "LoopedTransformer only supports num_layers=1"
        super(LoopedTransformerEncoder, self).__init__(
            vocab_size,
            hidden_size,
            embedding_size,
            num_layers,
            num_heads,
            inner_linear,
            inner_groups,
            prenormalized,
            mask_symbol,
            batch_first,
            layer_norm,
            weight_norm,
            dropout,
            embedder,
            time_dependent,
        )
        self.num_loops = num_loops

    def forward(self, inputs, hidden=None):
        batch_dim, time_dim = (0, 1) if self.batch_first else (1, 0)
        if self.mask_symbol is not None:
            padding_mask = inputs.eq(self.mask_symbol)
        else:
            padding_mask = None
        x = self.embedder(inputs).mul_(self.scale_embedding)
        if hasattr(self, "input_projection"):
            x = x @ self.input_projection
        pos_embedding = positional_embedding(x.size(time_dim), x.size(-1), device=x.device)
        x.add_(pos_embedding.unsqueeze(batch_dim))
        x = self.dropout(x)

        block = self.blocks[0]
        for _ in range(self.num_loops):
            block.set_mask(padding_mask)
            x = block(x)

        if hasattr(self, "lnorm"):
            x = self.lnorm(x)

        return State(outputs=x, mask=padding_mask, batch_first=self.batch_first)


class LoopedTransformerDecoder(TransformerAttentionDecoder):
    def __init__(
        self,
        vocab_size,
        hidden_size=512,
        embedding_size=None,
        num_layers=6,
        num_loops=12,
        num_heads=8,
        batch_first=True,
        dropout=0,
        inner_linear=2048,
        inner_groups=1,
        prenormalized=False,
        stateful=None,
        state_dim=None,
        mask_symbol=PAD,
        tie_embedding=True,
        layer_norm=True,
        weight_norm=False,
        embedder=None,
        classifier=True,
        permuted=False,
        learned_condition=False,
        max_length=512,
        time_dependent=False,
        **kwargs
    ):
        assert num_layers == 1, "LoopedTransformer only supports num_layers=1"
        super(LoopedTransformerDecoder, self).__init__(
            vocab_size,
            hidden_size,
            embedding_size,
            num_layers,
            num_heads,
            batch_first,
            dropout,
            inner_linear,
            inner_groups,
            prenormalized,
            stateful,
            state_dim,
            mask_symbol,
            tie_embedding,
            layer_norm,
            weight_norm,
            embedder,
            classifier,
            permuted,
            learned_condition,
            max_length,
            time_dependent,
            **kwargs
        )
        self.num_loops = num_loops

    def forward(
        self,
        inputs,
        state,
        time_multiply=1,
        get_attention=False,
        causal=None,
        input_order=None,
        output_order=None,
        output_reorder=None,
    ):
        context = state.context
        time_step = 0
        batch_dim, time_dim = (0, 1) if self.batch_first else (1, 0)

        if self.stateful:
            block_state = state.hidden
            if block_state is None:
                self.time_step = 0
            time_step = self.time_step
        else:
            block_state = state.inputs
            time_step = 0 if block_state is None else block_state[0][0].size(time_dim)
        if block_state is None:
            block_state = [None] * len(self.blocks)
        if self.mask_symbol is not None:
            padding_mask = inputs.eq(self.mask_symbol)
        else:
            padding_mask = None

        x = self.embedder(inputs).mul_(self.scale_embedding)

        if hasattr(self, "input_projection"):
            x = x @ self.input_projection

        pos_embedding = positional_embedding(
            x.size(time_dim), x.size(-1), offset=time_step, device=x.device
        ).unsqueeze(batch_dim)
        x.add_(pos_embedding)

        if self.permuted:
            if self.training:
                output_order, output_reorder = permuted_order(inputs, batch_first=self.batch_first)
                pos_target = output_order.narrow(time_dim, 1, x.size(time_dim))

                pos_input = output_order.narrow(time_dim, 0, x.size(time_dim))
                x = index_select_2d(x, pos_input)

                cond_embedding = self.conditioned_pos(pos_target)

            else:
                pos_target = torch.arange(x.size(time_dim) * time_multiply, device=x.device) + time_step + 1
                cond_embedding = self.conditioned_pos(pos_target).unsqueeze(batch_dim)
                output_reorder = None

            if time_multiply > 1:
                padding_mask = repeat(padding_mask, time_multiply).flatten(0, 1)
                x = repeat(x, time_multiply).flatten(0, 1)
                cond_embedding = (
                    repeat(cond_embedding.squeeze(batch_dim), inputs.size(batch_dim), dim=1)
                    .transpose(0, 1)
                    .contiguous()
                    .view_as(x)
                )
                context.mask = repeat(context.mask, time_multiply).flatten(0, 1)
                context.outputs = repeat(context.outputs, time_multiply).flatten(0, 1)
            x = x.add(cond_embedding)

        x = self.dropout(x)

        attention_scores = []
        updated_state = []

        block = self.blocks[0]
        # for i, block in enumerate(self.blocks):
        for _ in range(self.num_loops):
            if causal is not None:
                block.masked_attention.causal = causal
            block.set_mask(padding_mask, context.mask)
            x, attn_enc, block_s = block(x, context.outputs, block_state[0])
            updated_state.append(block_s)
            if get_attention:
                attention_scores.append(attn_enc)
            else:
                del attn_enc

        if hasattr(self, "lnorm"):
            x = self.lnorm(x)

        if output_reorder is not None:
            x = index_select_2d(x, output_reorder)

        if hasattr(self, "output_projection"):
            x = x @ self.output_projection.t()

        if self.classifier is not None:
            x = self.classifier(x)

        if self.stateful:
            state.hidden = tuple(updated_state)
            self.time_step += 1
        else:
            state.inputs = tuple(updated_state)
        if get_attention:
            state.attention_score = attention_scores

        if time_multiply > 1:
            x = x.view(time_multiply, inputs.size(batch_dim), -1)
            x = x.transpose(0, 1)

        return x, state


class LoopedTransformer(Seq2Seq):
    def __init__(
        self,
        vocab_size,
        hidden_size=512,
        embedding_size=None,
        num_layers=6,
        num_loops=12,
        num_heads=8,
        inner_linear=2048,
        inner_groups=1,
        dropout=0.1,
        prenormalized=False,
        tie_embedding=True,
        encoder=None,
        decoder=None,
        layer_norm=True,
        weight_norm=False,
        batch_first=True,
        stateful=None,
    ):
        super(LoopedTransformer, self).__init__()
        embedding_size = embedding_size or hidden_size
        # keeping encoder, decoder None will result with default configuration
        encoder = encoder or {}
        decoder = decoder or {}
        encoder = deepcopy(encoder)
        decoder = deepcopy(decoder)
        encoder.setdefault("embedding_size", embedding_size)
        encoder.setdefault("hidden_size", hidden_size)
        encoder.setdefault("num_layers", num_layers)
        encoder.setdefault("num_loops", num_loops)
        encoder.setdefault("num_heads", num_heads)
        encoder.setdefault("vocab_size", vocab_size)
        encoder.setdefault("layer_norm", layer_norm)
        encoder.setdefault("weight_norm", weight_norm)
        encoder.setdefault("dropout", dropout)
        encoder.setdefault("inner_linear", inner_linear)
        encoder.setdefault("inner_groups", inner_groups)
        encoder.setdefault("prenormalized", prenormalized)
        encoder.setdefault("batch_first", batch_first)

        decoder.setdefault("embedding_size", embedding_size)
        decoder.setdefault("hidden_size", hidden_size)
        decoder.setdefault("num_layers", num_layers)
        decoder.setdefault("num_loops", num_loops)
        decoder.setdefault("num_heads", num_heads)
        decoder.setdefault("tie_embedding", tie_embedding)
        decoder.setdefault("vocab_size", vocab_size)
        decoder.setdefault("layer_norm", layer_norm)
        decoder.setdefault("weight_norm", weight_norm)
        decoder.setdefault("dropout", dropout)
        decoder.setdefault("inner_linear", inner_linear)
        decoder.setdefault("inner_groups", inner_groups)
        decoder.setdefault("batch_first", batch_first)
        decoder.setdefault("prenormalized", prenormalized)
        decoder.setdefault("stateful", stateful)

        if isinstance(vocab_size, tuple):
            embedder = CharWordEmbedder(vocab_size[1], embedding_size, hidden_size)
            encoder.setdefault("embedder", embedder)
            decoder.setdefault("embedder", embedder)
            decoder["classifier"] = False

        self.batch_first = batch_first
        self.encoder = LoopedTransformerEncoder(**encoder)
        self.decoder = LoopedTransformerDecoder(**decoder)

        if tie_embedding and not isinstance(vocab_size, tuple):
            assert self.encoder.embedder.weight.shape == self.decoder.classifier.weight.shape
            self.encoder.embedder.weight = self.decoder.classifier.weight
            if embedding_size != hidden_size:
                self.encoder.input_projection = self.decoder.input_projection


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
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # t: (N,) tensor of timesteps
        # t_emb: (N, D) tensor of timestep embeddings
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class TimeDependentLoopedTransformerEncoder(LoopedTransformerEncoder):
    def __init__(self, *kargs, **kwargs):
        kwargs["time_dependent"] = True
        super().__init__(*kargs, **kwargs)
        hidden_size = kwargs["hidden_size"]
        self.timestep_embedder = TimestepEmbedder(hidden_size)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 4 * hidden_size, bias=True))
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, inputs, hidden=None):
        batch_dim, time_dim = (0, 1) if self.batch_first else (1, 0)
        if self.mask_symbol is not None:
            padding_mask = inputs.eq(self.mask_symbol)
        else:
            padding_mask = None
        x = self.embedder(inputs).mul_(self.scale_embedding)
        if hasattr(self, "input_projection"):
            x = x @ self.input_projection
        pos_embedding = positional_embedding(x.size(time_dim), x.size(-1), device=x.device)
        x.add_(pos_embedding.unsqueeze(batch_dim))
        x = self.dropout(x)

        block = self.blocks[0]
        for t in range(self.num_loops):
            block.set_mask(padding_mask)
            t_emb = self.timestep_embedder(torch.full((x.size(batch_dim),), t, device=x.device))
            gate_msa, gate_mlp, scale_msa, scale_mlp = self.adaLN_modulation(t_emb).chunk(4, dim=-1)
            x = block(x, gate_msa, gate_mlp, scale_msa, scale_mlp)

        if hasattr(self, "lnorm"):
            x = self.lnorm(x)

        return State(outputs=x, mask=padding_mask, batch_first=self.batch_first)


class TimeDependentLoopedTransformerDecoder(LoopedTransformerDecoder):
    def __init__(self, *kargs, **kwargs):
        kwargs["time_dependent"] = True
        super().__init__(*kargs, **kwargs)
        hidden_size = kwargs["hidden_size"]
        self.timestep_embedder = TimestepEmbedder(hidden_size)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 4 * hidden_size, bias=True))
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        inputs,
        state,
        time_multiply=1,
        get_attention=False,
        causal=None,
        input_order=None,
        output_order=None,
        output_reorder=None,
    ):
        context = state.context
        time_step = 0
        batch_dim, time_dim = (0, 1) if self.batch_first else (1, 0)

        if self.stateful:
            block_state = state.hidden
            if block_state is None:
                self.time_step = 0
            time_step = self.time_step
        else:
            block_state = state.inputs
            time_step = 0 if block_state is None else block_state[0][0].size(time_dim)
        if block_state is None:
            block_state = [None] * len(self.blocks)
        if self.mask_symbol is not None:
            padding_mask = inputs.eq(self.mask_symbol)
        else:
            padding_mask = None

        x = self.embedder(inputs).mul_(self.scale_embedding)

        if hasattr(self, "input_projection"):
            x = x @ self.input_projection

        pos_embedding = positional_embedding(
            x.size(time_dim), x.size(-1), offset=time_step, device=x.device
        ).unsqueeze(batch_dim)
        x.add_(pos_embedding)

        if self.permuted:
            if self.training:
                output_order, output_reorder = permuted_order(inputs, batch_first=self.batch_first)
                pos_target = output_order.narrow(time_dim, 1, x.size(time_dim))

                pos_input = output_order.narrow(time_dim, 0, x.size(time_dim))
                x = index_select_2d(x, pos_input)

                cond_embedding = self.conditioned_pos(pos_target)

            else:
                pos_target = torch.arange(x.size(time_dim) * time_multiply, device=x.device) + time_step + 1
                cond_embedding = self.conditioned_pos(pos_target).unsqueeze(batch_dim)
                output_reorder = None

            if time_multiply > 1:
                padding_mask = repeat(padding_mask, time_multiply).flatten(0, 1)
                x = repeat(x, time_multiply).flatten(0, 1)
                cond_embedding = (
                    repeat(cond_embedding.squeeze(batch_dim), inputs.size(batch_dim), dim=1)
                    .transpose(0, 1)
                    .contiguous()
                    .view_as(x)
                )
                context.mask = repeat(context.mask, time_multiply).flatten(0, 1)
                context.outputs = repeat(context.outputs, time_multiply).flatten(0, 1)
            x = x.add(cond_embedding)

        x = self.dropout(x)

        attention_scores = []
        updated_state = []

        block = self.blocks[0]
        # for i, block in enumerate(self.blocks):
        for t in range(self.num_loops):
            if causal is not None:
                block.masked_attention.causal = causal
            block.set_mask(padding_mask, context.mask)

            t_emb = self.timestep_embedder(torch.full((x.size(batch_dim),), t, device=x.device))
            gate_msa, gate_mlp, scale_msa, scale_mlp = self.adaLN_modulation(t_emb).chunk(4, dim=-1)
            x, attn_enc, block_s = block(x, context.outputs, gate_msa, gate_mlp, scale_msa, scale_mlp, block_state[0])
            updated_state.append(block_s)
            if get_attention:
                attention_scores.append(attn_enc)
            else:
                del attn_enc

        if hasattr(self, "lnorm"):
            x = self.lnorm(x)

        if output_reorder is not None:
            x = index_select_2d(x, output_reorder)

        if hasattr(self, "output_projection"):
            x = x @ self.output_projection.t()

        if self.classifier is not None:
            x = self.classifier(x)

        if self.stateful:
            state.hidden = tuple(updated_state)
            self.time_step += 1
        else:
            state.inputs = tuple(updated_state)
        if get_attention:
            state.attention_score = attention_scores

        if time_multiply > 1:
            x = x.view(time_multiply, inputs.size(batch_dim), -1)
            x = x.transpose(0, 1)

        return x, state


class TimeDependentLoopedTransformer(Seq2Seq):
    def __init__(
        self,
        vocab_size,
        hidden_size=512,
        embedding_size=None,
        num_layers=6,
        num_loops=12,
        num_heads=8,
        inner_linear=2048,
        inner_groups=1,
        dropout=0.1,
        prenormalized=False,
        tie_embedding=True,
        encoder=None,
        decoder=None,
        layer_norm=True,
        weight_norm=False,
        batch_first=True,
        stateful=None,
    ):
        super(TimeDependentLoopedTransformer, self).__init__()
        embedding_size = embedding_size or hidden_size
        # keeping encoder, decoder None will result with default configuration
        encoder = encoder or {}
        decoder = decoder or {}
        encoder = deepcopy(encoder)
        decoder = deepcopy(decoder)
        encoder.setdefault("embedding_size", embedding_size)
        encoder.setdefault("hidden_size", hidden_size)
        encoder.setdefault("num_layers", num_layers)
        encoder.setdefault("num_loops", num_loops)
        encoder.setdefault("num_heads", num_heads)
        encoder.setdefault("vocab_size", vocab_size)
        encoder.setdefault("layer_norm", layer_norm)
        encoder.setdefault("weight_norm", weight_norm)
        encoder.setdefault("dropout", dropout)
        encoder.setdefault("inner_linear", inner_linear)
        encoder.setdefault("inner_groups", inner_groups)
        encoder.setdefault("prenormalized", prenormalized)
        encoder.setdefault("batch_first", batch_first)

        decoder.setdefault("embedding_size", embedding_size)
        decoder.setdefault("hidden_size", hidden_size)
        decoder.setdefault("num_layers", num_layers)
        decoder.setdefault("num_loops", num_loops)
        decoder.setdefault("num_heads", num_heads)
        decoder.setdefault("tie_embedding", tie_embedding)
        decoder.setdefault("vocab_size", vocab_size)
        decoder.setdefault("layer_norm", layer_norm)
        decoder.setdefault("weight_norm", weight_norm)
        decoder.setdefault("dropout", dropout)
        decoder.setdefault("inner_linear", inner_linear)
        decoder.setdefault("inner_groups", inner_groups)
        decoder.setdefault("batch_first", batch_first)
        decoder.setdefault("prenormalized", prenormalized)
        decoder.setdefault("stateful", stateful)

        if isinstance(vocab_size, tuple):
            embedder = CharWordEmbedder(vocab_size[1], embedding_size, hidden_size)
            encoder.setdefault("embedder", embedder)
            decoder.setdefault("embedder", embedder)
            decoder["classifier"] = False

        self.batch_first = batch_first
        self.encoder = TimeDependentLoopedTransformerEncoder(**encoder)
        self.decoder = TimeDependentLoopedTransformerDecoder(**decoder)

        if tie_embedding and not isinstance(vocab_size, tuple):
            assert self.encoder.embedder.weight.shape == self.decoder.classifier.weight.shape
            self.encoder.embedder.weight = self.decoder.classifier.weight
            if embedding_size != hidden_size:
                self.encoder.input_projection = self.decoder.input_projection
