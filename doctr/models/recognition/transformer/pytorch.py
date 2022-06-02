# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

# This module 'transformer.py' is inspired by https://github.com/wenwenyu/MASTER-pytorch and has some borrowed code
# from https://github.com/codertimo/BERT-pytorch

import math
from typing import Optional, Tuple

import torch
from torch import nn

__all__ = ['Decoder', 'PositionalEncoding']


class PositionalEncoding(nn.Module):
    """ Compute positional encoding """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: embeddings (batch, max_len, d_model)

        Returns:
            positional embeddings (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]  # type: ignore
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    """ Position-wise Feed-Forward Network """
    def __init__(self, d_model: int, ffd: int, dropout=0.1) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.first_linear = nn.Linear(d_model, ffd)
        self.sec_linear = nn.Linear(ffd, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.first_linear(x))
        x = self.dropout(x)
        x = self.sec_linear(x)
        return x


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, num_heads: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.scaled_dot_product_attention = ScaledDotProductAttention()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask=None
    ) -> torch.Tensor:
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                             for linear, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.scaled_dot_product_attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        return self.output_linear(x)


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        dropout: float = 0.2,
        dff: int = 2048,
    ) -> None:

        super(Decoder, self).__init__()
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-5)

        self.attention = nn.ModuleList(
            [MultiHeadAttention(num_heads, d_model, dropout) for _ in range(self.num_layers)]
        )
        self.source_attention = nn.ModuleList(
            [MultiHeadAttention(num_heads, d_model, dropout) for _ in range(self.num_layers)]
        )
        self.position_feed_forward = nn.ModuleList(
            [PositionwiseFeedForward(d_model, dff, dropout) for _ in range(self.num_layers)]
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = tgt
        for i in range(self.num_layers):
            normed_output = self.layer_norm(tgt)
            output = tgt + self.dropout(
                self.attention[i](normed_output, normed_output, normed_output, target_mask)
            )
            normed_output = self.layer_norm(output)
            output = output + self.dropout(
                self.source_attention[i](normed_output, memory, memory, source_mask)
            )
            normed_output = self.layer_norm(output)
            output = output + self.dropout(self.position_feed_forward[i](normed_output))
        return self.layer_norm(output)

