# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from copy import deepcopy
from itertools import permutations
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

from doctr.datasets import VOCABS
from doctr.models.modules.transformer import MultiHeadAttention, PositionwiseFeedForward

from ...classification import vit_s
from ...utils.pytorch import load_pretrained_params
from .base import _PARSeq, _PARSeqPostProcessor

__all__ = ["PARSeq", "parseq"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    "parseq": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 128),
        "vocab": VOCABS["french"],
        "url": None,
    },
}


class CharEmbedding(nn.Module):
    """Implements the character embedding module

    Args:
        vocab_size: size of the vocabulary
        d_model: dimension of the model
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return math.sqrt(self.d_model) * self.embedding(x)


class PARSeqDecoder(nn.Module):
    """Implements decoder module of the PARSeq model

    Args:
        d_model: dimension of the model
        num_heads: number of attention heads
        ffd: dimension of the feed forward layer
        ffd_ratio: depth multiplier for the feed forward layer
        dropout: dropout rate
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 12,
        ffd: int = 2048,
        ffd_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, d_model, dropout=dropout)
        self.cross_attention = MultiHeadAttention(num_heads, d_model, dropout=dropout)
        self.position_feed_forward = PositionwiseFeedForward(d_model, ffd * ffd_ratio, dropout, nn.GELU())

        self.attention_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.cross_attention_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.query_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.content_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.feed_forward_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.output_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.attention_dropout = nn.Dropout(dropout)
        self.cross_attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(
        self,
        target,
        content,
        memory,
        target_mask: Optional[torch.Tensor] = None,
    ):
        query_norm = self.query_norm(target)
        content_norm = self.content_norm(content)
        target = target.clone() + self.attention_dropout(self.attention(query_norm, content_norm, content_norm, mask=target_mask))
        target = target.clone() + self.cross_attention_dropout(self.cross_attention(self.query_norm(target), memory, memory))
        target = target.clone() + self.feed_forward_dropout(self.position_feed_forward(self.feed_forward_norm(target)))
        return self.output_norm(target)


class PARSeq(_PARSeq, nn.Module):
    """Implements a PARSeq architecture as described in `"Scene Text Recognition
    with Permuted Autoregressive Sequence Models" <https://arxiv.org/pdf/2207.06966>`_.

    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        embedding_units: number of embedding units
        max_length: maximum word length handled by the model
        dropout_prob: dropout probability for the decoder
        dec_num_heads: number of attention heads in the decoder
        dec_ff_dim: dimension of the feed forward layer in the decoder
        dec_ffd_ratio: depth multiplier for the feed forward layer in the decoder
        input_shape: input shape of the image
        exportable: onnx exportable returns only logits
        cfg: dictionary containing information about the model
    """

    def __init__(
        self,
        feature_extractor,
        vocab: str,
        embedding_units: int,
        max_length: int = 25,
        dropout_prob: int = 0.1,
        dec_num_heads: int = 12,
        dec_ff_dim: int = 2048,
        dec_ffd_ratio: int = 4,
        input_shape: Tuple[int, int, int] = (3, 32, 128),
        exportable: bool = False,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.exportable = exportable
        self.cfg = cfg
        self.max_length = max_length + 1  # Add 1 step for EOS
        self.vocab_size = len(vocab) + 3  # Add 1 for EOS, 1 for SOS, 1 for PAD
        self.rng = np.random.default_rng()

        self.feat_extractor = feature_extractor
        self.decoder = PARSeqDecoder(embedding_units, dec_num_heads, dec_ff_dim, dec_ffd_ratio, dropout_prob)
        self.head = nn.Linear(embedding_units, self.vocab_size - 2)  # we ignore SOS and PAD
        self.text_embed = CharEmbedding(self.vocab_size, embedding_units)

        self.pos_queries = nn.Parameter(torch.Tensor(1, self.max_length, embedding_units))
        self.dropout = nn.Dropout(p=dropout_prob)

        self.postprocessor = PARSeqPostProcessor(vocab=self.vocab)

        nn.init.trunc_normal_(self.pos_queries, std=0.02)
        for n, m in self.named_modules():
            # Don't override the initialization of the backbone
            if n.startswith("feat_extractor."):
                continue
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # TODO: clean up + merge padding masks into masks  and refactor some parts !!

    def gen_tgt_perms(self, target: torch.Tensor) -> torch.Tensor:
        """Generates permutations of the target sequence.
        Slightly modified from https://github.com/baudm/parseq/blob/main/strhub/models/parseq/system.py"""
        max_num_chars = target.shape[1] - 2

        if max_num_chars == 1:
            return torch.arange(3, device=target.device).unsqueeze(0)

        perms = [torch.arange(max_num_chars, device=target.device)]

        max_perms = math.factorial(max_num_chars) // 2
        num_gen_perms = min(3, max_perms)

        if max_num_chars < 5:
            selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21] if max_num_chars == 4 else list(range(max_perms))
            perm_pool = torch.as_tensor(list(permutations(range(max_num_chars), max_num_chars)), device=target.device)[
                selector
            ][1:]

            perms = torch.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(len(perm_pool), size=num_gen_perms - len(perms), replace=False)
                perms = torch.cat([perms, perm_pool[i]])
        else:
            perms.extend([torch.randperm(max_num_chars, device=target.device) for _ in range(num_gen_perms - len(perms))])
            perms = torch.stack(perms)

        comp = perms.flip(-1)
        perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)

        bos_idx = torch.zeros(len(perms), 1, device=perms.device)
        eos_idx = torch.full((len(perms), 1), max_num_chars + 1, device=perms.device)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)

        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=target.device)
        return perms.int()  # (num_perms, max_length + 1)

    def generate_attention_masks(self, permutation: torch.Tensor):
        """Generate content and query masks for the decoder attention.

        Args:
            permutation: The permutation of the target sequence.

        Returns:
            content_mask: The content mask for the decoder attention.
            query_mask: The query mask for the decoder attention.
        """

        sz = permutation.shape[0]
        mask = torch.ones((sz, sz), device=permutation.device)

        for i in range(sz):
            query_idx = permutation[i]
            masked_keys = permutation[i + 1 :]
            mask[query_idx, masked_keys] = 1.0

        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=permutation.device)] = 1.0
        query_mask = mask[1:, :-1]

        return content_mask.int(), query_mask.int()

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        tgt_query_mask: Optional[torch.Tensor] = None,
    ):

        if tgt_padding_mask is not None:
            tgt_mask = tgt_padding_mask.unsqueeze(1).unsqueeze(1) & tgt_mask
            tgt_mask = tgt_mask.int()
        N, L = tgt.shape
        # apply positional information to the target sequence excluding the SOS token
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, : L - 1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        if target is None:
            target = self.pos_queries[:, :L].expand(N, -1, -1)
        target = self.dropout(target)
        return self.decoder(target, tgt_emb, memory, tgt_query_mask)

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[List[str]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> Dict[str, Any]:
        features = self.feat_extractor(x)["features"]  # (batch_size, patches_seqlen, d_model)
        pos_queries = self.pos_queries[:, : self.max_length].expand(features.size(0), -1, -1)

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        tgt_mask = query_mask = (
            torch.tril(torch.ones((self.max_length, self.max_length), device=features.device), diagonal=0)
            .to(dtype=torch.bool)
            .int()
        )

        if self.training and target is None:
            raise ValueError("Need to provide labels during training")

        if target is not None:
            # Build target tensor
            _gt, _seq_len = self.build_target(target)
            gt, seq_len = torch.from_numpy(_gt).to(dtype=torch.long).to(x.device), torch.tensor(_seq_len).to(x.device)

            # Generate target permutations
            tgt_perms = self.gen_tgt_perms(gt)
            tgt_in = gt[:, :-1]
            tgt_out = gt[:, 1:]

            # Create padding mask for target input
            tgt_padding_mask = (tgt_in == self.vocab_size) | (tgt_in == self.vocab_size - 2)

            for i, perm in enumerate(tgt_perms):
                # Generate attention masks for the permutation
                tgt_mask, query_mask = self.generate_attention_masks(perm)
                #print(query_mask)
                #print(query_mask.shape)

                # Decode target input and obtain logits
                out = self.decode(tgt_in, features, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
                logits = self.head(out)

                if i == 1:
                    # Replace [EOS] token with the vocab size in target output
                    tgt_out = torch.where(tgt_out == self.vocab_size - 2, self.vocab_size, tgt_out)
        else:
            # Initialize target input tensor with SOS token
            tgt_in = torch.full(
                (features.size(0), self.max_length), self.vocab_size, dtype=torch.long, device=features.device
            )
            tgt_in[:, 0] = self.vocab_size - 1  # <sos>

            logits = []
            for i in range(self.max_length):
                j = i + 1  # next token index

                # Efficient decoding: Input the context up to the ith token using one query at a time
                tgt_out = self.decode(
                    tgt_in[:, :j],
                    features,
                    tgt_mask[:j, :j],
                    target=pos_queries[:, i:j],
                    tgt_query_mask=query_mask[i:j, :j],
                )

                # Obtain the next token probability in the output's ith token position
                p_i = self.head(tgt_out)
                logits.append(p_i)

                if j < self.max_length:
                    # Greedy decode: Add the next token index to the target input
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)

                    # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                    if (tgt_in == self.vocab_size - 2).any(dim=-1).all():
                        break

            logits = torch.cat(logits, dim=1)

            # Update query mask
            query_mask[torch.triu(torch.ones(self.max_length, self.max_length, dtype=torch.bool, device=features.device), 2)] = 0

            # Prepare target input for 1 refine iteration
            bos = torch.full((features.size(0), 1), self.vocab_size - 1, dtype=torch.long, device=features.device)
            tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)

            # Create padding mask for refined target input
            tgt_padding_mask = (tgt_in == self.vocab_size - 2).int().cumsum(-1) > 0

            # Decode refined target input and obtain logits
            tgt_out = self.decode(
                tgt_in,
                features,
                tgt_mask,
                tgt_padding_mask,
                target=pos_queries,
                tgt_query_mask=query_mask[:, : tgt_in.shape[1]],
            )
            logits = self.head(tgt_out)

        # TODO: decoding -> decode_ar looks like the MASTER model positionwise decoding

        out: Dict[str, Any] = {}
        if self.exportable:
            out["logits"] = logits
            return out

        if return_model_output:
            out["out_map"] = logits

        if target is None or return_preds:
            # Post-process boxes
            out["preds"] = self.postprocessor(logits)

        if target is not None:
            out["loss"] = self.compute_loss(logits, gt, seq_len)

        return out

    @staticmethod
    def compute_loss(
        model_output: torch.Tensor,
        gt: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> torch.Tensor:
        """Compute categorical cross-entropy loss for the model.
        Sequences are masked after the EOS character.

        Args:
            model_output: predicted logits of the model
            gt: the encoded tensor with gt labels
            seq_len: lengths of each gt word inside the batch

        Returns:
            The loss of the model on the batch
        """
        # Input length : number of steps
        input_len = model_output.shape[1]
        # Add one for additional <eos> token (sos disappear in shift!)
        seq_len = seq_len + 1
        # Compute loss: don't forget to shift gt! Otherwise the model learns to output the gt[t-1]!
        # The "masked" first gt char is <sos>. Delete last logit of the model output.
        cce = F.cross_entropy(model_output[:, :-1, :].permute(0, 2, 1), gt[:, 1:-1], reduction="none")
        # Compute mask, remove 1 timestep here as well
        mask_2d = torch.arange(input_len - 1, device=model_output.device)[None, :] >= seq_len[:, None]
        cce[mask_2d] = 0

        ce_loss = cce.sum(1) / seq_len.to(dtype=model_output.dtype)
        return ce_loss.mean()


class PARSeqPostProcessor(_PARSeqPostProcessor):
    """Post processor for PARSeq architecture

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    def __call__(
        self,
        logits: torch.Tensor,
    ) -> List[Tuple[str, float]]:
        # compute pred with argmax for attention models
        out_idxs = logits.argmax(-1)
        # N x L
        probs = torch.gather(torch.softmax(logits, -1), -1, out_idxs.unsqueeze(-1)).squeeze(-1)
        # Take the minimum confidence of the sequence
        probs = probs.min(dim=1).values.detach().cpu()

        # Manual decoding
        word_values = [
            "".join(self._embedding[idx] for idx in encoded_seq).split("<eos>")[0]
            for encoded_seq in out_idxs.cpu().numpy()
        ]

        return list(zip(word_values, probs.numpy().tolist()))


def _parseq(
    arch: str,
    pretrained: bool,
    backbone_fn: Callable[[bool], nn.Module],
    layer: str,
    pretrained_backbone: bool = True,
    ignore_keys: Optional[List[str]] = None,
    **kwargs: Any,
) -> PARSeq:
    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["vocab"] = kwargs.get("vocab", _cfg["vocab"])
    _cfg["input_shape"] = kwargs.get("input_shape", _cfg["input_shape"])

    kwargs["vocab"] = _cfg["vocab"]
    kwargs["input_shape"] = _cfg["input_shape"]

    # Feature extractor
    feat_extractor = IntermediateLayerGetter(
        backbone_fn(pretrained_backbone, input_shape=_cfg["input_shape"]),  # type: ignore[call-arg]
        {layer: "features"},
    )

    # Build the model
    model = PARSeq(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if _cfg["vocab"] != default_cfgs[arch]["vocab"] else None
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    return model


def parseq(pretrained: bool = False, **kwargs: Any) -> PARSeq:
    """PARSeq architecture from
    `"Scene Text Recognition with Permuted Autoregressive Sequence Models" <https://arxiv.org/pdf/2207.06966>`_.

    >>> import torch
    >>> from doctr.models import parseq
    >>> model = parseq(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 32, 128))
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """

    return _parseq(
        "parseq",
        pretrained,
        vit_s,
        "1",
        embedding_units=384,
        ignore_keys=["head.weight", "head.bias"],
        **kwargs,
    )
