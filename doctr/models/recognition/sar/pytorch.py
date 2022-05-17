# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

from doctr.datasets import VOCABS

from ...classification import resnet31
from ...utils.pytorch import load_pretrained_params
from ..core import RecognitionModel, RecognitionPostProcessor

__all__ = ['SAR', 'sar_resnet31']

default_cfgs: Dict[str, Dict[str, Any]] = {
    'sar_resnet31': {
        'mean': (.5, .5, .5),
        'std': (1., 1., 1.),
        'input_shape': (3, 32, 128),
        'vocab': VOCABS['legacy_french'],
        'url': None,
    },
}


class SAREncoder(nn.Module):

    def __init__(self, in_feats: int, rnn_units: int, dropout_prob: float = 0.) -> None:

        super().__init__()
        self.rnn = nn.LSTM(in_feats, rnn_units, 2, batch_first=True, dropout=dropout_prob)
        self.linear = nn.Linear(rnn_units, rnn_units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # (N, L, C) --> (N, T, C)
        encoded = self.rnn(x)[0]
        # (N, C)
        return self.linear(encoded[:, -1, :])


class AttentionModule(nn.Module):

    def __init__(self, feat_chans: int, state_chans: int, attention_units: int) -> None:
        super().__init__()
        self.lstm_cell = nn.LSTMCell(feat_chans, state_chans)
        self.feat_conv = nn.Conv2d(feat_chans, attention_units, kernel_size=3, padding='same')
        # No need to add another bias since both tensors are summed together
        self.state_conv = nn.Conv2d(state_chans, attention_units, kernel_size=1, bias=False, padding='same')
        self.attention_projector = nn.Conv2d(attention_units, 1, kernel_size=1, bias=False, padding='same')

    def forward(self,
                prev_logit: torch.Tensor,  # (N, C)
                features: torch.Tensor,  # (N, C, H, W)
                hidden_state_init: torch.Tensor,  # (N, C)
                cell_state_init: torch.Tensor,  # (N, C)
                hidden_state: torch.Tensor,  # (N, C)
                cell_state: torch.Tensor  # (N, C)
                ) -> torch.Tensor:

        height_feat_map, width_feat_map = features.shape[-2:]
        # (N, state_chans), (N, state_chans)
        hidden_state, cell_state = self.lstm_cell(prev_logit, (hidden_state_init, cell_state_init))
        hidden_state, cell_state = self.lstm_cell(hidden_state_init, (hidden_state, cell_state))

        # (N, feat_chans, H, W) --> (N, attention_units, H, W)
        feat_projection = self.feat_conv(features)
        # (N, state_chans, 1, 1) --> (N, attention_units, 1, 1)
        hidden_state = hidden_state.view(hidden_state.size(0), hidden_state.size(1), 1, 1)
        state_projection = self.state_conv(hidden_state)
        # (N, attention_units, 1, 1) --> (N, attention_units, height_feat_map, width_feat_map)
        attention_weights = torch.tanh(torch.add(feat_projection, state_projection, alpha=1))
        # (N, attention_units, height_feat_map, width_feat_map) --> (N, 1, height_feat_map, width_feat_map)
        attention_weights = self.attention_projector(attention_weights)
        B, C, H, W = attention_weights.size()

        # (B, H, W) --> (B, 1, H, W)
        attention_weights = F.softmax(attention_weights.view(B, -1), dim=-1).view(B, C, H, W)
        # fuse features and attention weights (N, C)
        glimpse = torch.sum(torch.mul(features, attention_weights), dim=(2, 3), keepdim=False)
        return glimpse


class SARDecoder(nn.Module):
    """Implements decoder module of the SAR model
    Args:
        rnn_units: number of hidden units in recurrent cells
        max_length: maximum length of a sequence
        vocab_size: number of classes in the model alphabet
        embedding_units: number of hidden embedding units
        attention_units: number of hidden attention units
    """
    def __init__(
        self,
        rnn_units: int,
        max_length: int,
        vocab_size: int,
        embedding_units: int,
        attention_units: int = 512,
        feat_chans: int = 512,
        dropout_prob: float = 0.,
    ) -> None:

        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length

        self.embed = nn.Embedding(self.vocab_size + 1, embedding_units)
        self.attention_module = AttentionModule(feat_chans, rnn_units, attention_units)
        self.output_dense = nn.Linear(rnn_units, self.vocab_size + 1)
        self.dropout = nn.Dropout(dropout_prob)

        self.lstm_cell = nn.LSTMCell(embedding_units, rnn_units)

    def forward(self, features: torch.Tensor, encoded: torch.Tensor, gt):

        outputs = []

        if gt is not None:
            gt_embedding = self.embed(gt)
        else:
            # init symbol
            prev_symbol = self.embed(
                torch.full((features.size(0), ), fill_value=self.vocab_size, device=features.device, dtype=torch.long)
            )

        # init hidden state
        hidden_state_init, cell_state_init = self.lstm_cell(encoded)
        hidden_state, cell_state = self.lstm_cell(hidden_state_init)

        for t in range(self.max_length):
            if gt is not None:
                # (N, t, embedding_units)
                prev_symbol = gt_embedding[:, t, :]
            # (N, C)
            glimpse = self.attention_module(prev_symbol, features, hidden_state_init,
                                            cell_state_init, hidden_state, cell_state)
            # (N, vocab_size + 1)
            logits = self.output_dense(glimpse)
            if gt is not None:
                # (N, vocab_size + 1)
                logits = self.dropout(logits)
            else:
                # (N, vocab_size + 1)
                logits = F.softmax(input=logits, dim=-1)
                _, idx = torch.max(logits, dim=1, keepdim=False)
                # (N, rnn_units)
                prev_symbol = self.embed(idx)

            outputs.append(logits)

        return torch.stack(outputs, 1)


class SAR(nn.Module, RecognitionModel):
    """Implements a SAR architecture as described in `"Show, Attend and Read:A Simple and Strong Baseline for
    Irregular Text Recognition" <https://arxiv.org/pdf/1811.00751.pdf>`_.

    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        rnn_units: number of hidden units in both encoder and decoder LSTM
        embedding_units: number of embedding units
        attention_units: number of hidden units in attention module
        max_length: maximum word length handled by the model
        num_decoders: number of LSTM to stack in decoder layer
        dropout_prob: dropout probability of the encoder LSTM
        cfg: default setup dict of the model
    """

    def __init__(
        self,
        feature_extractor,
        vocab: str,
        rnn_units: int = 512,
        embedding_units: int = 512,
        attention_units: int = 512,
        max_length: int = 30,
        dropout_prob: float = 0.,
        input_shape: Tuple[int, int, int] = (3, 32, 128),
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__()
        self.vocab = vocab
        self.cfg = cfg

        self.max_length = max_length + 1  # Add 1 timestep for EOS after the longest word

        self.feat_extractor = feature_extractor

        # Size the LSTM
        self.feat_extractor.eval()
        with torch.no_grad():
            out_shape = self.feat_extractor(torch.zeros((1, *input_shape)))['features'].shape
        # Switch back to original mode
        self.feat_extractor.train()

        self.encoder = SAREncoder(out_shape[1], rnn_units, dropout_prob)
        self.decoder = SARDecoder(rnn_units, self.max_length, len(self.vocab),
                                  embedding_units, attention_units, dropout_prob=dropout_prob)

        self.postprocessor = SARPostProcessor(vocab=vocab)

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[List[str]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> Dict[str, Any]:

        features = self.feat_extractor(x)['features']
        # Vertical max pooling --> (N, C, W)
        pooled_features = F.max_pool2d(features, kernel_size=(features.shape[2], 1), stride=(1, 1))
        pooled_features = pooled_features.squeeze(2)
        # (N, W, C)
        pooled_features = pooled_features.permute(0, 2, 1).contiguous()
        # (N, C)
        encoded = self.encoder(pooled_features)
        if target is not None:
            _gt, _seq_len = self.build_target(target)
            gt, seq_len = torch.from_numpy(_gt).to(dtype=torch.long), torch.tensor(_seq_len)  # type: ignore[assignment]
            gt, seq_len = gt.to(x.device), seq_len.to(x.device)
        decoded_features = self.decoder(features, encoded, gt=None if target is None else gt)

        out: Dict[str, Any] = {}
        if return_model_output:
            out["out_map"] = decoded_features

        if target is None or return_preds:
            # Post-process boxes
            out["preds"] = self.postprocessor(decoded_features)

        if target is not None:
            out['loss'] = self.compute_loss(decoded_features, gt, seq_len)  # type: ignore[arg-type]

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
            gt: the encoded tensor with gt labels
            model_output: predicted logits of the model
            seq_len: lengths of each gt word inside the batch

        Returns:
            The loss of the model on the batch
        """
        # Input length : number of timesteps
        input_len = model_output.shape[1]
        # Add one for additional <eos> token
        seq_len = seq_len + 1
        # Compute loss
        # (N, L, vocab_size + 1)
        cce = F.cross_entropy(model_output.permute(0, 2, 1), gt, reduction='none')
        mask_2d = torch.arange(input_len, device=model_output.device)[None, :] >= seq_len[:, None]
        cce[mask_2d] = 0

        ce_loss = cce.sum(1) / seq_len.to(dtype=model_output.dtype)
        return ce_loss.mean()


class SARPostProcessor(RecognitionPostProcessor):
    """Post processor for SAR architectures"""

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
            ''.join(self._embedding[idx] for idx in encoded_seq).split("<eos>")[0]
            for encoded_seq in out_idxs.detach().cpu().numpy()
        ]

        return list(zip(word_values, probs.numpy().tolist()))


def _sar(
    arch: str,
    pretrained: bool,
    backbone_fn: Callable[[bool], nn.Module],
    layer: str,
    pretrained_backbone: bool = True,
    **kwargs: Any
) -> SAR:

    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg['vocab'] = kwargs.get('vocab', _cfg['vocab'])
    _cfg['input_shape'] = kwargs.get('input_shape', _cfg['input_shape'])

    # Feature extractor
    feat_extractor = IntermediateLayerGetter(
        backbone_fn(pretrained_backbone),
        {layer: 'features'},
    )
    kwargs['vocab'] = _cfg['vocab']
    kwargs['input_shape'] = _cfg['input_shape']

    # Build the model
    model = SAR(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'])

    return model


def sar_resnet31(pretrained: bool = False, **kwargs: Any) -> SAR:
    """SAR with a resnet-31 feature extractor as described in `"Show, Attend and Read:A Simple and Strong
    Baseline for Irregular Text Recognition" <https://arxiv.org/pdf/1811.00751.pdf>`_.

    >>> import torch
    >>> from doctr.models import sar_resnet31
    >>> model = sar_resnet31(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 32, 128))
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """

    return _sar('sar_resnet31', pretrained, resnet31, '10', **kwargs)
