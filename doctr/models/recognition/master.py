# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from typing import Tuple, List, Dict, Any, Optional
from copy import deepcopy

from .core import RecognitionModel, RecognitionPostProcessor
from ..backbones.resnet import ResnetStage
from ..utils import conv_sequence, load_pretrained_params
from .transformer import Decoder, positional_encoding, create_look_ahead_mask, create_padding_mask
from ...datasets import VOCABS

__all__ = ['MASTER', 'MASTERPostProcessor', 'master']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'master': {
        'mean': (.5, .5, .5),
        'std': (1., 1., 1.),
        'd_model': 512, 'headers': 1, 'dff': 2048, 'num_heads': 8, 'num_layers': 3, 'max_length': 50,
        'input_shape': (48, 160, 3),
        'post_processor': 'MASTERPostProcessor',
        'vocab': VOCABS['french'],
        'url': None,
    },
}


class MAGC(layers.Layer):

    """Implements the Multi-Aspect Global Context Attention, as described in
    <https://arxiv.org/pdf/1910.02562.pdf>`_.

    Args:
        inplanes: input channels
        headers: number of headers to split channels
        att_scale: if True, re-scale attention to counteract the variance distibutions
        **kwargs
    """

    def __init__(
        self,
        inplanes: int,
        headers: int = 1,
        att_scale: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.headers = headers  # h
        self.inplanes = inplanes  # C
        self.att_scale = att_scale

        self.single_header_inplanes = int(inplanes / headers)  # C / h

        self.conv_mask = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            kernel_initializer=tf.initializers.he_normal()
        )

        self.transform = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=self.inplanes,
                    kernel_size=1,
                    kernel_initializer=tf.initializers.he_normal()
                ),
                tf.keras.layers.LayerNormalization([1, 2, 3]),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(
                    filters=self.inplanes,
                    kernel_size=1,
                    kernel_initializer=tf.initializers.he_normal()
                ),
            ],
            name='transform'
        )

    @tf.function
    def context_modeling(self, inputs: tf.Tensor) -> tf.Tensor:
        B, H, W, C = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]

        # B, H, W, C -->> B*h, H, W, C/h
        x = tf.reshape(inputs, shape=(B, H, W, self.headers, self.single_header_inplanes))
        x = tf.transpose(x, perm=(0, 3, 1, 2, 4))
        x = tf.reshape(x, shape=(B * self.headers, H, W, self.single_header_inplanes))

        # Compute shorcut
        shortcut = x
        # B*h, 1, H*W, C/h
        shortcut = tf.reshape(shortcut, shape=(B * self.headers, 1, H * W, self.single_header_inplanes))
        # B*h, 1, C/h, H*W
        shortcut = tf.transpose(shortcut, perm=[0, 1, 3, 2])

        # Compute context mask
        # B*h, H, W, 1,
        context_mask = self.conv_mask(x)
        # B*h, 1, H*W, 1
        context_mask = tf.reshape(context_mask, shape=(B * self.headers, 1, H * W, 1))
        # scale variance
        if self.att_scale and self.headers > 1:
            context_mask = context_mask / tf.sqrt(self.single_header_inplanes)
        # B*h, 1, H*W, 1
        context_mask = tf.keras.activations.softmax(context_mask, axis=2)

        # Compute context
        # B*h, 1, C/h, 1
        context = tf.matmul(shortcut, context_mask)
        context = tf.reshape(context, shape=(B, 1, C, 1))
        # B, 1, 1, C
        context = tf.transpose(context, perm=(0, 1, 3, 2))
        # Set shape to resolve shape when calling this module in the Sequential MAGCResnet
        b, c = inputs.get_shape().as_list()[0], inputs.get_shape().as_list()[-1]
        context.set_shape([b, 1, 1, c])
        return context

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # Context modeling: B, H, W, C  ->  B, 1, 1, C
        context = self.context_modeling(inputs)
        # Transform: B, 1, 1, C  ->  B, 1, 1, C
        transformed = self.transform(context)
        return inputs + transformed


class MAGCResnet(Sequential):

    """Implements the modified resnet with MAGC layers, as described in paper.

    Args:
        headers: number of header to split channels in MAGC layers
        input_shape: shape of the model input (without batch dim)
    """

    def __init__(
        self,
        headers: int = 1,
        input_shape: Tuple[int, int, int] = (48, 160, 3),
    ) -> None:
        _layers = [
            # conv_1x
            *conv_sequence(out_channels=64, activation='relu', bn=True, kernel_size=3, input_shape=input_shape),
            *conv_sequence(out_channels=128, activation='relu', bn=True, kernel_size=3),
            layers.MaxPooling2D((2, 2), (2, 2)),
            # conv_2x
            ResnetStage(num_blocks=1, output_channels=256),
            MAGC(inplanes=256, headers=headers, att_scale=True),
            *conv_sequence(out_channels=256, activation='relu', bn=True, kernel_size=3),
            layers.MaxPooling2D((2, 2), (2, 2)),
            # conv_3x
            ResnetStage(num_blocks=2, output_channels=512),
            MAGC(inplanes=512, headers=headers, att_scale=True),
            *conv_sequence(out_channels=512, activation='relu', bn=True, kernel_size=3),
            layers.MaxPooling2D((2, 1), (2, 1)),
            # conv_4x
            ResnetStage(num_blocks=5, output_channels=512),
            MAGC(inplanes=512, headers=headers, att_scale=True),
            *conv_sequence(out_channels=512, activation='relu', bn=True, kernel_size=3),
            # conv_5x
            ResnetStage(num_blocks=3, output_channels=512),
            MAGC(inplanes=512, headers=headers, att_scale=True),
            *conv_sequence(out_channels=512, activation='relu', bn=True, kernel_size=3),
        ]
        super().__init__(_layers)


class MASTER(RecognitionModel):

    """Implements MASTER as described in paper: <https://arxiv.org/pdf/1910.02562.pdf>`_.
    Implementation based on the official TF implementation: <https://github.com/jiangxiluning/MASTER-TF>`_.

    Args:
        vocab_size: size of the vocabulary, (without EOS, SOS, PAD)
        d_model: d parameter for the transformer decoder
        headers: headers for the MAGC module
        dff: depth of the pointwise feed-forward layer
        num_heads: number of heads for the mutli-head attention module
        num_layers: number of decoder layers to stack
        max_length: maximum length of character sequence handled by the model
        input_shape: size of the image inputs
    """

    def __init__(
        self,
        vocab: str,
        d_model: int = 512,
        headers: int = 1,
        dff: int = 2048,
        num_heads: int = 8,
        num_layers: int = 3,
        max_length: int = 50,
        input_shape: tuple = (48, 160, 3),
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(vocab=vocab, cfg=cfg)

        self.max_length = max_length
        self.vocab_size = len(vocab)

        self.feature_extractor = MAGCResnet(headers=headers, input_shape=input_shape)
        self.seq_embedding = layers.Embedding(self.vocab_size + 1, d_model)  # One additional class for EOS

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=self.vocab_size,
            maximum_position_encoding=max_length,
        )
        self.feature_pe = positional_encoding(input_shape[0] * input_shape[1], d_model)
        self.linear = layers.Dense(self.vocab_size + 1, kernel_initializer=tf.initializers.he_uniform())

        self.postprocessor = MASTERPostProcessor(vocab=self.vocab)

    @tf.function
    def make_mask(self, target: tf.Tensor) -> tf.Tensor:
        look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
        target_padding_mask = create_padding_mask(target, self.vocab_size)  # Pad with EOS
        combined_mask = tf.maximum(target_padding_mask, look_ahead_mask)
        return combined_mask

    def compute_loss(
        self,
        model_output: tf.Tensor,
        gt: tf.Tensor,
        seq_len: tf.Tensor,
    ) -> tf.Tensor:
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
        input_len = tf.shape(model_output)[1]
        # Add one for additional <eos> token
        seq_len = seq_len + 1
        # One-hot gt labels
        oh_gt = tf.one_hot(gt, depth=model_output.shape[2])
        # Compute loss
        cce = tf.nn.softmax_cross_entropy_with_logits(oh_gt, model_output)
        # Compute mask
        mask_values = tf.zeros_like(cce)
        mask_2d = tf.sequence_mask(seq_len, input_len)
        masked_loss = tf.where(mask_2d, cce, mask_values)
        ce_loss = tf.math.divide(tf.reduce_sum(masked_loss, axis=1), tf.cast(seq_len, tf.float32))

        return tf.expand_dims(ce_loss, axis=1)

    def call(
        self,
        inputs: tf.Tensor,
        labels: Optional[List[str]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Call function for training

        Args:
            inputs: images
            labels: list of str labels
            return_model_output: if True, return logits
            return_preds: if True, decode logits

        Return:
            A dictionnary containing eventually loss, logits and predictions.
        """

        # Encode
        feature = self.feature_extractor(inputs, **kwargs)
        B, H, W, C = tf.shape(feature)[0], tf.shape(feature)[1], tf.shape(feature)[2], tf.shape(feature)[3]
        feature = tf.reshape(feature, shape=(B, H * W, C))
        encoded = feature + self.feature_pe[:, :H * W, :]

        out: Dict[str, tf.Tensor] = {}

        if labels is not None:
            # Compute target: tensor of gts and sequence lengths
            gt, seq_len = self.compute_target(labels)

            tgt_mask = self.make_mask(gt)

            # Compute logits
            output, _ = self.decoder(gt, encoded, tgt_mask, None, training=True)
            logits = self.linear(output)
            if return_model_output:
                out['out_map'] = logits

            # Compute loss
            out['loss'] = self.compute_loss(logits, gt, seq_len)

            if return_preds:
                predictions = self.postprocessor(logits)
                out['preds'] = predictions

        else:
            raw_predictions, logits = self.decode(encoded)
            if return_model_output:
                out['out_map'] = logits
            if return_preds:
                predictions = self.postprocessor(logits)
                out['preds'] = predictions

        return out

    @tf.function
    def decode(self, encoded: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Decode function for prediction

        Args:
            encoded: encoded features

        Return:
            A Tuple of tf.Tensor: predictions, logits
        """
        B = tf.shape(encoded)[0]
        max_len = tf.constant(self.max_length, dtype=tf.int32)
        start_symbol = tf.constant(self.vocab_size + 1, dtype=tf.int32)  # SOS (EOS = vocab_size)
        padding_symbol = tf.constant(self.vocab_size, dtype=tf.int32)

        ys = tf.fill(dims=(B, max_len - 1), value=padding_symbol)
        start_vector = tf.fill(dims=(B, 1), value=start_symbol)
        ys = tf.concat([start_vector, ys], axis=-1)

        final_logits = tf.zeros(shape=(B, max_len - 1, self.vocab_size + 1), dtype=tf.float32)  # don't fgt EOS
        # max_len = len + 2
        for i in range(self.max_length - 1):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(final_logits, tf.TensorShape([None, None, self.vocab_size + 1]))]
            )
            ys_mask = self.make_mask(ys)
            output, _ = self.decoder(ys, encoded, ys_mask, None, training=False)
            logits = self.linear(output)
            prob = tf.nn.softmax(logits, axis=-1)
            next_word = tf.argmax(prob, axis=-1, output_type=ys.dtype)

            # ys.shape = B, T
            i_mesh, j_mesh = tf.meshgrid(tf.range(B), tf.range(max_len), indexing='ij')
            indices = tf.stack([i_mesh[:, i + 1], j_mesh[:, i + 1]], axis=1)

            ys = tf.tensor_scatter_nd_update(ys, indices, next_word[:, i + 1])

            if i == (self.max_length - 2):
                final_logits = logits

        # ys predictions of shape B x max_length, final_logits of shape B x max_length x vocab_size + 1
        return ys, final_logits


class MASTERPostProcessor(RecognitionPostProcessor):
    """Post processor for MASTER architectures

    Args:
        vocab: string containing the ordered sequence of supported characters
        ignore_case: if True, ignore case of letters
        ignore_accents: if True, ignore accents of letters
    """

    def __call__(
        self,
        logits: tf.Tensor,
    ) -> List[Tuple[str, float]]:
        # compute pred with argmax for attention models
        out_idxs = tf.math.argmax(logits, axis=2)
        # N x L
        probs = tf.gather(tf.nn.softmax(logits, axis=-1), out_idxs, axis=-1, batch_dims=2)
        # Take the minimum confidence of the sequence
        probs = tf.math.reduce_min(probs, axis=1)

        # decode raw output of the model with tf_label_to_idx
        out_idxs = tf.cast(out_idxs, dtype='int32')
        decoded_strings_pred = tf.strings.reduce_join(inputs=tf.nn.embedding_lookup(self._embedding, out_idxs), axis=-1)
        decoded_strings_pred = tf.strings.split(decoded_strings_pred, "<eos>")
        decoded_strings_pred = tf.sparse.to_dense(decoded_strings_pred.to_sparse(), default_value='not valid')[:, 0]
        word_values = [word.decode() for word in decoded_strings_pred.numpy().tolist()]

        return list(zip(word_values, probs.numpy().tolist()))


def _master(arch: str, pretrained: bool, input_shape: Tuple[int, int, int] = None, **kwargs: Any) -> MASTER:

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg['input_shape'] = input_shape or _cfg['input_shape']
    _cfg['vocab'] = kwargs.get('vocab', _cfg['vocab'])
    _cfg['d_model'] = kwargs.get('d_model', _cfg['d_model'])
    _cfg['headers'] = kwargs.get('headers', _cfg['headers'])
    _cfg['dff'] = kwargs.get('dff', _cfg['dff'])
    _cfg['num_heads'] = kwargs.get('num_heads', _cfg['num_heads'])
    _cfg['num_layers'] = kwargs.get('num_layers', _cfg['num_layers'])
    _cfg['max_length'] = kwargs.get('max_length', _cfg['max_length'])

    kwargs['vocab'] = _cfg['vocab']
    kwargs['d_model'] = _cfg['d_model']
    kwargs['headers'] = _cfg['headers']
    kwargs['dff'] = _cfg['dff']
    kwargs['num_heads'] = _cfg['num_heads']
    kwargs['num_layers'] = _cfg['num_layers']
    kwargs['max_length'] = _cfg['max_length']

    # Build the model
    model = MASTER(cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'])

    return model


def master(pretrained: bool = False, **kwargs: Any) -> MASTER:
    """MASTER as described in paper: <https://arxiv.org/pdf/1910.02562.pdf>`_.

    Example::
        >>> import tensorflow as tf
        >>> from doctr.models import sar_vgg16_bn
        >>> model = master(pretrained=False)
        >>> input_tensor = tf.random.uniform(shape=[1, 48, 160, 3], maxval=1, dtype=tf.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """

    return _master('master', pretrained, **kwargs)
