# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.rnn.translate import data_utils
#from tensorflow.nn import rnn, rnn_cell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope

from .seq2seq import model_with_buckets
from .seq2seq import embedding_attention_decoder

class Seq2SeqModel(object):
    """Sequence-to-sequence model with attention and for multiple buckets.
    This class implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder. This is the same as the model described in
    this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
    or into the seq2seq library for complete model implementation.
    This class also allows to use GRU cells in addition to LSTM cells, and
    sampled softmax to handle large output vocabulary size. A single-layer
    version of this model, but with bi-directional encoder, was presented in
      http://arxiv.org/abs/1409.0473
    and sampled softmax is described in Section 3 of the following paper.
      http://arxiv.org/abs/1412.2007
    """

    def __init__(self, encoder_masks, encoder_inputs_tensor, 
            decoder_inputs,
            target_weights,
            target_vocab_size, 
            buckets,
            target_embedding_size,
            attn_num_layers,
            attn_num_hidden,
            forward_only,
            use_gru):
        """Create the model.

        Args:
          source_vocab_size: size of the source vocabulary.
          target_vocab_size: size of the target vocabulary.
          buckets: a list of pairs (I, O), where I specifies maximum input length
            that will be processed in that bucket, and O specifies maximum output
            length. Training instances that have inputs longer than I or outputs
            longer than O will be pushed to the next bucket and padded accordingly.
            We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
          size: number of units in each layer of the model.
          num_layers: number of layers in the model.
          max_gradient_norm: gradients will be clipped to maximally this norm.
          learning_rate: learning rate to start with.
          learning_rate_decay_factor: decay learning rate by this much when needed.
          use_lstm: if true, we use LSTM cells instead of GRU cells.
          num_samples: number of samples for sampled softmax.
          forward_only: if set, we do not construct the backward pass in the model.
        """
        self.encoder_inputs_tensor = encoder_inputs_tensor
        self.decoder_inputs = decoder_inputs
        self.target_weights = target_weights
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.encoder_masks = encoder_masks

        # Create the internal multi-layer cell for our RNN.
        single_cell = tf.nn.rnn_cell.BasicLSTMCell(attn_num_hidden, forget_bias=0.0, state_is_tuple=False)
        if use_gru:
            print("using GRU CELL in decoder")
            single_cell = tf.nn.rnn_cell.GRUCell(attn_num_hidden, state_is_tuple=False)
        cell = single_cell

        if attn_num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * attn_num_layers, state_is_tuple=False)

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(lstm_inputs, decoder_inputs, seq_length, do_decode):
            num_hidden = attn_num_layers * attn_num_hidden
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=0.0, state_is_tuple=False)
            # Backward direction cell
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=0.0, state_is_tuple=False)
            pre_encoder_inputs, output_state_fw, output_state_bw = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, lstm_inputs,
                initial_state_fw=None, initial_state_bw=None,
                dtype=tf.float32, sequence_length=None, scope=None)
            encoder_inputs = [e*f for e,f in zip(pre_encoder_inputs,encoder_masks[:seq_length])]
            top_states = [array_ops.reshape(e, [-1, 1, num_hidden*2])
                    for e in encoder_inputs]
            attention_states = array_ops.concat(1, top_states)
            initial_state = tf.concat(concat_dim=1, values=[output_state_fw, output_state_bw])
            outputs, _, attention_weights_history = embedding_attention_decoder(
                    decoder_inputs, initial_state, attention_states, cell,
                    num_symbols=target_vocab_size, 
                    embedding_size=target_embedding_size,
                    num_heads=1,
                    output_size=target_vocab_size, 
                    output_projection=None,
                    feed_previous=do_decode,
                    initial_state_attention=False,
                    attn_num_hidden = attn_num_hidden)
            return outputs, attention_weights_history

        # Our targets are decoder inputs shifted by one.
        targets = [decoder_inputs[i + 1]
                for i in xrange(len(decoder_inputs) - 1)]

        softmax_loss_function = None # default to tf.nn.sparse_softmax_cross_entropy_with_logits

        # Training outputs and losses.
        if forward_only:
            self.outputs, self.losses, self.attention_weights_histories = model_with_buckets(
                    encoder_inputs_tensor, decoder_inputs, targets,
                    self.target_weights, buckets, lambda x, y, z: seq2seq_f(x, y, z, True),
                    softmax_loss_function=softmax_loss_function)
        else:
            self.outputs, self.losses, self.attention_weights_histories = model_with_buckets(
                    encoder_inputs_tensor, decoder_inputs, targets,
                    self.target_weights, buckets, lambda x, y, z: seq2seq_f(x, y, z, False),
                    softmax_loss_function=softmax_loss_function)
