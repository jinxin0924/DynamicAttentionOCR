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

import tensorflow as tf


# from tensorflow.models.rnn.translate import data_utils
# from tensorflow.nn import rnn, rnn_cell


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
               mode,
               use_gru,
               beam_width):
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
    self.mode = mode

    # Create the internal multi-layer cell for our RNN.
    def _create_cell(attn_num_hidden, attn_num_layers, use_gru=False):
      if attn_num_layers == 1:
        if use_gru:
          return tf.contrib.rnn.GRUCell(attn_num_hidden)
        else:
          return tf.contrib.rnn.BasicLSTMCell(attn_num_hidden, forget_bias=0.0,
                                              state_is_tuple=False)
      else:
        cell_list = []
        for i in range(attn_num_layers):
          if use_gru:
            cell_list.append(tf.contrib.rnn.GRUCell(attn_num_hidden))
          else:
            cell_list.append(
              tf.contrib.rnn.BasicLSTMCell(attn_num_hidden, forget_bias=0.0,
                                           state_is_tuple=False))
        return tf.contrib.rnn.MultiRNNCell(cell_list, state_is_tuple=False)

    with tf.variable_scope("dynamic_seq2seq", dtype=dtype):
      # build encoder
      with tf.variable_scope("encoder"):
        fw_cell = _create_cell(num_bi_layers, num_bi_residual_layers, use_gru)
        bw_cell = _create_cell(num_bi_layers, num_bi_residual_layers, use_gru)

        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
          fw_cell,
          bw_cell,
          inputs,
          dtype=dtype,
          sequence_length=sequence_length,
          time_major=self.time_major,
          swap_memory=True)

        encoder_outputs, bi_encoder_state = tf.concat(bi_outputs, -1), bi_state
        if num_bi_layers == 1:
          encoder_state = bi_encoder_state
        else:
          # alternatively concat forward and backward states
          encoder_state = []
          for layer_id in range(num_bi_layers):
            encoder_state.append(bi_encoder_state[0][layer_id])  # forward
            encoder_state.append(bi_encoder_state[1][layer_id])  # backward
          encoder_state = tuple(encoder_state)

      ## Decoder.
      with tf.variable_scope("decoder") as decoder_scope:
        self.embedding_decoder = tf.get_variable(
          'embedding', [target_vocab_size, target_embedding_size], dtype)

        decoder_cell = _create_cell(attn_num_hidden, attn_num_layers, use_gru)

        if self.mode == tf.contrib.learn.ModeKeys.INFER and beam_width > 0:
          decoder_initial_state = tf.contrib.seq2seq.tile_batch(
            encoder_state, multiplier=beam_width)
        else:
          decoder_initial_state = encoder_state

        ## Train or eval
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
          # decoder_emp_inp: [max_time, batch_size, num_units]
          target_input = iterator.target_input
          if self.time_major:
            target_input = tf.transpose(target_input)

          decoder_emb_inp = tf.nn.embedding_lookup(
            self.embedding_decoder, target_input)

          # Helper
          helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_emb_inp, iterator.target_sequence_length,
            time_major=self.time_major)

          # Decoder
          my_decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell,
            helper,
            decoder_initial_state, )

          # Dynamic decoding
          outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=decoder_scope)

          sample_id = outputs.sample_id

          # Note: there's a subtle difference here between train and inference.
          # We could have set output_layer when create my_decoder
          #   and shared more code between train and inference.
          # We chose to apply the output_layer to all timesteps for speed:
          #   10% improvements for small models & 20% for larger ones.
          # If memory is a concern, we should apply output_layer per timestep.
          logits = self.output_layer(outputs.rnn_output)

        ## Inference
        else:
          length_penalty_weight = hparams.length_penalty_weight
          start_tokens = tf.fill([self.batch_size], tgt_sos_id)
          end_token = tgt_eos_id

          if beam_width > 0:
            my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
              cell=cell,
              embedding=self.embedding_decoder,
              start_tokens=start_tokens,
              end_token=end_token,
              initial_state=decoder_initial_state,
              beam_width=beam_width,
              output_layer=self.output_layer,
              length_penalty_weight=length_penalty_weight)
          else:
            # Helper
            sampling_temperature = hparams.sampling_temperature
            if sampling_temperature > 0.0:
              helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                self.embedding_decoder, start_tokens, end_token,
                softmax_temperature=sampling_temperature,
                seed=hparams.random_seed)
            else:
              helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.embedding_decoder, start_tokens, end_token)

            # Decoder
            my_decoder = tf.contrib.seq2seq.BasicDecoder(
              cell,
              helper,
              decoder_initial_state,
              output_layer=self.output_layer  # applied per timestep

            )

        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
          my_decoder,
          maximum_iterations=maximum_iterations,
          output_time_major=self.time_major,
          swap_memory=True,
          scope=decoder_scope)

        if beam_width > 0:
          logits = tf.no_op()
          sample_id = outputs.predicted_ids
        else:
          logits = outputs.rnn_output
          sample_id = outputs.sample_id
