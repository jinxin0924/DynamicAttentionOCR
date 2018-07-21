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


class Seq2SeqModel(object):
  def __init__(self, encoder_inputs_tensor,
               decoder_inputs,
               sequence_length,
               source_sequence_length,
               target_vocab_size,
               target_embedding_size,
               target_sequence_length,
               attn_num_layers,
               attn_num_hidden,
               mode,
               use_gru,
               beam_width,
               time_major,
               attention_option):
    self.encoder_inputs_tensor = encoder_inputs_tensor
    self.decoder_inputs = decoder_inputs
    self.target_vocab_size = target_vocab_size
    self.mode = mode
    self.sequence_length = sequence_length
    self.source_sequence_length = source_sequence_length
    self.beam_width = beam_width
    self.time_major = time_major

    dtype = tf.float32
    self.output_layer = tf.layers.Dense(
      target_vocab_size, use_bias=False, name="output_projection")

    self.embedding_decoder = tf.get_variable(
      'embedding', [target_vocab_size, target_embedding_size], dtype)

    with tf.variable_scope("dynamic_seq2seq"):
      # build encoder
      with tf.variable_scope("encoder"):
        encoder_outputs, encoder_state = self._build_encoder(
          encoder_inputs_tensor,
          sequence_length,
          attn_num_hidden,
          attn_num_layers,
          use_gru)

      ## Decoder.
      with tf.variable_scope("decoder"):
        logits, sample_id, final_context_state = self._build_decoder(
          target_input, target_sequence_length,
          attention_option,
          encoder_outputs, encoder_state, attn_num_hidden,
          attn_num_layers, use_gru)

      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        loss = self._compute_loss(target_input, logits, target_sequence_length)
      else:
        loss = None
    self.logits = logits
    self.loss = loss
    self.final_context_state = final_context_state
    self.sample_id = sample_id

  # Create the internal multi-layer cell for our RNN.
  def _create_cell(self, attn_num_hidden, attn_num_layers, use_gru=False):
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

  def _build_encoder(self, inputs, sequence_length, attn_num_hidden,
                     attn_num_layers, use_gru):
    num_bi_layers = attn_num_hidden // 2
    fw_cell = self.create_cell(num_bi_layers, attn_num_layers, use_gru)
    bw_cell = self.create_cell(num_bi_layers, attn_num_layers, use_gru)

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
    return encoder_outputs, encoder_state

  def _attention_mechanism_fn(self, attention_option, num_units, memory,
                              source_sequence_length):
    """Create attention mechanism based on the attention_option."""
    # Mechanism
    if attention_option == "luong":
      attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units, memory, memory_sequence_length=source_sequence_length)
    elif attention_option == "scaled_luong":
      attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units,
        memory,
        memory_sequence_length=source_sequence_length,
        scale=True)
    elif attention_option == "bahdanau":
      attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units, memory, memory_sequence_length=source_sequence_length)
    elif attention_option == "normed_bahdanau":
      attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units,
        memory,
        memory_sequence_length=source_sequence_length,
        normalize=True)
    else:
      raise ValueError("Unknown attention option %s" % attention_option)

    return attention_mechanism

  def _build_decoder(self, target_input, target_sequence_length,
                     attention_option, encoder_outputs,
                     encoder_state,
                     attn_num_hidden,
                     attn_num_layers, use_gru, sampling_temperature=0.0,
                     maximum_iterations=20):
    tgt_sos_id = tf.cast(0, tf.int32)
    tgt_eos_id = tf.cast(1, tf.int32)
    if self.time_major:
      memory = tf.transpose(encoder_outputs, [1, 0, 2])
    else:
      memory = encoder_outputs

    if self.mode == tf.contrib.learn.ModeKeys.INFER and self.beam_width > 0:
      memory = tf.contrib.seq2seq.tile_batch(
        memory, multiplier=self.beam_width)
      source_sequence_length = tf.contrib.seq2seq.tile_batch(
        self.source_sequence_length, multiplier=beam_width)
      encoder_state = tf.contrib.seq2seq.tile_batch(
        encoder_state, multiplier=self.beam_width)
      batch_size = self.batch_size * self.beam_width
    else:
      batch_size = self.batch_size

    attention_mechanism = self._attention_mechanism_fn(
      attention_option, attn_num_hidden, memory, self.source_sequence_length)

    decoder_cell = self._create_cell(attn_num_hidden, attn_num_layers, use_gru)

    alignment_history = (self.mode == tf.contrib.learn.ModeKeys.INFER and
                         self.beam_width == 0)

    cell = tf.contrib.seq2seq.AttentionWrapper(
      decoder_cell,
      attention_mechanism,
      attention_layer_size=attn_num_hidden,
      alignment_history=alignment_history,
      output_attention=True,
      name="attention")

    if self.mode == tf.contrib.learn.ModeKeys.INFER and self.beam_width > 0:
      decoder_initial_state = tf.contrib.seq2seq.tile_batch(
        encoder_state, multiplier=beam_width)
    else:
      decoder_initial_state = cell.clone(cell_state=encoder_state)

    ## Train or eval
    if self.mode != tf.contrib.learn.ModeKeys.INFER:
      # decoder_emp_inp: [max_time, batch_size, num_units]
      if self.time_major:
        target_input = tf.transpose(target_input)

      decoder_emb_inp = tf.nn.embedding_lookup(
        self.embedding_decoder, target_input)

      # Helper
      helper = tf.contrib.seq2seq.TrainingHelper(
        decoder_emb_inp, target_sequence_length,
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
      logits = self.output_layer(outputs.rnn_output)

    ## Inference
    else:
      length_penalty_weight = 0.0
      start_tokens = tf.fill([self.batch_size], tgt_sos_id)
      end_token = tgt_eos_id

      if self.beam_width > 0:
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
        sampling_temperature = sampling_temperature
        if sampling_temperature > 0.0:
          helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
            self.embedding_decoder, start_tokens, end_token,
            softmax_temperature=sampling_temperature)
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

    if self.beam_width > 0:
      logits = tf.no_op()
      sample_id = outputs.predicted_ids
    else:
      logits = outputs.rnn_output
      sample_id = outputs.sample_id
    return logits, sample_id, final_context_state


def _compute_loss(self, target_output, logits, target_sequence_length):
  """Compute optimization loss."""

  def get_max_time(self, tensor):
    time_axis = 0 if self.time_major else 1
    return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

  if self.time_major:
    target_output = tf.transpose(target_output)
  max_time = get_max_time(target_output)
  crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=target_output, logits=logits)
  target_weights = tf.sequence_mask(target_sequence_length, max_time,
                                    dtype=logits.dtype)
  if self.time_major:
    target_weights = tf.transpose(target_weights)

  loss = tf.reduce_sum(
    crossent * target_weights) / tf.to_float(self.batch_size)
  return loss
