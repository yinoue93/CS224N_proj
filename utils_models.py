import tensorflow as tf

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest


def attention_decoder_fn_sampled_inference(output_fn,
                                           encoder_state,
                                           attention_keys,
                                           attention_values,
                                           attention_score_fn,
                                           attention_construct_fn,
                                           embeddings,
                                           start_of_sequence_id,
                                           end_of_sequence_id,
                                           maximum_length,
                                           num_decoder_symbols,
                                           dtype=dtypes.int32,
                                           name=None):

    with ops.name_scope(name, "attention_decoder_fn_inference", [
        output_fn, encoder_state, attention_keys, attention_values,
        attention_score_fn, attention_construct_fn, embeddings,
        start_of_sequence_id, end_of_sequence_id, maximum_length,
        num_decoder_symbols, dtype
    ]):
        start_of_sequence_id = ops.convert_to_tensor(start_of_sequence_id, dtype)
        end_of_sequence_id = ops.convert_to_tensor(end_of_sequence_id, dtype)
        maximum_length = ops.convert_to_tensor(maximum_length, dtype)
        num_decoder_symbols = ops.convert_to_tensor(num_decoder_symbols, dtype)
        encoder_info = nest.flatten(encoder_state)[0]
        batch_size = encoder_info.get_shape()[0].value
        if output_fn is None:
            output_fn = lambda x: x
        if batch_size is None:
            batch_size = array_ops.shape(encoder_info)[0]

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        with ops.name_scope(
            name, "attention_decoder_fn_inference",
            [time, cell_state, cell_input, cell_output, context_state]
        ):
            if cell_input is not None:
                raise ValueError("Expected cell_input to be None, but saw: %s" %
                                cell_input)
            if cell_output is None:
                # invariant that this is time == 0
                next_input_id = array_ops.ones(
                    [batch_size,], dtype=dtype) * (start_of_sequence_id)
                done = array_ops.zeros([batch_size,], dtype=dtypes.bool)
                cell_state = encoder_state
                cell_output = array_ops.zeros(
                    [num_decoder_symbols], dtype=dtypes.float32)
                cell_input = array_ops.gather(embeddings, next_input_id)

                # init attention
                attention = _init_attention(encoder_state)
            else:
                # construct attention
                attention = attention_construct_fn(cell_output, attention_keys,
                                                   attention_values)
                cell_output = attention

                # sampled decoder
                cell_output = output_fn(cell_output)  # logits
                sampled_cell_output = random_ops.multinomial(cell_output, 1)
                sampled_cell_output = array_ops.reshape(sampled_cell_output, [-1])
                next_input_id = math_ops.cast(sampled_cell_output, dtype=dtype)
                done = math_ops.equal(next_input_id, end_of_sequence_id)
                cell_input = array_ops.gather(embeddings, next_input_id)

            # combine cell_input and attention
            next_input = array_ops.concat([cell_input, attention], 1)

            # if time > maxlen, return all true vector
            done = control_flow_ops.cond(
            math_ops.greater(time, maximum_length),
                lambda: array_ops.ones([batch_size,], dtype=dtypes.bool),
                lambda: done)
            return (done, cell_state, next_input, cell_output, context_state)

    return decoder_fn




def _init_attention(encoder_state):
    # Multi- vs single-layer
    if isinstance(encoder_state, tuple):
        top_state = encoder_state[-1]
    else:
        top_state = encoder_state

    # LSTM vs GRU
    if isinstance(top_state, core_rnn_cell_impl.LSTMStateTuple):
        attn = array_ops.zeros_like(top_state.h)
    else:
        attn = array_ops.zeros_like(top_state)

    return attn
