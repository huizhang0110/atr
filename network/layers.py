from atr.utils import label_map, sync_attention_wrapper

import functools
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2

slim = tf.contrib.slim

def combined_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.
    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.
    Args:
        tensor: A tensor of any type.
    Returns:
        A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.shape(tensor)
    combined_shape = []
    for index, dim in enumerate(static_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_shape[index])
    return combined_shape


def bilstm(layer_name, inputs, hidden_units):
    """Bidirectional LSTM network, 
    Args:
        layer_name: string, scope
        inputs: 3D tensor, [batch_size, max_time, num_features]
        hidden_units: intergate, The number of units in the LSTM cell
    Returns:
        A tuple (output, output_state) where:
        output:  Containing the forward and backward rnn output Tensor, 
                 3D tensor, [batch_size, max_time, hidden_units * 2]
        output_state: Containing the forward and backward final states of bidirection rnn.
    """
    with tf.variable_scope(layer_name):
        fw_lstm_cell = tf.contrib.rnn.LSTMCell(hidden_units)
        bw_lstm_cell = tf.contrib.rnn.LSTMCell(hidden_units)
        (output_fw, output_bw), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
            fw_lstm_cell, bw_lstm_cell, inputs, dtype=tf.float32
        )
        output = tf.concat((output_fw, output_bw), 2)
        output_state_c = tf.concat((output_state_fw.c, output_state_bw.c), 1)
        output_state_h = tf.concat((output_state_fw.h, output_state_bw.h), 1)
        output_state = tf.contrib.rnn.LSTMStateTuple(output_state_fw, output_state_bw)
        return output, output_state


def attention_based_decoder(encoder_outputs, groundtruth_text, label_map_obj, maximum_iterations=200):
    """A attention based decoder for seq2seq model.
    In the training phase, you also need to feed encoder_outputs and groundtruth_text. 
    In the test phase, you only need to feed encoder_outputs.
    """
    batch_size = combined_static_and_dynamic_shape(encoder_outputs)[0]
    attention_wrapper_class = sync_attention_wrapper.SyncAttentionWrapper

    def decoder(helper, scope, batch_size, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            # attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=512, memory=encoder_outputs)
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    num_units=512, memory=encoder_outputs)
            cell = tf.contrib.rnn.GRUCell(num_units=512)
            attn_cell = attention_wrapper_class(cell, 
                                                attention_mechanism, 
                                                output_attention=False,
                                                attention_layer_size=256)
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                attn_cell, num_classes, reuse=reuse
            )
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=out_cell, 
                helper=helper,
                initial_state=out_cell.zero_state(dtype=tf.float32, batch_size=batch_size)  # batch_size
            )
            outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, 
                output_time_major=False,
                impute_finished=True, 
                maximum_iterations=maximum_iterations
            )
            return outputs[0]
    
    with tf.name_scope('attention_decoder'):
        GO_TOKEN = 0
        END_TOKEN = 1
        UNK_TOKEN = 2
        
        start_tokens = tf.fill([batch_size, 1], tf.constant(GO_TOKEN, tf.int64))
        end_tokens = tf.fill([batch_size, 1], tf.constant(END_TOKEN, tf.int64))

        num_classes = label_map_obj.num_classes + 2
        embedding_fn = functools.partial(tf.one_hot, depth=num_classes)

        text_labels, text_lengths = label_map_obj.text_to_labels(groundtruth_text, pad_value=END_TOKEN, return_lengths=True)

        train_input = tf.concat([start_tokens, text_labels], axis=1)
        train_target = tf.concat([text_labels, end_tokens], axis=1)
        train_input_lengths = text_lengths + 1
        max_num_step = tf.reduce_max(train_input_lengths)

        # Train output
        train_helper = tf.contrib.seq2seq.TrainingHelper(
            embedding_fn(train_input), tf.to_int32(train_input_lengths)
        )
        train_outputs = decoder(train_helper, 'decoder', batch_size)
        train_logits = train_outputs.rnn_output
        train_labels = train_outputs.sample_id
        weights=tf.cast(tf.sequence_mask(train_input_lengths, max_num_step), tf.float32)
        train_loss = tf.contrib.seq2seq.sequence_loss(
            logits=train_outputs.rnn_output, targets=train_target, weights=weights,
            name='train_loss'
        )
        train_probabilities = tf.reduce_max(
            tf.nn.softmax(train_logits, name='probabilities'),
            axis=-1,
        )
        train_output_dict =  {
            'loss': train_loss,
            'logits': train_logits,
            'labels': train_labels,
            'predict_text': label_map_obj.labels_to_text(train_labels),
            'probabilities': train_probabilities,
        }
        tf.summary.scalar(name='train_loss', tensor=train_loss)

        # Eval output
        pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding_fn, 
            start_tokens=tf.fill([batch_size], GO_TOKEN),
            end_token=END_TOKEN
        )
        pred_outputs = decoder(pred_helper, 'decoder', batch_size, reuse=True)
        pred_logits = pred_outputs.rnn_output
        pred_labels = pred_outputs.sample_id
        eval_loss = tf.contrib.seq2seq.sequence_loss(
            logits=pred_outputs.rnn_output, 
            targets=train_target, 
            weights=weights,
            name='eval_loss'
        )
        pred_probabilities = tf.reduce_max(
            tf.nn.softmax(pred_logits, name='probabilities'),
            axis=-1,
        )
        pred_output_dict = {
            'logits': pred_logits,
            "probabilities": pred_probabilities,
            'labels': pred_labels,
            'predict_text': label_map_obj.labels_to_text(pred_labels),
        }
        return train_output_dict, pred_output_dict

