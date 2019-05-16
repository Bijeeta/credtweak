
'''
pass2path
A variant of seq2seq Encoder-Decoder RNN model that learns pairs of
(password, transition path), where given a password and a transition path, a
new password is generated.
'''

# imports
import numpy as np
import tensorflow as tf
import os
import pickle
import copy
import string
import json
import csv
import time
from word2keypress import Keyboard
from ast import literal_eval
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from pathlib import Path
from edit_distance_backtrace import find_med_backtrace

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2" # "0, 1" for multiple
devices = ["/gpu:0", "/gpu:1", "/gpu:2"]
num_devices = len(devices)

'''
Preprocessing:
We need to create look-up tables in order to translate characters to mathematical representation.
'''

# CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3 }
# chars = list(string.ascii_letters) + list(string.punctuation) +
# list(string.digits) + [" ", "\t", "\x03", "\x04"]
def create_lookup_tables_from_lst(char_lst):
    '''
    This function creates a dictionary out of a list with the added codes representing padding,
    unknows components, start and end of a sequence
    '''
    CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3 }
    # chars = list(string.ascii_letters) + list(string.punctuation) +
    # list(string.digits) + [" ", "\t", "\x03", "\x04"]
    # make a list of unique chars
    vocab = set(char_lst)

    # (1)
    # starts with the special tokens
    vocab_to_int = copy.copy(CODES)

    # the index (v_i) will starts from 4 (the 2nd arg in enumerate() specifies
    # the starting index)
    # since vocab_to_int already contains special tokens
    for v_i, v in enumerate(vocab, len(CODES)):
        vocab_to_int[str(v)] = v_i # opposite would be int()
    int_to_vocab = {v_i: v for v, v_i in vocab_to_int.items()}

    return vocab_to_int, int_to_vocab

def build_path_vocab(origin_vocab):
    '''
    This functions combines the path dictionary with the added codes.
    '''
    int_lst = origin_vocab.values()
    path_to_int_vocab, int_to_path_vocab = create_lookup_tables_from_lst(int_lst)
    return path_to_int_vocab, int_to_path_vocab

# Globals
kb = Keyboard()
thisfolder = Path(__file__).absolute().parent
TRANS_to_IDX = json.load((thisfolder / 'data/trans_dict_2idx.json').open()) # py
#with open(('./data/trans_dict_2idx.json')) as f:
#    TRANS_to_IDX = json.load(f) # jupyter
IDX_to_TRANS = {v: literal_eval(k) for k, v in TRANS_to_IDX.items()}
char_lst = list(string.ascii_letters) + list(string.digits) + list(string.punctuation) + [" ", "\t", "\x03", "\x04"]
source_vocab_to_int, source_int_to_vocab = create_lookup_tables_from_lst(char_lst)
target_vocab_to_int, target_int_to_vocab = build_path_vocab(TRANS_to_IDX)
NUM_SAMPLES_LARGE_DS = 117574260

def input_to_ids(source_pws_lst, target_paths_lst, source_vocab_to_int, target_vocab_to_int):
    """
    1st, 2nd args: lists passwords and paths to be converted
    3rd, 4th args: lookup tables for 1st and 2nd args respectively   
    return: A tuple of lists (source_id_pass, target_id_path) converted
    Will use for mini-batches of (passwords, paths)
    """
    # empty list of converted passwords and paths
    source_pass_id = []
    target_path_id = []
    
    max_source_pass_length = max([len(password) for password in source_pws_lst])
    max_target_path_length = max([len(path) for path in target_paths_lst])
    
    # iterating through each password (# of passwords & paths is the same)
    for i in range(len(source_pws_lst)):
        # extract password, one by one
        source_password = source_pws_lst[i]
#         print(target_paths_lst)
        target_path = target_paths_lst[i]
        
        # make a list of tokens/words (extraction) from the chosen password
        source_tokens = list(source_password)
        target_tokens = [str(t) for t in target_path]
        
        # empty list of converted words to index in the chosen password
        source_token_id = []
        target_token_id = []
        
        for index, token in enumerate(source_tokens):
            if (token != ""):
                if (source_vocab_to_int.get(token)):
                    source_token_id.append(source_vocab_to_int[token])
        
        for index, token in enumerate(target_tokens):
            if (token != ""):
                if (target_vocab_to_int.get(token)):
                    target_token_id.append(target_vocab_to_int[token])
                
        # put <EOS> token at the end of the chosen target sentence
        # this token suggests when to stop creating a sequence
        target_token_id.append(target_vocab_to_int['<EOS>'])
            
        # add each converted sentences in the final list
        source_pass_id.append(source_token_id)
        target_path_id.append(target_token_id)
    
    return source_pass_id, target_path_id


def enc_dec_model_inputs():
    '''
    A function that creates and returns parameters (TF placeholders) related to building model. 
    - inputs placeholder will be fed with passwords data, and its shape is `[None, None]`.
        The first `None` means the batch size, and the batch size is unknown since the user can set it. 
        The second `None` means the lengths of passwords. 
        The maximum length of password is different from batch to batch, so it cannot be set with the exact number. 
          - An alternative is to set the lengths of every password to the maximum length across all passwords in every batch.
              No matter which method you choose, you need to add special character, `<PAD>` in empty positions. 
              However, with the alternative option, there could be unnecessarily more `<PAD>` characters.
    - targets placeholder is similar to inputs placeholder except that it will be fed with transition paths data.
    - target_sequence_length placeholder represents the lengths of each path, so the shape is `None`,
        a column tensor, which is the same number to the batch size. 
        This particular value is required as an argument of TrainerHelper to build decoder model for training. 
    - max_target_len gets the maximum value out of lengths of all the target paths(sequences)
    '''
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets') 
    
    target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
    max_target_len = tf.reduce_max(target_sequence_length)    
    
    return inputs, targets, target_sequence_length, max_target_len


def hyperparam_inputs():
    '''
    A function that creates and returns parameters (TF placeholders) related to hyper-parameters of the model, 
    which are tunable.
    - lr_rate is learning rate
    - keep_prob is the keep probability for Dropouts (1 for testing, (0,1) for training)
    '''
    lr_rate = tf.placeholder(tf.float32, name='lr_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return lr_rate, keep_prob


def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for encoding by adding <GO> token
    return: Preprocessed target data
    """
    #'<GO>' id
    go_id = target_vocab_to_int['<GO>']
    '''
    extracts a slice of size (end-begin)/stride from the given input_ tensor.
    Starting at the location specified by begin the slice continues by adding stride to the index until 
    all dimensions are not less than end. 
    '''
    after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1]) # input_, begin, end, strides
    
    after_concat = tf.concat([tf.fill([batch_size, 1], go_id), after_slice], 1) # column tensor of "go_id" >> after_slice
    
    return after_concat

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, 
                   source_vocab_size, 
                   encoding_embedding_size):
    """
    This function creates the encoding layer of the network, by embedding the inputs and stacking RNN cells.
    returns: tuple (RNN output, RNN state)
    """
    embed = tf.contrib.layers.embed_sequence(rnn_inputs, 
                                             vocab_size=source_vocab_size, 
                                             embed_dim=encoding_embedding_size)

     # Here we can try to distribute the RNN layers across different GPUs
    '''
    devices = ["/gpu:0", "/gpu:1", "/gpu:2"]
    cells = [DeviceCellWrapper(dev, tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size), keep_prob)) for 
            dev in devices]
    stacked_cells = tf.contrib.rnn.MultiRNNCell(cells)
    '''
    max_layers_per_gpu = (num_layers + 1) // num_devices
    layers_used = 0
    cells = []
    for dev in devices:
        assigned = 0
        while (assigned < max_layers_per_gpu and layers_used < num_layers):
            cells.append(tf.contrib.rnn.DeviceWrapper(tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size), keep_prob), dev))
            assigned += 1
            layers_used += 1
    stacked_cells = tf.contrib.rnn.MultiRNNCell(cells)
    #stacked_cells =
    #tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size),
    #keep_prob) for _ in range(num_layers)])
    
    outputs, state = tf.nn.dynamic_rnn(cell=stacked_cells, 
                                       inputs=embed, 
                                       dtype=tf.float32)
    return outputs, state


def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, 
                         target_sequence_length, max_summary_length, 
                         output_layer, keep_prob):
    """
    Creates a training process for the decoding layer 
    returns: BasicDecoderOutput containing training logits and sample_id
    """
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, 
                                             output_keep_prob=keep_prob)
    
    # only for the input layer
    helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input, 
                                               sequence_length=target_sequence_length)
    
    decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell, 
                                              helper=helper, 
                                              initial_state=encoder_state, 
                                              output_layer=output_layer)
    # output_layer: optional layer to apply to the RNN output prior to storing
    # the result or sampling.

    # unrolling the decoder layer
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                      impute_finished=True, 
                                                      maximum_iterations=max_summary_length)
    # impute_finished: If True, then states for batch entries which are marked
    # as finished get copied through
    # and the corresponding outputs get zeroed out.
    # This causes some slowdown at each time step, but ensures that the final
    # state and outputs have the correct values
    # and that backprop ignores time steps that were marked as finished.
    return outputs

def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):
    """
    Create a inference process in decoding layer 
    return: BasicDecoderOutput containing inference logits and sample_id
    """
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, 
                                             output_keep_prob=keep_prob)
    
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=dec_embeddings, 
                                                      start_tokens=tf.fill([batch_size], start_of_sequence_id), 
                                                      end_token=end_of_sequence_id)
    # GreedyEmbeddingHelper uses the argmax of the output (treated as logits)
    # and passes the result through an embedding layer to get the next input.
    
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, 
                                              helper, 
                                              encoder_state, 
                                              output_layer)
    
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                      impute_finished=True, 
                                                      maximum_iterations=max_target_sequence_length)
    return outputs

def decoding_layer_infer_beam(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob, beam_width=10):
    """
    Create a inference process in decoding layer using Beam Search
    return: BasicDecoderOutput containing inference logits and sample_id
    """
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, 
                                             output_keep_prob=keep_prob)
    # Consider changing 128 to batch_size
    
    decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell = dec_cell,
                                                   embedding = dec_embeddings,
                                                   start_tokens = tf.tile(tf.constant([start_of_sequence_id], dtype=tf.int32), [batch_size]),
                                                   end_token = end_of_sequence_id,
                                                   initial_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width),
                                                   beam_width = beam_width,
                                                   output_layer = output_layer,
                                                   length_penalty_weight = 0.0)
    outputs, _, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                      impute_finished=False, 
                                                      maximum_iterations=max_target_sequence_length)
    return outputs, final_sequence_lengths


def decoding_layer(dec_input, encoder_state,
                   target_sequence_length, max_target_sequence_length,
                   rnn_size,
                   num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, decoding_embedding_size):
    """
    Create decoding layer by assembling the training and inference decoders.
    return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    target_vocab_size = len(target_vocab_to_int)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

     # Here we can try to distribute the RNN layers across different GPUs
    '''
    devices = ["/gpu:0", "/gpu:1", "/gpu:2"]
    cells = [DeviceCellWrapper(dev, tf.contrib.rnn.LSTMCell(rnn_size)) for 
            dev in devices]
    cells = tf.contrib.rnn.MultiRNNCell(cells)
    '''
    
    cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(num_layers)])
    
    with tf.variable_scope("decode"):
        output_layer = tf.layers.Dense(target_vocab_size)
        train_output = decoding_layer_train(encoder_state, 
                                            cells, 
                                            dec_embed_input, 
                                            target_sequence_length, 
                                            max_target_sequence_length, 
                                            output_layer, 
                                            keep_prob)

    with tf.variable_scope("decode", reuse=True):
        infer_output = decoding_layer_infer(encoder_state, 
                                            cells, 
                                            dec_embeddings, 
                                            target_vocab_to_int['<GO>'], 
                                            target_vocab_to_int['<EOS>'], 
                                            max_target_sequence_length, 
                                            target_vocab_size, 
                                            output_layer,
                                            batch_size,
                                            keep_prob)

    return (train_output, infer_output)


def decoding_layer_beam(dec_input, encoder_state,
                   target_sequence_length, max_target_sequence_length,
                   rnn_size,
                   num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, decoding_embedding_size, beam_width=10):
    """
    Create decoding layer by assembling the training and inference beam search decoders.
    return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    target_vocab_size = len(target_vocab_to_int)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

     # Here we can try to distribute the RNN layers across different GPUs
    '''
    devices = ["/gpu:0", "/gpu:1", "/gpu:2"]
    cells = [DeviceCellWrapper(dev, tf.contrib.rnn.LSTMCell(rnn_size)) for 
            dev in devices]
    cells = tf.contrib.rnn.MultiRNNCell(cells)
    '''
    max_layers_per_gpu = (num_layers + 1) // num_devices
    layers_used = 0
    cells = []
    for dev in devices:
        assigned = 0
        while (assigned < max_layers_per_gpu and layers_used < num_layers):
            cells.append(tf.contrib.rnn.DeviceWrapper(tf.contrib.rnn.LSTMCell(rnn_size), dev))
            assigned += 1
            layers_used += 1
    cells = tf.contrib.rnn.MultiRNNCell(cells)
    #cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(rnn_size) for
    #_ in range(num_layers)])
    
    with tf.variable_scope("decode"):
        output_layer = tf.layers.Dense(target_vocab_size)
        train_output = decoding_layer_train(encoder_state, 
                                            cells, 
                                            dec_embed_input, 
                                            target_sequence_length, 
                                            max_target_sequence_length, 
                                            output_layer, 
                                            keep_prob)

    with tf.variable_scope("decode", reuse=True):
        infer_output, infer_lengths = decoding_layer_infer_beam(encoder_state, 
                                                                cells, 
                                                                dec_embeddings, 
                                                                target_vocab_to_int['<GO>'], 
                                                                target_vocab_to_int['<EOS>'], 
                                                                max_target_sequence_length, 
                                                                target_vocab_size, 
                                                                output_layer,
                                                                batch_size,
                                                                keep_prob,
                                                                beam_width)

    return (train_output, infer_output, infer_lengths)


def pass2path_model(input_data, target_data, keep_prob, batch_size,
                  target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int):
    """
    Assemble the Password-to-Path model
    return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    enc_outputs, enc_states = encoding_layer(input_data, 
                                             rnn_size, 
                                             num_layers, 
                                             keep_prob, 
                                             source_vocab_size, 
                                             enc_embedding_size)
    
    dec_input = process_decoder_input(target_data, 
                                      target_vocab_to_int, 
                                      batch_size)
    
    train_output, infer_output = decoding_layer(dec_input, 
                                                enc_states, 
                                                target_sequence_length, 
                                                max_target_sentence_length,
                                                rnn_size,
                                                num_layers,
                                                target_vocab_to_int,
                                                target_vocab_size,
                                                batch_size,
                                                keep_prob,
                                                dec_embedding_size)
    
    return train_output, infer_output


def pass2path_model_beam(input_data, target_data, keep_prob, batch_size,
                  target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int, beam_width=10):
    """
    Assemble the Password-to-Path model
    return: Tuple of (Training BasicDecoderOutput, Inference FinalBeamSearchDecoderOutput)
    """
    enc_outputs, enc_states = encoding_layer(input_data, 
                                             rnn_size, 
                                             num_layers, 
                                             keep_prob, 
                                             source_vocab_size, 
                                             enc_embedding_size)
    
    dec_input = process_decoder_input(target_data, 
                                      target_vocab_to_int, 
                                      batch_size)
    
    train_output, infer_output, infer_lengths = decoding_layer_beam(dec_input, 
                                                    enc_states, 
                                                    target_sequence_length, 
                                                    max_target_sentence_length,
                                                    rnn_size,
                                                    num_layers,
                                                    target_vocab_to_int,
                                                    target_vocab_size,
                                                    batch_size,
                                                    keep_prob,
                                                    dec_embedding_size, beam_width)
    
    return train_output, infer_output, infer_lengths

def _parse_line(line):
    '''
    This helper function parses a line from the CSV dataset to a dictionary and a target.
    '''
    # Metadata describing the text columns
    
    COLUMNS = ['pass_1', 'pass_2', 'path']
    FIELD_DEFAULTS = [[""], [""], [""]]
    
    # Decode the line into its fields
    fields = tf.decode_csv(line, FIELD_DEFAULTS)

    # Pack the result into a dictionary
    features = dict(zip(COLUMNS,fields))

    # Separate the label from the features
    path = features.pop('path')

    return features, path

def csv_input_fn(csv_path, batch_size, skip_lines):
    '''
    This function builds a batch from the dataset in the CSV file.
    '''
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(skip_lines)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

def csv_input_fn_filter(csv_path, batch_size, skip_lines, edit_distance, limit=-1):
    '''
    This function builds a batch from the dataset in the CSV file, and filters out
    samples that do not satisfy the edit distance condition.
    '''
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(skip_lines).take(limit)

    # Parse each line.
    dataset = dataset.map(_parse_line)
    
    # Filter
    dataset = dataset.filter(lambda feat, path : tf.less_equal(tf.size((tf.string_split(tf.reshape(path, shape=(1,)), " []"))), edit_distance))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

def pad_sequence_batch(sequence_batch, pad_int):
    """
    Pad sequences with <PAD> so that each sequence of a batch has the same length
    """
    max_sequence = max([len(seq) for seq in sequence_batch])
    return [seq + [pad_int] * (max_sequence - len(seq)) for seq in sequence_batch]


def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    """
    Batch targets, sources, and the lengths of their sentences together
    """
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths
        
def get_batches_from_file(sess, filename, batch_size, source_vocab_to_int, target_vocab_to_int):
    """
    Batch targets, sources, and the lengths of their sequences together, where
    the dataset originates from a CSV file.
    """
    num_samples = 0
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        num_samples = sum(1 for row in csv_reader)
    print("# Samples: {}".format(num_samples))
    dataset = csv_input_fn(filename, batch_size, skip_lines = 0)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    sess.run(iterator.initializer)
    total_batches = num_samples // batch_size
    print("Total batches: {}".format(total_batches))
    
    for batch_i in range(0, total_batches):
#         start_i = batch_i * batch_size
        batch = sess.run(next_element)
        pass_batch = [d.decode('utf-8') for d in batch[0]['pass_1']]
        path_batch = [json.loads(j) for j in [d.decode('utf-8') for d in batch[1]]]
#         if (len(pass_batch) != len(path_batch)):
#             print(pass_batch)
#             print(path_batch)
        
        pass_batch_ids, path_batch_ids = input_to_ids(pass_batch, path_batch,
                                                      source_vocab_to_int, target_vocab_to_int)
        
        source_pad_int = source_vocab_to_int['<PAD>']
        target_pad_int = target_vocab_to_int['<PAD>']

        # Pad
        pad_sources_batch = np.array(pad_sequence_batch(pass_batch_ids, source_pad_int))
        pad_targets_batch = np.array(pad_sequence_batch(path_batch_ids, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths

def get_batches_from_file_filter(sess, filename, batch_size, source_vocab_to_int, target_vocab_to_int, edit_distance=3):
    """
    Batch targets, sources, and the lengths of their sequences together, where
    the dataset originates from a CSV file. It filters out samples that do not satisfy the
    edit distance condition.
    """
    num_samples = 0
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        num_samples = sum(1 for row in csv_reader)
#     print("# Samples: {}".format(num_samples))
    dataset = csv_input_fn_filter(filename, batch_size, skip_lines = 0, edit_distance = edit_distance)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    sess.run(iterator.initializer)
    total_batches = num_samples // batch_size
#     print("Total batches: {}".format(total_batches))
    
    for batch_i in range(0, total_batches):
#         start_i = batch_i * batch_size
        batch = sess.run(next_element)
        pass_batch = [d.decode('utf-8') for d in batch[0]['pass_1']]
        path_batch = [json.loads(j) for j in [d.decode('utf-8') for d in batch[1]]]
#         if (len(pass_batch) != len(path_batch)):
#             print(pass_batch)
#             print(path_batch)
        
        pass_batch_ids, path_batch_ids = input_to_ids(pass_batch, path_batch,
                                                      source_vocab_to_int, target_vocab_to_int)
        
        source_pad_int = source_vocab_to_int['<PAD>']
        target_pad_int = target_vocab_to_int['<PAD>']

        # Pad
        pad_sources_batch = np.array(pad_sequence_batch(pass_batch_ids, source_pad_int))
        pad_targets_batch = np.array(pad_sequence_batch(path_batch_ids, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths
        
def preprocess_batch(batch, source_vocab_to_int, target_vocab_to_int):
    #kb = Keyboard()
    pass_batch = [kb.word_to_keyseq(d.decode('utf-8')) for d in batch[0]['pass_1']]
    pass2_batch = [kb.word_to_keyseq(d.decode('utf-8')) for d in batch[0]['pass_2']]
    path_batch = [json.loads(j) for j in [d.decode('utf-8') for d in batch[1]]]
    pass_batch_ids, path_batch_ids = input_to_ids(pass_batch, path_batch,
                                                      source_vocab_to_int, target_vocab_to_int)
    source_pad_int = source_vocab_to_int['<PAD>']
    target_pad_int = target_vocab_to_int['<PAD>']
    # Pad
    pad_sources_batch = np.array(pad_sequence_batch(pass_batch_ids, source_pad_int))
    pad_targets_batch = np.array(pad_sequence_batch(path_batch_ids, target_pad_int))

    # Need the lengths for the _lengths parameters
    pad_targets_lengths = []
    for target in pad_targets_batch:
        pad_targets_lengths.append(len(target))

    pad_source_lengths = []
    for source in pad_sources_batch:
        pad_source_lengths.append(len(source))
    return (pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths, pass_batch, pass2_batch)

def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(logits,
            [(0,0),(0,max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))
def get_accuracy_beam(target, logits, pad_int=0):
    """
    Calculate accuracy of BeamSearch output as follows: if one of the K outputs is correct,
    it counts as a correct prediction (positive contribution to the accuracy).
    "logits" are now of the shape: [batches, max_seq, K]
    """
#     print(target.shape)
#     print(logits.shape)
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant', constant_values = pad_int)
    if max_seq - logits.shape[1]:
        logits = np.pad(logits,
            [(0,0), (0,max_seq - logits.shape[1]), (0,0)],
            'constant', constant_values = pad_int)
#     print(target)
#     print(logits)
    acc = [np.mean(np.equal(target, np.reshape(logits[:,:,i], target.shape))) for i in range(logits.shape[2])]
#     if (max(acc) > 0.6):
#         print(target)
#         print(np.reshape(np.transpose(logits[:,:,acc.index(max(acc))]),
#         target.shape))
    return max(acc)


def save_params(params):
    with open('params.p', 'wb') as out_file:
        pickle.dump(params, out_file)


def load_params():
    with open('params.p', mode='rb') as in_file:
        return pickle.load(in_file)


def pass_to_seq(password, vocab_to_int):
    '''
    This function transforms password to sequence of integers, in order to make the tested password
    applicable to the network's input.
    '''
    results = []
    for c in list(password):
        if c in vocab_to_int:
            results.append(vocab_to_int[c])
        else:
            results.append(vocab_to_int['<UNK>'])
            
    return results

def path2word_kb_feasible(word, path, print_path=False):
    '''
    This function decodes the word in which the given path transitions the input word into.
    This is the KeyPress version, which handles the keyboard representations.
    If one of the parts components is not feasible (e.g removing a char from out of range index), it skips it
    Input parameters: original word, transition path
    Output: decoded word
    '''
    #kb = Keyboard()
    word = kb.word_to_keyseq(word)
    if not path:
        return kb.keyseq_to_word(word)
    #path = [literal_eval(p) for p in path]
    if (print_path):
        print(path)
#     print(type(path))
#     print(word)
    final_word = []
    word_len = len(word)
    path_len = len(path)
    i = 0
    j = 0
    while (i < word_len or j < path_len):
        if ((j < path_len and path[j][2] == i) or (i >= word_len and path[j][2] >= i)):
            if (path[j][0] == "s"):
                # substitute
                final_word.append(path[j][1])
                i += 1
                j += 1
            elif (path[j][0] == "d"):
                # delete
                i += 1
                j += 1
            else:
                # "i", insert
                final_word.append(path[j][1])
                j += 1
        else:
            if (i < word_len):
                final_word.append(word[i])
                i += 1
            if (j < path_len and i > path[j][2]):
                j += 1
    return (kb.keyseq_to_word(''.join(final_word)))

def path_to_pass(password, idx_path, trans_dict):
    '''
    This function decodes the password in which the given path transitions the input password into.
    Input parameters: original password, transition path, transition dictionary
    Output: decoded password
    '''
#     print(idx_path)
    str_path = []
    for i in idx_path:
        if (i != '<PAD>' and i != '<UNK>') and trans_dict.get(int(i)):
            str_path.append(trans_dict[int(i)])
#         else:
#             print("could not find " + str(i) + " in dictionary")
#     print(str_path)
    output_pass = path2word_kb_feasible(password, str_path)
    return output_pass

def get_accuracy_beam_decode(logits, pass1_batch, pass2_batch, target_int_to_vocab, trans_dict_2path, bias=False):
    """
    Calculate accuracy of BeamSearch output as follows: if one of the K outputs is correct,
    it counts as a correct prediction (positive contribution to the accuracy).
    "logits" are now of the shape: [batches, max_seq, K]
    """
#     print(logits.shape)
    beam_width = logits.shape[2]
    match_vec = np.zeros((1, logits.shape[0]), dtype=bool)
    for i in range(beam_width):
        decode_pred_batch = logits[:,:,i]
        for k in range(logits.shape[0]):
            decode_pred = logits[k,:,i]
#             print(decode_pred)
            path_idx_pred = [target_int_to_vocab[j] for j in decode_pred]
#             print(path_idx_pred)
            if ('<EOS>' in path_idx_pred):
                path_idx_pred = path_idx_pred[:path_idx_pred.index('<EOS>')]
            prediction = path_to_pass(pass1_batch[k], path_idx_pred, trans_dict_2path)
#             print("pred: {}, targ: {}".format(prediction, pass2_batch[k]))
            if (prediction == kb.keyseq_to_word(pass2_batch[k])):
#                 print(prediction)
                match_vec[0, k] = True
            if (bias and (not match_vec[0, k]) and (k == logits.shape[0] - 1)):
                if (kb.keyseq_to_word(pass2_batch[k]) == kb.keyseq_to_word(pass1_batch[k])):
                    match_vec[0, k] = True       
    acc = np.mean(match_vec)
#     if (max(acc) > 0.6):
#         print(target)
#         print(np.reshape(np.transpose(logits[:,:,acc.index(max(acc))]),
#         target.shape))
    return acc


def train_pass2path(ds_csv_path, display_step=10, epochs=80, batch_size=50, rnn_size=128, num_layers=3, embed_size=200,
                   beam_width=10, edit_distance=3, learning_rate=0.001, keep_probability=0.8):
    # Hyperparameters
    encoding_embedding_size = embed_size
    decoding_embedding_size = embed_size
    
    # Beam Graph
    
    trans_dict_2path = IDX_to_TRANS
    save_path = 'checkpoints/dev'

    train_graph_beam = tf.Graph()
    with train_graph_beam.as_default():
        input_data, targets, target_sequence_length, max_target_sequence_length = enc_dec_model_inputs()
        lr, keep_prob = hyperparam_inputs()

        train_logits, inference_logits, infer_lengths = pass2path_model_beam(tf.reverse(input_data, [-1]),
                                                              targets,
                                                              keep_prob,
                                                              batch_size,
                                                              target_sequence_length,
                                                              max_target_sequence_length,
                                                              len(source_vocab_to_int),
                                                              len(target_vocab_to_int),
                                                              encoding_embedding_size,
                                                              decoding_embedding_size,
                                                              rnn_size,
                                                              num_layers,
                                                              target_vocab_to_int,beam_width)

        training_logits = tf.identity(train_logits.rnn_output, name='logits')
    #     print(inference_logits)
        inference_masks = tf.transpose(tf.sequence_mask(infer_lengths,dtype=tf.int32), perm=[0, 2, 1])
        inference_logits = tf.identity(inference_logits.predicted_ids * inference_masks, name='predictions')

        # https://www.tensorflow.org/api_docs/python/tf/sequence_mask
        # - Returns a mask tensor representing the first N positions of each
        # cell.
        masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

        with tf.name_scope("optimization"):
            # Loss function - weighted softmax cross entropy
            cost = tf.contrib.seq2seq.sequence_loss(training_logits,
                targets,
                masks)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
            
    # Beam version v.2
    # Locations and loads:
    f_path = ds_csv_path
    with tf.Session(graph=train_graph_beam) as sess:
        sess.run(tf.global_variables_initializer())
        if ("cleaned_pw_paths_tr.csv" in f_path):
            num_samples = NUM_SAMPLES_LARGE_DS
        else:
            num_samples = 0
            with open(f_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                num_samples = sum(1 for row in csv_reader if len(json.loads(row[2])) <= edit_distance)

        total_batches = num_samples // batch_size

        print("# Samples: {}".format(num_samples))
        print("Total batches: {}".format(total_batches))

        # Split data to training and validation sets
        num_validation = int(0.2 * num_samples)
        total_valid_batches = num_validation // batch_size
        total_train_batches = total_batches - total_valid_batches

        print("Total validation batches: {}".format(total_valid_batches))
        print("Total training batches: {}".format(total_train_batches))

        valid_dataset = csv_input_fn_filter(f_path, batch_size, skip_lines = 0,
                                            edit_distance = edit_distance, limit = num_validation)
        train_dataset = csv_input_fn_filter(f_path, batch_size, skip_lines = num_validation,
                                            edit_distance = edit_distance, limit = -1)

        valid_iterator = valid_dataset.make_initializable_iterator()
        train_iterator = train_dataset.make_initializable_iterator()

        next_valid_element = valid_iterator.get_next()
        next_train_element = train_iterator.get_next()

        sess.run(valid_iterator.initializer)
        sess.run(train_iterator.initializer)

        start = time.clock()
        ckpt_time = start
        for epoch_i in range(epochs):
            for batch_i in range(total_train_batches):
                batch_train = sess.run(next_train_element)
                batch_valid = sess.run(next_valid_element)
                (source_batch, target_batch, sources_lengths, targets_lengths, pass1_train_batch,
                 pass2_train_batch) = preprocess_batch(batch_train, 
                                                                                                  source_vocab_to_int,
                                                                                                  target_vocab_to_int)

                (valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths,
                 pass1_valid_batch, pass2_valid_batch) = preprocess_batch(batch_valid, 
                    source_vocab_to_int,
                    target_vocab_to_int)

                _, loss = sess.run([train_op, cost],
                    {input_data: source_batch,
                     targets: target_batch,
                     lr: learning_rate,
                     target_sequence_length: targets_lengths,
                     keep_prob: keep_probability})


                if batch_i % display_step == 0 and batch_i > 0:
                    batch_train_logits = sess.run(inference_logits,
                        {input_data: source_batch,
                         target_sequence_length: targets_lengths,
                         keep_prob: 1.0})

                    batch_valid_logits = sess.run(inference_logits,
                        {input_data: valid_sources_batch,
                         target_sequence_length: valid_targets_lengths,
                         keep_prob: 1.0})
    #                 print(batch_train_logits)

                    train_acc = get_accuracy_beam(target_batch, batch_train_logits)
                    train2_acc = get_accuracy_beam_decode(batch_train_logits, pass1_train_batch, pass2_train_batch,
                                                          target_int_to_vocab, trans_dict_2path)
                    valid_acc = get_accuracy_beam(valid_targets_batch, batch_valid_logits)
                    valid2_acc = get_accuracy_beam_decode(batch_valid_logits, pass1_valid_batch, pass2_valid_batch,
                                                          target_int_to_vocab, trans_dict_2path)
                    current_time = time.clock()

                    print('Time: {:>6.1f}/{:>6.1f} Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                          .format(current_time - ckpt_time ,current_time - start, epoch_i, batch_i, total_train_batches,
                                  train2_acc, valid2_acc, loss))
                    # Checkpoint:
                    saver = tf.train.Saver()
                    saver.save(sess, save_path)
                    save_params(save_path)
                    ckpt_time = current_time

        # Save Model
        saver = tf.train.Saver()
        saver.save(sess, save_path)
        print('Model Trained and Saved')
        # Save parameters for checkpoint
        save_params(save_path)
        print('Path to parameters saved')
        print('Total time: {:>6.1f}'.format(time.clock() - start))
        
def predict_pass2path(password, batch_size=50, beam_width=10):
    trans_dict_2path = IDX_to_TRANS
    load_path = load_params()

    original_pass = password

    decoded_origin_pass = pass_to_seq(original_pass, source_vocab_to_int)
#     print(type([decoded_origin_pass]*batch_size))

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)

        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

        decode_logits = sess.run(logits, {input_data: [decoded_origin_pass] * batch_size,
                                             target_sequence_length: [len(decoded_origin_pass) * 2] * batch_size,
                                             keep_prob: 1.0})[0]
    #     print(decode_logits)
        print('Input')
        print('  Password Ids:      {}'.format([i for i in decoded_origin_pass]))
        print('  Password Characters: {}'.format([source_int_to_vocab[i] for i in decoded_origin_pass]))

        decode_logits = np.array(decode_logits)
        print('Predictions')
        for i in range(beam_width):
            decode_pred = decode_logits[:,i]
            path_idx_pred = [target_int_to_vocab[j] for j in decode_pred]
            if ('<EOS>' in path_idx_pred):
                path_idx_pred = path_idx_pred[:path_idx_pred.index('<EOS>')]
            print('--------------')
            print('  Path Ids:      {}'.format([i for i in decode_pred]))
            print('  Path Componenets: {}'.format(" ".join([target_int_to_vocab[i] for i in decode_pred])))
            print('  Decoded Password: {}'.format(path_to_pass(original_pass, path_idx_pred, trans_dict_2path)))
            print('--------------')
            
def predict_list_pass2path(password, batch_size=50, beam_width=10):      

    trans_dict_2path = IDX_to_TRANS
    load_path = load_params()

    original_pass = password

    decoded_origin_pass = pass_to_seq(original_pass, source_vocab_to_int)
    
    
    predictions = []

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)

        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

        decode_logits = sess.run(logits, {input_data: [decoded_origin_pass] * batch_size,
                                             target_sequence_length: [len(decoded_origin_pass) * 2] * batch_size,
                                             keep_prob: 1.0})[0]
        decode_logits = np.array(decode_logits)
        for i in range(beam_width):
            decode_pred = decode_logits[:,i]
            path_idx_pred = [target_int_to_vocab[j] for j in decode_pred]
            if ('<EOS>' in path_idx_pred):
                path_idx_pred = path_idx_pred[:path_idx_pred.index('<EOS>')]
            predictions.append(path_to_pass(original_pass, path_idx_pred, trans_dict_2path))
    return predictions


def test_samples_gen(fpath):
    with open(fpath) as tst_file:
        for line in tst_file:
            p = line.split('\t')
            yield p[0], p[1][:-2]
def test_batches_gen(fpath, batch_size):
    samples_gen = test_samples_gen(fpath)
    curr_batch_size = 0
    source_batch = []
    target_batch = []
    for i, sample in enumerate(samples_gen):
        if (curr_batch_size < batch_size):
            source_batch.append(sample[0])
            target_batch.append(sample[1])
            curr_batch_size += 1
        else:
            yield source_batch, target_batch
            curr_batch_size = 0
            source_batch = []
            target_batch = []
    yield source_batch, target_batch
    
def preprocess_batch_prediction(batch, source_vocab_to_int):
    pass_batch = [kb.word_to_keyseq(d) for d in batch[0]]
    pass2_batch = [kb.word_to_keyseq(d) for d in batch[1]]

    pass_batch_ids = [pass_to_seq(p, source_vocab_to_int) for p in pass_batch]
    source_pad_int = source_vocab_to_int['<PAD>']
    # Pad
    pad_sources_batch = np.array(pad_sequence_batch(pass_batch_ids, source_pad_int))
    return (pad_sources_batch, pass_batch, pass2_batch)

def path2pass_run_testset(fpath, batch_size=50):
    '''
    This function loads a pre-trained pass2path model and outputs the accuracy of the model
    on the input test set
    '''

    trans_dict_2path = IDX_to_TRANS
    load_path = load_params()
    
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)

        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        
        acc = 0
        b_gen = test_batches_gen(fpath, batch_size)
        total_samples = 0
        
        for batch_i, batch in enumerate(b_gen):
            total_samples += len(batch[0])
#             print(max_seq_len)
            if len(batch[0]) < batch_size:
#                 print(s_batch)
                valid_size = len(batch[0])
                # this is the last batch
                while (len(batch[0]) < batch_size):
                    batch[0].append('0')
                    batch[1].append('0')
#                 print(batch)
                pad_encoded_s_batch, s_batch, t_batch = preprocess_batch_prediction(batch, source_vocab_to_int)
                max_seq_len = max([len(s) for s in pad_encoded_s_batch])
                decode_logits = sess.run(logits, {input_data: pad_encoded_s_batch,
                                                  target_sequence_length: [max_seq_len * 2] * batch_size,
                                                  keep_prob: 1.0})
                decode_logits = decode_logits[:valid_size,:,:]
                b_acc = get_accuracy_beam_decode(decode_logits, s_batch[:valid_size], t_batch[:valid_size],
                                                 target_int_to_vocab, trans_dict_2path, bias=True)
                acc += valid_size * b_acc
            else:
                pad_encoded_s_batch, s_batch, t_batch = preprocess_batch_prediction(batch, source_vocab_to_int)
                max_seq_len = max([len(s) for s in pad_encoded_s_batch])
                decode_logits = sess.run(logits, {input_data: pad_encoded_s_batch,
                                                  target_sequence_length: [max_seq_len * 2] * batch_size,
                                                  keep_prob: 1.0})
                b_acc = get_accuracy_beam_decode(decode_logits, s_batch, t_batch,
                                                 target_int_to_vocab, trans_dict_2path, bias=True)
                acc += batch_size * b_acc
        acc = acc / total_samples
        return acc

def filter_edit_distance_tst(fpath, ed=3):
    samples_match = 0
    with open(fpath) as tst_file:
        with open('edited_tst_file.txt', 'w') as efile:
            for line in tst_file:
                p = line.split('\t')
                edit_distance, _ = find_med_backtrace(p[0], p[1][:-2])
                if (edit_distance <= ed):
                    samples_match += 1
                    efile.write(line)
    print("Found {} samples that matched to edit distance: {}, file saved as 'edited_tst_file.txt' ".format(samples_match, ed))




