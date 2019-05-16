'''
pass2path - training module
A variant of seq2seq Encoder-Decoder RNN model that learns pairs of
(password, transition path), where given a password and a transition path, a
new password is generated.

This model is based on JayPark's seq2seq model (Python 2): https://github.com/JayParks/tf-seq2seq
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
import math
from word2keypress import Keyboard
from ast import literal_eval
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from pathlib import Path
from edit_distance_backtrace import find_med_backtrace
import random
from collections import OrderedDict
from pass2path_model import Pass2PathModel

# GPU Config -TF allocates memory according to this.
# Note: the GPUs in the list must be the ones you actually use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # "0, 1" for multiple
devices = ["/gpu:2"]
num_devices = len(devices)

'''
Preprocessing:
We need to create look-up tables in order to translate characters to mathematical representation.
'''

# CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3 }
CODES = {'<GO>': 0, '<EOS>': 1, '<UNK>': 2}
# chars = list(string.ascii_letters) + list(string.punctuation) +
# list(string.digits) + [" ", "\t", "\x03", "\x04"]
def create_lookup_tables_from_lst(char_lst):
    '''
    This function creates a dictionary out of a list with the added codes representing padding,
    unknows components, start and end of a sequence
    '''
    # <EOS> acts as <PAD>
    # CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3 }
    # chars = list(string.ascii_letters) + list(string.punctuation) +
    # list(string.digits) + [" ", "\t", "\x03", "\x04"]

    # make a list of unique chars (from https://stackoverflow.com/a/480227)
    seen = set()
    seen_add = seen.add
    vocab = [x for x in char_lst if not (x in seen or seen_add(x))]

    if len(vocab) < 100:
        print(vocab)
        print(char_lst)
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
    int_lst = sorted(list(origin_vocab.values()))
    path_to_int_vocab, int_to_path_vocab = create_lookup_tables_from_lst(int_lst)
    return path_to_int_vocab, int_to_path_vocab

# Globals
kb = Keyboard()
# thisfolder = Path(__file__).absolute().parent
# TRANS_to_IDX = json.load((thisfolder / 'data/trans_dict_2idx.json').open()) #
# py
with open(('./data/trans_dict_2idx.json')) as f:
    TRANS_to_IDX = json.load(f) # jupyter
IDX_to_TRANS = {v: literal_eval(k) for k, v in TRANS_to_IDX.items()}
char_lst = list(string.ascii_letters) + list(string.digits) + list(string.punctuation) + [" ", "\t", "\x03", "\x04"]
source_vocab_to_int, source_int_to_vocab = create_lookup_tables_from_lst(char_lst)
target_vocab_to_int, target_int_to_vocab = build_path_vocab(TRANS_to_IDX)
NUM_SAMPLES_LARGE_DS = 117574260
NUM_SAMPLES_CORNELL_DS = 11667608
tf.app.flags.FLAGS.__flags.clear()

def set_train_flags(cell_type='lstm', hidden_units=128, num_layers=3, embed_size=200, path_to_ds='cleaned_pw_paths_tr.csv',
                   use_residual=False, keep_prob=0.6, edit_distance=2, lr=0.001, batch_size=50, epochs=10,
                   display_step=100, save_freq=11500):
    
    tf.app.flags.FLAGS.__flags.clear()

    # Network parameters
    tf.app.flags.DEFINE_string('cell_type', cell_type , 'RNN cell for encoder and decoder, default: lstm')
    tf.app.flags.DEFINE_integer('hidden_units', hidden_units, 'Number of hidden units in each layer')
    tf.app.flags.DEFINE_integer('depth', num_layers , 'Number of layers in each encoder and decoder')
    tf.app.flags.DEFINE_integer('embedding_size', embed_size, 'Embedding dimensions of encoder and decoder inputs')
    tf.app.flags.DEFINE_integer('num_encoder_symbols', len(source_int_to_vocab), 'Source vocabulary size')
    tf.app.flags.DEFINE_integer('num_decoder_symbols', len(target_int_to_vocab), 'Target vocabulary size')
    tf.app.flags.DEFINE_string('ds_csv_path', path_to_ds, 'path to the csv file containing the dataset')
    tf.app.flags.DEFINE_boolean('use_residual', use_residual, 'Use residual connection between layers')
    tf.app.flags.DEFINE_boolean('use_dropout', True, 'Use dropout in each rnn cell')
    tf.app.flags.DEFINE_float('dropout_rate', 1 - keep_prob, 'Dropout probability for input/output/state units (0.0: no dropout)')
    tf.app.flags.DEFINE_integer('edit_distance', edit_distance, 'Filter out samples with edit distance greater than this number')
    # Training parameters
    tf.app.flags.DEFINE_float('learning_rate', lr, 'Learning rate')
    tf.app.flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')
    tf.app.flags.DEFINE_integer('batch_size', batch_size, 'Batch size')
    tf.app.flags.DEFINE_integer('max_epochs', epochs, 'Maximum # of training epochs')
    # tf.app.flags.DEFINE_integer('max_load_batches', 20, 'Maximum # of batches
    # to load at one time')
    tf.app.flags.DEFINE_integer('max_seq_length', 62, 'Maximum sequence length')
    tf.app.flags.DEFINE_integer('display_freq', display_step, 'Display training status every this iteration')
    tf.app.flags.DEFINE_integer('save_freq', save_freq, 'Save model checkpoint every this iteration')
    tf.app.flags.DEFINE_integer('valid_freq', 3 * display_step, 'Evaluate model every this iteration: valid_data needed')
    tf.app.flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training: (adadelta, adam, rmsprop)')
    tf.app.flags.DEFINE_string('model_dir', './model/', 'Path to save model checkpoints')
    tf.app.flags.DEFINE_string('summary_dir', './model/summary', 'Path to save model summary')
    tf.app.flags.DEFINE_string('model_name', 'pass2path.ckpt', 'File name used for model checkpoints')
    # Ignore Cmmand Line
    tf.app.flags.DEFINE_string('t', '', '')
    tf.app.flags.DEFINE_string('d', '', '')
    tf.app.flags.DEFINE_string('b', '', '')
    tf.app.flags.DEFINE_string('q', '', '')
    tf.app.flags.DEFINE_string('c', '', '')
    tf.app.flags.DEFINE_string('s', '', '')
    tf.app.flags.DEFINE_string('e', '', '')
    tf.app.flags.DEFINE_string('z', '', '')
    tf.app.flags.DEFINE_string('l', '', '')
    tf.app.flags.DEFINE_string('m', '', '')
    tf.app.flags.DEFINE_string('w', '', '')
    tf.app.flags.DEFINE_string('i', '', '')
    tf.app.flags.DEFINE_string('f', '', '')
    tf.app.flags.DEFINE_string('r', '', '')
    tf.app.flags.DEFINE_string('k', '', '')
    # tf.app.flags.DEFINE_boolean('shuffle_each_epoch', True, 'Shuffle training
    # dataset for each epoch')
    # tf.app.flags.DEFINE_boolean('sort_by_length', True, 'Sort pre-fetched
    # minibatches by their target sequence lengths')
    tf.app.flags.DEFINE_boolean('use_fp16', False, 'Use half precision float16 instead of float32 as dtype')

    # Runtime parameters
    tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft placement')
    tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')

FLAGS = tf.app.flags.FLAGS


def create_model(session, FLAGS):

    config = OrderedDict(sorted((dict([(key,val.value) for key,val in FLAGS.__flags.items()])).items()))
    model = Pass2PathModel(config, 'train')

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if (ckpt):
        print("Found a checkpoint state...")
        print(ckpt.model_checkpoint_path)
    if (ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
        print('Reloading model parameters..')
        model.restore(session, ckpt.model_checkpoint_path)
        
    else:
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
        print('Created new model parameters..')
        session.run(tf.global_variables_initializer())
   
    return model

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
    Pad sequences with <PAD> = <EOS> so that each sequence of a batch has the same length
    """
    max_sequence = max([len(seq) for seq in sequence_batch])
    return [seq + [pad_int] * (max_sequence - len(seq)) for seq in sequence_batch]

def preprocess_batch(batch, source_vocab_to_int, target_vocab_to_int):
    #kb = Keyboard()
    pass_batch = [kb.word_to_keyseq(d.decode('utf-8')) for d in batch[0]['pass_1']]
    pass2_batch = [kb.word_to_keyseq(d.decode('utf-8')) for d in batch[0]['pass_2']]
    path_batch = [json.loads(j) for j in [d.decode('utf-8') for d in batch[1]]]
    pass_batch_ids, path_batch_ids = input_to_ids(pass_batch, path_batch,
                                                      source_vocab_to_int, target_vocab_to_int)
    source_pad_int = source_vocab_to_int['<EOS>']
    target_pad_int = target_vocab_to_int['<EOS>']
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
    return (np.array(pad_sources_batch), np.array(pad_targets_batch), 
            np.array(pad_source_lengths), np.array(pad_targets_lengths), pass_batch, pass2_batch)

def train():
    trans_dict_2path = IDX_to_TRANS
    #print(FLAGS.__flags.items())
    f_path = FLAGS.ds_csv_path
    batch_size = FLAGS.batch_size
    edit_distance = FLAGS.edit_distance
    # Initiate TF session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, 
        log_device_placement=FLAGS.log_device_placement, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        print('Loading training data..')
        if ("cleaned_pw_paths_tr.csv" in f_path and edit_distance == 3):
            num_samples = NUM_SAMPLES_LARGE_DS
        elif ("cleaned_pw_paths_tr.l8c3.csv" in f_path and edit_distance == 3):
            num_samples = NUM_SAMPLES_CORNELL_DS
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

        # Create a log writer object
        log_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)
        
        # Create a new model or reload existing checkpoint
        model = create_model(sess, FLAGS)

        step_time, loss = 0.0, 0.0
        words_seen, sents_seen = 0, 0
        start_time = time.time()
        global_start_time = start_time

        # Training loop
        print('Training..')
        for epoch_idx in range(FLAGS.max_epochs):
            if (model.global_epoch_step.eval() >= FLAGS.max_epochs):
                print('Training is already complete.', \
                      'current epoch:{}, max epoch:{}'.format(model.global_epoch_step.eval(), FLAGS.max_epochs))
                break

            for batch_i in range(total_train_batches):    
                # Get a batch
                batch_train = sess.run(next_train_element)
                
        
                (source_batch, target_batch, sources_lengths, targets_lengths, pass1_train_batch,
                 pass2_train_batch) = preprocess_batch(batch_train, 
                                                       source_vocab_to_int,
                                                       target_vocab_to_int)

                # Execute a single training step
                step_loss, summary = model.train(sess, encoder_inputs=source_batch, encoder_inputs_length=sources_lengths, 
                                                 decoder_inputs=target_batch, decoder_inputs_length=targets_lengths)
#                 print(step_loss)

                loss += float(step_loss) / FLAGS.display_freq
                words_seen += float(np.sum(sources_lengths + targets_lengths))
                sents_seen += float(source_batch.shape[0]) # batch_size

                if (model.global_step.eval() % FLAGS.display_freq == 0):
                    # CHANGE TO ACCURACY
                    avg_perplexity = math.exp(float(loss)) if loss < 300 else float("inf")

                    time_elapsed = time.time() - start_time
                    step_time = time_elapsed / FLAGS.display_freq

                    words_per_sec = words_seen / time_elapsed
                    sents_per_sec = sents_seen / time_elapsed

                    print('Epoch ', model.global_epoch_step.eval(), 'Batch: {}/{}'.format(batch_i, total_train_batches) , 'Step ', model.global_step.eval(), \
                          'Step-time {0:.2f}'.format(step_time), 'Total-time: {0:.2f}'.format(time.time() - global_start_time), \
                          '{0:.2f} passwords/s'.format(sents_per_sec), '{0:.2f} tokens/s'.format(words_per_sec), \
                            'Loss: {0:.3f}'.format(loss))

                    loss = 0
                    words_seen = 0
                    sents_seen = 0
                    start_time = time.time()
                    
                    # Record training summary for the current batch
                    log_writer.add_summary(summary, model.global_step.eval())

                # Execute a validation step
                if (model.global_step.eval() % FLAGS.valid_freq == 0):
                    print('Validation step')
                    batch_valid = sess.run(next_valid_element)
                    (valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths,
                        pass1_valid_batch, pass2_valid_batch) = preprocess_batch(batch_valid, 
                                                                                 source_vocab_to_int,
                                                                                 target_vocab_to_int)
                    valid_loss = model.eval(sess, encoder_inputs=valid_sources_batch, encoder_inputs_length=valid_sources_lengths,
                                               decoder_inputs=valid_targets_batch, decoder_inputs_length=valid_targets_lengths)
                    print('Valid Loss: {0:.2f}'.format(valid_loss[0]))

                # Save the model checkpoint
                if (model.global_step.eval() % FLAGS.save_freq == 0):
                    print('Saving the model..')
                    checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                    model.save(sess, checkpoint_path, global_step=model.global_step)
                    json.dump(model.config,
                              open('%s-%d.json' % (checkpoint_path, model.global_step.eval()), 'w'),
                              indent=2)
            # Increase the epoch index of the model
            model.global_epoch_step_op.eval()
            print('Epoch {0:} DONE'.format(model.global_epoch_step.eval()))
        
        print('Saving the last model..')
        checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
        model.save(sess, checkpoint_path, global_step=model.global_step)
        json.dump(model.config,
                  open('%s-%d.json' % (checkpoint_path, model.global_step.eval()), 'w'),
                  indent=2)
        total_time = time.time() - global_start_time
        
    print('Training Terminated, Total time: {} seconds'.format(total_time))