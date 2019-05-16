'''
pass2path - decode module
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1" for multiple
devices = ["/gpu:1"]
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
    # print("TRANS_t_IDX:{}".format(int_lst[:10]))
    path_to_int_vocab, int_to_path_vocab = create_lookup_tables_from_lst(int_lst)
    return path_to_int_vocab, int_to_path_vocab

# Globals:
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
latest_ckpt = tf.train.latest_checkpoint('./model/')
#latest_ckpt = './model/pass2path.ckpt-1667500'
def set_decode_flags(goal='decode', checkpoint=-1, beam_width=10, decode_batch_size=80, tst_file_path='test_new-opw_5000.txt'):
    tf.app.flags.FLAGS.__flags.clear()

    tf.app.flags.DEFINE_integer('beam_width', beam_width, 'Beam width used in beamsearch')
    if (goal == 'decode'):
        batch_size = decode_batch_size
    else:
        batch_size = 1
    tf.app.flags.DEFINE_integer('decode_batch_size', batch_size, 'Batch size used for decoding')
    tf.app.flags.DEFINE_integer('max_decode_step', 500, 'Maximum time step limit to decode')
    # tf.app.flags.DEFINE_boolean('write_n_best', False, 'Write n-best list
    # (n=beam_width)')
    if (checkpoint == -1):
        ckpt = latest_ckpt
    else:
        ckpt = './model/pass2path.ckpt-' + str(checkpoint)
    tf.app.flags.DEFINE_string('model_path',ckpt, 'Path to a specific model checkpoint.')

    tf.app.flags.DEFINE_string('decode_input', tst_file_path, 'Decoding input path')
    tf.app.flags.DEFINE_string('decode_output', 'data/pass2path_' + str(checkpoint) + '_' + (tst_file_path.split('/'))[-1][:-4] + '.predictions', 'Decoding output path')

    # Runtime parameters
    tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft placement')
    tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')

    # Ignore Cmmand Line
    tf.app.flags.DEFINE_string('x', '', '')
    tf.app.flags.DEFINE_string('d', '', '')
    tf.app.flags.DEFINE_string('p', '', '')
    tf.app.flags.DEFINE_string('o', '', '')
    tf.app.flags.DEFINE_string('w', '', '')
    tf.app.flags.DEFINE_string('a', '', '')
    tf.app.flags.DEFINE_string('b', '', '')
    tf.app.flags.DEFINE_string('j', '', '')
    tf.app.flags.DEFINE_string('u', '', '')

FLAGS = tf.app.flags.FLAGS

def load_config(FLAGS):
    
    config = json.load(open('%s.json' % FLAGS.model_path, 'r'))
    for key, value in FLAGS.__flags.items():
        config[key] = value.value

    return config


def load_model(session, config):
    
    model = Pass2PathModel(config, 'decode')
    if tf.train.checkpoint_exists(FLAGS.model_path):
        print('Reloading model parameters..')
        model.restore(session, FLAGS.model_path)
    else:
        raise ValueError('No such file:[{}]'.format(FLAGS.model_path))
    return model


def pass_to_seq(password, vocab_to_int):
    '''
    This function transforms password to sequence of integers, in order to make the tested password
    applicable to the network's input.
    '''
    results = []
    #print(">>>", password)
    for c in password:
        results.append(vocab_to_int.get(c, 2)) # <UNK> is 2
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
            #print(k)
            decode_pred = logits[k,:,i]
#             print(decode_pred)
            path_idx_pred = [target_int_to_vocab[j] for j in decode_pred]
#             print(path_idx_pred)
            if ('<EOS>' in path_idx_pred):
                path_idx_pred = path_idx_pred[:path_idx_pred.index('<EOS>')]
            prediction = path_to_pass(pass1_batch[k], path_idx_pred, trans_dict_2path)
            #print("pred: {}, targ: {}".format(prediction, pass2_batch[k]))
            if (prediction == kb.keyseq_to_word(pass2_batch[k])):
#                 if not match_vec[0, k]:
#                     print(prediction)
                match_vec[0, k] = True
            if (bias and (not match_vec[0, k]) and (i == beam_width - 1)):
                #print(k)
                #print(pass2_batch[k])
                #print(kb.keyseq_to_word(pass2_batch[k]))
                #print(pass1_batch[k])
                #print(kb.keyseq_to_word(pass1_batch[k]))
                if (kb.keyseq_to_word(pass2_batch[k]) == kb.keyseq_to_word(pass1_batch[k])):
                    match_vec[0, k] = True       
    acc = np.mean(match_vec)
#     if (max(acc) > 0.6):
#         print(target)
#         print(np.reshape(np.transpose(logits[:,:,acc.index(max(acc))]),
#         target.shape))
    return acc

def pad_sequence_batch(sequence_batch, pad_int):
    """
    Pad sequences with <PAD> = <EOS> so that each sequence of a batch has the same length
    """
    max_sequence = max([len(seq) for seq in sequence_batch])
    return [seq + [pad_int] * (max_sequence - len(seq)) for seq in sequence_batch]
def test_samples_gen(fpath):
    with open(fpath) as tst_file:
        for line in tst_file:
            if not all(0 < ord(c) < 255 for c in line): 
                continue
            if ' ' in line:
               continue
            p = line.rstrip().split('\t')
            #if (len(p) != 2):
            #    continue
            #if ('\n' in p[1]):
            #    p[1] = p[1][:-1]
            #print(line)
            #print(p[0])
            #print(p[1])
            yield p[0], p[1]
def test_batches_gen(fpath, batch_size):
    samples_gen = test_samples_gen(fpath)
    curr_batch_size = 0
    source_batch = []
    target_batch = []
    for i, sample in enumerate(samples_gen):
        #print('Sample: ', i)
        if (curr_batch_size < batch_size):
            source_batch.append(sample[0])
            target_batch.append(sample[1])
            curr_batch_size += 1
        else:
            yield source_batch, target_batch
            curr_batch_size = 1
            source_batch = [sample[0]]
            target_batch = [sample[1]]
    yield source_batch, target_batch

def predict_batches_gen(fpath, batch_size):
    curr_batch_size = 0
    predict_batch = []
    with open(fpath) as in_file:
        for line in in_file:
            password = line.split('\n')[0]
            if (curr_batch_size < batch_size):
                predict_batch.append(password)
                curr_batch_size += 1
            else:
                yield predict_batch
                predict_batch = [password]
                curr_batch_size = 1
        yield predict_batch

def preprocess_batch_prediction(batch, source_vocab_to_int):
    pass_batch = [kb.word_to_keyseq(d) for d in batch[0]]
    pass2_batch = [kb.word_to_keyseq(d) for d in batch[1]]
    
    batch_lengths = np.array([len(p) for p in pass_batch])
    pass_batch_ids = [pass_to_seq(p, source_vocab_to_int) for p in pass_batch]
    source_pad_int = source_vocab_to_int['<EOS>']
    # Pad
    pad_sources_batch = np.array(pad_sequence_batch(pass_batch_ids, source_pad_int))
    return (pad_sources_batch, batch_lengths,pass_batch, pass2_batch)

def preprocess_batch_prediction_single_pass(batch, source_vocab_to_int):
    pass_batch = [kb.word_to_keyseq(d) for d in batch]
    batch_lengths = np.array([len(p) for p in pass_batch])
    pass_batch_ids = [pass_to_seq(p, source_vocab_to_int) for p in pass_batch]
    source_pad_int = source_vocab_to_int['<EOS>']
    # Pad
    pad_sources_batch = np.array(pad_sequence_batch(pass_batch_ids, source_pad_int))
    return (pad_sources_batch, batch_lengths, pass_batch)

def decode(write_to_file=False):
    # Load model config
    config = load_config(FLAGS)

    # Load source data to decode
    trans_dict_2path = IDX_to_TRANS
    b_gen = test_batches_gen(FLAGS.decode_input, FLAGS.decode_batch_size)
    # Initiate TF session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, 
        log_device_placement=FLAGS.log_device_placement, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        # Reload existing checkpoint
        model = load_model(sess, config)

        try:
            print('Decoding and Testing {}..'.format(FLAGS.decode_input))
            if (write_to_file):
                print('Saving predictions in {}'.format(FLAGS.decode_output))
                fout = open(FLAGS.decode_output, 'w')
            acc = 0
            total_samples = 0
            for batch_i, batch in enumerate(b_gen):
                print("Decoding batch # {}".format(batch_i))
                batch_size = len(batch[0])
                total_samples += batch_size
    #             print(max_seq_len)
                pad_encoded_s_batch, batch_lengths, s_batch, t_batch = preprocess_batch_prediction(batch, source_vocab_to_int)
                # predicted_ids: GreedyDecoder; [batch_size, max_time_step, 1]
                # BeamSearchDecoder; [batch_size, max_time_step, beam_width]
                decode_logits, decode_scores = model.predict_scores(sess, encoder_inputs=pad_encoded_s_batch, 
                                                                    encoder_inputs_length=batch_lengths)
                b_acc = get_accuracy_beam_decode(decode_logits, s_batch, t_batch,
                                                 target_int_to_vocab, trans_dict_2path, bias=True)
                acc += batch_size * b_acc
                if (write_to_file):
                    # Write decoding results
                    for i in range(decode_logits.shape[0]):
                        predictions = []
                        for k in range(FLAGS.beam_width):
                            decode_pred = decode_logits[i,:,k]
                #             print(decode_pred)
                            path_idx_pred = [target_int_to_vocab[j] for j in decode_pred]
                #             print(path_idx_pred)
                            if ('<EOS>' in path_idx_pred):
                                path_idx_pred = path_idx_pred[:path_idx_pred.index('<EOS>')]
                            predictions.append(path_to_pass(s_batch[i], path_idx_pred, trans_dict_2path))
                        pred_with_scores = list(zip(predictions, (decode_scores.ravel()).tolist()))
                        fout.write(kb.keyseq_to_word(s_batch[i]) + '\t' + json.dumps(pred_with_scores) + '\n')
                        predictions = []

            acc = acc / total_samples
#             print("Test Accuracy: {0:.4f}".format(acc))
        except IOError:
            pass
        finally:
            if (write_to_file):
                fout.close()
                print("Predictions saved at {}".format(FLAGS.decode_output))
            print("Test Accuracy: {0:.4f}".format(acc))

def predict_batch(num_uniuqe_predictions=100, bias=False):
    # Load model config
    config = load_config(FLAGS)
    # Load source data to decode
    trans_dict_2path = IDX_to_TRANS
    b_gen = predict_batches_gen(FLAGS.decode_input, FLAGS.decode_batch_size)
    # Initiate TF session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, 
        log_device_placement=FLAGS.log_device_placement, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        # Reload existing checkpoint
        model = load_model(sess, config)
        try:
            print('Decoding and Testing {}..'.format(FLAGS.decode_input))
            print('Saving predictions in {}'.format(FLAGS.decode_output))
            fout = open(FLAGS.decode_output, 'w')
            total_samples = 0
            for batch_i, batch in enumerate(b_gen):
                print("Decoding batch # {}".format(batch_i))
                batch_size = len(batch)
                total_samples += batch_size
    #             print(max_seq_len)
                pad_encoded_s_batch, batch_lengths, s_batch = preprocess_batch_prediction_single_pass(batch, source_vocab_to_int)
                # predicted_ids: GreedyDecoder; [batch_size, max_time_step, 1]
                # BeamSearchDecoder; [batch_size, max_time_step, beam_width]
                decode_logits, decode_scores = model.predict_scores(sess, encoder_inputs=pad_encoded_s_batch, 
                                                                    encoder_inputs_length=batch_lengths)
                # Write decoding results
                for i in range(decode_logits.shape[0]):
                    predictions = []
                    for k in range(FLAGS.beam_width):
                        decode_pred = decode_logits[i,:,k]
                        path_idx_pred = [target_int_to_vocab[j] for j in decode_pred]
                        if ('<EOS>' in path_idx_pred):
                            path_idx_pred = path_idx_pred[:path_idx_pred.index('<EOS>')]
                        predictions.append(path_to_pass(s_batch[i], path_idx_pred, trans_dict_2path))
                    #pred_with_scores = list(zip(predictions,
                    #(decode_scores.ravel()).tolist()))
                    # Take uniuqe predicions and add the original password as a
                    # guess
                    orig_pass = kb.keyseq_to_word(s_batch[i])
                    seen = set()
                    seen_add = seen.add
                    if (bias):
                        unq_predictions = [orig_pass]
                        seen_add(orig_pass)
                    else:
                        unq_predictions = []
                    unq_predictions += [x for x in predictions if not (x in seen or seen_add(x))]
                    unq_predictions = unq_predictions[:num_uniuqe_predictions]
                    fout.write(orig_pass + '\t' + json.dumps(unq_predictions) + '\n')
                    predictions = []
        except IOError:
            pass
        finally:
            fout.close()
            print("Predictions saved at {}".format(FLAGS.decode_output))

def predict(original_pass):
    # Load model config
    config = load_config(FLAGS)
    trans_dict_2path = IDX_to_TRANS
    
    decoded_origin_pass = pass_to_seq(original_pass, source_vocab_to_int)
    # Initiate TF session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, 
            log_device_placement=FLAGS.log_device_placement,
            gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

        # Reload existing checkpoint
        model = load_model(sess, config)
        print('Decoding...', decoded_origin_pass)
        decode_logits = model.predict(sess, encoder_inputs=np.array([decoded_origin_pass]), 
                                      encoder_inputs_length=np.array([len(decoded_origin_pass)]))
        for i in range(decode_logits.shape[0]):
            predictions = []
            for k in range(FLAGS.beam_width):
                decode_pred = decode_logits[i, :, k]
                path_idx_pred = [target_int_to_vocab[j] for j in decode_pred]
                if ('<EOS>' in path_idx_pred):
                    path_idx_pred = path_idx_pred[:path_idx_pred.index('<EOS>')]
                predictions.append(path_to_pass(original_pass, path_idx_pred, trans_dict_2path))
        print("Predictions for {}: {}".format(original_pass, ', '.join(sorted(predictions))))

def test_samples_csv_gen(csv_path):
    '''
    This function generates pairs of passwods from a given csv file,
    assuming the first 2 columns are words
    '''
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            yield row[0], row[1]

def create_test_file(csv_path, num_samples=5000, prob=0.001):
    '''
    This function creates a test file format ("pass1<tab>pass2") from a csv dataset
    '''
    coin = [True, False]
    probs = [prob, 1 - prob]
    sample_gen = test_samples_csv_gen(csv_path)
    count = 0
    with open(csv_path[:-4] + '_test.txt', 'w') as tfile:
        for i, sample in enumerate(sample_gen):
            if (count >= num_samples):
                break
            if (np.random.choice(coin, 1, p=probs)[0]):
                tfile.write(sample[0] + '\t' + sample[1] + '\n')
                count += 1
    print("Created test file of {} samples in {}".format(count, csv_path[:-4] + '_test.txt'))
