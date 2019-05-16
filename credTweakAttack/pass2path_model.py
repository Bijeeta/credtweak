'''
pass2path
A variant of seq2seq Encoder-Decoder RNN model that learns pairs of
(password, transition path), where given a password and a transition path, a
new password is generated.

This model is based on JayPark's seq2seq model (Python 2): https://github.com/JayParks/tf-seq2seq

Number of parameters:
m - dimension of embeddings
n - number of hidden units
C_dict - size of charcters dictionary
P_dict - size of transitions (paths) dictionary

For a single-stacked LSTM (including bias, each LSTM has 4 gates/layers):
# params = 4*(n*m + n^2 + n)

From input to embedings:
# params = C_dict * m

Softmax:
# params = n * P_dict

In our case:
1. Inputs -> Embeddings: 100 * 200 = 20000 (Encoder)
2. Embeddings -> Layer 1: 4*(200*128 + 128^2 + 128) = 168448 (Encoder)
3. Layer 1 -> Layer 2: 4*(128*128 + 128^2 +128) = 131584 (Encoder)
4. Layer 2 -> Layer 3:  131584 (Encoder)

5. Layer 3 -> Embeddings: 128 * 200 = 25600 (Decoder)
6. Embeddings -> Layer 1: 4*(200*128 + 128^2 + 128) = 168448 (Decoder)
7. Layer 1 -> Layer 2: 4*(128*128 + 128^2 +128) = 131584 (Decoder)
8. Layer 2 -> Layer 3:  131584 (Decoder)

9. Layer 3 -> Softmax: 128 * 12017 = 1538176 (Decoder)

Total # params: 2447008 (2.447M)
'''

# imports
import numpy as np

import math
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper, ResidualWrapper, DeviceWrapper

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest

#from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops

# Globals
_GO = '_GO'
EOS = '_EOS' # also function as PAD
UNK = '_UNK'

EXTRA_TOKENS = [_GO, EOS, UNK]

START_TOKEN = EXTRA_TOKENS.index(_GO)	# start_token = 0
END_TOKEN = EXTRA_TOKENS.index(EOS)	# end_token = 1
UNK_TOKEN = EXTRA_TOKENS.index(UNK)

class Pass2PathModel():

    def __init__(self, config, mode):

        '''
        mode: train or decode
        config: dictionary consisting of network's parameters
        config uses tf's flags
        '''

        assert mode.lower() in ['train', 'decode']

        self.config = config
        self.mode = mode.lower()

        self.cell_type = config['cell_type']
        self.hidden_units = config['hidden_units']
        self.depth = config['depth']
        #self.attention_type = config['attention_type']
        self.embedding_size = config['embedding_size']
        #self.bidirectional = config.bidirectional
       
        self.num_encoder_symbols = config['num_encoder_symbols'] # Embedding size
        self.num_decoder_symbols = config['num_decoder_symbols'] # Embedding size

        self.use_residual = config['use_residual']
        #self.attn_input_feeding = config['attn_input_feeding']
        self.use_dropout = config['use_dropout']
        self.keep_prob = 1.0 - config['dropout_rate']

        self.optimizer = config['optimizer']
        self.learning_rate = config['learning_rate']
        self.max_gradient_norm = config['max_gradient_norm']
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
	    tf.assign(self.global_epoch_step, self.global_epoch_step + 1)

        self.dtype = tf.float16 if config['use_fp16'] else tf.float32 # for faster learning
        self.keep_prob_placeholder = tf.placeholder(self.dtype, shape=[], name='keep_prob')

        # BeamSearch only needed fo decoding
        self.use_beamsearch_decode = False 
        if (self.mode == 'decode'):
            self.beam_width = config['beam_width']
            self.use_beamsearch_decode = True if self.beam_width > 1 else False
            self.max_decode_step = config['max_decode_step']
 
        self.build_model()

       
    def build_model(self):
        print("building model..")

        # Building encoder and decoder networks
        self.init_placeholders()
        self.build_encoder()
        self.build_decoder()

        # Merge all the training summaries
        self.summary_op = tf.summary.merge_all()


    def init_placeholders(self):       
        # encoder_inputs: [batch_size, max_time_steps]
        self.encoder_inputs = tf.placeholder(dtype=tf.int32,
            shape=(None, None), name='encoder_inputs')

        # encoder_inputs_length: [batch_size]
        self.encoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(None,), name='encoder_inputs_length')

        # get dynamic batch_size
        self.batch_size = tf.shape(self.encoder_inputs)[0]
        if (self.mode == 'train'):

            # decoder_inputs: [batch_size, max_time_steps]
            self.decoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name='decoder_inputs')
            # decoder_inputs_length: [batch_size]
            self.decoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(None,), name='decoder_inputs_length')

            decoder_start_token = tf.ones(shape=[self.batch_size, 1], dtype=tf.int32) * START_TOKEN
            decoder_end_token = tf.ones(shape=[self.batch_size, 1], dtype=tf.int32) * END_TOKEN           

            # decoder_inputs_train: [batch_size , max_time_steps + 1]
            # insert _GO symbol in front of each decoder input
            self.decoder_inputs_train = tf.concat([decoder_start_token,
                                                  self.decoder_inputs], axis=1)

            # decoder_inputs_length_train: [batch_size]
            self.decoder_inputs_length_train = self.decoder_inputs_length + 1

            # decoder_targets_train: [batch_size, max_time_steps + 1]
            # insert EOS symbol at the end of each decoder input
            self.decoder_targets_train = tf.concat([self.decoder_inputs,
                                                   decoder_end_token], axis=1)


    def build_encoder(self):
        print("building encoder..")
        with tf.variable_scope('encoder'):
            # Building encoder_cell
            self.encoder_cell = self.build_encoder_cell()
            
            # Initialize encoder_embeddings to have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=self.dtype)
             
            self.encoder_embeddings = tf.get_variable(name='embedding',
                shape=[self.num_encoder_symbols, self.embedding_size],
                initializer=initializer, dtype=self.dtype)
            
            # Embedded_inputs: [batch_size, time_step, embedding_size]
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(params=self.encoder_embeddings, ids=self.encoder_inputs)
       
            # Input projection layer to feed embedded inputs to the cell
            # ** Essential when use_residual=True to match input/output dims
            input_layer = Dense(self.hidden_units, dtype=self.dtype, name='input_projection')

            # Embedded inputs having gone through input projection layer
            self.encoder_inputs_embedded = input_layer(self.encoder_inputs_embedded)
    
            # Encode input sequences into context vectors:
            # encoder_outputs: [batch_size, max_time_step, cell_output_size]
            # encoder_state: [batch_size, cell_output_size]
            self.encoder_outputs, self.encoder_last_state = tf.nn.dynamic_rnn(cell=self.encoder_cell, inputs=self.encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length, dtype=self.dtype,
                time_major=False)


    def build_decoder(self):
        print("building decoder...")
        with tf.variable_scope('decoder'):
            # Building decoder_cell and decoder_initial_state
            self.decoder_cell, self.decoder_initial_state = self.build_decoder_cell()

            # Initialize decoder embeddings to have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=self.dtype)
             
            self.decoder_embeddings = tf.get_variable(name='embedding',
                shape=[self.num_decoder_symbols, self.embedding_size],
                initializer=initializer, dtype=self.dtype)

            # Input projection layer to feed embedded inputs to the cell
            # ** Essential when use_residual=True to match input/output dims
            input_layer = Dense(self.hidden_units, dtype=self.dtype, name='input_projection')

            # Output projection layer to convert cell_outputs to logits
            output_layer = Dense(self.num_decoder_symbols, name='output_projection')

            if self.mode == 'train':
                # decoder_inputs_embedded: [batch_size, max_time_step + 1,
                # embedding_size]
                self.decoder_inputs_embedded = tf.nn.embedding_lookup(params=self.decoder_embeddings, ids=self.decoder_inputs_train)
               
                # Embedded inputs having gone through input projection layer
                self.decoder_inputs_embedded = input_layer(self.decoder_inputs_embedded)

                # Helper to feed inputs for training: read inputs from dense
                # ground truth vectors
                training_helper = seq2seq.TrainingHelper(inputs=self.decoder_inputs_embedded,
                                                   sequence_length=self.decoder_inputs_length_train,
                                                   time_major=False,
                                                   name='training_helper')

                training_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                   helper=training_helper,
                                                   initial_state=self.decoder_initial_state,
                                                   output_layer=output_layer)
                                                   #output_layer=None)
                    
                # Maximum decoder time_steps in current batch
                max_decoder_length = tf.reduce_max(self.decoder_inputs_length_train)

               
                (self.decoder_outputs_train, self.decoder_last_state_train, 
                 self.decoder_outputs_length_train) = (seq2seq.dynamic_decode(decoder=training_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=max_decoder_length))
                 
                # More efficient to do the projection on the
                # batch-time-concatenated tensor
                # logits_train: [batch_size, max_time_step + 1,
                # num_decoder_symbols]
                # self.decoder_logits_train =
                # output_layer(self.decoder_outputs_train.rnn_output)
                self.decoder_logits_train = tf.identity(self.decoder_outputs_train.rnn_output) 
                # Use argmax to extract decoder symbols to emit
                self.decoder_pred_train = tf.argmax(self.decoder_logits_train, axis=-1,
                                                    name='decoder_pred_train')

                # masks: masking for valid and padded time steps, [batch_size,
                # max_time_step + 1]
                masks = tf.sequence_mask(lengths=self.decoder_inputs_length_train, 
                                         maxlen=max_decoder_length, dtype=self.dtype, name='masks')

                # Computes per word average cross-entropy over a batch
                # Internally calls
                # 'nn_ops.sparse_softmax_cross_entropy_with_logits' by default
                self.loss = seq2seq.sequence_loss(logits=self.decoder_logits_train, 
                                                  targets=self.decoder_targets_train,
                                                  weights=masks,
                                                  average_across_timesteps=True,
                                                  average_across_batch=True,)
                # Training summary for the current batch_loss
                tf.summary.scalar('loss', self.loss)

                # Contruct graphs for minimizing loss
                self.init_optimizer()

            elif (self.mode == 'decode'):
        
                # Start_tokens: [batch_size,] `int32` vector
                start_tokens = tf.ones([self.batch_size,], tf.int32) * START_TOKEN
                end_token = END_TOKEN

                def embed_and_input_proj(inputs):
                    return input_layer(tf.nn.embedding_lookup(self.decoder_embeddings, inputs))
                    
                if not self.use_beamsearch_decode:
                    # Helper to feed inputs for greedy decoding: uses the
                    # argmax of the output
                    decoding_helper = seq2seq.GreedyEmbeddingHelper(start_tokens=start_tokens,
                                                                    end_token=end_token,
                                                                    embedding=embed_and_input_proj)
                    # Basic decoder performs greedy decoding at each time step
                    print("building greedy decoder..")
                    inference_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                             helper=decoding_helper,
                                                             initial_state=self.decoder_initial_state,
                                                             output_layer=output_layer)
                else:
                    # Beamsearch is used to approximately find the most likely
                    # translation
                    print("building beamsearch decoder..")
                    inference_decoder = beam_search_decoder.BeamSearchDecoder(cell=self.decoder_cell,
                                                               embedding=embed_and_input_proj,
                                                               start_tokens=start_tokens,
                                                               end_token=end_token,
                                                               initial_state=self.decoder_initial_state,
                                                               beam_width=self.beam_width,
                                                               output_layer=output_layer,)

                (self.decoder_outputs_decode, self.decoder_last_state_decode,
                 self.decoder_outputs_length_decode) = (seq2seq.dynamic_decode(decoder=inference_decoder,
                    output_time_major=False,
                    #impute_finished=True, # error occurs
                    maximum_iterations=self.max_decode_step))

                if not self.use_beamsearch_decode:
                    # decoder_outputs_decode.sample_id: [batch_size,
                    # max_time_step]
                    # Or use argmax to find decoder symbols to emit:
                    # self.decoder_pred_decode =
                    # tf.argmax(self.decoder_outputs_decode.rnn_output,
                    #                                      axis=-1,
                    #                                      name='decoder_pred_decode')

                    # Here, we use expand_dims to be compatible with the result
                    # of the beamsearch decoder
                    # decoder_pred_decode: [batch_size, max_time_step, 1]
                    # (output_major=False)
                    self.decoder_pred_decode = tf.expand_dims(self.decoder_outputs_decode.sample_id, -1)

                else:
                    # Use beam search to approximately find the most likely
                    # translation
                    # decoder_pred_decode: [batch_size, max_time_step,
                    # beam_width] (output_major=False)
                    self.decoder_pred_decode = self.decoder_outputs_decode.predicted_ids
                    self.decoder_pred_scores = self.decoder_outputs_decode.beam_search_decoder_output.scores


    def build_single_cell(self):
        cell_type = LSTMCell
        if (self.cell_type.lower() == 'gru'):
            cell_type = GRUCell
        cell = cell_type(self.hidden_units)

        if self.use_dropout:
            cell = DropoutWrapper(cell, dtype=self.dtype,
                                  output_keep_prob=self.keep_prob_placeholder,)
        if self.use_residual:
            cell = ResidualWrapper(cell)
            
        return cell


    # Building encoder cell
    def build_encoder_cell(self):
        # ADD GPU SUPPORT
        return MultiRNNCell([self.build_single_cell() for i in range(self.depth)])


    # Building decoder cell and attention.  Also returns decoder_initial_state
    def build_decoder_cell(self):

        encoder_outputs = self.encoder_outputs
        encoder_last_state = self.encoder_last_state
        encoder_inputs_length = self.encoder_inputs_length

        # To use BeamSearchDecoder, encoder_outputs, encoder_last_state,
        # encoder_inputs_length
        # needs to be tiled so that: [batch_size, .., ..] -> [batch_size x
        # beam_width, .., ..]
        if self.use_beamsearch_decode:
            print("use beamsearch decoding..")
            encoder_outputs = seq2seq.tile_batch(self.encoder_outputs, multiplier=self.beam_width)
            encoder_last_state = nest.map_structure(lambda s: seq2seq.tile_batch(s, self.beam_width), self.encoder_last_state)
            encoder_inputs_length = seq2seq.tile_batch(self.encoder_inputs_length, multiplier=self.beam_width)
 
        # Building decoder_cell
        self.decoder_cell_list = [
            self.build_single_cell() for i in range(self.depth)]
        # ADD GPU SUPPORT FOR DISTRIBUION
        decoder_initial_state = encoder_last_state

        # Also if beamsearch decoding is used, the batch_size argument in
        # .zero_state
        # should be ${decoder_beam_width} times to the origianl batch_size
        batch_size = self.batch_size if not self.use_beamsearch_decode \
                     else self.batch_size * self.beam_width
        initial_state = [state for state in encoder_last_state]

        initial_state[-1] = self.decoder_cell_list[-1].zero_state(batch_size=batch_size, dtype=self.dtype)
        decoder_initial_state = tuple(initial_state)

        return MultiRNNCell(self.decoder_cell_list), decoder_initial_state


    def init_optimizer(self):
        print("setting optimizer..")
        # Gradients and SGD update operation for training the model
        trainable_params = tf.trainable_variables()
        if self.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        else:
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        # Compute gradients of loss w.r.t. all trainable variables
        #gradients = tf.gradients(self.loss, trainable_params) # OLD
        gradients = self.opt.compute_gradients(self.loss)

        # Clip gradients by a given maximum_gradient_norm
        #clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm) # OLD
        clip_gradients = [(tf.clip_by_value(grad, -self.max_gradient_norm, self.max_gradient_norm), var) for grad, var in gradients if grad is not None]
        # Update the model
        self.updates = self.opt.apply_gradients(clip_gradients, global_step=self.global_step)

    def save(self, sess, path, var_list=None, global_step=None):
        # var_list = None returns the list of all saveable variables
        saver = tf.train.Saver(var_list)

        save_path = saver.save(sess, save_path=path, global_step=global_step)
        print('model saved at %s' % save_path)
        

    def restore(self, sess, path, var_list=None):
        # var_list = None returns the list of all saveable variables
        saver = tf.train.Saver(var_list)
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)


    def train(self, sess, encoder_inputs, encoder_inputs_length, 
              decoder_inputs, decoder_inputs_length):
        """Run a train step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size] 
              to feed as sequence lengths for each element in the given batch
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average accuracy, and the outputs.
        """
        # Check if the model is 'training' mode
        if self.mode.lower() != 'train':
            raise ValueError("train step can only be operated in train mode")

        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs, decoder_inputs_length, False)
        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = self.keep_prob
 
        output_feed = [self.updates,	# Update Op that does optimization
                       self.loss,	# Loss for current batch
                       self.summary_op]	# Training summary
        
        outputs = sess.run(output_feed, input_feed)
        return outputs[1], outputs[2]	# loss, summary


    def eval(self, sess, encoder_inputs, encoder_inputs_length,
             decoder_inputs, decoder_inputs_length):
        """Run a evaluation step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size] 
              to feed as sequence lengths for each element in the given batch
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average accuracy, and the outputs.
        """
        
        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs, decoder_inputs_length, False)
        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = 1.0

        output_feed = [self.loss,	# Loss for current batch
                       self.summary_op]	# Training diary
        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1]	# loss


    def predict(self, sess, encoder_inputs, encoder_inputs_length):

        # To reproduce results we have to keep a constant random seed:
        # tf.set_random_seed(1729)
        # np.random.seed(1729)
        
        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length, 
                                      decoder_inputs=None, decoder_inputs_length=None, 
                                      decode=True)

        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = 1.0
 
        output_feed = [self.decoder_pred_decode]
        outputs = sess.run(output_feed, input_feed)

	    # GreedyDecoder: [batch_size, max_time_step]
        # BeamSearchDecoder: [batch_size, max_time_step, beam_width]
        return outputs[0]	

    def predict_scores(self, sess, encoder_inputs, encoder_inputs_length):

        # To reproduce results we have to keep a constant random seed:
        # tf.set_random_seed(1729)
        # np.random.seed(1729)
        
        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length, 
                                      decoder_inputs=None, decoder_inputs_length=None, 
                                      decode=True)

        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = 1.0
 
        output_feed = [self.decoder_pred_decode, self.decoder_pred_scores]
        outputs = sess.run(output_feed, input_feed)
        #scores = np.exp(np.sum(outputs[1], axis=1))
        scores = np.sum(outputs[1], axis=1)
        #print(outputs[1].shape)
        #print(scores.shape)
        #print(scores)
        #sanity_check = np.sum(scores, axis=1) 
        #print(sanity_check.shape)
        #print(sanity_check)
	    # GreedyDecoder: [batch_size, max_time_step]
        # BeamSearchDecoder: [batch_size, max_time_step, beam_width]
        return outputs[0], scores	


    def check_feeds(self, encoder_inputs, encoder_inputs_length, 
                    decoder_inputs, decoder_inputs_length, decode):
        """
        Args:
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size] 
              to feed as sequence lengths for each element in the given batch
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          decode: a scalar boolean that indicates decode mode
        Returns:
          A feed for the model that consists of encoder_inputs, encoder_inputs_length,
          decoder_inputs, decoder_inputs_length
        """ 
        #print(encoder_inputs)
        input_batch_size = encoder_inputs.shape[0]
        if input_batch_size != encoder_inputs_length.shape[0]:
            raise ValueError("Encoder inputs and their lengths must be equal in their "
                "batch_size, %d != %d" % (input_batch_size, encoder_inputs_length.shape[0]))

        if not decode:
            target_batch_size = decoder_inputs.shape[0]
            if target_batch_size != input_batch_size:
                raise ValueError("Encoder inputs and Decoder inputs must be equal in their "
                    "batch_size, %d != %d" % (input_batch_size, target_batch_size))
            if target_batch_size != decoder_inputs_length.shape[0]:
                raise ValueError("Decoder targets and their lengths must be equal in their "
                    "batch_size, %d != %d" % (target_batch_size, decoder_inputs_length.shape[0]))

        input_feed = {}
    
        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.encoder_inputs_length.name] = encoder_inputs_length

        if not decode:
            input_feed[self.decoder_inputs.name] = decoder_inputs
            input_feed[self.decoder_inputs_length.name] = decoder_inputs_length

        return input_feed 

