import tensorflow as tf

tf_ver = tf.__version__
SHERLOCK = (str(tf_ver) == '0.12.1')

from tensorflow.contrib import rnn
if SHERLOCK:
	from tensorflow.python.ops import rnn_cell as rnn
	from tensorflow.contrib.metrics import confusion_matrix as tf_confusion_matrix
else:
	from tensorflow.contrib import rnn


from tensorflow.contrib import seq2seq
import numpy as np
import sys
import os
import logging
import math

import utils_hyperparam


class Config(object):

	def __init__(self, hyperparam_path):
		self.batch_size = 100
		self.lr = 0.001

		self.songtype = 20#20
		self.sign = 16
		self.notesize = 3
		self.flats = 11
		self.mode = 6
		self.len = 1
		self.complex = 1
		self.max_length = 8

		self.vocab_size = 81
		# self.meta_embed = self.songtype
		self.meta_embed = 100 #self.songtype/2
		self.hidden_size = self.meta_embed*5 #+ 2
		self.embedding_dims = self.vocab_size*3/4
		self.vocab_meta = self.songtype + self.sign + self.notesize+ self.flats + self.mode
		self.num_meta = 7
		self.num_layers = 2
		self.keep_prob = 0.6

		# Only for CBOW model
		self.embed_size = 32

		# Only for Seq2Seq Attention Models
		self.num_encode = 8
		self.num_decode = 4
		self.attention_option = 'luong'
		self.bidirectional = False

		# Discriminator Parameters
		self.numFilters = 32
		self.hidden_units = 100
		self.num_outputs = 2
		self.cnn_lr = 0.001
		self.label_smooth = 0.15
		self.generator_prob = 0.1
		self.num_classes = 19
		self.gan_lr = 0.001

		if len(hyperparam_path)!=0:
			print "Setting hyperparameters from a file %s" %hyperparam_path
			utils_hyperparam.setHyperparam(self, hyperparam_path)


class CBOW(object):

	def __init__(self, input_size, batch_size, vocab_size, hyperparam_path):
		self.config = Config(hyperparam_path)
		self.input_size = input_size
		self.config.batch_size = batch_size
		self.config.vocab_size = vocab_size
		self.input_placeholder = tf.placeholder(tf.int32, shape=[None, self.input_size], name="Inputs")
		self.label_placeholder = tf.placeholder(tf.int32, shape=[None], name="Labels")
		self.embeddings = tf.Variable(tf.random_uniform([self.config.vocab_size,
								self.config.embed_size], -1.0, 1.0))

		print("Completed Initializing the CBOW Model.....")

	def create_model(self):
		weight = tf.get_variable("Wout", shape=[self.config.embed_size, self.config.vocab_size],
					initializer=tf.contrib.layers.xavier_initializer())
		bias = tf.Variable(tf.zeros([self.config.vocab_size]))

		word_vec =  tf.nn.embedding_lookup(self.embeddings, self.input_placeholder)
		average_embedding = tf.reduce_sum(word_vec, reduction_indices=1)

		self.logits_op = tf.add(tf.matmul(average_embedding, weight), bias)
		self.probabilities_op = tf.nn.softmax(self.logits_op)
		print("Built the CBOW Model.....")

	def train(self):
		self.loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_op, labels=self.label_placeholder))
		tf.summary.scalar('Loss', self.loss_op)
		self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(self.loss_op)

		print("Setup the training mechanism for the CBOW model.....")

	def metrics(self):
		last_axis = len(self.probabilities_op.get_shape().as_list())
		self.prediction_op = tf.to_int32(tf.argmax(self.probabilities_op, axis=last_axis-1))
		difference = self.label_placeholder - self.prediction_op
		zero = tf.constant(0, dtype=tf.int32)
		boolean_difference = tf.cast(tf.equal(difference, zero), tf.float64)
		self.accuracy_op = tf.reduce_mean(boolean_difference)
		tf.summary.scalar('Accuracy', self.accuracy_op)

		self.summary_op = tf.summary.merge_all()

		if SHERLOCK:
			self.confusion_matrix = tf_confusion_matrix(tf.reshape(self.label_placeholder, [-1]), tf.reshape(self.prediction_op, [-1]), num_classes=self.config.vocab_size, dtype=tf.int32)
		else:
			self.confusion_matrix = tf.confusion_matrix(tf.reshape(self.label_placeholder, [-1]), tf.reshape(self.prediction_op, [-1]), num_classes=self.config.vocab_size, dtype=tf.int32)

	def _feed_dict(self, feed_values):
		input_batch = feed_values[0]
		label_batch = feed_values[1]
		feed_dict = {
			self.input_placeholder: input_batch,
			self.label_placeholder: label_batch
		}
		return feed_dict

	def run(self, args, session, feed_values):
		feed_dict = self._feed_dict(feed_values)

		if args.train == "train":
			_, summary, loss, probabilities, prediction, accuracy, confusion_matrix = session.run([self.train_op, self.summary_op, self.loss_op, self.probabilities_op, self.prediction_op, self.accuracy_op, self.confusion_matrix], feed_dict=feed_dict)
		else: # Sample case not necessary b/c function will only be called during normal runs
			summary, loss, probabilities, prediction, accuracy, confusion_matrix = session.run([self.summary_op, self.loss_op, self.probabilities_op, self.prediction_op, self.accuracy_op, self.confusion_matrix], feed_dict=feed_dict)

		print "Average accuracy per batch {0}".format(accuracy)
		print "Batch Loss: {0}".format(loss)
		# print "Output Predictions: {0}".format(prediction)
		# print "Output Prediction Probabilities: {0}".format(probabilities)

		return summary, confusion_matrix, accuracy


	def sample(self, session, feed_values):
		feed_dict = self._feed_dict(feed_values)

		logits = session.run([self.logits_op], feed_dict=feed_dict)[0]
		return logits, np.zeros((1, 1)) # dummy value



class CharRNN(object):

	def __init__(self, input_size, label_size, batch_size, vocab_size, cell_type, hyperparam_path, gan_inputs=None):
		self.input_size = input_size
		self.label_size = label_size
		self.cell_type = cell_type
		self.config = Config(hyperparam_path)
		self.config.batch_size = batch_size
		self.config.vocab_size = vocab_size
		self.gan_inputs = gan_inputs

		# self.initial_state = self.cell.zero_state(self.config.batch_size, dtype=tf.int32)
		# input_shape = (None,) + tuple([self.config.max_length,input_size])
		input_shape = (None,) + tuple([input_size])
		# output_shape = (None,) + tuple([self.config.max_length,label_size])
		output_shape = (None,) + tuple([label_size])

		self.input_placeholder = tf.placeholder(tf.int32, shape=input_shape, name='Input')
		self.label_placeholder = tf.placeholder(tf.int32, shape=output_shape, name='Output')
		self.meta_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.num_meta], name='Meta')
		self.use_meta_placeholder = tf.placeholder(tf.bool, name='State_Initialization_Bool')

		if cell_type == 'rnn':
			self.cell = rnn.BasicRNNCell(self.config.hidden_size)
			self.initial_state_placeholder = tf.placeholder(tf.float32, shape=[None, self.config.hidden_size], name="Initial_State")
		elif cell_type == 'gru':
			self.cell = rnn.GRUCell(self.config.hidden_size)
			self.initial_state_placeholder = tf.placeholder(tf.float32, shape=[None, self.config.hidden_size], name="Initial_State")
		elif cell_type == 'lstm':
			self.cell = rnn.BasicLSTMCell(self.config.hidden_size)
			self.initial_state_placeholder = tf.placeholder(tf.float32, shape=[self.config.num_layers, None, self.config.hidden_size], name="Initial_State")

		print "Completed Initializing the Char RNN Model using a {0} cell".format(cell_type.upper())


	def create_model(self, is_train=True):
		if is_train:
			self.cell = rnn.DropoutWrapper(self.cell, input_keep_prob=1.0, output_keep_prob=self.config.keep_prob)
		rnn_model = rnn.MultiRNNCell([self.cell]*self.config.num_layers, state_is_tuple=True)

		# Embedding lookup for ABC format characters
		embeddings_var = tf.Variable(tf.random_uniform([self.config.vocab_size, self.config.embedding_dims],
										 0, 10, dtype=tf.float32, seed=3), name='char_embeddings')
		true_inputs = self.input_placeholder if (self.gan_inputs == None) else self.gan_inputs

		embeddings = tf.nn.embedding_lookup(embeddings_var, true_inputs)

		# Embedding lookup for Metadata
		embeddings_var_meta = tf.Variable(tf.random_uniform([self.config.vocab_meta, self.config.meta_embed],
									 0, 10, dtype=tf.float32, seed=3), name='char_embeddings_meta')
		embeddings_meta = tf.nn.embedding_lookup(embeddings_var_meta, self.meta_placeholder[:, :5])

		# Putting all the word embeddings together and then appending the numerical constants at the end of the word embeddings
		embeddings_meta = tf.reshape(embeddings_meta, shape=[-1, self.config.hidden_size]) # -2
		# 	embeddings_meta = tf.concat([embeddings_meta, tf.convert_to_tensor(self.meta_placeholder[:, 5:])], axis=0)

		if self.cell_type == 'lstm':
			initial_added = tf.cond(self.use_meta_placeholder,
									lambda: [embeddings_meta for layer in xrange(self.config.num_layers)],
									lambda: tf.unstack(self.initial_state_placeholder, axis=0)) # [self.initial_state_placeholder[layer] for layer in xrange(self.config.num_layers)])
			[initial_added[idx].set_shape([self.config.batch_size, self.config.hidden_size]) for idx in xrange(self.config.num_layers)]
			initial_tuple = tuple([rnn.LSTMStateTuple(initial_added[idx], np.zeros((self.config.batch_size, self.config.hidden_size), dtype=np.float32)) for idx in xrange(self.config.num_layers)])
		else:
			initial_added = tf.cond(self.use_meta_placeholder,
									lambda: embeddings_meta,
									lambda: self.initial_state_placeholder)
			initial_tuple = (initial_added, np.zeros((self.config.batch_size, self.config.hidden_size), dtype=np.float32))

		rnn_output, self.state_op = tf.nn.dynamic_rnn(rnn_model, embeddings, dtype=tf.float32, initial_state=initial_tuple)

		decode_var = tf.Variable(tf.random_uniform([self.config.hidden_size, self.config.vocab_size],
										 0, 10, dtype=tf.float32, seed=3), name='char_decode')
		decode_bias = tf.Variable(tf.random_uniform([self.config.vocab_size],
										 0, 10, dtype=tf.float32, seed=3), name='char_decode_bias')
		decode_list = []
		for i in xrange(self.input_size):
			decode_list.append(tf.matmul(rnn_output[:, i, :], decode_var) + decode_bias)

		self.logits_op = tf.stack(decode_list, axis=1)
		self.rnn_output = rnn_output
		self.probabilities_op = tf.nn.softmax(self.logits_op)

		print("Built the Char RNN Model...")


	def train(self, max_norm=5, op='adam'):
		self.loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_op, labels=self.label_placeholder))
		tf.summary.scalar('Loss', self.loss_op)
		tvars = tf.trainable_variables()

		# Gradient clipping
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_op, tvars),max_norm)
		optimizer = tf.train.AdamOptimizer(self.config.lr)
		self.train_op = optimizer.apply_gradients(zip(grads, tvars))

		print("Setup the training mechanism for the Char RNN Model...")


	def metrics(self):
		# Same function, did not make a general one b/c need to store _ops within class
		last_axis = len(self.probabilities_op.get_shape().as_list())
		self.prediction_op = tf.to_int32(tf.argmax(self.probabilities_op, axis=last_axis-1))
		difference = self.label_placeholder - self.prediction_op
		zero = tf.constant(0, dtype=tf.int32)
		boolean_difference = tf.cast(tf.equal(difference, zero), tf.float64)
		self.accuracy_op = tf.reduce_mean(boolean_difference)
		tf.summary.scalar('Accuracy', self.accuracy_op)

		self.summary_op = tf.summary.merge_all()

		if SHERLOCK:
			self.confusion_matrix = tf_confusion_matrix(tf.reshape(self.label_placeholder, [-1]), tf.reshape(self.prediction_op, [-1]), num_classes=self.config.vocab_size, dtype=tf.int32)
		else:
			self.confusion_matrix = tf.confusion_matrix(tf.reshape(self.label_placeholder, [-1]), tf.reshape(self.prediction_op, [-1]), num_classes=self.config.vocab_size, dtype=tf.int32)


	def _feed_dict(self, feed_values):
		input_batch = feed_values[0]
		label_batch = feed_values[1]
		meta_batch = feed_values[2]
		initial_state_batch = feed_values[3]
		use_meta_batch = feed_values[4]

		feed_dict = {
			self.input_placeholder: input_batch,
			self.label_placeholder: label_batch,
			self.meta_placeholder: meta_batch,
			self.initial_state_placeholder: initial_state_batch,
			self.use_meta_placeholder: use_meta_batch
		}

		return feed_dict


	def run(self, args, session, feed_values):
		feed_dict = self._feed_dict(feed_values)

		if args.train == "train":
			_, summary, loss, probabilities, prediction, accuracy, confusion_matrix = session.run([self.train_op, self.summary_op, self.loss_op, self.probabilities_op, self.prediction_op, self.accuracy_op, self.confusion_matrix], feed_dict=feed_dict)
		else: # Sample case not necessary b/c function will only be called during normal runs
			summary, loss, probabilities, prediction, accuracy, confusion_matrix = session.run([self.summary_op, self.loss_op, self.probabilities_op, self.prediction_op, self.accuracy_op, self.confusion_matrix], feed_dict=feed_dict)

		print "Average accuracy per batch {0}".format(accuracy)
		print "Batch Loss: {0}".format(loss)
		# print "Output Predictions: {0}".format(prediction)
		# print "Output Prediction Probabilities: {0}".format(probabilities)

		return summary, confusion_matrix, accuracy


	def sample(self, session, feed_values):
		feed_dict = self._feed_dict(feed_values)

		logits, state = session.run([self.logits_op, self.state_op], feed_dict=feed_dict)
		return logits, state






class Seq2SeqRNN(object):

	def __init__(self,input_size, label_size, batch_size, vocab_size, cell_type,
						hyperparam_path, start_encode, end_encode):
		self.input_size = input_size
		self.label_size = label_size
		self.cell_type = cell_type
		self.config = Config(hyperparam_path)
		self.config.batch_size = batch_size
		self.config.vocab_size = vocab_size

		# input_shape = (None,) + tuple([input_size])
		input_shape = (None, None)
		# output_shape = (None,) + tuple([label_size])
		output_shape = (None, None)

		self.input_placeholder = tf.placeholder(tf.int32, shape=input_shape, name='Input')
		self.label_placeholder = tf.placeholder(tf.int32, shape=output_shape, name='Output')
		self.meta_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.num_meta], name='Meta')
		self.use_meta_placeholder = tf.placeholder(tf.bool, name='State_Initialization_Bool')
		self.num_encode = tf.placeholder(tf.int32, shape=(None,), name='Num_encode')
		self.num_decode = tf.placeholder(tf.int32, shape=(None,),  name='Num_decode')

		if cell_type == 'rnn':
			self.cell = rnn.BasicRNNCell(self.config.hidden_size)
			self.initial_state_placeholder = tf.placeholder(tf.float32, shape=[None, self.config.hidden_size], name="Initial_State")
		elif cell_type == 'gru':
			self.cell = rnn.GRUCell(self.config.hidden_size)
			self.initial_state_placeholder = tf.placeholder(tf.float32, shape=[None, self.config.hidden_size], name="Initial_State")
		elif cell_type == 'lstm':
			self.cell = rnn.BasicLSTMCell(self.config.hidden_size)
			self.initial_state_placeholder = tf.placeholder(tf.float32, shape=[self.config.num_layers, None, self.config.hidden_size], name="Initial_State")

		print "Completed Initializing the Seq2Seq RNN Model using a {0} cell".format(cell_type.upper())

		# Seq2Seq specific initializers
		self.attention_option = "luong"
		self.start_encode = start_encode
		self.end_encode = end_encode


	# Based on the example model presented in https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/model_new.py
	def create_model(self, is_train):
		with tf.variable_scope("Seq2Seq") as scope:

			def output_fn(outputs):
				return tf.contrib.layers.linear(outputs, self.config.vocab_size, scope=scope)

			if is_train:
				self.cell = rnn.DropoutWrapper(self.cell, input_keep_prob=1.0, output_keep_prob=self.config.keep_prob)

			self.cell = rnn.MultiRNNCell([self.cell]*self.config.num_layers, state_is_tuple=True)

			# GO_SLICE = tf.ones([tf.shape(self.input_placeholder)[0],1], dtype=tf.int32)*self.start_encode

			# self.decoder_train_inputs = self.label_placeholder[:,:self.input_size-1]
			self.go_token = tf.constant(self.config.vocab_size-1, dtype=tf.int32, shape=[1, self.config.batch_size])
			self.decoder_train_inputs = tf.concat([self.go_token, self.label_placeholder[:self.input_size-1, :]], axis=0)

			self.decoder_train_targets = self.label_placeholder

			self.loss_weights = tf.ones([self.config.batch_size, self.input_size], dtype=tf.float32, name="loss_weights")
			sqrt3 = math.sqrt(3)
			initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

			# Creating the embeddings and deriving the embeddings for the encoder and decoder
			self.embedding_matrix = tf.get_variable(name="embedding_matrix",
				shape=[self.config.vocab_size, self.config.embedding_dims], initializer=initializer,
				dtype=tf.float32)

			self.encoder_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.input_placeholder)

			self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.decoder_train_inputs)

			# Embedding lookup for Metadata
			embeddings_var_meta = tf.Variable(tf.random_uniform([self.config.vocab_meta, self.config.meta_embed],
										 0, 10, dtype=tf.float32, seed=3), name='char_embeddings_meta')
			embeddings_meta = tf.nn.embedding_lookup(embeddings_var_meta, self.meta_placeholder[:, :5])

			# Putting all the word embeddings together and then appending the numerical constants at the end of the word embeddings
			embeddings_meta = tf.reshape(embeddings_meta, shape=[-1, self.config.hidden_size]) # -2
			# 	embeddings_meta = tf.concat([embeddings_meta, tf.convert_to_tensor(self.meta_placeholder[:, 5:])], axis=0)

			# Create initial_state
			if self.cell_type == 'lstm':
				initial_added = tf.cond(self.use_meta_placeholder,
										lambda: [embeddings_meta for layer in xrange(self.config.num_layers)],
										lambda: tf.unstack(self.initial_state_placeholder, axis=0)) # [self.initial_state_placeholder[layer] for layer in xrange(self.config.num_layers)])
				[initial_added[idx].set_shape([self.config.batch_size, self.config.hidden_size]) for idx in xrange(self.config.num_layers)]
				initial_tuple = tuple([rnn.LSTMStateTuple(initial_added[idx], np.zeros((self.config.batch_size, self.config.hidden_size), dtype=np.float32)) for idx in xrange(self.config.num_layers)])
			else:
				initial_added = tf.cond(self.use_meta_placeholder,
										lambda: embeddings_meta,
										lambda: self.initial_state_placeholder)
				initial_tuple = (initial_added, np.zeros((self.config.batch_size, self.config.hidden_size), dtype=np.float32))

			if not self.config.bidirectional:
				self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(cell=self.cell, inputs=self.encoder_embedded,
									  sequence_length=self.num_encode,time_major=True, dtype=tf.float32, initial_state=initial_tuple)
			else:
				encoder_fw_outputs,encoder_bw_outputs,
				encoder_fw_state, encoder_bw_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell,
												cell_bw=self.cell, inputs=self.encoder_embedded,
												sequence_length=self.num_encode, time_major=True, dtype=tf.float32)
				self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
				if isinstance(encoder_fw_state, LSTMStateTuple):
					encoder_state_c = tf.concat( (encoder_fw_state.c, encoder_bw_state.c), 1, name='bidirectional_concat_c')
					encoder_state_h = tf.concat( (encoder_fw_state.h, encoder_bw_state.h), 1, name='bidirectional_concat_h')
					self.encoder_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

				elif isinstance(encoder_fw_state, tf.Tensor):
					self.encoder_state = tf.concat((encoder_fw_state, encoder_bw_state), 1, name='bidirectional_concat')


			# Setting up the Attention mechanism
			attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])

			attention_keys, attention_values, attention_score_fn, \
					attention_construct_fn = seq2seq.prepare_attention( attention_states=attention_states,
										attention_option=self.attention_option, num_units=self.config.hidden_size)

			decoder_fn_train = seq2seq.attention_decoder_fn_train( encoder_state=self.encoder_state,
						attention_keys=attention_keys, attention_values=attention_values,
						attention_score_fn=attention_score_fn, attention_construct_fn=attention_construct_fn,
						name='attention_decoder')

			decoder_fn_inference = seq2seq.attention_decoder_fn_inference( output_fn=output_fn, encoder_state=self.encoder_state,
						attention_keys=attention_keys, attention_values=attention_values, attention_score_fn=attention_score_fn,
						attention_construct_fn=attention_construct_fn, embeddings=self.embedding_matrix,
						start_of_sequence_id=self.start_encode, end_of_sequence_id=self.end_encode,
						maximum_length=tf.reduce_max(self.num_encode) + 3, num_decoder_symbols=self.config.vocab_size)

			self.decoder_outputs_train, self.decoder_state_train, \
			self.decoder_context_state_train =  seq2seq.dynamic_rnn_decoder( cell=self.cell,
						decoder_fn=decoder_fn_train, inputs=self.decoder_inputs_embedded,
						sequence_length=self.num_decode, time_major=True, scope=scope)

			self.decoder_logits_train = tf.contrib.layers.linear(self.decoder_outputs_train, self.config.vocab_size, scope=scope)
			self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_prediction_train')

			# self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_prediction_train')

			scope.reuse_variables()

			self.decoder_logits_inference, self.decoder_state_inference, \
			self.decoder_context_state_inference = seq2seq.dynamic_rnn_decoder(cell=self.cell,
					decoder_fn=decoder_fn_inference, time_major=True, scope=scope)


			self.decoder_prediction_inference = tf.argmax(self.decoder_logits_inference, axis=-1, name='decoder_prediction_inference')

			print("Built the Seq2Seq RNN Model...")


	def train(self, op='adam', max_norm=5):
		logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
		targets = tf.transpose(self.decoder_train_targets, [1, 0])
		# print self.decoder_logits_train.get_shape().as_list()
		# print self.decoder_train_targets.get_shape().as_list()
		# print logits.get_shape().as_list()
		# print targets.get_shape().as_list()

		self.loss_op = seq2seq.sequence_loss(logits=logits, targets=targets,
										  weights=self.loss_weights)
		tf.summary.scalar('Loss', self.loss_op)
		self.train_op = tf.train.AdamOptimizer().minimize(self.loss_op)
		print("Setup the training mechanism for the Seq2Seq RNN Model...")


	def metrics(self):
		# Same function, did not make a general one b/c need to store _ops within class
		difference = self.decoder_train_targets - tf.cast(self.decoder_prediction_train, tf.int32)
		zero = tf.constant(0, dtype=tf.int32)
		boolean_difference = tf.cast(tf.equal(difference, zero), tf.float64)
		self.accuracy_op = tf.reduce_mean(boolean_difference)
		tf.summary.scalar('Accuracy', self.accuracy_op)

		self.summary_op = tf.summary.merge_all()

		if SHERLOCK:
			self.confusion_matrix = tf_confusion_matrix(tf.reshape(self.label_placeholder, [-1]), tf.reshape(self.decoder_prediction_train, [-1]), num_classes=self.config.vocab_size, dtype=tf.int32)
		else:
			self.confusion_matrix = tf.confusion_matrix(tf.reshape(self.label_placeholder, [-1]), tf.reshape(self.decoder_prediction_train, [-1]), num_classes=self.config.vocab_size, dtype=tf.int32)


	def _feed_dict(self, feed_values):
		input_batch = feed_values[0]
		label_batch = feed_values[1]
		meta_batch = feed_values[2]
		initial_state_batch = feed_values[3]
		use_meta_batch = feed_values[4]
		num_encode = feed_values[5]
		num_decode = feed_values[6]

		feed_dict = {
			self.input_placeholder: input_batch,
			self.label_placeholder: label_batch,
			self.meta_placeholder: meta_batch,
			self.initial_state_placeholder: initial_state_batch,
			self.use_meta_placeholder: use_meta_batch,
			self.num_encode: num_encode,
			self.num_decode: num_decode
		}

		return feed_dict


	def run(self, args, session, feed_values):
		feed_dict = self._feed_dict(feed_values)

		if args.train == "train":
			_, summary, loss, prediction, accuracy, confusion_matrix = session.run([self.train_op, self.summary_op, self.loss_op, self.decoder_prediction_train, self.accuracy_op, self.confusion_matrix], feed_dict=feed_dict)
		else: # Sample case not necessary b/c function will only be called during normal runs
			summary, loss, prediction, accuracy, confusion_matrix = session.run([self.summary_op, self.loss_op, self.decoder_prediction_train, self.accuracy_op, self.confusion_matrix], feed_dict=feed_dict)

		print "Average accuracy per batch {0}".format(accuracy)
		print "Batch Loss: {0}".format(loss)
		# print "Output Predictions: {0}".format(prediction)
		# print "Output Prediction Probabilities: {0}".format(probabilities)

		return summary, confusion_matrix, accuracy


	def sample(self, session, feed_values):
		feed_dict = self._feed_dict(feed_values)

		logits = tf.transpose(self.decoder_logits_inference, [1, 0, 2])
		# self.logits_op = tf.nn.softmax(logits)
		predictions = tf.argmax(tf.nn.softmax(logits), axis=-1)
		# logits, state = session.run([self.logits_op, self.self.decoder_context_state_inference], feed_dict=feed_dict)
		pred = session.run(predictions, feed_dict=feed_dict)
		# return logits, state
		return pred





class Discriminator(object):

	def __init__(self, inputs, labels_size, is_training, batch_size, hyperparam_path, use_lrelu=True, use_batchnorm=False, dropout=None, reuse=True):
		self.input = inputs
		self.labels_size = labels_size
		self.batch_size = batch_size
		self.is_training = is_training
		self.reuse = reuse
		self.dropout = dropout
		self.use_batchnorm = use_batchnorm
		self.use_lrelu = use_lrelu
		self.config = Config(hyperparam_path)

	def lrelu(self, x, leak=0.2, name='lrelu'):
		return tf.maximum(x, leak*x)

	def conv_layer(self, inputs, filterSz, strideSz, padding, use_lrelu, use_batchnorm, scope):
		filterWeights = tf.Variable(tf.random_normal(filterSz))
		l1 = tf.nn.conv2d(inputs,filterWeights,strideSz,padding='SAME')
		if use_batchnorm:
			l1 = tf.contrib.layers.batch_norm(l1, decay=0.9,center=True,scale=True,
						epsilon=1e-8,is_training=is_training, reuse=self.reuse, trainable=True, scope=scope)

		if use_lrelu:
			l2 = self.lrelu(l1)
		else:
			l2 = tf.nn.relu(l1)

		if self.dropout is not None and self.is_training == True:
			l2 = tf.nn.dropout(l2, self.dropout)

		return l2


	def create_model(self):
		with tf.variable_scope("discriminator") as scope:
			if self.reuse:
				scope.reuse_variables()


			filterSz1 = [3, self.config.embedding_dims,1, self.config.numFilters]
			strideSz1 = [1,1,1,1]

			conv_layer1 = self.conv_layer(self.input,filterSz1,strideSz1, padding='SAME', use_lrelu=True, use_batchnorm=False, scope=scope)

			filterSz2 = [3, 1, self.config.numFilters, self.config.numFilters]
			strideSz2 = [1,1,1,1]
			conv_layer2 = self.conv_layer(conv_layer1,filterSz2,strideSz2,padding='SAME',use_lrelu=True, use_batchnorm=False, scope=scope)

			win_size = [1,3,1,1]
			strideSz3 = [1,1,1,1]
			conv_layer3 = tf.nn.max_pool(conv_layer2,ksize=win_size,strides=strideSz3, padding='SAME')

			layerShape = conv_layer3.get_shape().as_list()
			numParams = reduce(lambda x, y: x*y, layerShape[1:])

			layer_flatten = tf.reshape(conv_layer3, [-1, numParams])

			fully_conn_weights_1 = tf.get_variable("weights_fully_conn_1", [numParams, self.config.hidden_units],
										initializer=tf.random_normal_initializer())
			fully_conn_bias_1 = tf.get_variable("bias_fully_conn_1", [self.config.hidden_units,],
										initializer=tf.random_normal_initializer())
			layer4 = tf.matmul(layer_flatten,fully_conn_weights_1 ) + fully_conn_bias_1

			if self.dropout is not None and self.is_training == True:
				layer4 = tf.nn.dropout(layer4, self.dropout)

			fully_conn_weights_2 = tf.get_variable("weights_fully_conn_2", [self.config.hidden_units,self.labels_size ],
										initializer=tf.random_normal_initializer())
			fully_conn_bias_2 = tf.get_variable("bias_fully_conn_2", [self.labels_size,],
										initializer=tf.random_normal_initializer())
			layer5 = tf.matmul(layer4,fully_conn_weights_2 ) + fully_conn_bias_2

			self.output = layer5
			self.pred = tf.nn.softmax(layer5)

			return self.pred


	# def train(self, op='adam'):
	# 	self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output,
	# 					 labels=self.labels))


	# 	train_op = tf.train.AdamOptimizer(self.config.cnn_lr).minimize(self.loss)
	# 	return train_op



class GenAdversarialNet(object):


	def __init__(self, input_size, label_size ,num_classes, cell_type, is_training, batch_size, vocab_size,
					 hyperparam_path, use_lrelu=True, use_batchnorm=False, dropout=None):
		self.input_size = input_size
		self.label_size = label_size
		self.is_training = is_training
		self.cell_type = cell_type
		self.hyperparam_path = hyperparam_path
		self.use_lrelu = use_lrelu
		self.use_batchnorm = use_batchnorm
		self.dropout = dropout
		self.config = Config(hyperparam_path)
		self.config.batch_size = batch_size
		self.config.vocab_size = vocab_size
		self.config.num_classes = num_classes

		output_shape = (None,)
		self.label_placeholder = tf.placeholder(tf.int32, shape=output_shape, name='Output')

		input_shape = (None,) + tuple([self.input_size])
		self.input_placeholder = tf.placeholder(tf.int32, shape=input_shape, name='Input')

		print "Completed Initializing the GAN Model using a {0} cell".format(cell_type.upper())


	# Function taken from Goodfellow's Codebase on Training of GANs: https://github.com/openai/improved-gan/
	def sigmoid_kl_with_logits(self, logits, targets):
	# broadcasts the same target value across the whole batch
	# this is implemented so awkwardly because tensorflow lacks an x log x op
		if targets in [0., 1.]:
			entropy = 0.
		else:
			entropy = - targets * np.log(targets) - (1. - targets) * np.log(1. - targets)
		return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits) * targets) - entropy

	# Ideas for function taken from Goodfellow's Codebase on Training of GANs: https://github.com/openai/improved-gan/
	def normalize_class_outputs(self,logits):
		generated_class_logits = tf.squeeze(tf.slice(logits, [0, self.config.num_classes - 1], [self.config.batch_size/2, 1]))
		positive_class_logits = tf.slice(logits, [0, 0], [self.config.batch_size/2, self.config.num_classes - 1])
		mx = tf.reduce_max(positive_class_logits, 1, keep_dims=True)
		safe_pos_class_logits = positive_class_logits - mx

		gan_logits = tf.log(tf.reduce_sum(tf.exp(safe_pos_class_logits), 1)) + tf.squeeze(mx) - generated_class_logits
		return gan_logits


	def create_model(self):
		generator_inputs = tf.slice(self.input_placeholder, [0,0], [self.config.batch_size/2,self.input_size ])

		self.generator_model = CharRNN(self.input_size, self.label_size, self.config.batch_size/2 ,self.config.vocab_size,
								self.cell_type, self.hyperparam_path, gan_inputs = generator_inputs)
		self.generator_model.create_model(is_train = True)
		self.generator_model.train()
		self.generator_output = self.generator_model.logits_op

		# Sample the output of the GAN to find the correct prediction of each character
		generator_samples = []
		for i in xrange(self.input_size):
			generator_samples.append(tf.multinomial(self.generator_output[:,i,:], num_samples=1))

		self.current_policy = tf.stack(generator_samples, axis=1)

		# Create the Discriminator embeddings and sample the Generator output and Real input from these embeddings
		real_inputs = tf.slice(self.input_placeholder, [self.config.batch_size/2,0], [self.config.batch_size/2,self.input_size ])
		self.embeddings_disc = tf.Variable(tf.random_uniform([self.config.vocab_size, self.config.embedding_dims],
										 0, 10, dtype=tf.float32, seed=3), name='char_embeddings')
		embeddings_generator_out = tf.nn.embedding_lookup(self.embeddings_disc, self.current_policy)
		embeddings_real_input = tf.nn.embedding_lookup(self.embeddings_disc, real_inputs)

		# Inputs the fake examples from the CharRNN to the CNN Discriminator
		embeddings_generator_out = tf.expand_dims(embeddings_generator_out[:,:,0,:], -1)
		self.discriminator_gen_model = Discriminator(embeddings_generator_out, self.config.num_classes, is_training=self.is_training,
				 batch_size=self.config.batch_size/2, hyperparam_path=self.hyperparam_path, use_lrelu=self.use_lrelu, use_batchnorm=self.use_batchnorm,
				 dropout=self.dropout, reuse=False)
		discriminator_gen_pred = self.discriminator_gen_model.create_model()

		# Inputs the real sequences from the text files to the CNN Discriminator
		embeddings_real_input = tf.expand_dims(embeddings_real_input, -1)
		self.discriminator_real_samp = Discriminator(embeddings_real_input,  self.config.num_classes, is_training=self.is_training,
				 batch_size=self.config.batch_size/2, hyperparam_path=self.hyperparam_path, use_lrelu=self.use_lrelu, use_batchnorm=self.use_batchnorm,
				 dropout=self.dropout, reuse=True)
		discriminator_real_pred = self.discriminator_real_samp.create_model()


		# Collecting outputs and finding losses
		self.gan_real_output = self.discriminator_real_samp.output
		self.gan_fake_output = self.discriminator_gen_model.output

		self.gan_logits_real = self.normalize_class_outputs(self.gan_real_output)
		self.gan_logits_fake = self.normalize_class_outputs(self.gan_fake_output)

		self.gan_pred_real = self.sigmoid_kl_with_logits(self.gan_logits_real, 1. - self.config.label_smooth)
		self.gan_pred_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.gan_logits_fake,
								labels=tf.zeros_like(self.gan_logits_fake))

		print("Built the GAN Model...")
		return self.gan_pred_real, self.gan_pred_fake



	def train(self):
		class_loss_weight = 1

		self.gan_logits = tf.concat([self.gan_real_output, self.gan_fake_output],axis=0)

		loss_class = tf.reduce_mean(class_loss_weight*tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.gan_logits,
					labels=self.label_placeholder))

		tot_d_loss = tf.reduce_mean(self.gan_pred_real + self.gan_pred_fake) +  loss_class
		# tot_g_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.generator_output, 
		# 					labels=self.label_placeholder[self.config.batch_size/2:]))

		fool_examples = tf.cast(tf.equal(tf.argmax(self.gan_fake_output, axis=-1), self.config.num_classes-1), tf.float32)*(-1)
		real_examples = tf.cast(tf.not_equal(tf.argmax(self.gan_fake_output, axis=-1), self.config.num_classes-1), tf.float32)

		combined_labels = fool_examples + real_examples

		combined_labels = tf.expand_dims(combined_labels, -1)
		combined_labels = tf.expand_dims(combined_labels, -1)
		prob_grads = tf.multiply(combined_labels, tf.nn.softmax(self.generator_output))

		self.d_gen_grad = tf.gradients(tot_d_loss, self.embeddings_disc)
		self.train_op_d = tf.train.AdamOptimizer(self.config.gan_lr).apply_gradients(zip(self.d_gen_grad, [self.embeddings_disc]))
		self.train_op_gan = tf.train.AdamOptimizer(self.config.gan_lr).minimize(prob_grads)

		print "Completed setup of training mechanism for the GAN...."
		return self.generator_model.input_placeholder, self.generator_model.label_placeholder, \
				self.generator_model.meta_placeholder, self.generator_model.initial_state_placeholder, \
				 self.generator_model.use_meta_placeholder, self.train_op_d, self.train_op_gan
