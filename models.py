import tensorflow as tf
tf_ver = tf.__version__
if str(tf_ver) != '0.12.1':
	from tensorflow.contrib import rnn
else:
	from tensorflow.python.ops import rnn_cell as rnn

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
		self.meta_embed = 1#00 #self.songtype/2
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

		# Discriminator Parameters
		self.numFilters = 32
		self.hidden_units = 100
		self.num_outputs = 2
		self.cnn_lr = 0.001
		self.label_smooth = 0.15
		self.generator_prob = 0.1
		self.num_classes = 5
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
		self.summary_op = tf.summary.merge_all()
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

		logits = session.run(self.logits_op, feed_dict=feed_dict)
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
		# print embeddings_meta.get_shape().as_list()
		# print embeddings.get_shape().as_list()
		# print self.input_placeholder.get_shape().as_list()

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
		self.probabilities_op = tf.nn.softmax(self.logits_op)

		print("Built the Char RNN Model...")


	def train(self, max_norm=5, op='adam'):
		self.loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_op, labels=self.label_placeholder))
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
		self.summary_op = tf.summary.merge_all()
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

	def __init__(self,input_size, label_size,batch_size, vocab_size, cell_type, hyperparam_path, start_encode, end_encode):
		self.input_size = input_size
		self.label_size = label_size
		self.cell_type = cell_type
		self.config = Config(hyperparam_path)
		self.config.batch_size = batch_size
		self.config.vocab_size = vocab_size

		input_shape = (None,) + tuple([input_size])
		output_shape = (None,) + tuple([label_size])

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

			self.decoder_train_inputs = self.label_placeholder[:,:self.input_size-1]
			self.decoder_train_targets = self.label_placeholder[:,1:]
			self.loss_weights = tf.ones([self.config.batch_size, self.input_size], dtype=tf.float32, name="loss_weights")
			sqrt3 = math.sqrt(3)
			initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

			# Creating the embeddings and deriving the embeddings for the encoder and decoder
			self.embedding_matrix = tf.get_variable(name="embedding_matrix",
			    shape=[self.config.vocab_size, self.config.embedding_dims], initializer=initializer,
			    dtype=tf.float32)

			self.encoder_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.input_placeholder)

			self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.decoder_train_inputs)

			self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(cell=self.cell, inputs=self.encoder_embedded,
			                          sequence_length=self.num_encode ,time_major=True, dtype=tf.float32)

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

			self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_prediction_train')

			scope.reuse_variables()

			self.decoder_logits_inference, self.decoder_state_inference, \
            self.decoder_context_state_inference = seq2seq.dynamic_rnn_decoder(cell=self.cell,
                    decoder_fn=decoder_fn_inference, time_major=True, scope=scope)


			self.decoder_prediction_inference = tf.argmax(self.decoder_logits_inference, axis=-1, name='decoder_prediction_inference')

			print("Built the Seq2Seq RNN Model...")
			return self.decoder_prediction_train, self.decoder_prediction_inference



	def train(self, op='adam', max_norm=5):
		logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
		targets = tf.transpose(self.decoder_train_targets, [1, 0])
		self.loss = seq2seq.sequence_loss(logits=logits, targets=targets,
		                                  weights=self.loss_weights)
		self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
		print("Setup the training mechanism for the Seq2Seq RNN Model...")

		return self.input_placeholder, self.label_placeholder, self.meta_placeholder, self.initial_state_placeholder,\
			self.use_meta_placeholder, self.num_encode, self.num_decode, self.train_op, self.loss



class Discriminator(object):

	def __init__(self, inputs, labels, is_training, batch_size, hyperparam_path, use_lrelu=True, use_batchnorm=False, dropout=None, reuse=True):
		self.input = inputs
		self.labels = labels
		self.batch_size = batch_size
		self.is_training = is_training
		self.reuse = reuse
		self.dropout = dropout
		self.use_batchnorm = use_batchnorm
		self.use_lrelu = use_lrelu
		self.config = Config(hyperparam_path)

	def lrelu(self, x, leak=0.2, name='lrelu'):
		return tf.maximum(x, leak*x)

	def conv_layer(self, inputs, filterSz1, strideSz1, scope):
		l1 = tf.nn.conv2d(inputs,filterSz1,strideSz1,padding='SAME')
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


			filterSz1 = [3, self.config.embedding_dims,1, self.config.numFilters1]
			strideSz1 = [1,1,1,1]
			conv_layer1 = self.conv_layer(self.input,filterSz1,strideSz1,padding='SAME', scope=scope)

			filterSz2 = [3, 1, self.config.numFilters, self.config.numFilters]
			strideSz2 = [1,1,1,1]
			conv_layer2 = self.conv_layer(conv_layer1,filterSz2,strideSz2,padding='SAME', scope=scope)

			win_size = [1,3,1,1]
			strideSz3 = [1,1,1,1]
			conv_layer3 = tf.nn.max_pool(conv_layer2,ksize=win_size,strides=strideSz3, padding='SAME')

			layerShape = conv_layer3.get_shape().as_list()
			numParams = reduce(lambda x, y: x*y, layerShape[1:])

			layer_flatten = tf.reshape(conv_layer3, [-1, numParams])

			layer4 = tf.contrib.layers.fully_connected(layer_flatten, num_outputs=self.config.hidden_units,
						reuse=self.reuse,trainable=True, scope=scope)

			if self.dropout is not None and self.is_training == True:
				layer4 = tf.nn.dropout(layer4, self.dropout)


			layer5 = tf.contrib.layers.fully_connected(layer4, num_outputs=self.config.num_outputs,
						reuse=self.reuse,trainable=True, scope=scope)

			self.output = layer5
			self.pred = tf.nn.softmax(layer5)

			return self.pred


	def train(self, op='adam'):
		self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output,
						 labels=self.labels))


		train_op = tf.train.AdamOptimizer(self.config.cnn_lr).minimize(self.loss)
		return train_op



class GenAdversarialNet(object):


	def __init__(self, input_size, label_size ,is_training, cell_type,
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

		output_shape = (None,) + tuple([self.label_size])
		self.label_placeholder = tf.placeholder(tf.float32, shape=output_shape, name='Output')

		input_shape = (None,) + tuple([self.input_size])
		self.input_placeholder = tf.placeholder(tf.float32, shape=input_shape, name='Input')


	# Function taken from Goodfellow's Codebase on Training of GANs: https://github.com/openai/improved-gan/
	def sigmoid_kl_with_logits(logits, targets):
    # broadcasts the same target value across the whole batch
    # this is implemented so awkwardly because tensorflow lacks an x log x op
	    if targets in [0., 1.]:
	        entropy = 0.
	    else:
	        entropy = - targets * np.log(targets) - (1. - targets) * np.log(1. - targets)
	    return tf.nn.sigmoid_cross_entropy_with_logits(logits, tf.ones_like(logits) * targets) - entropy

	# Ideas for function taken from Goodfellow's Codebase on Training of GANs: https://github.com/openai/improved-gan/
	def normalize_class_outputs(logits):
		generated_class_logits = tf.squeeze(tf.slice(logits, [0, self.config.num_classes - 1], [self.config.batch_size, 1]))
		positive_class_logits = tf.slice(logits, [0, 0], [self.config.batch_size, self.config.num_classes - 1])
		mx = tf.reduce_max(positive_class_logits, 1, keep_dims=True)
		safe_pos_class_logits = positive_class_logits - mx

		gan_logits = tf.log(tf.reduce_sum(tf.exp(safe_pos_class_logits), 1)) + tf.squeeze(mx) - generated_class_logits
		return gan_logits


	def create_model(self):
		generator_inputs = tf.slice(self.input_placeholder, [0,0], [self.config.batch_size/2,self.input_size ])

		generator_model = CharRNN(self.input_size, self.label_size, self.batch_size,self.vocab_size,
								self.cell_type, self.hyperparam_path, gan_inputs = self.input_placeholder)
		generator_model = generator_model.create_model(is_train = True)
		self.rnn_placeholder, self.rnn_label_placeholder, self.rnn_meta_placeholder, \
			self.rnn_initial_state_placeholder, self.rnn_use_meta_placeholder, \
			self.rnn_train_op, self.rnn_loss = generator_model.train()
		generator_output = generator_model.output

		# Inputs the fake examples from the CharRNN to the CNN Discriminator
		discriminator_model = Discriminator(generator_output, self.labels, is_training=self.is_training,
				 batch_size=self.batch_size, hyperparam_path=self.hyperparam_path, use_lrelu=self.use_lrelu, use_batchnorm=self.use_batchnorm,
				 dropout=self.dropout, reuse=False)
		discriminator_model = discriminator_model.create_model()
		self.discriminator_fake = discriminator_model.train()

		# Inputs the real sequences from the text files to the CNN Discriminator
		embeddings_real = tf.Variable(tf.random_uniform([self.config.vocab_size, self.config.embedding_dims],
										 0, 10, dtype=tf.float32, seed=3), name='char_embeddings')
		embeddings = tf.nn.embedding_lookup(embeddings_real, self.input_placeholder)
		discriminator_real_samp = Discriminator(embeddings, self.labels, is_training=self.is_training,
				 batch_size=self.batch_size,use_lrelu=self.use_lrelu, use_batchnorm=self.use_batchnorm,
				 dropout=self.dropout, reuse=True)
		discriminator_real_samp = discriminator_real_samp.create_model()
		self.discriminator_real = discriminator_real_samp.train()


		self.gan_real_output = self.discriminator_real.output
		self.gan_fake_output = self.discriminator_fake.output

		self.gan_logits_real = self.normalize_class_outputs(self.gan_real_output)
		self.gan_logits_fake = self.normalize_class_outputs(self.gan_fake_output)

		self.gan_pred_real = self.sigmoid_kl_with_logits(self.gan_logits_real, 1. - self.label_smooth)
		self.gan_pred_fake = tf.nn.sigmoid_cross_entropy_with_logits(self.gan_logits_fake,
            					tf.zeros_like(self.gan_logits_fake))

		return self.gan_pred_real, self.gan_pred_fake



	def train(self):
		class_loss_weight = 1

		self.gan_logits = tf.concat([self.gan_logits_real, self.gan_logits_fake],axis=0)
		loss_class = class_loss_weight*tf.nn.sparse_softmax_cross_entropy_with_logits(self.gan_logits,
            		self.label_placeholder)

		tot_d_loss = tf.reduce_mean(self.gan_pred_real + self.gan_pred_fake + loss_class)
		tot_g_loss = tf.reduce_mean(self.sigmoid_kl_with_logits(self.gan_pred_fake, self.config.generator_prob))

		self.train_op_d = tf.train.AdamOptimizer(self.config.gan_lr).minimize(tot_d_loss)
		self.train_op_gan = tf.train.AdamOptimizer(self.config.gan_lr).minimize(tot_g_loss)

		return self.input_placeholder, self.label_placeholder, \
				self.rnn_meta_placeholder, self.rnn_initial_state_placeholder, \
				 self.rnn_use_meta_placeholder, self.train_op_d, self.train_op_gan
