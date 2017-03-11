import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
import numpy as np
import sys
import os
import logging

import utils_hyperparam


class Config(object):

	def __init__(self, set_hyperparam):
		self.batch_size = 100
		self.lr = 0.001

		self.songtype = 20
		self.sign = 16
		self.notesize = 3
		self.flats = 11
		self.mode = 6
		self.len = 1
		self.complex = 1
		self.max_length = 8

		self.vocab_size = 124
		# self.meta_embed = self.songtype
		self.meta_embed = 100 #self.songtype
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

		if set_hyperparam:
			print "Running in development mode"
			utils_hyperparam.setHyperparam(self)


class CBOW(object):

	def __init__(self, input_size, label_size):
		self.config = Config()
		self.input_size = input_size
		self.label_size = label_size
		self.input_placeholder = tf.placeholder(tf.int32, shape=[None, self.input_size])
		self.label_placeholder = tf.placeholder(tf.int32, shape=[None, self.label_size])
		self.embeddings = tf.Variable(tf.random_uniform([self.config.vocab_size,
								self.config.embed_size], -1.0, 1.0))

		print("Completed Initializing the CBOW Model.....")


	def create_model(self):
		weight = tf.get_variable("Wout", shape=[self.config.embed_size, self.config.vocab_size],
					initializer=tf.contrib.layers.xavier_initializer())
		bias = tf.Variable(tf.zeros([self.config.vocab_size]))

		word_vec =  tf.nn.embedding_lookup(self.embeddings, self.input_placeholder)
		average_embedding = tf.reduce_sum(word_vec, reduction_indices=1)
		model_output = tf.add(tf.matmul(average_embedding, weight), bias)
		self.pred = model_output
		print("Built the CBOW Model.....")

		return model_output

	def train(self):
		self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred, labels=self.label_placeholder))
		self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)

		print("Setup the training mechanism for the CBOW model.....")

		return self.input_placeholder, self.label_placeholder, self.train_op, self.loss



class CharRNN(object):

	def __init__(self, input_size, label_size, batch_size, vocab_size, cell_type, set_hyperparam):
		self.input_size = input_size
		self.label_size = label_size
		self.cell_type = cell_type
		self.config = Config(set_hyperparam)
		self.config.batch_size = batch_size
		self.config.vocab_size = vocab_size

		if cell_type == 'rnn':
			self.cell = rnn.BasicRNNCell(self.config.hidden_size)
		elif cell_type == 'gru':
			self.cell = rnn.GRUCell(self.config.hidden_size)
		elif cell_type == 'lstm':
			self.cell = rnn.BasicLSTMCell(self.config.hidden_size)

		# self.initial_state = self.cell.zero_state(self.config.batch_size, dtype=tf.int32)
		# input_shape = (None,) + tuple([self.config.max_length,input_size])
		input_shape = (None,) + tuple([input_size])
		# output_shape = (None,) + tuple([self.config.max_length,label_size])
		output_shape = (None,) + tuple([label_size])
		self.input_placeholder = tf.placeholder(tf.int32, shape=input_shape, name='Input')
		self.label_placeholder = tf.placeholder(tf.int32, shape=output_shape, name='Output')
		self.meta_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.num_meta], name='Meta')
		self.initial_state_placeholder = tf.placeholder(tf.float32, shape=[None, self.config.hidden_size], name="Initial_State")
		self.use_meta_placeholder = tf.placeholder(tf.bool, name='State_Initialization_Bool')

		print "Completed Initializing the Char RNN Model using a {0} cell".format(cell_type.upper())


	def create_model(self, is_train=True):
		if is_train:
			self.cell = rnn.DropoutWrapper(self.cell, input_keep_prob=1.0, output_keep_prob=1.0)
		rnn_model = rnn.MultiRNNCell([self.cell]*self.config.num_layers, state_is_tuple=True)

		# Embedding lookup for ABC format characters
		embeddings_var = tf.Variable(tf.random_uniform([self.config.vocab_size, self.config.embedding_dims],
										 0, 10, dtype=tf.float32, seed=3), name='char_embeddings')
		embeddings = tf.nn.embedding_lookup(embeddings_var, self.input_placeholder)

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

		initial_added = tf.cond(self.use_meta_placeholder,
								lambda: embeddings_meta,
								lambda: self.initial_state_placeholder)

		if self.cell_type == 'lstm':
			initial_tuple = tuple([rnn.LSTMStateTuple(initial_added, np.zeros((self.config.batch_size, self.config.hidden_size), dtype=np.float32)) for idx in xrange(self.config.num_layers)])
		else:
			initial_tuple = (initial_added, np.zeros((self.config.batch_size, self.config.hidden_size), dtype=np.float32))
		rnn_output, state = tf.nn.dynamic_rnn(rnn_model, embeddings, dtype=tf.float32, initial_state=initial_tuple)

		decode_var = tf.Variable(tf.random_uniform([self.config.hidden_size, self.config.vocab_size],
										 0, 10, dtype=tf.float32, seed=3), name='char_decode')
		decode_bias = tf.Variable(tf.random_uniform([self.config.vocab_size],
										 0, 10, dtype=tf.float32, seed=3), name='char_decode_bias')
		decode_list = []
		for i in xrange(self.input_size):
			decode_list.append(tf.matmul(rnn_output[:, i, :], decode_var) + decode_bias)
		self.output = tf.stack(decode_list, axis=1)
		self.pred = tf.nn.softmax(self.output)

		print("Built the Char RNN Model...")

		return self.pred, state



	def train(self, max_norm=5, op='adam'):
		self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.label_placeholder))
		tvars = tf.trainable_variables()

		# Gradient clipping
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),max_norm)
		optimizer = tf.train.AdamOptimizer(self.config.lr)
		self.train_op = optimizer.apply_gradients(zip(grads, tvars))

		print("Setup the training mechanism for the Char RNN Model...")

		return self.input_placeholder, self.label_placeholder, self.meta_placeholder, self.initial_state_placeholder, self.use_meta_placeholder, self.train_op, self.loss



# class Seq2SeqRNN(object):

# 	def __init__(self,input_size, label_size, cell_type):
# 		self.cell_type = cell_type
# 		self.config = Config()

# 		if cell_type == 'rnn':
# 			self.cell = rnn.BasicRNNCell(self.config.hidden_size)
# 			self.initial_state = self.cell.zero_state(batch_size)
# 		elif cell_type == 'gru':
# 			self.cell = rnn.GRUCell(self.config.hidden_size)
# 			self.initial_state = self.cell.zero_state(batch_size)
# 		elif cell_type == 'lstm':
# 			self.cell = rnn.BasicLSTMCell(self.config.hidden_size)
# 			self.initial_state = self.cell.zero_state(batch_size)

# 		input_shape = (None,) + tuple([max_length,input_size])
# 		output_shape = (None,) + tuple([max_length,label_size])
# 		self.input_placeholder = tf.placeholder(tf.float32, shape=input_shape, name='Input')
# 		self.label_placeholder = tf.placeholder(tf.float32, shape=output_shape, name='Output')

# 		# Seq2Seq specific initializers
# 		self.num_encode = num_encode
# 		self.num_decode = num_decode
# 		self.num_meta = num_meta
# 		self.attention_option = "luong"

# 		return self.input_placeholder, self.label_placeholder, self.initial_state


# 	def create_model(self, is_train):

# 	 	if is_train:
# 	 		self.cell = rnn.DropoutWrapper(self.cell, output_keep_prob=self.config.keep_prob,
# 	 						input_keep_prob=1.0, output_keep_prob=1.0)
# 	 	rnn_model = rnn.MultiRNNCell([self.cell]*self.config.num_layers, state_is_tuple=True)

# 	 	# Embedding lookup for ABC format characters
# 	 	num_dims = self.config.vocab_size/2
# 	 	embeddings_var = tf.Variable(tf.random_uniform([self.config.batch_size, num_dims, self.config.vocab_size],
# 	 									 0, 10, dtype=tf.float32, seed=3), name='char_embeddings')
# 	 	embeddings_char = tf.nn.embedding_lookup(embeddings_var, self.input_placeholder)

# 	 	# Embedding lookup for Metadata
# 	 	num_dims_meta = self.meta_size/2
# 	 	embeddings_var_meta = tf.Variable(tf.random_uniform([self.config.batch_size, num_dims_meta, self.config.meta_size],
# 	 									 0, 10, dtype=tf.float32, seed=3), name='char_embeddings_meta')
# 	 	embeddings_meta = tf.nn.embedding_lookup(embeddings_var_meta, self.initial_state)

# 	 	# Unrolling the timesteps of the RNN
# 	 	encoder_outputs, encoder_state = rnn.dynamic_rnn( cell=rnn_model ,inputs=embeddings_char,
# 	 								 dtype=tf.float32, time_major=True)

# 	 	# Prepare Attention mechanism
# 	 	attention_keys, attention_values, attention_score_fn, \
# 	 					 attention_construct_fn = seq2seq.prepare_attention(encoder_outputs,
# 	 					 				self.config.attention_option, self.config.num_decode)

# 	 	# Training mechanism of decoder and attention mechanism
# 	 	if is_train:
# 		 	decoder_fn_train = seq2seq.attention_decoder_fn_train(encoder_state=encoder_state,
# 	              					attention_keys=attention_keys,
# 	              					attention_values=attention_values,
# 	              					attention_score_fn=attention_score_fn,
# 	              					attention_construct_fn=attention_construct_fn)


# 		 	decoder_outputs_train, decoder_state_train, _ = seq2seq.dynamic_rnn_decoder(
# 								            cell=, decoder_fn=decoder_fn_train,
# 								            inputs=decoder_inputs, sequence_length=decoder_length,
# 								            time_major=True)
# 		 else:
# 		 	decoder_fn_inference = attention_decoder_fn.attention_decoder_fn_inference(
#                   		output_fn=output_fn, encoder_state=encoder_state, attention_keys=attention_keys,
#                   		attention_values=attention_values, attention_score_fn=attention_score_fn,
#                   		attention_construct_fn=attention_construct_fn, embeddings=decoder_embeddings,
#                   		start_of_sequence_id=start_of_sequence_id, end_of_sequence_id=end_of_sequence_id,
#                   		maximum_length=decoder_sequence_length - 1,
#                   		num_decoder_symbols=num_decoder_symbols, dtype=dtypes.int32)

#           	decoder_outputs_inference, decoder_state_inference, _ = seq2seq.dynamic_rnn_decoder(
#                   					cell=decoder_cell, decoder_fn=decoder_fn_inference,
#                   					time_major=True)



	# def train(self, op='adam', max_norm):



# class Discriminator(object):

# 	def __init__(self, inputs, labels, is_training, batch_size,use_lrelu=True, use_batchnorm=False, dropout=None, reuse=True):
# 		self.input = inputs
# 		self.labels = labels
# 		self.batch_size = batch_size
# 		self.is_training = is_training
# 		self.reuse = reuse
# 		self.dropout = dropout
# 		self.use_batchnorm = use_batchnorm
# 		self.use_lrelu = use_lrelu
# 		self.config = Config()

# 	def lrelu(self, x, leak=0.2, name='lrelu'):
# 		return tf.maximum(x, leak*x)

# 	def conv_layer(self, inputs, filterSz1, strideSz1, scope):
# 		l1 = tf.nn.conv2d(inputs,filterSz1,strideSz1,padding='SAME')
# 		if use_batchnorm:
# 			l1 = tf.contrib.layers.batch_norm(l1, decay=0.9,center=True,scale=True,
# 						epsilon=1e-8,is_training=is_training, reuse=self.reuse, trainable=True, scope=scope)

# 		if use_lrelu:
# 			l2 = self.lrelu(l1)
# 		else:
# 			l2 = tf.nn.relu(l1)

# 		if self.dropout is not None and self.is_training == True:
# 			l2 = tf.nn.dropout(l2, self.dropout)

# 		return l2


# 	def create_model(self):
# 		with tf.variable_scope("discriminator") as scope:
# 			if self.reuse:
# 				scope.reuse_variables()


# 			filterSz1 = [3, self.config.embedding_dims,1, self.config.numFilters1]
# 			strideSz1 = [1,1,1,1]
# 			conv_layer1 = self.conv_layer(self.input,filterSz1,strideSz1,padding='SAME', scope)

# 			filterSz2 = [3, 1, self.config.numFilters, self.config.numFilters]
# 			strideSz2 = [1,1,1,1]
# 			conv_layer2 = self.conv_layer(conv_layer1,filterSz2,strideSz2,padding='SAME', scope)

# 			win_size = [1,3,1,1]
# 			strideSz3 = [1,1,1,1]
# 			conv_layer3 = tf.nn.max_pool(conv_layer2,ksize=win_size,strides=strideSz3, padding='SAME')

# 			layerShape = conv_layer3.get_shape().as_list()
# 			numParams = reduce(lambda x, y: x*y, layerShape[1:])

# 			layer_flatten = tf.reshape(conv_layer3, [-1, numParams])

# 			layer4 = tf.contrib.layers.fully_connected(layer_flatten, num_outputs=self.config.hidden_units,
# 						reuse=self.reuse,trainable=True, scope=scope)

# 			if self.dropout is not None and self.is_training == True:
# 				layer4 = tf.nn.dropout(layer4, self.dropout)


# 			layer5 = tf.contrib.layers.fully_connected(layer4, num_outputs=self.config.num_outputs,
# 						reuse=self.reuse,trainable=True, scope=scope)

# 			self.output = layer5
# 			self.pred = tf.nn.softmax(layer5)

# 			return self.pred
			


# 	def train(self, op='adam'):
# 		self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output,
# 						 labels=self.labels))


# 		train_op = tf.train.AdamOptimizer(self.config.cnn_lr).minimize(self.loss)
# 		return train_op



# class GenAdversarialNet(object):
