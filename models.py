import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
import numpy as np
import sys
import os
import logging


# class Config(object):

# 	def __init__(self):
# 		self.batch_size = 32
# 		self.lr = 
# 		self.hidden_size = 
# 		self.num_layers = 
# 		self.num_epochs = 
# 		self.vocab_size = 
# 		self.meta_size = 
# 		self.keep_prob = 

# 		# Only for Seq2Seq Attention Models
# 		self.num_encode = 
# 		self.num_decode = 



class CharRNN(object):

	def __init__(self,input_size, label_size, cell_type):
		self.cell_type = cell_type
		self.config = Config()

		if cell_type == 'rnn':
			self.cell = rnn.BasicRNNCell(self.config.hidden_size, state_is_tuple=True)
			self.initial_state = self.cell.zero_state(batch_size)
		elif cell_type == 'gru':
			self.cell = rnn.GRUCell(self.config.hidden_size)
			self.initial_state = self.cell.zero_state(batch_size, state_is_tuple=True)
		elif cell_type == 'lstm':
			self.cell = rnn.BasicLSTMCell(self.config.hidden_size, state_is_tuple=True)
			self.initial_state = self.cell.zero_state(batch_size)

		input_shape = (None,) + tuple([max_length,input_size])
		output_shape = (None,) + tuple([max_length,label_size])
		self.input_placeholder = tf.placeholder(tf.int32, shape=input_shape, name='Input')
		self.label_placeholder = tf.placeholder(tf.int32, shape=output_shape, name='Output')
		return self.input_placeholder, self.label_placeholder, self.initial_state


	def create_model(self, is_train):
		 # with tf.variable_scope(self.cell_type):
	 	if is_train:
	 		self.cell = rnn.DropoutWrapper(self.cell, output_keep_prob=self.config.keep_prob, 
	 						input_keep_prob=1.0, output_keep_prob=1.0)
	 	rnn_model = rnn.MultiRNNCell([self.cell]*self.config.num_layers, state_is_tuple=True)

	 	# Embedding lookup for ABC format characters
	 	num_dims = self.config.vocab_size/2
	 	embeddings_var = tf.Variable(tf.random_uniform([self.config.batch_size,num_dims, self.config.vocab_size], 0, 10, dtype=tf.float32, seed=3), name='char_embeddings')
	 	embeddings = tf.nn.embedding_lookup(embeddings_var, self.input_placeholder)

	 	# Embedding lookup for Metadata
	 	num_dims_meta = self.config.meta_size/2
	 	embeddings_var_meta = tf.Variable(tf.random_uniform([self.config.batch_size, num_dims_meta, self.config.meta_size],
	 								 0, 10, dtype=tf.float32, seed=3), name='char_embeddings_meta')
	 	embeddings_meta = tf.nn.embedding_lookup(embeddings_var_meta, self.initial_state)


	 	output, state = tf.nn.dynamic_rnn(rnn_model, embeddings, dtype=tf.float32, initial_state=embeddings_meta)
	 	self.pred = output
	 	return output, state



	def train(self, op='adam', max_norm):
		loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(preds, self.label_placeholder))

		tvars = tf.trainable_variables()
		# Gradient clipping
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),max_norm)
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

		return self.train_op



class Seq2SeqRNN(object):

	def __init__(self,input_size, label_size, cell_type):
		self.cell_type = cell_type
		self.config = Config()

		if cell_type == 'rnn':
			self.cell = rnn.BasicRNNCell(self.config.hidden_size)
			self.initial_state = self.cell.zero_state(batch_size)
		elif cell_type == 'gru':
			self.cell = rnn.GRUCell(self.config.hidden_size)
			self.initial_state = self.cell.zero_state(batch_size)
		elif cell_type == 'lstm':
			self.cell = rnn.BasicLSTMCell(self.config.hidden_size)
			self.initial_state = self.cell.zero_state(batch_size)

		input_shape = (None,) + tuple([max_length,input_size])
		output_shape = (None,) + tuple([max_length,label_size])
		self.input_placeholder = tf.placeholder(tf.float32, shape=input_shape, name='Input')
		self.label_placeholder = tf.placeholder(tf.float32, shape=output_shape, name='Output')

		# Seq2Seq specific initializers
		self.num_encode = num_encode
		self.num_decode = num_decode
		self.num_meta = num_meta
		self.attention_option = "luong"

		return self.input_placeholder, self.label_placeholder, self.initial_state


	def create_model(self, is_train):

	 	if is_train:
	 		self.cell = rnn.DropoutWrapper(self.cell, output_keep_prob=self.config.keep_prob, 
	 						input_keep_prob=1.0, output_keep_prob=1.0)
	 	rnn_model = rnn.MultiRNNCell([self.cell]*self.config.num_layers, state_is_tuple=True)

	 	# Embedding lookup for ABC format characters
	 	num_dims = self.config.vocab_size/2
	 	embeddings_var = tf.Variable(tf.random_uniform([self.config.batch_size, num_dims, self.config.vocab_size],
	 									 0, 10, dtype=tf.float32, seed=3), name='char_embeddings')
	 	embeddings_char = tf.nn.embedding_lookup(embeddings_var, self.input_placeholder)

	 	# Embedding lookup for Metadata
	 	num_dims_meta = self.meta_size/2
	 	embeddings_var_meta = tf.Variable(tf.random_uniform([self.config.batch_size, num_dims_meta, self.config.meta_size],
	 									 0, 10, dtype=tf.float32, seed=3), name='char_embeddings_meta')
	 	embeddings_meta = tf.nn.embedding_lookup(embeddings_var_meta, self.initial_state) 

	 	# Unrolling the timesteps of the RNN
	 	encoder_outputs, encoder_state = rnn.dynamic_rnn( cell=rnn_model ,inputs=embeddings_char,
	 								 dtype=tf.float32, time_major=True)

	 	# Prepare Attention mechanism
	 	attention_keys, attention_values, attention_score_fn, \
	 					 attention_construct_fn = seq2seq.prepare_attention(
               										attention_states, attention_option, decoder_hidden_size)

	 	# Training mechanism of decoder and attention mechanism
	 	if train:
		 	decoder_fn_train = seq2seq.attention_decoder_fn_train(encoder_state=encoder_state,
	              					attention_keys=attention_keys,
	              					attention_values=attention_values,
	              					attention_score_fn=attention_score_fn,
	              					attention_construct_fn=attention_construct_fn)

		 	
		 	decoder_outputs_train, decoder_state_train, _ = seq2seq.dynamic_rnn_decoder(
								            cell=, decoder_fn=decoder_fn_train,
								            inputs=decoder_inputs, sequence_length=decoder_length,
								            time_major=True)
		 else



	def train(self, op='adam', max_norm):








# class GenAdversarialNet(object):








 













	
