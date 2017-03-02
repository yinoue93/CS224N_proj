import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
import numpy as np
import sys
import os
import logging



class CharRNN(model):

	def __init__(self,input_size, label_size, batch_size, vocab_size, cell_type, num_layers, max_length ,dropout, lr):
		self.batch_size = batch_size
		self.lr = lr
		self.max_length = max_length
		self.num_layers = num_layers
		self.num_epochs = num_epochs
		self.cell_type = cell_type
		self.vocab_size = vocab_size
		self.keep_prob = dropout
		if cell_type == 'rnn':
			self.cell = rnn.BasicRNNCell
		elif cell_type == 'gru':
			self.cell = rnn.GRUCell
		elif cell_type == 'lstm':
			self.cell = rnn.BasicLSTMCell

		input_shape = (None,) + tuple([max_length,input_size])
		output_shape = (None,) + tuple([max_length,label_size])
		self.input_placeholder = tf.placeholder(tf.float32, shape=input_shape, name='Input')
		self.label_placeholder = tf.placeholder(tf.float32, shape=output_shape, name='Output')
		return self.input_placeholder, self.label_placeholder


	def create_model(self):
		 # with tf.variable_scope(self.cell_type):
	 	first_layer = self.cell(self.max_length, state_is_tuple=True)
	 	dropout_layer = tf.nn.rnn_cell.DropoutWrapper(first_layer, output_keep_prob=self.keep_prob, 
	 						input_keep_prob=1.0, output_keep_prob=1.0)
	 	rnn_model = rnn.MultiRNNCell([dropout_layer]*self.num_layers, state_is_tuple=True)
	 	self.initial_state = rnn_model.zero_state(self.batch_size, tf.float32)

	 	num_dims = self.vocab_size/2
	 	embeddings_var = tf.Variable(tf.random_uniform([num_dims, self.vocab_size], 0.1, 10, dtype=tf.float32, seed=3), name='char_embeddings')
	 	embeddings = tf.nn.embedding_lookup(embeddings_var, self.input_placeholder)

	 	output, state = tf.nn.dynamic_rnn(rnn_model, embeddings, dtype=tf.float32)
	 	self.pred = output
	 	return output, state, self.initial_state



	def train(self, op='adam'):
		loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(preds, self.label_placeholder))
		train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
		return train_op



# class Seq2SeqRNN(model):



















	
