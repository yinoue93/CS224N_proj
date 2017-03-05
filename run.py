import tensorflow as tf
import numpy as np
import os
import sys
from argparse import ArgumentParser
from models import CharRNN
from utils_preprocess import hdf52dict




def run_model(train, datapath, batch_size, max_steps):
    pass
    # x, y = build_model() # placeholder variables


def main(args):
	input_size = 8
	initial_size = 7
	label_size = 1

	if args.model == 'seq2seq':
		curModel = Seq2SeqRNN(input_size, label_size, 'rnn')
	elif args.model == 'char':
		curModel = CharRNN(input_size, label_size, 'rnn')

	print("Reading in data from HDF5 File....")
	data_dict = hdf52dict('encoded_data.h5')

	output, state = curModel.build_model(is_train = args.train)
	input_placeholder, label_placeholder, initial_placeholder, train_op = curModel.train()

	



   

def parseCommandLine():
	desc = u'{0} [Args] [Options]\nDetailed options -h or --help'.format(__file__)
	parser = ArgumentParser(description=desc)

	print("Parsing Command Line Arguments...")
	requiredModel = parser.add_argument_group('Required Model arguments')
	requiredModel.add_argument('-m', choices = ["seq2seq", "char"], type = str, 
						dest = 'model', required = True, help = 'Type of model to run')
	requiredTrain = parser.add_argument_group('Required Train/Test arguments')
	requiredTrain.add_argument('-p', choices = ["train", "test"], type = str, 
						dest = 'train', required = True, help = 'Training or Testing phase to be run')

	args = parser.parse_args()
	return args


if __name__ == "__main__":
    args = parseCommandLine()
    main(args)
