import tensorflow as tf
import numpy as np
import os
import sys
from argparse import ArgumentParser
from models import CharRNN
# from utils_preprocess import hdf52dict
import pickle
import reader
import random


MAX_STEPS = 10000
TRAIN_DATA = '/afs/ir/users/a/x/axelsly/cs224n/project/small_processed/small_processed/nn_input_train'
TEST_DATA = '/afs/ir/users/a/x/axelsly/cs224n/project/small_processed/small_processed/nn_input_test'
BATCH_SIZE = 32 # should be dynamically passed in to Config
NUM_EPOCHS = 5
GPU_OPTIONS = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)


def main(args):
    input_size = 8
    initial_size = 7
    label_size = 1

    if args.model == 'seq2seq':
        curModel = Seq2SeqRNN(input_size, label_size, 'rnn')
    elif args.model == 'char':
        curModel = CharRNN(input_size, label_size, 'rnn')


    output, state = curModel.build_model(is_train = args.train)
    input_placeholder, label_placeholder, initial_placeholder, train_op = curModel.train()

    print "Running {0} model for {1} epochs.".format(args.model, NUM_EPOCHS)
    if args.train:
        print "Reading in training filenames."
        train_filenames = reader.abc_filenames(TRAIN_DATA)
        print "Creating training batches"
        train_batches = abc_batch(train_filenames)
    else:
        print "Reading in testing filenames."
        test_filenames = reader.abc_filenames(TEST_DATA)
        print "Creating testing batches."
        test_batches = abc_batch(test_filenames)



    with tf.Session(gpu_options=GPU_OPTIONS) as session:
        print "Inititialized TF Session!"
        
        if args.train:
            init_op = tf.initialize_all_variables() #tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
            init_op.run()

            for i in xrange(NUM_EPOCHS):
                random.shuffle(train_batches)
                for train_batch in train_batches:
                    # Get train data - into feed_dict
                    random.shuffle(train_batch)
                    data = map(reader.read_abc_pickle, train_batch)
                    meta_batch, input_window_batch, output_window_batch = tuple([list(tup) for tup in zip(*data)])

                    feed_dict = {
                        input_placeholder: input_window_batch,
                        initial_placeholder: meta_batch,
                        label_placeholder: output_window_batch
                    }

                    _, output_pred, output_state = sess.run([train_op, output, state], feed_dict=feed_dict)
                    # _, batch_loss = sess.run([train_op, loss], feed_dict=feed_dict)

                    print "Output Predicion: " + str(output_pred)
                    print "Output State: " + str(output_state)

        else: # Testing
            for test_batch in test_batches:
                # Get test data - into feed_dict
                data = map(reader.read_abc_pickle, test_batch)
                meta_batch, input_window_batch, output_window_batch = tuple([list(tup) for tup in zip(*data)])

                # TODO: testing code
                pass








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
