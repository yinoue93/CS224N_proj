import tensorflow as tf
import numpy as np
import os
import sys
from argparse import ArgumentParser
from models import CharRNN, Config
# from utils_preprocess import hdf52dict
import pickle
import reader
import random


# TRAIN_DATA = '/data/small_processed/nn_input_train'
# TEST_DATA = '/data/small_processed/nn_input_test'
TRAIN_DATA = '/data/the_session_processed/nn_input_train_stride_25_window_25_nnType_char_rnn_shuffled'
TEST_DATA = '/data/the_session_processed/nn_input_dev_stride_25_window_25_nnType_char_rnn_shuffled'

CKPT_DIR = '/data/ckpt'
SUMMARY_DIR = '/data/summary'


BATCH_SIZE = 32 # should be dynamically passe into Config
NUM_EPOCHS = 50
GPU_CONFIG = tf.ConfigProto()
GPU_CONFIG.gpu_options.per_process_gpu_memory_fraction = 0.5


def run_model(args):
    input_size = 25
    initial_size = 7
    label_size = 25

    if args.model == 'seq2seq':
        curModel = Seq2SeqRNN(input_size, label_size, 'rnn')
    elif args.model == 'char':
        # curModel = CharRNN(input_size, label_size, 'rnn')
        # curModel = CharRNN(input_size, label_size, 'gru')
        curModel = CharRNN(input_size, label_size, 'lstm')


    output, state = curModel.create_model(is_train = args.train)
    input_placeholder, label_placeholder, initial_placeholder, train_op, loss = curModel.train()

    print "Running {0} model for {1} epochs.".format(args.model, NUM_EPOCHS)
    if args.train:
        print "Reading in training filenames."
        train_filenames = reader.abc_filenames(TRAIN_DATA)
        # print "Creating training batches"
        # train_batches = reader.abc_batch(train_filenames, n=BATCH_SIZE)
    else:
        print "Reading in testing filenames."
        test_filenames = reader.abc_filenames(TEST_DATA)
        # print "Creating testing batches."
        # test_batches = reader.abc_batch(test_filenames, n=BATCH_SIZE)


    global_step = tf.Variable(0, trainable=False, name='global_step') #tf.contrib.framework.get_or_create_global_step()

    # saver = tf.train.Saver(tf.all_variables(), max_to_keep=50)
    saver = tf.train.Saver(max_to_keep=NUM_EPOCHS)

    tf.summary.scalar('Loss', loss)
    summary_op = tf.summary.merge_all()
    step = 0

    # Checkpoint
    ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        # print(ckpt.model_checkpoint_path)
        i_stopped = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
        print('No checkpoint file found!')
        i_stopped = 0

    with tf.Session(config=GPU_CONFIG) as session:
        print "Inititialized TF Session!"

        # Train model
        if args.train:
            init_op = tf.global_variables_initializer() # tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
            init_op.run()

            train_writer = tf.summary.FileWriter(SUMMARY_DIR, graph=session.graph, max_queue=10, flush_secs=30)

            for i in xrange(i_stopped, NUM_EPOCHS):
                print "Running epoch ({0})...".format(i)
                random.shuffle(train_filenames)
                for j, train_file in enumerate(train_filenames):
                    # Get train data - into feed_dict
                    data = reader.read_abc_pickle(train_file)
                    random.shuffle(data)
                    train_batches = reader.abc_batch(data, n=BATCH_SIZE)
                    for k, train_batch in enumerate(train_batches):
                        meta_batch, input_window_batch, output_window_batch = tuple([list(tup) for tup in zip(*train_batch)])

                        feed_dict = {
                            input_placeholder: input_window_batch,
                            initial_placeholder: meta_batch,
                            label_placeholder: output_window_batch
                        }

                        _, summary, batch_loss, output_pred, output_state = session.run([train_op, summary_op, loss, output, state], feed_dict=feed_dict)
                        train_writer.add_summary(summary, step)

                        prediction = np.argmax(output_pred, axis=2)
                        difference = output_window_batch - prediction

                        correct_per_batch = np.sum(difference == 0, axis=1)
                        accuracy_per_batch = correct_per_batch / float(difference.shape[1])
                        accuaracy = np.mean(accuracy_per_batch)

                        print "Average accuracy per batch {0}".format(accuaracy)
                        print "Batch Loss: {0}".format(batch_loss)
                        # print "Output Predictions: {0}".format(prediction)
                        # print "Input Labels: {0}".format(output_window_batch)
                        # print "Output Prediction Probabilities: {0}".format(output_pred)
                        # print "Output State: {0}".format(output_state)

                        # Processed another batch
                        step += 1

                # Checkpoint model - every epoch
                checkpoint_path = os.path.join(CKPT_DIR, 'model.ckpt')
                saver.save(session, checkpoint_path, global_step=i)

        # Test Model
        else:
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



def main(_):
    if tf.gfile.Exists(SUMMARY_DIR):
        tf.gfile.DeleteRecursively(SUMMARY_DIR)
    tf.gfile.MakeDirs(SUMMARY_DIR)

    args = parseCommandLine()
    run_model(args)


if __name__ == "__main__":
    tf.app.run()
