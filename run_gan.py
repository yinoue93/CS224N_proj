import tensorflow as tf
import numpy as np
import os
import sys
from models import CharRNN, Config, Seq2SeqRNN, CBOW
# from utils_preprocess import hdf52dict
import pickle
import reader
import random
import utils_runtime
import utils_hyperparam
import utils

tf_ver = tf.__version__
SHERLOCK = (str(tf_ver) == '0.12.1')

# TRAIN_DATA = '/data/small_processed/nn_input_train'
# DEVELOPMENT_DATA = '/data/small_processed/nn_input_test'

# for Sherlock
if SHERLOCK:
    DIR_MODIFIER = '/scratch/users/nipuna1'
    from tensorflow.contrib.metrics import confusion_matrix as tf_confusion_matrix
# for Azure
else:
    DIR_MODIFIER = '/data'

TRAIN_DATA = DIR_MODIFIER + '/full_dataset/char_rnn_dataset/nn_input_train_stride_25_window_25_nnType_char_rnn_shuffled'
TEST_DATA = DIR_MODIFIER + '/full_dataset/char_rnn_dataset/nn_input_test_stride_25_window_25_nnType_char_rnn_shuffled'
DEVELOPMENT_DATA = DIR_MODIFIER + '/full_dataset/char_rnn_dataset/nn_input_dev_stride_25_window_25_nnType_char_rnn_shuffled'

GAN_TRAIN_DATA = DIR_MODIFIER + '/full_dataset/gan_dataset/nn_input_train_stride_25_window_25_nnType_seq2seq_output_sz_25_shuffled'
GAN_TEST_DATA = DIR_MODIFIER + '/full_dataset/gan_dataset/nn_input_test_stride_25_window_25_nnType_seq2seq_output_sz_25_shuffled'
GAN_DEVELOPMENT_DATA = DIR_MODIFIER + '/full_dataset/gan_dataset/nn_input_dev_stride_25_window_25_nnType_seq2seq_output_sz_25_shuffled'

VOCAB_DATA = DIR_MODIFIER + '/full_dataset/global_map_music.p'
META_DATA = DIR_MODIFIER + '/full_dataset/global_map_meta.p'

SUMMARY_DIR = DIR_MODIFIER + '/cbow_summary'

BATCH_SIZE = 100 # should be dynamically passed into Config
NUM_EPOCHS = 50
GPU_CONFIG = tf.ConfigProto()
GPU_CONFIG.gpu_options.per_process_gpu_memory_fraction = 0.3

# For T --> inf, p is uniform. Easy to sample from!
# For T --> 0, p "concentrates" on arg max. Hard to sample from!
TEMPERATURE = 1.0




def run_gan(args):
    input_size = 1 if args.train == "sample" else 25
    initial_size = 7
    label_size = 1 if args.train == "sample" else 25
    batch_size = 1 if args.train == "sample" else BATCH_SIZE
    NUM_EPOCHS = args.num_epochs
    print "Using checkpoint directory: {0}".format(args.ckpt_dir)

    # Getting vocabulary mapping:
    vocabulary = reader.read_abc_pickle(VOCAB_DATA)
    vocabulary_keys = vocabulary.keys() + ["<start>", "<end>"]
    vocabulary = dict(zip(vocabulary_keys, range(len(vocabulary_keys))))
    vocabulary_size = len(vocabulary)
    vocabulary_decode = dict(zip(vocabulary.values(), vocabulary.keys()))

    # Getting meta mapping:
    meta_map = pickle.load(open(META_DATA, 'rb'))

    cell_type = 'lstm'
    # cell_type = 'gru'
    # cell_type = 'rnn'

    gan_label_size = len(meta_map['R'])
    curModel = GenAdversarialNet(input_size, gan_label_size,
                                args.train=='train', batch_size, cell_type,
                                 args.set_config, use_lrelu=True, use_batchnorm=False,
                                 dropout=None)

    probabilities_real_op, probabilities_fake_op = curModel.create_model()
    self.input_placeholder, self.label_placeholder, \
                self.rnn_meta_placeholder, self.rnn_initial_state_placeholder, \
                 self.rnn_use_meta_placeholder, self.train_op_d, self.train_op_gan = curModel.train()

    print "Running {0} model for {1} epochs.".format(args.model, NUM_EPOCHS)

    print "Reading in {0}-set filenames.".format(args.train)
    if args.train == 'train':
        dataset_dir = GAN_TRAIN_DATA
    elif args.train == 'test':
        dataset_dir = GAN_TEST_DATA
    else: # args.train == 'dev' or 'sample' (which has no dataset, but we just read anyway)
        dataset_dir = GAN_DEVELOPMENT_DATA
    dateset_filenames = reader.abc_filenames(dataset_dir)

    global_step = tf.Variable(0, trainable=False, name='global_step') #tf.contrib.framework.get_or_create_global_step()

    saver = tf.train.Saver(tf.all_variables(), max_to_keep=NUM_EPOCHS)

    tf.summary.scalar('Loss', loss_op)
    summary_op = tf.summary.merge_all()
    step = 0

    with tf.Session(config=GPU_CONFIG) as session:
        print "Inititialized TF Session!"

        # Checkpoint
        i_stopped, found_ckpt = utils_runtime.get_checkpoint(args, session, saver)

        file_writer = tf.summary.FileWriter(SUMMARY_DIR, graph=session.graph, max_queue=10, flush_secs=30)
        confusion_matrix = np.zeros((vocabulary_size, vocabulary_size))
        batch_accuracies = []

        if args.train == "train":
            init_op = tf.global_variables_initializer() # tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
            init_op.run()
        else:
            # Exit if no checkpoint to test
            if not found_ckpt:
                return
            NUM_EPOCHS = i_stopped + 1

        # Sample Model
        if args.train == "sample":
            # Sample Model
            warm_length = 20
            warm_meta, warm_chars = utils_runtime.genWarmStartDataset(warm_length)

            warm_meta_array = [warm_meta[:] for idx in xrange(3)]
            warm_meta_array[1][4] = 1 - warm_meta_array[1][4]
            warm_meta_array[1][3] = np.random.choice(11)

            print "Sampling from single RNN cell using warm start of ({0})".format(warm_length)
            for meta in warm_meta_array:
                print "Current Metadata: {0}".format(meta)
                generated = warm_chars[:]

                # Warm Start
                for j, c in enumerate(warm_chars):
                    if cell_type == 'lstm':
                        if j == 0:
                            initial_state_sample = [[np.zeros(curModel.config.hidden_size) for entry in xrange(batch_size)] for layer in xrange(curModel.config.num_layers)]
                        else:
                            initial_state_sample = []
                            for lstm_tuple in state:
                                initial_state_sample.append(lstm_tuple[0])
                    else:
                        initial_state_sample = [np.zeros(curModel.config.hidden_size) for entry in xrange(batch_size)] if (j == 0) else state[0]

                    feed_dict = create_feed_dict(args, curModel, [[c]], [[0]], [meta],
                                                initial_state_sample, (j == 0))
                    loss, logits, state = session.run([loss_op, logits_op, state_op], feed_dict=feed_dict)

                # Sample
                sampled_character = utils_runtime.sample_with_temperature(logits, TEMPERATURE)
                while sampled_character != 81 and len(generated) < 200:
                    if cell_type == 'lstm':
                        initial_state_sample = []
                        for lstm_tuple in state:
                            initial_state_sample.append(lstm_tuple[0])
                    else:
                        initial_state_sample = state[0]

                    feed_dict = create_feed_dict(args, curModel, [[sampled_character]],
                                                [[0]], [np.zeros_like(meta)],
                                                initial_state_sample, False)

                    loss, logits, state = session.run([loss_op, logits_op, state_op], feed_dict=feed_dict)
                    sampled_character = utils_runtime.sample_with_temperature(logits, TEMPERATURE)
                    generated.append(sampled_character)

                decoded_characters = [vocabulary_decode[char] for char in generated]

                # Currently chopping off the last char regardless if its <end> or not
                encoding = utils.encoding2ABC(meta, generated[1:-1])

        # Train, dev, test model
        else:
            for i in xrange(i_stopped, NUM_EPOCHS):
                print "Running epoch ({0})...".format(i)
                random.shuffle(dateset_filenames)
                for j, data_file in enumerate(dateset_filenames):
                    # Get train data - into feed_dict
                    data = reader.read_abc_pickle(data_file)
                    random.shuffle(data)
                    data_batches = reader.abc_batch(data, n=batch_size)
                    for k, data_batch in enumerate(data_batches):
                        meta_batch, input_window_batch, output_window_batch = tuple([list(tup) for tup in zip(*data_batch)])
                        initial_state_batch = [[np.zeros(curModel.config.hidden_size) for entry in xrange(batch_size)] for layer in xrange(curModel.config.num_layers)]

                        feed_dict = create_feed_dict(args, curModel, input_window_batch,
                                                    output_window_batch, meta_batch,
                                                    initial_state_batch, True)

                        if args.train == "train":
                            _, summary, loss, probabilities, state, prediction, accuracy, conf = session.run([train_op, summary_op, loss_op, probabilities_op, state_op, prediction_op, accuracy_op, conf_op], feed_dict=feed_dict)
                        else:
                            summary, loss, probabilities, state, prediction, accuracy, conf = session.run([summary_op, loss_op, probabilities_op, state_op, prediction_op, accuracy_op, conf_op], feed_dict=feed_dict)
                        file_writer.add_summary(summary, step)

                        # Update confusion matrix
                        confusion_matrix += conf

                        # Record batch accuracies for test code
                        if args.train == "test" or args.train == 'dev':
                            batch_accuracies.append(accuracy)

                        # Print out some stats
                        print_model_outputs(accuracy, loss, prediction, output_window_batch, probabilities)

                        # Processed another batch
                        step += 1

                if args.train == "train":
                    # Checkpoint model - every epoch
                    utils_runtime.save_checkpoint(session, saver, i)
                    confusion_suffix = i
                else: # dev or test (NOT sample)
                    test_accuracy = np.mean(batch_accuracies)
                    print "Model {0} accuracy: {1}".format(args.train, test_accuracy)
                    confusion_suffix = "_{0}-set".format(args.train)

                    if args.train == 'dev':
                        # Update the file for choosing best hyperparameters
                        curFile = open(curModel.config.dev_filename, 'a')
                        curFile.write("Dev set accuracy: {0}".format(test_accuracy))
                        curFile.write('\n')
                        curFile.close()

                # Plot Confusion Matrix
                utils_runtime.plot_confusion(confusion_matrix, vocabulary, confusion_suffix, characters_remove=['|', '2'])















def main(_):

    args = utils_runtime.parseCommandLine()
    if args.model == 'gan':
        run_gan(args)
    else:
        run_model(args)

    if args.train != "sample":
        if tf.gfile.Exists(SUMMARY_DIR):
            tf.gfile.DeleteRecursively(SUMMARY_DIR)
        tf.gfile.MakeDirs(SUMMARY_DIR)

if __name__ == "__main__":
    tf.app.run()
