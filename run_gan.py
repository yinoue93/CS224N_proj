import tensorflow as tf
import numpy as np
import os
import sys
from models import Config, GenAdversarialNet
# from utils_preprocess import hdf52dict
import pickle
import reader
import random
import utils_runtime
import utils_hyperparam
import utils
import re

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
    use_seq2seq_data = (args.model == 'seq2seq')
    if args.data_dir != '':
        dataset_dir = args.data_dir
    elif args.train == 'train':
        dataset_dir = GAN_TRAIN_DATA
    elif args.train == 'test':
        dataset_dir = GAN_TEST_DATA
    else: # args.train == 'dev' or 'sample' (which has no dataset, but we just read anyway)
        dataset_dir = GAN_DEVELOPMENT_DATA


    print 'Using dataset %s' %dataset_dir
    dateset_filenames = reader.abc_filenames(dataset_dir)

    # figure out the input data size
    window_sz = int(re.findall('[0-9]+', re.findall('window_[0-9]+', dataset_dir)[0])[0])
    if 'output_sz' in dataset_dir:
        label_sz = int(re.findall('[0-9]+', re.findall('output_sz_[0-9]+', dataset_dir)[0])[0])
    else:
        label_sz = window_sz

    input_size = 1 if (args.train == "sample" and args.model!='cbow') else window_sz
    initial_size = 7
    label_size = 1 if args.train == "sample" else label_sz
    batch_size = 1 if args.train == "sample" else BATCH_SIZE
    NUM_EPOCHS = args.num_epochs
    print "Using checkpoint directory: {0}".format(args.ckpt_dir)

    # Getting vocabulary mapping:
    vocabulary = reader.read_abc_pickle(VOCAB_DATA)
    vocab_sz = len(vocabulary)
    vocabulary["<start>"] = vocab_sz
    vocabulary["<end>"] = vocab_sz+1
    if use_seq2seq_data:
        vocabulary["<go>"] = vocab_sz+2

    # Vocabulary info
    vocabulary_size = len(vocabulary)
    vocabulary_decode = dict(zip(vocabulary.values(), vocabulary.keys()))
    meta_vocabulary = reader.read_abc_pickle(META_DATA)
    num_classes = len(meta_vocabulary['R'])

    gan_label_size = 1

    start_encode = vocabulary["<go>"] if (args.train == "sample" and use_seq2seq_data) else vocabulary["<start>"]
    end_encode = vocabulary["<end>"]
    # Getting meta mapping:
    meta_map = pickle.load(open(META_DATA, 'rb'))

    cell_type = 'lstm'
    # cell_type = 'gru'
    # cell_type = 'rnn'

    curModel = GenAdversarialNet(input_size, gan_label_size, num_classes, cell_type,
                                args.train=='train', batch_size, vocabulary_size,
                                 args.set_config, use_lrelu=True, use_batchnorm=False,
                                 dropout=None)

    probabilities_real_op, probabilities_fake_op = curModel.create_model()
    input_placeholder, label_placeholder, \
        rnn_meta_placeholder, rnn_initial_state_placeholder, \
            rnn_use_meta_placeholder, train_op_d, train_op_gan = curModel.train()

    print "Reading in {0}-set filenames.".format(args.train)
    print "Running {0} model for {1} epochs.".format(args.model, NUM_EPOCHS)

    global_step = tf.Variable(0, trainable=False, name='global_step') #tf.contrib.framework.get_or_create_global_step()
    saver = tf.train.Saver(max_to_keep=NUM_EPOCHS)
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
            pass
            # # Sample Model
            # warm_length = 20
            # warm_meta, warm_chars = utils_runtime.genWarmStartDataset(warm_length)
            #
            # warm_meta_array = [warm_meta[:] for idx in xrange(5)]
            #
            # # Change Key
            # warm_meta_array[1][4] = 1 - warm_meta_array[1][4]
            # # Change Number of Flats/Sharps
            # warm_meta_array[2][3] = np.random.choice(11)
            # # Lower Complexity
            # warm_meta_array[3][6] = 50
            # # Higher Complexity
            # warm_meta_array[4][6] = 350
            #
            # new_warm_meta = utils_runtime.encode_meta_batch(meta_vocabulary, warm_meta_array)
            # new_warm_meta_array = zip(warm_meta_array, new_warm_meta)
            #
            # print "Sampling from single RNN cell using warm start of ({0})".format(warm_length)
            # for old_meta, meta in new_warm_meta_array:
            #     print "Current Metadata: {0}".format(meta)
            #     generated = warm_chars[:]
            #
            #     if args.model == 'char':
            #         # Warm Start
            #         for j, c in enumerate(warm_chars):
            #             if cell_type == 'lstm':
            #                 if j == 0:
            #                     initial_state_sample = [[np.zeros(curModel.config.hidden_size) for entry in xrange(batch_size)] for layer in xrange(curModel.config.num_layers)]
            #                 else:
            #                     initial_state_sample = []
            #                     for lstm_tuple in state:
            #                         initial_state_sample.append(lstm_tuple[0])
            #             else:
            #                 initial_state_sample = [np.zeros(curModel.config.hidden_size) for entry in xrange(batch_size)] if (j == 0) else state[0]
            #
            #             feed_values = utils_runtime.pack_feed_values(args, [[c]],
            #                                         [[0]], [meta],
            #                                         initial_state_sample, (j == 0),
            #                                         None, None)
            #             logits, state = curModel.sample(session, feed_values)
            #
            #         # Sample
            #         sampled_character = utils_runtime.sample_with_temperature(logits, TEMPERATURE)
            #         while sampled_character != vocabulary["<end>"] and len(generated) < 100:
            #             if cell_type == 'lstm':
            #                 initial_state_sample = []
            #                 for lstm_tuple in state:
            #                     initial_state_sample.append(lstm_tuple[0])
            #             else:
            #                 initial_state_sample = state[0]
            #
            #             feed_values = utils_runtime.pack_feed_values(args, [[sampled_character]],
            #                                         [[0]], [np.zeros_like(meta)],
            #                                         initial_state_sample, False,
            #                                         None, None)
            #             logits, state = curModel.sample(session, feed_values)
            #
            #             sampled_character = utils_runtime.sample_with_temperature(logits, TEMPERATURE)
            #             generated.append(sampled_character)
            #
            #     elif args.model == 'seq2seq':
            #         prediction = sample_Seq2Seq(args, curModel, cell_type, session, warm_chars, vocabulary, meta, batch_size)
            #         generated.extend(prediction.flatten())
            #
            #
            #     decoded_characters = [vocabulary_decode[char] for char in generated]
            #
            #     # Currently chopping off the last char regardless if its <end> or not
            #     encoding = utils.encoding2ABC(old_meta, generated)

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
                        new_meta_batch = utils_runtime.encode_meta_batch(meta_vocabulary, meta_batch)

                        initial_state_batch = [[np.zeros(curModel.config.hidden_size) for entry in xrange(batch_size)] for layer in xrange(curModel.config.num_layers)]
                        gan_labels = [m[0] for m in meta_batch]

                        feed_dict = {
                            input_placeholder: input_window_batch,
                            label_placeholder: gan_labels,
                            rnn_meta_placeholder: new_meta_batch,
                            rnn_initial_state_placeholder: initial_state_batch,
                            rnn_use_meta_placeholder: True
                        }

                        # summary, conf, accuracy = curModel.run(args, session, feed_values)

                        if args.train == "train":
                    		_, _ = session.run([train_op_d, train_op_gan], feed_dict=feed_dict)
                    	else: # Sample case not necessary b/c function will only be called during normal runs
                            pass
                            # summary, loss, probabilities, prediction, accuracy, confusion_matrix = session.run([self.summary_op, self.loss_op, self.probabilities_op, self.prediction_op, self.accuracy_op, self.confusion_matrix], feed_dict=feed_dict)


                        # file_writer.add_summary(summary, step)
                        #
                        # # Update confusion matrix
                        # confusion_matrix += conf
                        #
                        # # Record batch accuracies for test code
                        # if args.train == "test" or args.train == 'dev':
                        #     batch_accuracies.append(accuracy)
                        #
                        # # Processed another batch
                        # step += 1

                if args.train == "train":
                    # Checkpoint model - every epoch
                    utils_runtime.save_checkpoint(args, session, saver, i)
                    confusion_suffix = str(i)
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
                plot_confusion(confusion_matrix, vocabulary, confusion_suffix+"_all")
                plot_confusion(confusion_matrix, vocabulary, confusion_suffix+"_removed", characters_remove=['|', '2', '<end>'])



def main(_):

    args = utils_runtime.parseCommandLine()
    run_gan(args)

    if args.train != "sample":
        if tf.gfile.Exists(SUMMARY_DIR):
            tf.gfile.DeleteRecursively(SUMMARY_DIR)
        tf.gfile.MakeDirs(SUMMARY_DIR)

if __name__ == "__main__":
    tf.app.run()
