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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils_runtime
import utils_hyperparam
import utils



# TRAIN_DATA = '/data/small_processed/nn_input_train'
# DEVELOPMENT_DATA = '/data/small_processed/nn_input_test'
TRAIN_DATA = '/data/the_session_processed/nn_input_train_stride_25_window_25_nnType_char_rnn_shuffled'
TEST_DATA = '/data/the_session_processed/nn_input_test_stride_25_window_25_nnType_char_rnn_shuffled'
DEVELOPMENT_DATA = '/data/the_session_processed/nn_input_dev_stride_25_window_25_nnType_char_rnn_shuffled'
VOCAB_DATA = '/data/the_session_processed/vocab_map_music.p'

SUMMARY_DIR = '/data/summary'

BATCH_SIZE = 100 # should be dynamically passed into Config
NUM_EPOCHS = 50
GPU_CONFIG = tf.ConfigProto()
GPU_CONFIG.gpu_options.per_process_gpu_memory_fraction = 0.3

# For T --> inf, p is uniform. Easy to sample from!
# For T --> 0, p "concentrates" on arg max. Hard to sample from!
TEMPERATURE = 1.0



def create_metrics_op(probabilities, labels, vocabulary_size):
    prediction = tf.to_int32(tf.argmax(probabilities, axis=2))

    difference = labels - prediction
    zero = tf.constant(0, dtype=tf.int32)
    boolean_difference = tf.cast(tf.equal(difference, zero), tf.float64)
    accuracy = tf.reduce_mean(boolean_difference)
    tf.summary.scalar('Accuracy', accuracy)

    confusion_matrix = tf.confusion_matrix(tf.reshape(labels, [-1]), tf.reshape(prediction, [-1]), num_classes=vocabulary_size, dtype=tf.int32)

    return prediction, accuracy, confusion_matrix



def sample_with_temperature(logits, temperature):
    flattened_logits = logits.flatten()
    unnormalized = np.exp((flattened_logits - np.max(flattened_logits)) / temperature)
    probabilities = unnormalized / float(np.sum(unnormalized))
    sample = np.random.choice(len(probabilities), p=probabilities)
    return sample



def plot_confusion(confusion_matrix, vocabulary, epoch, characters_remove=[], annotate=False):
    # Get vocabulary components
    vocabulary_keys = vocabulary.keys()
    removed_indicies = []
    for c in characters_remove:
        i = vocabulary_keys.index(c)
        vocabulary_keys.remove(c)
        removed_indicies.append(i)

    # Delete unnecessary rows
    conf_temp = np.delete(confusion_matrix, removed_indicies, axis=0)
    # Delete unnecessary cols
    new_confusion = np.delete(conf_temp, removed_indicies, axis=1)

    vocabulary_values = range(len(vocabulary_keys))
    vocabulary_size = len(vocabulary_keys)

    fig, ax = plt.subplots(figsize=(10, 10))
    res = ax.imshow(new_confusion.astype(int), interpolation='nearest', cmap=plt.cm.jet)
    cb = fig.colorbar(res)

    if annotate:
        for x in xrange(vocabulary_size):
            for y in xrange(vocabulary_size):
                ax.annotate(str(new_confusion[x, y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=4)

    plt.xticks(vocabulary_values, vocabulary_keys)
    plt.yticks(vocabulary_values, vocabulary_keys)
    fig.savefig('confusion_matrix_epoch{0}.png'.format(epoch))



def get_checkpoint(args, session, saver):
    # Checkpoint
    found_ckpt = False

    if args.override:
        if tf.gfile.Exists(args.ckpt_dir):
            tf.gfile.DeleteRecursively(args.ckpt_dir)
        tf.gfile.MakeDirs(args.ckpt_dir)

    ckpt = tf.train.get_checkpoint_state(args.ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
        i_stopped = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        print "Found checkpoint for epoch ({0})".format(i_stopped)
        found_ckpt = True
    else:
        print('No checkpoint file found!')
        i_stopped = 0

    return i_stopped, found_ckpt



def save_checkpoint(session, saver, i):
    checkpoint_path = os.path.join(args.ckpt_dir, 'model.ckpt')
    saver.save(session, checkpoint_path, global_step=i)



def print_model_outputs(accuracy, loss, prediction, output_window_batch, probabilities):
    print "Average accuracy per batch {0}".format(accuracy)
    print "Batch Loss: {0}".format(loss)
    # print "Output Predictions: {0}".format(prediction)
    # print "Input Labels: {0}".format(output_window_batch)
    # print "Output Prediction Probabilities: {0}".format(probabilities)



def create_feed_dict(args, curModel, input_batch, label_batch, meta_batch,
                            initial_state_batch, use_meta_batch): # add more for GAN
    if args.model == 'seq2seq':
        feed_dict = {
            curModel.input_placeholder: input_batch,
            curModel.meta_placeholder: meta_batch,
            curModel.label_placeholder: label_batch,
            curModel.initial_state_placeholder: initial_state_batch,
            curModel.use_meta_placeholder: use_meta_batch
            # attention
        }
    elif args.model == 'char':
        feed_dict = {
            curModel.input_placeholder: input_batch,
            curModel.meta_placeholder: meta_batch,
            curModel.label_placeholder: label_batch,
            curModel.initial_state_placeholder: initial_state_batch,
            curModel.use_meta_placeholder: use_meta_batch
        }
    elif args.model == 'cbow':
        feed_dict = {
            curModel.input_placeholder: input_batch,
            curModel.label_placeholder: label_batch
        }
    elif args.model == 'gan':
        feed_dict = {
            curModel.input_placeholder: input_batch,
            curModel.meta_placeholder: meta_batch,
            curModel.label_placeholder: label_batch,
            curModel.initial_state_placeholder: initial_state_batch,
            curModel.use_meta_placeholder: use_meta_batch
            # MORE
        }
    return feed_dict



def run_model(args):
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

    cell_type = 'lstm'
    # cell_type = 'gru'
    # cell_type = 'rnn'

    if args.model == 'seq2seq':
        curModel = Seq2SeqRNN(input_size, label_size, cell_type, args.set_config)
    elif args.model == 'char':
        curModel = CharRNN(input_size, label_size, batch_size, vocabulary_size, cell_type, args.set_config)
        probabilities_op, logits_op, state_op = curModel.create_model(is_train = (args.train=='train'))
        input_placeholder, label_placeholder, meta_placeholder, initial_state_placeholder, use_meta_placeholder, train_op, loss_op = curModel.train()
    elif args.model == 'cbow':
        curModel = CBOW(input_size, label_size, batch_size, vocab_size, args.set_config)
        probabilities_op, logits_op = curModel.create_model()
        input_placeholder, label_placeholder, train_op, loss_op = curModel.train()
    elif args.model == 'gan':
        curModel = GenAdversarialNet(fake_input_size, real_input_size, label_size, is_training, batch_size, vocab_size, cell_type, hyperparam_path)

    prediction_op, accuracy_op, conf_op = create_metrics_op(probabilities_op, label_placeholder, vocabulary_size)

    print "Running {0} model for {1} epochs.".format(args.model, NUM_EPOCHS)

    print "Reading in {0}-set filenames.".format(args.train)
    if args.train == 'train':
        dataset_dir = TRAIN_DATA
    elif args.train == 'test':
        dataset_dir = TEST_DATA
    else: # args.train == 'dev' or 'sample' (which has no dataset, but we just read anyway)
        dataset_dir = DEVELOPMENT_DATA
    dateset_filenames = reader.abc_filenames(dataset_dir)

    global_step = tf.Variable(0, trainable=False, name='global_step') #tf.contrib.framework.get_or_create_global_step()

    saver = tf.train.Saver(max_to_keep=NUM_EPOCHS)

    tf.summary.scalar('Loss', loss_op)
    summary_op = tf.summary.merge_all()
    step = 0

    with tf.Session(config=GPU_CONFIG) as session:
        print "Inititialized TF Session!"

        # Checkpoint
        i_stopped, found_ckpt = get_checkpoint(args, session, saver)

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
                sampled_character = sample_with_temperature(logits, TEMPERATURE)
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
                    sampled_character = sample_with_temperature(logits, TEMPERATURE)
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
                    save_checkpoint(session, saver, i)
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
                plot_confusion(confusion_matrix, vocabulary, confusion_suffix)#, characters_remove=['|', '2'])


















def parseCommandLine():
    desc = u'{0} [Args] [Options]\nDetailed options -h or --help'.format(__file__)
    parser = ArgumentParser(description=desc)

    print("Parsing Command Line Arguments...")
    requiredModel = parser.add_argument_group('Required Model arguments')
    requiredModel.add_argument('-m', choices = ["seq2seq", "char", "cbow", "gan"], type = str,
    					dest = 'model', required = True, help = 'Type of model to run')
    requiredTrain = parser.add_argument_group('Required Train/Test arguments')
    requiredTrain.add_argument('-p', choices = ["train", "test", "sample", "dev"], type = str,
    					dest = 'train', required = True, help = 'Training or Testing phase to be run')

    requiredTrain.add_argument('-c', type = str, dest = 'set_config',
                               help = 'Set hyperparameters', default='')

    parser.add_argument('-o', dest='override', action="store_true", help='Override the checkpoints')
    parser.add_argument('-e', dest='num_epochs', default=50, type=int, help='Set the number of Epochs')
    parser.add_argument('-ckpt', dest='ckpt_dir', default='/data/temp_ckpt/', type=str, help='Set the checkpoint directory')
    args = parser.parse_args()
    return args



def main(_):

    args = parseCommandLine()
    run_model(args)

    if args.train != "sample":
        if tf.gfile.Exists(SUMMARY_DIR):
            tf.gfile.DeleteRecursively(SUMMARY_DIR)
        tf.gfile.MakeDirs(SUMMARY_DIR)

if __name__ == "__main__":
    tf.app.run()
