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


# TRAIN_DATA = '/data/small_processed/nn_input_train'
# DEVELOPMENT_DATA = '/data/small_processed/nn_input_test'
TRAIN_DATA = '/data/the_session_processed/nn_input_train_stride_25_window_25_nnType_char_rnn_shuffled'
DEVELOPMENT_DATA = '/data/the_session_processed/nn_input_dev_stride_25_window_25_nnType_char_rnn_shuffled'
VOCAB_DATA = '/data/the_session_processed/vocab_map_music.p'

CKPT_DIR =  '/data/ckpt'
SUMMARY_DIR = '/data/summary'


BATCH_SIZE = 100 # should be dynamically passed into Config
NUM_EPOCHS = 50
GPU_CONFIG = tf.ConfigProto()
GPU_CONFIG.gpu_options.per_process_gpu_memory_fraction = 0.5



def create_metrics_op(output, labels, vocabulary_size):
    prediction = tf.to_int32(tf.argmax(output, axis=2))

    difference = labels - prediction
    zero = tf.constant(0, dtype=tf.int32)
    boolean_difference = tf.cast(tf.equal(difference, zero), tf.float64)
    accuracy = tf.reduce_mean(boolean_difference)
    tf.summary.scalar('Accuracy', accuracy)

    confusion_matrix = tf.confusion_matrix(tf.reshape(labels, [-1]), tf.reshape(prediction, [-1]), num_classes=vocabulary_size, dtype=tf.int32)
    # conf matrix summary? tensorboard image?

    return prediction, accuracy, confusion_matrix



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



# def run_once(writer, )



def run_model(args):
    input_size = 1 if args.train == "sample" else 25
    initial_size = 7
    label_size = 1 if args.train == "sample" else 25
    batch_size = 1 if args.train == "sample" else BATCH_SIZE
    NUM_EPOCHS = args.num_epochs

    # Getting vocabulary mapping:
    vocabulary = reader.read_abc_pickle(VOCAB_DATA)
    vocabulary_keys = vocabulary.keys() + ["<start>", "<end>"]
    vocabulary = dict(zip(vocabulary_keys, range(len(vocabulary_keys))))
    vocabulary_size = len(vocabulary)
    vocabulary_decode = dict(zip(vocabulary.values(), vocabulary.keys()))

    if args.model == 'seq2seq':
        curModel = Seq2SeqRNN(input_size, label_size, 'rnn')
    elif args.model == 'char':
        # curModel = CharRNN(input_size, label_size, batch_size, vocabulary_size, 'rnn')
        curModel = CharRNN(input_size, label_size, batch_size, vocabulary_size, 'lstm')
        # curModel = CharRNN(input_size, label_size, batch_size, vocabulary_size, 'lstm')

    if args.train == 'dev':
        print "Running in development mode"
        setHyperparameters(curModel)



    output_op, state_op = curModel.create_model(is_train = args.train)
    input_placeholder, label_placeholder, meta_placeholder, initial_state_placeholder, use_meta_placeholder, train_op, loss_op = curModel.train()
    prediction_op, accuracy_op, conf_op = create_metrics_op(output_op, label_placeholder, vocabulary_size)

    print "Running {0} model for {1} epochs.".format(args.model, NUM_EPOCHS)
    if args.train:
        print "Reading in training filenames."
        train_filenames = reader.abc_filenames(TRAIN_DATA)
    else:
        print "Reading in testing filenames."
        test_filenames = reader.abc_filenames(DEVELOPMENT_DATA)


    global_step = tf.Variable(0, trainable=False, name='global_step') #tf.contrib.framework.get_or_create_global_step()

    # saver = tf.train.Saver(tf.all_variables(), max_to_keep=50)
    saver = tf.train.Saver(max_to_keep=NUM_EPOCHS)

    tf.summary.scalar('Loss', loss_op)
    summary_op = tf.summary.merge_all()
    step = 0
    found_ckpt = False

    with tf.Session(config=GPU_CONFIG) as session:
        print "Inititialized TF Session!"

        # Checkpoint
        if args.override:
            if tf.gfile.Exists(CKPT_DIR):
                tf.gfile.DeleteRecursively(CKPT_DIR)
            tf.gfile.MakeDirs(CKPT_DIR)

        ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            i_stopped = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            found_ckpt = True
        else:
            print('No checkpoint file found!')
            i_stopped = 0

        # Train model
        if args.train == "train":
            init_op = tf.global_variables_initializer() # tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
            init_op.run()

            train_writer = tf.summary.FileWriter(SUMMARY_DIR, graph=session.graph, max_queue=10, flush_secs=30)

            for i in xrange(i_stopped, NUM_EPOCHS):
                confusion_matrix = np.zeros((vocabulary_size, vocabulary_size))
                print "Running epoch ({0})...".format(i)
                random.shuffle(train_filenames)
                for j, train_file in enumerate(train_filenames):
                    # Get train data - into feed_dict
                    data = reader.read_abc_pickle(train_file)
                    random.shuffle(data)
                    train_batches = reader.abc_batch(data, n=batch_size)
                    for k, train_batch in enumerate(train_batches):
                        meta_batch, input_window_batch, output_window_batch = tuple([list(tup) for tup in zip(*train_batch)])

                        feed_dict = {
                            input_placeholder: input_window_batch,
                            meta_placeholder: meta_batch,
                            initial_state_placeholder: [np.zeros(curModel.config.hidden_size) for entry in xrange(batch_size)],
                            use_meta_placeholder: True,
                            label_placeholder: output_window_batch
                        }

                        _, summary, loss, output, state, prediction, accuracy, conf = session.run([train_op, summary_op, loss_op, output_op, state_op, prediction_op, accuracy_op, conf_op], feed_dict=feed_dict)
                        train_writer.add_summary(summary, step)

                        confusion_matrix += conf

                        print "Average accuracy per batch {0}".format(accuracy)
                        print "Batch Loss: {0}".format(loss)
                        # print "Output Predictions: {0}".format(prediction)
                        # print "Input Labels: {0}".format(output_window_batch)
                        # print "Output Prediction Probabilities: {0}".format(output_pred)
                        # print "Output State: {0}".format(output_state)

                        # Processed another batch
                        step += 1

                # Checkpoint model - every epoch
                checkpoint_path = os.path.join(CKPT_DIR, 'model.ckpt')
                saver.save(session, checkpoint_path, global_step=i)

                plot_confusion(confusion_matrix, vocabulary, i, characters_remove=['|', '2'])

        # Test Model
        if args.train == "test" or args.train == 'dev':
            # Exit if no checkpoint to test
            if not found_ckpt and args.train != 'dev':
                return

            confusion_matrix = np.zeros((vocabulary_size, vocabulary_size))
            batch_accuracies = []
            test_writer = tf.summary.FileWriter(SUMMARY_DIR, graph=session.graph, max_queue=10, flush_secs=30)

            print "Running test set from file {0}".format(DEVELOPMENT_DATA)
            random.shuffle(test_filenames)
            for j, test_file in enumerate(test_filenames):
                # Get test data - into feed_dict
                data = reader.read_abc_pickle(test_file)
                random.shuffle(data)
                test_batches = reader.abc_batch(data, n=batch_size)
                for k, train_batch in enumerate(train_batches):
                    meta_batch, input_window_batch, output_window_batch = tuple([list(tup) for tup in zip(*test_batches)])

                    feed_dict = {
                        input_placeholder: input_window_batch,
                        meta_placeholder: meta_batch,
                        initial_state_placeholder: [np.zeros(curModel.config.hidden_size) for entry in xrange(batch_size)],
                        use_meta_placeholder: True,
                        label_placeholder: output_window_batch
                    }

                    summary, loss, output, state, prediction, accuracy, conf = session.run([summary_op, loss_op, output_op, state_op, prediction_op, accuracy_op, conf_op], feed_dict=feed_dict)
                    train_writer.add_summary(summary, step)

                    confusion_matrix += conf
                    batch_accuracies.append(accuracy)

                    print "Average accuracy per batch {0}".format(accuracy)
                    print "Batch Loss: {0}".format(loss)
                    # print "Output Predictions: {0}".format(prediction)
                    # print "Input Labels: {0}".format(output_window_batch)
                    # print "Output Prediction Probabilities: {0}".format(output_pred)
                    # print "Output State: {0}".format(output_state)

                    # Processed another batch
                    step += 1

            plot_confusion(confusion_matrix, vocabulary, "_dev-set", characters_remove=['|', '2'])
            test_accuracy = np.mean(batch_accuracies)
            print "Model TEST accuracy: {0}".format(test_accuracy)
            if args.train == 'dev':
                # Update the file for choosing best hyperparameters
                curFile = open(curModel.dev_filename, 'a')
                curFile.write("Model development accuracy: {0}".format(test_accuracy))
                curFile.write('\n')
                curFile.close()


        # Sample Model
        else:
            # Exit if no checkpoint to sample
            if not found_ckpt:
                return

            warm_length = 20
            warm_meta, warm_chars = utils_runtime.genWarmStartDataset(warm_length)
            generated = warm_chars[1:]

            print "Sampling from single RNN cell using warm start of ({0})".format(warm_length)
            for j, c in enumerate(warm_chars):
                initial_state_sample = [np.zeros(curModel.config.hidden_size) for entry in xrange(batch_size)] if (j == 0) else state[0]

                feed_dict = {
                    input_placeholder: [[c]],
                    meta_placeholder: [warm_meta],
                    initial_state_placeholder: initial_state_sample,
                    use_meta_placeholder: j == 0,
                    label_placeholder: [[0]]   # TODO: revisit
                }

                loss, output, state, prediction = session.run([loss_op, output_op, state_op, prediction_op], feed_dict=feed_dict)

            sampled_character = prediction[0, 0]
            while True:
                feed_dict = {
                    input_placeholder: [[sampled_character]],
                    meta_placeholder: [np.zeros_like(warm_meta)],
                    initial_state_placeholder: state[0],
                    use_meta_placeholder: False,
                    label_placeholder: [[0]]   # TODO: revisit
                }

                loss, output, state, prediction = session.run([loss_op, output_op, state_op, prediction_op], feed_dict=feed_dict)
                if prediction == 81 or len(generated) > 100:
                    break
                # sample from "output" (probabilities) instead of finding the argmax?
                sampled_character = np.random.choice(len(output.flatten()), p=output.flatten())
                generated.append(sampled_character)

            decoded_characters = [vocabulary_decode[char] for char in generated]
            print ''.join(decoded_characters)







def parseCommandLine():
    desc = u'{0} [Args] [Options]\nDetailed options -h or --help'.format(__file__)
    parser = ArgumentParser(description=desc)

    print("Parsing Command Line Arguments...")
    requiredModel = parser.add_argument_group('Required Model arguments')
    requiredModel.add_argument('-m', choices = ["seq2seq", "char"], type = str,
    					dest = 'model', required = True, help = 'Type of model to run')
    requiredTrain = parser.add_argument_group('Required Train/Test arguments')
    requiredTrain.add_argument('-p', choices = ["train", "test", "sample", "dev"], type = str,
    					dest = 'train', required = True, help = 'Training or Testing phase to be run')

    parser.add_argument('-o', dest='override', action="store_true", help='Override the checkpoints')
    parser.add_argument('-e', dest='num_epochs', default=50, type=int, help='Set the number of Epochs')

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
