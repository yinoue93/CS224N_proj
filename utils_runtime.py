import pickle
import os
import random

import tensorflow as tf
from utils_preprocess import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tf_ver = tf.__version__
SHERLOCK = (str(tf_ver) == '0.12.1')

# for Sherlock
if SHERLOCK:
    DIR_MODIFIER = '/scratch/users/nipuna1'
# for Azure
else:
    DIR_MODIFIER = '/data'

def genWarmStartDataset(data_len,
			dataFolder=os.path.join(DIR_MODIFIER, 'full_dataset/warmup_dataset/checked')):
	"""
	Generates metadata and music data for the use in warm starting the RNN models

	A file gets sampled from @dataFolder under ./checked file, and gets encoded using
	the 'vocab_map_meta.p' and 'vocab_map_music.p' files under @vocab_dir.

	The first @data_len characters in the music data is returned.
	"""

	oneHotHeaders = ('R', 'M', 'L', 'K_key', 'K_mode')
	otherHeaders = ('len', 'complexity')

	meta_map = pickle.load(open(os.path.join(DIR_MODIFIER, 'full_dataset/global_map_meta.p'),'rb'))
	music_map = pickle.load(open(os.path.join(DIR_MODIFIER, 'full_dataset/global_map_music.p'),'rb'))

	# while loop here, just in case that the file we choose contains characters that
	# does not appear in the original dataset
	while True:
		# pick a random file in dataFolder
		abc_list = os.listdir(dataFolder)
		abc_file = os.path.join(dataFolder, random.choice(abc_list))

		meta,music = loadCleanABC(abc_file)
		warm_str = music[:data_len-1]

		# start encoding
		meta_enList = []
		music_enList = []
		encodeSuccess = True

		# encode the metadata info
		for header in oneHotHeaders:
			if meta[header] not in meta_map[header]:
				encodeSuccess = False
				break
			else:
				meta_enList.append(meta_map[header][meta[header]])

		for header in otherHeaders:
			meta_enList.append(meta[header])

		# encode music data
		# add the BEGIN token
		music_enList.append(len(music_map))
		for i in range(data_len-1):
			c = music[i]
			if c not in music_map:
				encodeSuccess = False
				break
			else:
				music_enList.append(music_map[c])

		if encodeSuccess:
			break

	print '-'*50
	print 'Generating the warm-start sequence...'
	print 'Chose %s to warm-start...' % abc_file
	print 'Meta Data is: %s' % str(meta)
	print 'The associated encoding is: %s' % str(meta_enList)
	print 'Music to warm-start with is: %s' % warm_str
	print 'The associated encoding is: %s' % str(music_enList)
	print '-'*50

	return meta_enList,music_enList


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


def save_checkpoint(args, session, saver, i):
    checkpoint_path = os.path.join(args.ckpt_dir, 'model.ckpt')
    saver.save(session, checkpoint_path, global_step=i)
    # saver.save(session, os.path.join(SUMMARY_DIR,'model.ckpt'), global_step=i)





if __name__ == "__main__":
	genWarmStartDataset(20)
