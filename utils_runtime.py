import pickle
import os
import random

from utils_preprocess import *

def genWarmStartDataset(data_len, vocab_dir='/data/the_session_processed', 
						dataFolder='/data/montreal_plus_local_processed'):
	"""
	Generates metadata and music data for the use in warm starting the RNN models

	A file gets sampled from @dataFolder under ./checked file, and gets encoded using
	the 'vocab_map_meta.p' and 'vocab_map_music.p' files under @vocab_dir.

	The first @data_len characters in the music data is returned.
	"""

	oneHotHeaders = ('R', 'M', 'L', 'K_key', 'K_mode')
	otherHeaders = ('len', 'complexity')
	
	meta_map = pickle.load(open(os.path.join(vocab_dir, 'vocab_map_meta.p'), 'r'))
	music_map = pickle.load(open(os.path.join(vocab_dir, 'vocab_map_music.p'), 'r'))

	# while loop here, just in case that the file we choose contains characters that
	# does not appear in the original dataset
	while True:
		# pick a random file in checkedFolder
		checkedFolder = os.path.join(dataFolder, CHECK_DIR)
		abc_list = os.listdir(checkedFolder)
		abc_file = os.path.join(checkedFolder, random.choice(abc_list))

		meta,music = loadCleanABC(abc_file)
		warm_str = music[:data_len]

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

if __name__ == "__main__":
	genWarmStartDataset(20)