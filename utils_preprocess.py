import matplotlib.pyplot as plt
import os
import re
import pickle

import numpy as np

from collections import Counter
from multiprocessing import Pool

from utils import *

def eraseUnreadable(folderN):
	iterf = 0
	for fileN in os.listdir(folderN):
		fileAbsN = folderN+'/'+fileN
		try:
			if iterf%100 == 0:
				print iterf
			iterf += 1
			midi_data = pretty_midi.PrettyMIDI(fileAbsN)
			#roll = midi_data.get_piano_roll()
		except:
			print 'failed: '+fileN
			os.remove(fileAbsN)

def extractABCtxtWorker(dataPack):
	filename,outputname = dataPack
	header = True
	headerDict = {}
	headerTup = ('T', 'R', 'M', 'L', 'K', 'Q')
	headerDefault = {'T':'none', 'R':'none', 'M':'none', 'L':'none', 'K':'C', 'Q':'100'}
	print filename
	with open(filename,'r') as infile:
		fileStr = infile.read().replace(' ','').replace('\r','\n')
		if 'X:' not in fileStr:
			return
		print repr(fileStr)

		with open(outputname,'w') as outfile:
			for line in fileStr.split('\n'):
				line = line.strip()
				# skip empty lines
				if line=='':
					continue

				if header:
					headerStr = re.match('[a-zA-Z]:',line)

					if headerStr is None:
						header = False
						for head in headerTup:
							if head in headerDict:
								if head=='R':
									headerDict[head] = headerDict[head].lower()
								outfile.write(headerDict[head]+'\n')
							else:
								outfile.write('%s:%s\n' %(head,headerDefault[head]))
						outfile.write(line)

					elif headerStr.group()[0] in headerTup:
						headerDict[headerStr.group()[0]] = line

				else:
					outfile.write(line)

			outfile.write('\n')


def extractABCtxt(folderName):
	outputFolder = folderName+'_cleaned'
	makedir(outputFolder)

	p = Pool(8)
	filenames = [re.sub(r'[^\x00-\x7f]',r'',fname) for fname in os.listdir(folderName)]
	mapList = [(os.path.join(folderName,fname),os.path.join(outputFolder,fname2))
										for fname,fname2 in zip(os.listdir(folderName),filenames)]

	p.map(extractABCtxtWorker, mapList)

MIN_MEASURES = 10
def abcFileCheckerWorker(dataPack):
	filename,outputname = dataPack
	header = True
	headerDict = {}
	headerTup = ('T', 'R', 'M', 'L', 'K', 'Q')
	with open(filename,'r') as infile:
		fileStr = infile.read()
		fileList = fileStr.split('\n')

		# checking stage
		#-----------------------------
		# each .abc file needs to be 8 lines long (6 metadata, 1 music, and 1 empty line)
		if len(fileList)!=8:
			print filename+': Does not have 8 lines'
			return

		# check that the file contains all metadata tags
		for i,header in enumerate(headerTup):
			if fileList[i][0] != header:
				print filename+': Does not contain the metadata '+header
				return

		# make sure that there are more than MIN_MEASURES measures in the song
		if fileList[6].replace('||','|').count('|')<MIN_MEASURES+1:
			print filename+': Song too short'
			return
		#-----------------------------

		# augmentation stage
		#-----------------------------
		fileStr = 'X:1\n' + fileStr

		# save the non-augmented song
		with open(outputname,'w') as outfile:
			outfile.write(fileStr)

		# check if the file just saved was correctly formed .abc file
		if not passesABC2ABC(outputname):
			print "Doesn't pass abc2abc: " + outputname
			os.remove(outputname)
			return

		for shift in xrange(-5,6):
			transposeABC(outputname, outputname.replace('.abc','_%s.abc'%shift), shift)


def abcFileChecker(folderName):
	outputFolder = folderName+'_checked'
	makedir(outputFolder)

	p = Pool(8)
	mapList = [(os.path.join(folderName,fname),os.path.join(outputFolder,fname))
										for fname in os.listdir(folderName)]

	p.map(abcFileCheckerWorker, mapList)

def generateVocab(foldername):
	"""
	Creates the vocabulary under the @foldername
	"""
	# tally up the metadata
	metaCount = {}
	headerTup = ('T', 'R', 'M', 'L', 'K_key', 'K_mode', 'Q', 'len', 'complexity')
	for header in headerTup:
		metaCount[header] = {}

	musicDict = {}

	filenames = [os.path.join(foldername,filename) for filename in os.listdir(foldername)]
	for i,filename in enumerate(filenames):
		if i%1000==0:
			print i

		try:
			meta,music = loadCleanABC(filename)
		except:
			print filename

		if '\xc5' in music:
			print filename
			exit(0)

		for header in headerTup:
			newMeta = str(meta[header])
			if newMeta not in metaCount[header]:
				metaCount[header][newMeta] = 0

			musicChars = Counter(music)
			for c in musicChars:
				if c not in musicDict:
					musicDict[c] = 0
				musicDict[c] += musicChars[c]

			metaCount[header][newMeta] += 1

	meta2Store = {'R':{}, 'M':{}, 'L':{}, 'K_key':{}, 'K_mode':{}}
	for header in meta2Store:
		for i,key in enumerate(metaCount[header]):
			meta2Store[header][key] = i

		# uncomment to print out the result
		# print header
		# print len(meta2Store[header])
		# print meta2Store[header]
		# print metaCount[header]
		# print '-'*40

	music2Store = {}
	for i,letter in enumerate(musicDict):
		music2Store[letter] = i

	# write out to a file
	pickle.dump(meta2Store, open('vocab_map_meta.p','wb'))
	pickle.dump(music2Store, open('vocab_map_music.p','wb'))

def encodeABCWorker(dataPack):
	oneHotHeaders = ('R', 'M', 'L', 'K_key', 'K_mode')
	otherHeaders = ('len', 'complexity')

	filename,outputname,meta_map,music_map = dataPack
	meta,music = loadCleanABC(filename)

	encodeList = []
	# encode the metadata info
	for header in oneHotHeaders:
		encodeList.append(meta_map[header][meta[header]])
	for header in otherHeaders:
		encodeList.append(meta[header])

	for c in music:
		encodeList.append(music_map[c])

	np.save(outputname,np.asarray(encodeList))


def encodeABC(folderName):
	meta_map = pickle.load(open('vocab_map_meta.p','rb'))
	music_map = pickle.load(open('vocab_map_music.p','rb'))

	outputFolder_test = folderName+'_test_encoded'
	makedir(outputFolder_test)
	outputFolder_train = folderName+'_train_encoded'
	makedir(outputFolder_train)

	p = Pool(8)
	testSongs = pickle.load(open('test_songs.p','rb'))
	mapList = []

	for filename in os.listdir(folderName):
		fromName = os.path.join(folderName,filename)
		outputFolder = outputFolder_test if (filename[:filename.find('_')] in testSongs) else outputFolder_train
		toName = os.path.join(outputFolder,filename.replace('.abc','.npy'))

		mapList.append((fromName,toName,meta_map,music_map))

	p.map(encodeABCWorker, mapList)

def npy2nnInputWorker(dataPack):
	stride_sz, window_sz, nnType, output_sz, outputFolder, meta, music, basename = dataPack

	count = 0
	while True:
		start_indx = count*stride_sz
		input_window = music[start_indx:start_indx+window_sz]

		if nnType=='char_rnn':
			output_start = start_indx+1
			output_end = start_indx+window_sz+1
		elif nnType=='seq2seq':
			output_start = start_indx+window_sz+1
			output_end = output_start + output_sz
		elif nnType=='BOW':
			output_start = start_indx+window_sz+1
			output_end = output_start+1
		else:
			print 'specify the correct nnType...'
			exit(0)

		if output_end>len(music):
			break
		output_window = music[output_start:output_end]

		tup = (meta, input_window, output_window)
		filename = os.path.join(outputFolder, basename.replace('.npy','')+'_'+str(count))
		pickle.dump(tup, open(filename+'.p','wb'))

		count += 1

def npy2nnInput(h5file, stride_sz, window_sz, nnType, output_sz=0, outputFolder=''):
	"""
	Converts encoded npy to an array of tuples for NN input

	@h5file 	- string / filename of h5 file to read from
	@stride_sz 	- int / stride size
	@window_sz 	- int / window size of the input
	@output_sz 	- int / window size of the output (only used for nnType='seq2seq')
	@nnType 	- string / nn to feed the generated data to.
			  	  'BOW' 'seq2seq' 'char_rnn'
	@outputFolder - string / filename of the output
	"""

	makedir(outputFolder)

	mapList = []
	with h5py.File(h5file,'r') as hf_in:
		for key in hf_in.keys():
			data = np.array(hf_in.get(key))
			meta,music = data[:7],data[7:]

			mapList.append((stride_sz, window_sz, nnType, output_sz, outputFolder, meta, music, key))

	p = Pool(8)
	p.map(npy2nnInputWorker, mapList)

def loadNNInput(foldername):
	input_list = []
	filenames = os.listdir(foldername)
	ten_percent = int(len(filenames)*0.1)
	run_sum = 0
	for i,filename in enumerate(filenames):
		# print a dot for every 10 percent of data read
		if i==run_sum:
			run_sum += ten_percent
			print '.'
		with open(os.path.join(foldername,filename)) as f:
			input_list.append(pickle.load(f))

	pickle.dump(input_list, open(foldername+'.p','wb'))
	return input_list

if __name__ == "__main__":
	print len(loadNNInput('/data/the_session_nn_input_train_window_50_stride_25'))
	print len(loadNNInput('/data/the_session_nn_input_test_window_50_stride_25'))
	print len(loadNNInput('/data/the_session_nn_input_test_window_100_stride_50'))

	# preprocessing pipeline
	#-----------------------------------
	# extractABCtxt('the_session')
	# abcFileChecker('the_session_cleaned')
	# generateVocab('the_session_cleaned_checked')
	# encodeABC('the_session_cleaned_checked')
	# abc2h5('/data/the_session_cleaned_checked_test_encoded','/data/encoded_data_test.h5')
	# abc2h5('/data/the_session_cleaned_checked_train_encoded','/data/encoded_data_train.h5')
	# npy2nnInput('/data/encoded_data_train.h5', 50, 100, 'char_rnn', outputFolder='/data/the_session_nn_input_train_window_100_stride_50')
	#-----------------------------------
	pass
