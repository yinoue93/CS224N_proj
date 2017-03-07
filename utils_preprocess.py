import matplotlib.pyplot as plt
import os
import re
import pickle

import numpy as np

from collections import Counter
from multiprocessing import Pool

from utils import *

FORMAT_DIR = 'formatted'
CHECK_DIR = 'checked'
ENCODE_TEST_DIR = 'test_encoded'
ENCODE_TRAIN_DIR = 'train_encoded'
ENCODE_DEV_DIR = 'dev_encoded'
NN_INPUT_TEST_DIR = 'nn_input_test'
NN_INPUT_TRAIN_DIR = 'nn_input_train'
NN_INPUT_DEV_DIR = 'nn_input_dev'

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

def formatABCtxtWorker(dataPack):
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

		with open(outputname,'w') as outfile:
			for line in fileStr.split('\n'):
				line = line.strip()
				# skip empty lines
				if line=='' or '%' in line:
					continue

				if header:
					headerStr = re.match('[a-zA-Z]:',line)

					if headerStr is None:
						header = False
						for head in headerTup:
							if head in headerDict:
								if head=='R':
									headerDict[head] = headerDict[head][0:2]+headerDict[head][2:].lower()
								outfile.write(headerDict[head]+'\n')
							else:
								outfile.write('%s:%s\n' %(head,headerDefault[head]))

					elif headerStr.group()[0] in headerTup:
						headerDict[headerStr.group()[0]] = line

				if not header:
					headerStr = re.match('[a-zA-Z]:',line)
					if (headerStr is not None) or line[:3]=='[V:':
						break

					# remove stuff inside of double quotes
					quotes = False
					addStr = ''
					for ch in line:
						if not quotes:
							addStr += ch
						if ch=='"':
							if quotes:
								addStr = addStr[:-1]
							quotes = not quotes

					outfile.write(addStr)

			outfile.write('\n')

def formatABCtxt(folderName, outputFolder):
	outputFolder = os.path.join(outputFolder, FORMAT_DIR)
	makedir(outputFolder)

	p = Pool(8)
	filenames = [re.sub(r'[^\x00-\x7f]',r'',fname) for fname in os.listdir(folderName)]
	mapList = [(os.path.join(folderName,fname),os.path.join(outputFolder,fname2)) 
										for fname,fname2 in zip(os.listdir(folderName),filenames)]

	p.map(formatABCtxtWorker, mapList)

MIN_MEASURES = 10
NUM_TRANSPOSITIONS = 3
def checkABCtxtWorker(dataPack):
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

		shift_cands = np.linspace(-5,6,12)
		np.delete(shift_cands,5)
		for shift in np.random.choice(shift_cands, NUM_TRANSPOSITIONS):
			transposeABC(outputname, outputname.replace('.abc','_%s.abc'%shift), shift)


def checkABCtxt(outputFolder):
	"""
	Checks if the file under @outputFolder meets requirements
	Also augments the file by transposing to 4 random keys
	"""
	folderName = os.path.join(outputFolder, FORMAT_DIR)
	outputFolder = os.path.join(outputFolder, CHECK_DIR)
	makedir(outputFolder)

	p = Pool(8)
	mapList = [(os.path.join(folderName,fname),os.path.join(outputFolder,fname)) 
										for fname in os.listdir(folderName)]

	p.map(checkABCtxtWorker, mapList)

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

	inputFolderName = os.path.join(foldername, CHECK_DIR)
	filenames = [os.path.join(inputFolderName,filename) for filename in os.listdir(inputFolderName)]
	for i,filename in enumerate(filenames):
		if i%1000==0:
			print i

		try:
			meta,music = loadCleanABC(filename)
		except:
			print filename
			continue

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
	pickle.dump(meta2Store, open(os.path.join(foldername, 'vocab_map_meta.p'),'wb'))
	pickle.dump(music2Store, open(os.path.join(foldername, 'vocab_map_music.p'),'wb'))

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

	# add the BEGIN token
	encodeList.append(len(music_map))

	# encode music data
	for c in music:
		encodeList.append(music_map[c])

	# add the END token
	encodeList.append(len(music_map)+1)

	np.save(outputname,np.asarray(encodeList))


def encodeABC(outputFolder):
	folderName = os.path.join(outputFolder, CHECK_DIR)

	meta_map = pickle.load(open(os.path.join(outputFolder, 'vocab_map_meta.p'),'rb'))
	music_map = pickle.load(open(os.path.join(outputFolder, 'vocab_map_music.p'),'rb'))

	outputFolder_test = os.path.join(outputFolder, ENCODE_TEST_DIR)
	makedir(outputFolder_test)
	outputFolder_train = os.path.join(outputFolder, ENCODE_TRAIN_DIR)
	makedir(outputFolder_train)
	outputFolder_dev = os.path.join(outputFolder, ENCODE_DEV_DIR)
	makedir(outputFolder_dev)

	p = Pool(8)
	testSongs = pickle.load(open(os.path.join(outputFolder, 'test_songs.p'),'rb'))
	trainSongs = pickle.load(open(os.path.join(outputFolder, 'train_songs.p'),'rb'))
	devSongs = pickle.load(open(os.path.join(outputFolder, 'dev_songs.p'),'rb'))
	mapList = []

	for filename in os.listdir(folderName):
		fromName = os.path.join(folderName,filename)
		song_basename = filename[:filename.find('_')]
		if song_basename in testSongs:
			outFolder = outputFolder_test
		elif song_basename in trainSongs:
			outFolder = outputFolder_train
		elif song_basename in devSongs:
			outFolder = outputFolder_dev
		toName = os.path.join(outFolder,filename.replace('.abc','.npy'))

		mapList.append((fromName,toName,meta_map,music_map))

	p.map(encodeABCWorker, mapList)

def npy2nnInputWorkerWorker(dataPack):
	stride_sz, window_sz, nnType, output_sz, fname = dataPack

	tupList = []

	# open the song (in numpy form)
	data = np.load(fname)
	meta,music = data[:7],data[7:]

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
		tupList.append(tup)

		count += 1

	# add another window which includes the last element
	if nnType=='char_rnn' or nnType=='BOW':
		start_indx = len(music)-window_sz-1
		output_start = start_indx+1
		output_end = start_indx+window_sz+1 if nnType=='char_rnn' else output_start+1 
	elif nnType=='seq2seq':
		start_indx = len(music)-window_sz-output_sz-1
		output_start = start_indx+window_sz+1
		output_end = output_start+output_sz

	tupList.append((meta, music[start_indx:start_indx+window_sz], music[output_start:output_end]))

	return tupList

def npy2nnInputWorker(dataPack):
	outfname,tupList = dataPack
	windowList = []
	for tup in tupList:
		windowList += npy2nnInputWorkerWorker(tup)

	pickle.dump(windowList, open(outfname,'wb'))

def npy2nnInput(outputFolder, stride_sz, window_sz, nnType, output_sz=0, num_buckets=8):
	"""
	Converts encoded npy to an array of tuples for NN input

	@outputFolder 	- string / filename of h5 file to read from
	@stride_sz 		- int / stride size
	@window_sz 		- int / window size of the input
	@output_sz 		- int / window size of the output (only used for nnType='seq2seq')
	@nnType 		- string / nn to feed the generated data to.
			 		  'BOW' 'seq2seq' 'char_rnn'
	@num_buckets	- int / number of files to generate
	"""

	dir_list = [(NN_INPUT_TEST_DIR, ENCODE_TEST_DIR), 
				(NN_INPUT_TRAIN_DIR, ENCODE_TRAIN_DIR), 
				(NN_INPUT_DEV_DIR, ENCODE_DEV_DIR)]

	for outDir,inDir in dir_list:
		inputList = []
		nnFolder = os.path.join(outputFolder, outDir)
		makedir(nnFolder)

		encodedDir = os.path.join(outputFolder, inDir)
		for fname in os.listdir(encodedDir):
			inputList.append((stride_sz, window_sz, nnType, output_sz, os.path.join(encodedDir, fname)))

		random.shuffle(inputList)
		mapList = []
		for i in range(num_buckets):
			mapList.append((os.path.join(nnFolder,'stride_%d_window_%d_nnType_%s_%d.p'%(stride_sz,window_sz,nnType,i)), 
							inputList[int(i*len(inputList)/num_buckets)
										:int((i+1)*len(inputList)/num_buckets)]))

		p = Pool(8)
		p.map(npy2nnInputWorker, mapList)

def shuffleDataset(foldername):
	dir_list = (NN_INPUT_TEST_DIR, NN_INPUT_TRAIN_DIR, NN_INPUT_DEV_DIR)

	for outdir in dir_list:
		originalDir = os.path.join(foldername,outdir)
		outFolder = originalDir+'_shuffled'
		makedir(outFolder)

		input_list = []
		filenames = os.listdir(originalDir)
		num_buckets = len(filenames)
		print 'Loading data'
		for filename in filenames:
			print filename
			with open(os.path.join(originalDir,filename),'r') as f:
				input_list += pickle.load(f)

		random.shuffle(input_list)

		print 'Done shuffling, saving the shuffled data...'
		for i,filename in enumerate(filenames):
			print filename
			with open(os.path.join(outFolder,filename),'w') as f:
				input_frac = input_list[int(i*len(input_list)/len(filenames))
										:int((i+1)*len(input_list)/len(filenames))]
				pickle.dump(input_frac, f)

if __name__ == "__main__":
	# formatABCtxtWorker(('tmp.abc','tmp2.abc'))
	# print passesABC2ABC('tmp2.abc')

	# preprocessing pipeline
	#-----------------------------------
	originalDataDir = 'small'
	processedDir = originalDataDir+'_processed'
	formatABCtxt(originalDataDir, processedDir)
	checkABCtxt(processedDir)
	datasetSplit(processedDir, (0.8,0.1,0.1))
	generateVocab(processedDir)
	encodeABC(processedDir)
	npy2nnInput(processedDir, 50, 100, 'char_rnn')
	shuffleDataset(processedDir)
	#-----------------------------------
	pass