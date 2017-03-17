from collections import Counter
from utils import makedir

import pickle
import os
import re
import random

import numpy as np

def generateVocab(foldername, filename):

	with open(filename, 'r') as f:
		inputStr = f.readline()

		charList = Counter(inputStr).keys()

	dict2Store = {}
	for i,letter in enumerate(charList):
		dict2Store[letter] = i

	# write out to a file
	pickle.dump(dict2Store, open(os.path.join(foldername, 'vocab_map_baseline.p'), 'wb'))

def encode(foldername, filename):
	outname = os.path.join(foldername, 'encoded.p')
	encodedList = []
	encodeMap = pickle.load(open(os.path.join(foldername, 'vocab_map_baseline.p'), 'rb'))

	with open(filename, 'r') as f:
		inputStr = f.readline()

		for inStr in inputStr:
			encodedList.append(encodeMap[inStr])

	with open(outname, 'wb') as f:
		pickle.dump(encodedList, f)

def datasetNNInput(foldername, inputSz):
	encodedName = os.path.join(foldername, 'encoded.p')
	encodedList = pickle.load(open(encodedName, 'rb'))

	makedir(os.path.join(foldername, 'inputs'))

	iterNum = int((len(encodedList)-1)/inputSz) - 1
	empty_meta = np.asarray([0]*7)
	for i in range(iterNum):
		if i%100==0:
			print '%d/%d' %(i,iterNum)

		startIndx = i*inputSz
		endIndx = (i+1)*inputSz
		inData = np.asarray(encodedList[startIndx:endIndx])
		labelData = np.asarray(encodedList[startIndx+1:endIndx+1])

		data = [empty_meta, inData, labelData]
		outname = os.path.join(foldername, 'inputs/input_%d.p' % i)

		with open(outname, 'wb') as f:
			pickle.dump(data, f)

def datasetSplit(folderName, setRatio):
	"""
	Split the dataset into training, testing, and dev sets.
	Usage: testTrainSplit('the_session_cleaned', (0.8,0.1,0.1))
	"""
	if sum(setRatio)!=1:
		print '[ERROR] datasetSplit(): %f+%f+%f does not equal 1...' \
				%(setRatio[0],setRatio[1],setRatio[2])
		exit(0)

	inputsFname = os.path.join(folderName, 'inputs')
	filelist = os.listdir(inputsFname)

	random.shuffle(filelist)

	train_test_split_indx = int(len(filelist)*setRatio[0])
	test_dev_split_indx = int(len(filelist)*(setRatio[0]+setRatio[1]))
	trainFiles = filelist[:train_test_split_indx]
	testFiles = filelist[train_test_split_indx:test_dev_split_indx]
	devFiles = filelist[test_dev_split_indx:]

	trainFilename = os.path.join(folderName, 'train')
	testFilename = os.path.join(folderName, 'test')
	devFilename = os.path.join(folderName, 'dev')
	makedir(trainFilename)
	makedir(testFilename)
	makedir(devFilename)

	inputfileList = [testFiles, trainFiles, devFiles]
	dirNames = [testFilename, trainFilename, devFilename]

	for itr in range(len(inputfileList)):
		nextSkip = len(inputfileList[itr]) / 8.0 + 1

		list2Save = []
		count = 0
		for i,fname in enumerate(inputfileList[itr]):
			fromfname = os.path.join(inputsFname, fname)
			with open(fromfname, 'rb') as f:
				list2Save.append(pickle.load(f))

			if i>nextSkip:
				outDirname = os.path.join(dirNames[itr], '%d.p' % count)
				with open(outDirname, 'wb') as f:
					pickle.dump(list2Save, f)

				list2Save = []
				count += 1
				nextSkip += len(inputfileList[itr]) / 8.0

		outDirname = os.path.join(dirNames[itr], '%d.p' % 7)
		with open(outDirname, 'wb') as f:
			pickle.dump(list2Save, f)


if __name__ == "__main__":
	# preprocessing pipeline
	#-----------------------------------
	filename = 'war_peace_cleaned.txt'
	processedDir = '/data/full_dataset/baseline'

	# print '-'*20 + 'GENERATING VOCAB' + '-'*20
	# generateVocab(processedDir, filename)

	# print '-'*20 + 'ENCODING' + '-'*20
	# encode(processedDir, filename)

	# print '-'*20 + 'FORMING NNINPUTS' + '-'*20
	# datasetNNInput(processedDir, 25)

	print '-'*20 + 'SPLITTING' + '-'*20
	datasetSplit(processedDir, (0.8,0.1,0.1))
	#-----------------------------------