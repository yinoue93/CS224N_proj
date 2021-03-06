import os
import numpy as np
import pretty_midi
import re
import pickle
import random


import tensorflow as tf

from collections import Counter

CHECK_DIR = 'checked'

tf_ver = tf.__version__
SHERLOCK = (str(tf_ver) == '0.12.1')

# for Sherlock
if SHERLOCK:
    DIR_MODIFIER = '/scratch/users/nipuna1'
# for Azure
else:
    DIR_MODIFIER = '/data'

# Midi Related
#------------------------------------
def plotMIDI(file_name):
	"""
	Usage: plotMIDI('video_games/dw2.mid')
	"""
	midi_data = pretty_midi.PrettyMIDI(file_name)
	roll = midi_data.get_piano_roll()

	plt.matshow(roll[:,:2000], aspect='auto', origin='lower', cmap='magma')
	plt.show()


# File Manipulation Related
#------------------------------------
def makedir(outputFolder):
	if not os.path.exists(outputFolder):
		os.makedirs(outputFolder)

import h5py
def write2hdf5(filename, dict2store, compression="lzf"):
	"""
	Write items in a dictionary to an hdf5file
	@type   filename    :   String
	@param  filename    :   Filename of the hdf5 file to output to.
	@type   dict2store  :   Dict
	@param  dict2store  :   Dictionary of items to store. The value should be an array.

	Usage: write2hdf5('encoded_data.h5',{'data':os.listdir('the_session_cleaned_checked_encoded')})
	"""
	with h5py.File(filename,'w') as hf:
		for key,value in dict2store.iteritems():
			hf.create_dataset(key, data=value,compression=compression)


def hdf52dict(hdf5Filename):
	"""
	Loads an HDF5 file of a game and returns a dictionary of the contents
	@type   hdf5Filename:   String
	@param  hdf5Filename:   Filename of the hdf5 file.
	"""
	retDict = {}
	with h5py.File(hdf5Filename,'r') as hf:
		for key in hf.keys():
			retDict[key] = np.array(hf.get(key))

	return retDict

def abc2h5(folderName='the_session_cleaned_checked_encoded', outputFile='encoded_data.h5'):
	encodeDict = {}
	for filestr in os.listdir(folderName):
		encodeDict[filestr] = np.load(os.path.join(folderName,filestr))
	write2hdf5(outputFile,encodeDict)

def find_basename(filename):
	firstNum = re.search('[0-9]', filename).group(0)
	if filename[0]==firstNum:
		firstNum = re.search('_', filename).group(0)
	return filename[:(filename.find(firstNum)-1)]

def datasetSplit(folderName, setRatio):
	"""
	Split the dataset into training, testing, and dev sets.
	Usage: testTrainSplit('the_session_cleaned', (0.8,0.1,0.1))
	"""
	if sum(setRatio)!=1:
		print '[ERROR] datasetSplit(): %f+%f+%f does not equal 1...' \
				%(setRatio[0],setRatio[1],setRatio[2])
		exit(0)

	songlist = set()
	for filename in os.listdir(os.path.join(folderName, CHECK_DIR)):
		basename = find_basename(filename)
		songlist.add(basename)

	songlist = list(songlist)
	random.shuffle(songlist)

	train_test_split_indx = int(len(songlist)*setRatio[0])
	test_dev_split_indx = int(len(songlist)*(setRatio[0]+setRatio[1]))
	trainSongs = songlist[:train_test_split_indx]
	testSongs = songlist[train_test_split_indx:test_dev_split_indx]
	devSongs = songlist[test_dev_split_indx:]

	pickle.dump(trainSongs, open(os.path.join(folderName, 'train_songs.p'),'wb'))
	pickle.dump(testSongs, open(os.path.join(folderName, 'test_songs.p'),'wb'))
	pickle.dump(devSongs, open(os.path.join(folderName, 'dev_songs.p'),'wb'))

#------------------------------------

# .abc Related
#------------------------------------
def findNumMeasures(music):
	return music.replace('||','|').replace('|||','|').count('|')

def transposeABC(fromFile, toFile, shiftLvl):
	"""
	Transposes the .abc file in @fromFile by @shiftLvl and saves it to @toFile

	abc2abc.exe taken from http://ifdo.ca/~seymour/runabc/top.html
	"""

	# am I being ran on a Windows machine ('nt'), or a linux machine('posix')?
	if os.name=='posix':
		exe_cmd = 'abc2abc'
	elif os.name=='nt':
		exe_cmd = 'abcmidi_win32\\abc2abc.exe'

	cmd = '%s "%s" -t %d -e > "%s"' \
			%(exe_cmd,fromFile,shiftLvl,toFile)
	print toFile

	os.system(cmd)

MODE_MAJ = 0
MODE_MIN = 1
MODE_MIX = 2
MODE_DOR = 3
MODE_PHR = 4
MODE_LYD = 5
MODE_LOC = 6
def keySigDecomposer(line):
	"""
	Decompose the key signature into two portions- key and mode

	Returns:
	key - number of flats, negative for sharps
	mode - as defined by MODE_ constants
	"""

	# first determine the mode
	mode = MODE_MAJ

	searchList = [('mix',MODE_MIX),('dor',MODE_DOR),('phr',MODE_PHR),('lyd',MODE_LYD),
				  ('loc',MODE_LOC),('maj',MODE_MAJ),('min',MODE_MIN),('m',MODE_MIN),
				  ('p',MODE_PHR)]

	lower = line.lower()
	for searchTup in searchList:
		if searchTup[0] in lower:
			mode = searchTup[1]
			line = line[:lower.rfind(searchTup[0])]
			break

	# then determine the key
	keys = ['B#','E#','A#','D#','G#','C#','F#','B','E','A','D','G','C',
			'F','Bb','Eb','Ab','Db','Gb','Cb','Fb']
	mode_modifier = {MODE_MAJ:-12, MODE_MIN:-9, MODE_MIX:-11, MODE_DOR:-10,
					 MODE_PHR:-8, MODE_LYD:-13, MODE_LOC:-7}

	key = keys.index(line) + mode_modifier[mode]

	return str(key),str(mode)

def keySigComposer(num_flats, mode):
	"""
	Reverses keySigDecomposer
	"""
	num_flats = int(num_flats)
	mode = int(mode)

	keys = ['B#','E#','A#','D#','G#','C#','F#','B','E','A','D','G','C',
			'F','Bb','Eb','Ab','Db','Gb','Cb','Fb']
	mode_modifier = {MODE_MAJ:12, MODE_MIN:9, MODE_MIX:11, MODE_DOR:10, 
					 MODE_PHR:8, MODE_LYD:13, MODE_LOC:7}
	mode_name = {MODE_MAJ:'', MODE_MIN:'m', MODE_MIX:'mix', MODE_DOR:'dor', 
				 MODE_PHR:'phr', MODE_LYD:'lyd', MODE_LOC:'loc'}

	return 'K:%s%s\n' %(keys[mode_modifier[mode]+num_flats], mode_name[mode])

def loadCleanABC(abcname):
	"""
	Loads a file in .abc format (cleaned), and returns the meta data and music contained
	in the file.

	@meta - dictionary of metadata, key is the metadata type (ex. 'K')
	@music - string of the music
	"""
	headerTup = ('X', 'T', 'R', 'M', 'L', 'K', 'Q')

	meta = {}
	music = ''
	counter = len(headerTup)
	with open(abcname,'r') as abcfile:
		for line in abcfile:
			# break down the key signature into # of sharps and flats
			# and mode
			if counter>0:
				if line[0]=='K':
					try:
						meta['K_key'],meta['K_mode'] = keySigDecomposer(line[2:-1])
					except:
						print 'Key signature decomposition failed for file: ' + abcname
						raise Exception('Key signature decomposition failed for file: ' + abcname)
				elif line[0]=='M':
					if 'C' in line[2:-1]:
						meta['M'] = '4/4'
					else:
						meta['M'] = line[2:-1]
				else:
					meta[line[0]] = line[2:-1]
				counter -= 1
			else:
				music = line[:-1]

	notes = [chr(i) for i in range(ord('a'),ord('g')+1)]
	notes += [c.upper() for c in notes]
	# add metadata that we manually create
	meta['len'] = findNumMeasures(music)
	countList = Counter(music)
	timeSigNumerator = int(meta['M'][:meta['M'].find('/')])
	meta['complexity'] = (sum(countList[c] for c in notes)*100)/(meta['len']*timeSigNumerator)

	return meta,music

def writeCleanABC(meta, music, outfile=''):
	headerTup = ('X', 'T', 'R', 'M', 'L', 'K', 'Q')

	abcStr = ''
	for header in headerTup:
		if header=='K':
			abcStr += keySigComposer(meta['K_key'], meta['K_mode'])
		else:
			abcStr += '%s:%s\n' %(header, meta[header])

	abcStr += music + '\n'

	if len(outfile)!=0:
		with open(outfile,'w') as f:
			f.write(abcStr)

	return abcStr

import subprocess
def passesABC2ABC(fromFile):
	"""
	Returns true if the .abc file in @fromFile passes the abc2abc.exe check
	"""

	# am I being ran on a Windows machine ('nt'), or a linux machine('posix')?
	if os.name=='posix':
		cmd = 'abc2abc'
	elif os.name=='nt':
		cmd = 'abcmidi_win32\\abc2abc.exe'

	cmdlist = [cmd, fromFile]
	proc = subprocess.Popen(cmdlist, stdout=subprocess.PIPE)

	(out, err) = proc.communicate()

	# error check
	errorCnt_bar = out.count('Error : Bar')
	errorCnt = out.count('Error') - out.count('ignored')
	if errorCnt_bar>2 or errorCnt!=errorCnt_bar:
		return False
	elif errorCnt>0:
		barErrorList = re.findall('Bar [0-9]+', out)
		for i,barStr in enumerate(barErrorList):
			barErrorList[i] = int(re.search('[0-9]+',barStr).group(0))

		if barErrorList[0] == 1:
			errorCnt -= 1

		if abs(findNumMeasures(out)-barErrorList[-1])<3:
			errorCnt -= 1

	return errorCnt==0

def encoding2ABC(metaList, musicList, meta_map, music_map, outputname=None, 
				 vocab_dir=os.path.join(DIR_MODIFIER, 'the_session_processed')):
	"""
	Converts lists encoding of .abc song into .abc string

	@metaList 	- A list of encoded metadata
	@musicList	- A list of encoded music
	"""

	oneHotHeaders = ('R', 'M', 'L')

	meta_reverse = {}
	for header in meta_map.keys():
		meta_ori = meta_map[header]
		meta_reverse[header] = dict(zip(meta_ori.values(), meta_ori.keys()))

	music_reverse = dict(zip(music_map.values(), music_map.keys()))

	abcStr = 'X: 1\n'
	for i in range(len(oneHotHeaders)):
		header = oneHotHeaders[i]
		abcStr += '%s: %s\n' %(header, str(meta_reverse[header][metaList[i]]))

	num_flats = int(meta_reverse['K_key'][metaList[len(oneHotHeaders)]])
	mode = int(meta_reverse['K_mode'][metaList[len(oneHotHeaders)+1]])

	abcStr += keySigComposer(num_flats, mode)

	for music_ch in musicList:
		if music_ch<len(music_reverse):
			abcStr += music_reverse[music_ch]

	print 'Generated .abc file is: \n%s' %abcStr

	if outputname is not None:
		# am I being ran on a Windows machine ('nt'), or a linux machine('posix')?
		if os.name=='posix':
			exe_cmd = 'abc2midi'
		elif os.name=='nt':
			exe_cmd = 'abcmidi_win32\\abc2midi.exe'

		cmd = '%s "%s" -t %d -o %s' %(exe_cmd, outputname)
		print toFile

		os.system(cmd)

	return abcStr

def convertR(folderName, rName):
	"""
	Need to convert: china, france, march
	"""
	outfolder = folderName + '_R_fixed'
	makedir(outfolder)

	for fname in os.listdir(folderName):
		try:
			meta,music = loadCleanABC(os.path.join(folderName, fname))
			meta['R'] = rName
			writeCleanABC(meta, music, os.path.join(outfolder, fname))
		except:
			pass

def mvSongGenre(folderName, outputFolder, rName):
	"""
	Need to move: jig, waltz, polka, hornpipe, reel
	"""
	for fname in os.listdir(folderName):
		meta,music = loadCleanABC(os.path.join(folderName, fname))

		if meta['R']==rName:
			writeCleanABC(meta, music, os.path.join(outputFolder, fname))


def mergeDictionaries(dict1, dict2):
	"""
	To merge music and meta dictionaries:

	dict1 = pickle.load(open('/data/the_session_processed/vocab_map_music.p','rb'))
	dict2 = pickle.load(open('/data/gan_dataset/vocab_map_music.p','rb'))
	pickle.dump(mergeDictionaries(dict1, dict2), open('/data/global_map_music.p','wb'))

	dict1 = pickle.load(open('/data/the_session_processed/vocab_map_meta.p','rb'))
	dict2 = pickle.load(open('/data/gan_dataset/vocab_map_meta.p','rb'))

	dictN = {}
	for header in dict1.keys():
		dictN[header] = mergeDictionaries(dict1[header], dict2[header])

	pickle.dump(dictN, open('/data/global_map_meta.p','wb'))

	"""
	keys = set(dict1.keys()+dict2.keys())
	newDict = {}
	for i,k in enumerate(keys):
		newDict[k] = i

	return newDict

def randomABCGeneration(meta_map, music_map):
	metaList = []
	musicList = []

	oneHotHeaders = ('R', 'M', 'L', 'K_key', 'K_mode')
	for header in oneHotHeaders:
		metaList.append(random.choice(range(len(meta_map[header]))))

	c = 0
	while c!=len(music_map):
		c = random.choice(range(len(music_map)+1))
		musicList.append(c)

	return encoding2ABC(metaList, musicList)

if __name__ == "__main__":
	# convertR('/data/france_processed/checked', 'france')
	# for rname in ['jig', 'waltz', 'polka', 'hornpipe', 'reel']:
	# 	print rname
	# 	mvSongGenre('/data/the_session_processed/checked', '/data/gan_dataset', rname)

	pass
