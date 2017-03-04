import pretty_midi
import matplotlib.pyplot as plt
import os
import re
import pickle

import numpy as np

from collections import Counter
from multiprocessing import Pool

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

def plotMIDI(file_name):
	midi_data = pretty_midi.PrettyMIDI(file_name)
	roll = midi_data.get_piano_roll()

	plt.matshow(roll[:,:2000], aspect='auto', origin='lower', cmap='magma')
	plt.show()

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
	if not os.path.exists(outputFolder):
		os.makedirs(outputFolder)

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

		with open(outputname,'w') as outfile:
			outfile.write(fileStr)


def abcFileChecker(folderName):
	outputFolder = folderName+'_checked'
	if not os.path.exists(outputFolder):
		os.makedirs(outputFolder)

	p = Pool(8)
	mapList = [(os.path.join(folderName,fname),os.path.join(outputFolder,fname)) 
										for fname in os.listdir(folderName)]

	p.map(abcFileCheckerWorker, mapList)


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

def loadCleanABC(abcname):
	"""
	Loads a file in .abc format (cleaned), and returns the meta data and music contained
	in the file. 
	
	@meta - dictionary of metadata, key is the metadata type (ex. 'K')
	@music - string of the music
	"""
	meta = {}
	counter = 6
	with open(abcname,'r') as abcfile:
		for line in abcfile:
			# break down the key signature into # of sharps and flats
			# and mode
			if counter==2:
				try:
					meta['K_key'],meta['K_mode'] = keySigDecomposer(line[2:-1])
				except:
					print 'Key signature decomposition failed for file: ' + abcname
					exit(0)
			if counter>0:
				meta[line[0]] = line[2:-1]
				counter -= 1
			else:
				music = line[:-1]

	notes = [chr(i) for i in range(ord('a'),ord('g')+1)]
	notes += [c.upper() for c in notes]
	# add metadata that we manually create
	meta['len'] = music.replace('||','|').replace('|||','|').count('|')
	countList = Counter(music)
	timeSigNumerator = int(meta['M'][:meta['M'].find('/')])
	meta['complexity'] = (sum(countList[c] for c in notes)*100)/(meta['len']*timeSigNumerator)

	return meta,music

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
	for filename in filenames:
		meta,music = loadCleanABC(filename)
		
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

	outputFolder = folderName+'_encoded'
	if not os.path.exists(outputFolder):
		os.makedirs(outputFolder)

	p = Pool(8)
	mapList = [(os.path.join(folderName,fname),os.path.join(outputFolder,fname.replace('.abc','.npy')),
				meta_map,music_map) for fname in os.listdir(folderName)]

	p.map(encodeABCWorker, mapList)

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

if __name__ == "__main__":
	# encodeABC('the_session_cleaned_checked')
	#generateVocab('the_session_cleaned_checked')
	#abcFileChecker('the_session_cleaned')
	#extractABCtxtWorker(("the_session/The Bugle Horn polka_3.abc",'tmp.abc'))
	# eraseUnreadable('video_games')
	#plotMIDI('video_games/dw2.mid')
	pass