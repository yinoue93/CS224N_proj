import pretty_midi
import os
from shutil import copy
import numpy as np

import matplotlib.pyplot as plt

from multiprocessing import Pool, Lock, Array

import h5py
def write2hdf5(filename, dict2store, compression="lzf"):
	"""
	Write items in a dictionary to an hdf5file
	@type   filename    :   String
	@param  filename    :   Filename of the hdf5 file to output to.
	@type   dict2store  :   Dict
	@param  dict2store  :   Dictionary of items to store. The value should be an array.
	"""
	with h5py.File(filename,'w') as hf:
		for key,value in dict2store.iteritems():
			hf.create_dataset(key, data=value,compression=compression)

def eraseUnreadable(folderN):
	iterf = 0
	for fileN in os.listdir(folderN):
		fileAbsN = folderN+'/'+fileN
		try:
			if iterf%100 == 0:
				print iterf
			iterf += 1
			midi_data = pretty_midi.PrettyMIDI(fileAbsN)
		except:
			print 'failed: '+fileN
			os.remove(fileAbsN)

def plotMIDI(file_name):
	midi_data = pretty_midi.PrettyMIDI(file_name)
	roll = midi_data.get_piano_roll()

	plt.matshow(roll[:,:2000], aspect='auto', origin='lower', cmap='magma')
	plt.show()

def initChecker(l,clist):
	global LOCK,CHECKLIST
	LOCK = l
	CHECKLIST = clist

def executeParallel(func, mapList):
	"""
	multiprocess a function
	@type   func    :   function handle
	@param  func    :   function to multiprocess
	@type   mapList :   list
	@param  mapList :   arguments to be given to the function
	Usage:
		executeParallel(reduceNpySz,os.listdir('/data/augmented_roi_original'))
	"""

	p = Pool(8)
	p.map(func, mapList)

def checkTimeSignature(midi_data):
	timeSigs = midi_data.time_signature_changes

	return (len(timeSigs)==1) and (timeSigs[0].numerator==4 and timeSigs[0].denominator==4)

def checkerWorker(dataPack):
	filename,outputFolder = dataPack
	midi_data = pretty_midi.PrettyMIDI(filename)
	timeSigPass = checkTimeSignature(midi_data)

	checkPass = timeSigPass

	LOCK.acquire()
	if checkPass:
		CHECKLIST[0] = CHECKLIST[0]+1
	else:
		CHECKLIST[1] = CHECKLIST[1]+1
	if sum(CHECKLIST)%250==0:
		print CHECKLIST[:]
	LOCK.release()

	if outputFolder!=None:
		copy(filename,outputFolder)

def checker(folderName, outputFolder=None):
	if outputFolder != None:
		if not os.path.exists(outputFolder):
			os.makedirs(outputFolder)

	checkList = Array('i', [0,0])
	LOCK = Lock()
	mapList = [(os.path.join(folderName,fname),outputFolder) for fname in os.listdir(folderName)]
	p = Pool(8, initializer=initChecker, initargs=(LOCK,checkList))
	p.map(checkerWorker, mapList)

def convert2pianoRollWorker(dataPack):
	h5name,filenames = dataPack
	h5Dict = {}
	for fname in filenames:
		midi_data = pretty_midi.PrettyMIDI(fname)
		outname = os.path.basename(fname)
		outname = outname[:outname.rfind('.mid')]+'.npy'
		h5Dict[outname] = midi_data.get_piano_roll()
	
	write2hdf5(h5name, h5Dict)

def convert2pianoRoll(folderName):
	"""
	Converts .mid files to pianoroll .npys, and save them in .hdf5 format.
	"""
	outputFolder = folderName + '_pianoroll'
	if not os.path.exists(outputFolder):
		os.makedirs(outputFolder)

	p = Pool(8)
	filenames = [os.path.join(folderName,fname) for fname in os.listdir(folderName)][0:13]
	batchNum = 2000
	indx = 0
	mapList = []
	while indx<len(filenames):
		h5name = 'batch_%s_%d.h5' %(folderName,int(indx/batchNum))
		h5name = os.path.join(outputFolder,h5name)
		mapList.append((h5name,filenames[indx:min(indx+batchNum, len(filenames))]))
		indx += batchNum

	p.map(convert2pianoRollWorker, mapList)

def decompressHDF5Worker(dataPack):
	h5name,outname = dataPack

	with h5py.File(h5name,'r') as hf:
		for key in hf.keys():
			filename = os.path.join(outname,os.path.basename(key))
			np.save(filename, np.array(hf.get(key)))

def decompressHDF5(hdf5Name):
	"""
	Decompresses the HDF5 files in @hdf5Name, and saves the .npy files
	"""
	outputFolder = hdf5Name+'_decompressed'
	if not os.path.exists(outputFolder):
		os.makedirs(outputFolder)

	p = Pool(8)
	mapList = [(os.path.join(hdf5Name,fname),outputFolder) for fname in os.listdir(hdf5Name)]

	p.map(decompressHDF5Worker, mapList)

def segmentNpyWorker(dataPack):
	h5name,outname,segment_dur = dataPack

	with h5py.File(h5name,'r') as hf:
		for key in hf.keys():
			whole_npy = np.array(hf.get(key))
			base_filename = os.path.join(outname,os.path.basename(key)).replace('.npy','')

			# segment
			start_t = 0
			counter = 0
			num_notes,midi_duration = whole_npy.shape
			while start_t+segment_dur<midi_duration:
				filename = base_filename + ('_%d.npy' %counter)
				np.save(filename, whole_npy[:,start_t:(start_t+segment_dur)])

				start_t += segment_dur
				counter += 1

def segmentNpy(hdf5Name, segment_sec, fs):
	"""
	Segments the HDF5 files in @hdf5Name by @segment_sec*@fs, and saves the .npy files

	Usage: segmentNpy('video_games_pianoroll', 15, 100)
	"""
	outputFolder = hdf5Name+'_segmented'
	if not os.path.exists(outputFolder):
		os.makedirs(outputFolder)

	p = Pool(8)
	mapList = [(os.path.join(hdf5Name,fname),outputFolder,segment_sec*fs) 
				for fname in os.listdir(hdf5Name)]

	p.map(segmentNpyWorker, mapList)

def pianoroll2midi(pianoroll, outfilename, use_velocity=True, fs=100):
	"""
	Converts @pianoroll to a midi, and saves it to @outfilename
	Only uses 1 instrument

	@pianoroll: numpy matrix
	@outfilename: string

	Usage: pianoroll2midi(np.load('video_games_pianoroll_segmented/%28clockt%290.npy'),
					'sample.mid', use_velocity=False)
	"""
	pm = pretty_midi.PrettyMIDI(resolution=960, initial_tempo=120)
	instrument = pretty_midi.Instrument(0)

	if np.max(pianoroll)>127:
		pianoroll = pianoroll/np.max(pianoroll)*127
	if not use_velocity:
		pianoroll[np.nonzero(pianoroll)] = 100

	num_notes,midi_duration = pianoroll.shape
	duration_history = [0]*num_notes
	velocity_history = [0]*num_notes
	deltaT = 1.0/fs
	for timeIndx in range(midi_duration):
		timeshot = pianoroll[:,timeIndx]

		for noteIndx in range(num_notes):
			if (velocity_history[noteIndx]!=timeshot[noteIndx]) \
			   or (timeIndx==midi_duration-1 and timeshot[noteIndx]>0):
				if velocity_history[noteIndx]>0:
					start_t = duration_history[noteIndx]*deltaT
					end_t = timeIndx*deltaT
					note = pretty_midi.Note(velocity=int(velocity_history[noteIndx]), 
											pitch=noteIndx, 
											start=start_t, end=end_t)
					instrument.notes.append(note)

				velocity_history[noteIndx] = timeshot[noteIndx]
				duration_history[noteIndx] = timeIndx

	pm.instruments.append(instrument)
	pm.write(outfilename)
	
if __name__ == "__main__":
	#eraseUnreadable('video_games')
	#plotMIDI('video_games/dw2.mid')

	# checker('video_games','video_games_cleaned')

	# midi_data = pretty_midi.PrettyMIDI('video_games/Zelda_2-_Battle_Stage.mid')
	# piano = midi_data.get_piano_roll()
	# pianoroll2midi(piano, 'sample.mid')
	pass