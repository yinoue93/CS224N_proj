import pretty_midi
import matplotlib.pyplot as plt
import os
import re

from multiprocessing import Pool

def drummer():
	iterf = 0
	for kk in os.listdir('video_games'):
		try:
			if iterf%100 == 0:
				print iterf
			iterf += 1
			midi_data = pretty_midi.PrettyMIDI('video_games/'+kk)
			roll = midi_data.get_piano_roll()

			count = 0
			for i in midi_data.instruments:
				if i.is_drum:
					count += 1

			if count>1:
				ccount += 1
				#print kk
				#print '--'+str(count)
		except:
			print 'failed: '+kk
	print ccount

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
	with open(filename,'r') as infile:
		with open(outputname,'w') as outfile:
			for line in infile:
				line = line.replace(' ','')
				if line=='\n':
					continue
				if header:
					headerStr = re.match('[a-zA-Z]:',line)
					if headerStr is None:
						header = False
						for head in headerTup:
							if head in headerDict:
								outfile.write(headerDict[head])
						outfile.write(line.replace('\n', ''))
					elif headerStr.group()[0] in headerTup:
						headerDict[headerStr.group()[0]] = line
				else:
					outfile.write(line.replace('\n', ''))

			outfile.write('\n')


def extractABCtxt(folderName):
	outputFolder = folderName+'_txt_extracted'
	if not os.path.exists(outputFolder):
		os.makedirs(outputFolder)

	p = Pool(8)
	mapList = [(os.path.join(folderName,fname),os.path.join(outputFolder,fname)) 
										for fname in os.listdir(folderName)]

	p.map(extractABCtxtWorker, mapList)

if __name__ == "__main__":
	extractABCtxt('/data/the_session')
	# eraseUnreadable('video_games')
	#plotMIDI('video_games/dw2.mid')