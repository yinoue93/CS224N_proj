import pretty_midi
import matplotlib.pyplot as plt
import os
import re

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
		fileStr = infile.read().replace(' ','')
		if fileStr.replace('\n','')=='':
			return

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
								outfile.write(headerDict[head]+'\n')
							else:
								outfile.write(headerDefault[head]+'\n')
						outfile.write(line)

					elif headerStr.group()[0] in headerTup:
						headerDict[headerStr.group()[0]] = line

				else:
					outfile.write(line)

			outfile.write('\n')


def extractABCtxt(folderName):
	outputFolder = folderName+'_txt_extracted'
	if not os.path.exists(outputFolder):
		os.makedirs(outputFolder)

	p = Pool(8)
	filenames = [re.sub(r'[^\x00-\x7f]',r'',fname) for fname in os.listdir(folderName)]
	mapList = [(os.path.join(folderName,fname),os.path.join(outputFolder,fname2)) 
										for fname,fname2 in zip(os.listdir(folderName),filenames)]

	p.map(extractABCtxtWorker, mapList)


if __name__ == "__main__":
	#extractABCtxt('the_session')
	extractABCtxtWorker(('the_session/A French Dance jig_0.abc','tmp.abc'))
	# eraseUnreadable('video_games')
	#plotMIDI('video_games/dw2.mid')