import pretty_midi

import matplotlib.pyplot as plt

import os

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


if __name__ == "__main__":
	eraseUnreadable('video_games')
	#plotMIDI('video_games/dw2.mid')