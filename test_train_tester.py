import re
import os
import pickle
import datetime

from utils_hyperparam import OUTPUT_FILE

TMP_HYPER_PICKLE = 'tmp_hyperparam.p'

#-----------CHANGE THESE PARAMETERS--------------------------
TRAIN = '/data/full_dataset/char_rnn_dataset/nn_input_train_stride_25_window_10_nnType_char_rnn_shuffled'
CKPT_DIR = '/data/another/char_10/'
MODEL_TYPE = 'char'
#------------------------------------------------------------

TEST = TRAIN.replace('train', 'test')
DEV = TRAIN.replace('train', 'dev')

def runTests(ckptList, dataset):
	for ckptPath in ckptList:
		cmd = 'python run.py -p dev -ckpt %s -m %s -c %s -data %s' \
							%(ckptPath, MODEL_TYPE, TMP_HYPER_PICKLE, dataset)

		os.system(cmd)

def getTestTrainAccuracies():
	if os.path.exists(OUTPUT_FILE):
		os.remove(OUTPUT_FILE)

	# first scrape the model names
	ckptSet = set()
	for filename in os.listdir(CKPT_DIR):
		modelName = re.findall('model.ckpt-[0-9]+', filename)
		if len(modelName)==0:
			continue
		ckptSet.add(modelName[0])

	ckptList = []
	for i in range(len(ckptSet)):
		for j,cName in enumerate(ckptSet):
			if str(i) in cName:
				break

		ckptList.append(cName)

	ckptList = [os.path.join(CKPT_DIR, ckptName) for ckptName in ckptList]

	# dump a fake pickle file to trick run.py to think that we are doing
	# hyperparameter tuning
	emptyDict = {}
	pickle.dump(emptyDict, open(TMP_HYPER_PICKLE, 'wb'))

	with open(OUTPUT_FILE, 'a') as f:
		f.write('Train Dataset:\n')
	runTests(ckptList, TRAIN)

	with open(OUTPUT_FILE, 'a') as f:
		f.write('\nTest Dataset:\n')
	runTests(ckptList, TEST)

	with open(OUTPUT_FILE, 'a') as f:
		f.write('\nDev Dataset:\n')
	runTests(ckptList, DEV)

	# rename the result file with a timestamp
	now = datetime.datetime.now()
	resultName = '%s_%s_%s.txt' %(TRAIN[(TRAIN.rfind('/')+1):], MODEL_TYPE, 
							   now.strftime("%B_%d_%H_%M_%S"))

	with open(OUTPUT_FILE, 'r') as f, open(resultName, 'w') as g:
		txt = f.read()
		g.write(txt.replace('Dev set accuracy: ',''))

	os.remove(OUTPUT_FILE)



if __name__ == "__main__":
	getTestTrainAccuracies()