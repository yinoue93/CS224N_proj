import itertools
import pickle
import os
import datetime

import numpy as np

TMP_HYPER_PICKLE = 'tmp_hyperparam.p'
OUTPUT_FILE = 'grid_search_result.txt'
DEV_CKPT_DIR = '/data/dev_ckpt'

def parseHyperTxt(paramTxtF):
	nameList = []
	paramList = []

	with open(paramTxtF, 'r') as paramF:
		count = 0
		for line in paramF:
			if count==0:
				modelType = line.replace('\n','')
				count += 1
				continue
			elif count==1:
				num_epochs = int(line.replace('\n',''))
				count += 1
				continue

			name,start,end,step = [s.strip() for s in line.split(',')]

			params = list(np.arange(float(start),float(end),float(step)))
			# python list is not inclusive
			params.append(float(end))

			nameList.append(name)
			paramList.append(params)

	return modelType,num_epochs,nameList,paramList

def runHyperparam(paramTxtF):
	"""
	Runs the gridsearch for hyperparameter tuning search.
	The grids are as defined in @paramTxtF
	"""

	if os.path.exists(OUTPUT_FILE):
		os.remove(OUTPUT_FILE)
	
	# parse the paramTxtF
	modelType, num_epochs, nameList, paramList = parseHyperTxt(paramTxtF)

	# create all combinations of params
	param_all_combos = list(itertools.product(*paramList))

	for param in param_all_combos:
		# create the param list and pickle it to TMP_HYPER_PICKLE
		paramDict = {}
		paramStrList = []
		for name,par in zip(nameList,param):
			paramDict[name] = par

			if name=='meta_embed':
				# hidden_size is always 5 times meta_embed
				paramDict['hidden_size'] = par*5
				paramStrList.append('hidden_size: %f' %(par*5))
			
			paramStrList.append('%s: %f' %(name,par))

		paramStr = ','.join(paramStrList) + '\n'

		pickle.dump(paramDict, open(TMP_HYPER_PICKLE, 'wb'))

		with open(OUTPUT_FILE, 'a') as f:
			f.write(paramStr)

		# run the model
		print '='*80
		print 'Testing model with param: %s' % str(paramDict)
		print '='*80

		# train the model using the new hyperparameters
		cmd = 'python run.py -p train -o -ckpt %s -m %s -e %d -c %s' %(DEV_CKPT_DIR, modelType, num_epochs, TMP_HYPER_PICKLE)
		os.system(cmd)

		# test the model on the dev set
		cmd = 'python run.py -p dev -ckpt %s -m %s -c %s' %(DEV_CKPT_DIR, modelType, TMP_HYPER_PICKLE)
		os.system(cmd)

	os.remove(TMP_HYPER_PICKLE)

	# rename the result file with a timestamp
	now = datetime.datetime.now()
	resultName = '%s_%s.txt' %(OUTPUT_FILE.replace('.txt',''), 
							   now.strftime("%B_%d_%H_%M_%S"))
	os.rename(OUTPUT_FILE, resultName)


def setHyperparam(config, hyperparam_path):
	"""
	To be called by a Config class.
	Reads TMP_HYPER_PICKLE, and sets the parameters of the @model accordingly.
	"""

	paramDict = pickle.load(open(hyperparam_path, 'rb'))

	for key,val in paramDict.iteritems():
		if type(getattr(config, key)) == int:
			setattr(config, key, int(val))
		else:
			setattr(config, key, val)

	setattr(config, 'dev_filename', OUTPUT_FILE)


if __name__ == "__main__":
	hyperparamTxt = 'hyperparameters.txt'
	runHyperparam(hyperparamTxt)