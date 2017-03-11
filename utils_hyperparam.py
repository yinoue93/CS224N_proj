import itertools
import pickle
import os
import datetime
import re

from argparse import ArgumentParser

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
			params = [round(a,5) for a in params]
			# python list is not inclusive
			if params[-1]!=float(end):
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

	print '[INFO] There are %d combinations of hyperparameters...' %len(param_all_combos)

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

def resultParser(resultFname, top_N=3):
	"""
	Usage: python utils_hyperparam.py -m results -f grid_search_result_tmp.txt
	"""
	
	top_N_acc = np.array([0]*top_N, dtype=np.float32)
	top_N_str = ['']*top_N
	prev_str = ''
	with open(resultFname, 'r') as f:
		for line in f:
			line = line.replace('\n', '')
			if 'Dev set accuracy' in line:
				accuracy = float(re.search('0.[0-9]+', line).group(0))

				if accuracy>top_N_acc[0]:
					top_N_acc[0] = accuracy
					top_N_str[0] = prev_str

					indx = np.argsort(top_N_acc)
					top_N_acc = top_N_acc[indx]
					top_N_str = [top_N_str[ind] for ind in list(indx)]
			else:
				prev_str = line

	print np.flipud(top_N_acc)
	print '\n'.join(list(reversed(top_N_str)))

if __name__ == "__main__":
	desc = u'{0} [Args] [Options]\nDetailed options -h or --help'.format(__file__)
	parser = ArgumentParser(description=desc)

	parser.add_argument('-m', choices = ['results','tune'], type = str,
						dest = 'mode', required = True, help = 'Specify which mode to run')
	parser.add_argument('-f', type = str, default='', 
						dest = 'filename', help = 'Filename to read results from')
	parser.add_argument('-n', type = int, default=3,
						dest = 'top_N', help = 'Top N accuracies')

	args = parser.parse_args()

	if args.mode == 'tune':
		hyperparamTxt = 'hyperparameters.txt'
		runHyperparam(hyperparamTxt)
	elif args.mode == 'results':
		resultParser(args.filename, args.top_N)