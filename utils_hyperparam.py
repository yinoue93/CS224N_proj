import itertools
import pickle
import os
import datetime
import re
import ast
import tensorflow as tf

from argparse import ArgumentParser

import numpy as np

TMP_HYPER_PICKLE = 'tmp_hyperparam.p'
OUTPUT_FILE = 'grid_search_result.txt'


tf_ver = tf.__version__
SHERLOCK = (str(tf_ver) == '0.12.1')

if SHERLOCK:
    DIR_MODIFIER = '/scratch/users/nipuna1'
else:
    DIR_MODIFIER = '/data'

DEV_CKPT_DIR = DIR_MODIFIER + '/dev_ckpt'

def parseHyperTxt(paramTxtF):
	nameList = []
	paramList = []

	with open(paramTxtF, 'r') as paramF:
		count = 0
		for line in paramF:
			if count==0:
				modelType = line.replace('\n','').replace('\r','')
				count += 1
				continue
			elif count==1:
				num_epochs = int(line.replace('\n','').replace('\r',''))
				count += 1
				continue

			if '[' in line:
				name = [s.strip() for s in line.split(',')][0]
				listStr = line[line.find('['):(line.rfind(']')+1)]
				params = ast.literal_eval(listStr)
			else:
				name,start,end,step = [s.strip() for s in line.split(',')]

				startNum = float(start)
				endNum = float(end)
				stepNum = float(step)

				if startNum == endNum:
					params = [startNum]
				else:
					params = list(np.arange(startNum,endNum,stepNum))
					params = [round(a,5) for a in params]
					# python list is not inclusive
					if params[-1]!=endNum:
						params.append(endNum)

				params = [(int(par) if par.is_integer() else par) for par in params]

			nameList.append(name)
			paramList.append(params)

	return modelType,num_epochs,nameList,paramList

def runHyperparam(paramTxtF, dataset):
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

	dataset_train = dataset
	dataset_dev = dataset.replace('train','dev')

	count = 0
	for param in param_all_combos:
		# create the param list and pickle it to TMP_HYPER_PICKLE
		paramDict = {}
		paramStrList = []
		for name,par in zip(nameList,param):
			paramDict[name] = par
			
			paramStrList.append('{}: {}'.format(name, par))

		paramStr = ','.join(paramStrList) + '\n'

		pickle.dump(paramDict, open(TMP_HYPER_PICKLE, 'wb'))

		with open(OUTPUT_FILE, 'a') as f:
			f.write(paramStr)

		# run the model
		print '='*80
		print 'Testing model with param: %s' % str(paramDict)
		print '='*80

		# train the model using the new hyperparameters
		print '-'*30 + 'TRAINING' + '-'*30
		if dataset=='':
			cmd = 'python run.py -p train -o -ckpt %s -m %s -e %d -c %s' \
								%(DEV_CKPT_DIR, modelType, num_epochs, TMP_HYPER_PICKLE)
		else:
			cmd = 'python run.py -p train -o -ckpt %s -m %s -e %d -c %s -data %s' \
								%(DEV_CKPT_DIR, modelType, num_epochs, TMP_HYPER_PICKLE, dataset_train)
		os.system(cmd)

		# test the model on the dev set
		print '-'*30 + 'TESTING DEV' + '-'*30
		if dataset=='':
			cmd = 'python run.py -p dev -ckpt %s -m %s -c %s' \
								%(DEV_CKPT_DIR, modelType, TMP_HYPER_PICKLE)
		else:
			cmd = 'python run.py -p dev -ckpt %s -m %s -c %s -data %s' \
								%(DEV_CKPT_DIR, modelType, TMP_HYPER_PICKLE, dataset_dev)
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
		setattr(config, key, val)

	setattr(config, 'dev_filename', OUTPUT_FILE)

def resultParser(resultFname, top_N=3):
	"""
	Usage: python utils_hyperparam.py -m results -f grid_search_result_tmp.txt

	"""
	
	top_N_acc = np.array([0]*top_N, dtype=np.float32)
	top_N_str = ['']*top_N
	prev_str = ''
	config_map = None
	accuracy_list = []
	with open(resultFname, 'r') as f:
		for line in f:
			line = line.replace('\n', '')
			if 'Dev set accuracy' in line:
				accuracy = float(re.search('0.[0-9]+', line).group(0))
				accuracy_list.append(accuracy)

				if accuracy>top_N_acc[0]:
					top_N_acc[0] = accuracy
					top_N_str[0] = prev_str

					indx = np.argsort(top_N_acc)
					top_N_acc = top_N_acc[indx]
					top_N_str = [top_N_str[ind] for ind in list(indx)]
			else:
				prev_str = line

				tokenList = re.findall('[a-zA-Z_]+: [0-9]+\.*[0-9]*', line)
				configList = [re.match('[a-zA-Z_]+', token).group(0) for token in tokenList]
				valList = [float(re.findall('[0-9]+\.*[0-9]*', token)[0]) 
													for token in tokenList]

				if config_map==None:
					config_map = []
					for configStr in configList:
						config_map.append([])

				for i,val in enumerate(valList):
					config_map[i].append(val)


	print '-'*25 + 'Config. with top accuracy:' + '-'*25
	print np.flipud(top_N_acc)
	print '\n'.join(list(reversed(top_N_str)))
	print '-'*50

	print '='*25 + '"Flattened" Accuracies:' + '='*25
	accuracy_list = np.asarray(accuracy_list)
	for config_idx,configStr in enumerate(configList):
		print '*'*50
		print configStr

		config_list = np.asarray(config_map[config_idx])
		for level in sorted(list(set(config_list))):
			indices = (config_list==level)
			print '%0.3f:%0.5f' %(level,np.mean(accuracy_list[indices]))

	print '='*50

if __name__ == "__main__":
	desc = u'{0} [Args] [Options]\nDetailed options -h or --help'.format(__file__)
	parser = ArgumentParser(description=desc)

	parser.add_argument('-m', choices = ['results','tune'], type = str,
						dest = 'mode', required = True, help = 'Specify which mode to run')
	parser.add_argument('-f', type = str, default='', 
						dest = 'filename', help = 'Filename to read results from')
	parser.add_argument('-data', type = str, default='', 
						dest = 'dataset', help = 'Dataset to run run.py')
	parser.add_argument('-n', type = int, default=3,
						dest = 'top_N', help = 'Top N accuracies')

	args = parser.parse_args()

	if args.mode == 'tune':
		hyperparamTxt = 'hparams_seq2seq.txt'
		runHyperparam(hyperparamTxt, args.dataset)
	elif args.mode == 'results':
		resultParser(args.filename, args.top_N)