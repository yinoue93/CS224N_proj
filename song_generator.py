import sys
import os
import re
import pickle

from argparse import ArgumentParser

from run import run_model

class ArgumentParserWannabe(object):
    pass

def generateSong(args):
	args_fake = ArgumentParserWannabe()
	args_fake.train = 'sample'
	args_fake.data_dir = ''
	args_fake.num_epochs = 1
	args_fake.ckpt_dir = ''
	args_fake.set_config = 'song_generator.p'
	args_fake.override = False
	args_fake.ran_from_script = True
	args_fake.warm_len = args.warm_len

	if args.temperature==0 and (args.model=='seq2seq' or args.model=='duet'):
		args_fake.temperature = None
	else:
		args_fake.temperature = args.temperature

	sys.stdout = open(os.devnull, "w")

	if len(args.real_song) != 0:
		args_fake.warmupData = '/data/full_dataset/handmade/' + args.real_song

	ckpt_modifier = '' if args.ckpt_num==-1 else ('model.ckpt-'+str(args.ckpt_num))
	
	if args.model=='seq2seq':
		args_fake.model = 'seq2seq'
		args_fake.ckpt_dir = '/data/another/seq2seq_25_2/'+ckpt_modifier

		paramDict = {'meta_embed':160, 'embedding_dims':100, 'keep_prob':0.8,
					 'attention_option':'bahnadau', 'bidirectional':False}
		with open(args_fake.set_config,'wb') as f:
			pickle.dump(paramDict, f)

		generated = run_model(args_fake)

	elif args.model=='char':
		args_fake.model = 'char'
		args_fake.ckpt_dir = '/data/another/char_50_2/'+ckpt_modifier

		paramDict = {'meta_embed':160, 'embedding_dims':20, 'keep_prob':0.8}
		with open(args_fake.set_config,'wb') as f:
			pickle.dump(paramDict, f)
		
		generated = run_model(args_fake)

	elif args.model=='cbow':
		args_fake.model = 'cbow'
		args_fake.ckpt_dir = '/data/another/cbow_ckpt/model.ckpt-8'

		paramDict = {'meta_embed':100, 'embedding_dims':60, 'keep_prob':0.8}
		with open(args_fake.set_config,'wb') as f:
			pickle.dump(paramDict, f)
		
		generated = run_model(args_fake)

	elif args.model=='duet':
		args_fake.model = 'seq2seq'
		args_fake.ckpt_dir = '/data/another/seq2seq_duet/'+ckpt_modifier
		args_fake.meta_map = 'full_dataset/duet_processed/vocab_map_meta.p'
		args_fake.music_map = 'full_dataset/duet_processed/vocab_map_music.p'
		args_fake.warmupData = '/data/full_dataset/duet_processed/checked'

		paramDict = {'meta_embed':160, 'embedding_dims':100, 'keep_prob':0.8,
					 'attention_option':'bahnadau', 'bidirectional':False}
		with open(args_fake.set_config,'wb') as f:
			pickle.dump(paramDict, f)
		
		generated = run_model(args_fake).replace('%','\n')

	generated = generated.replace('<start>','').replace('<end>','')

	long_num = re.findall('[0-9][0-9]+', generated)
	for longint in long_num:
		generated = generated.replace(longint, longint[0])

	sys.stdout = sys.__stdout__
	print '-'*50
	print generated

def parseCommandLineSong():
	desc = u'{0} [Args] [Options]\nDetailed options -h or --help'.format(__file__)
	parser = ArgumentParser(description=desc)

	print("Parsing Command Line Arguments...")
	requiredModel = parser.add_argument_group('Required Model arguments')
	requiredModel.add_argument('-m', choices = ["seq2seq", "char", "cbow", "duet"], type = str,
						dest = 'model', required = True, help = 'Type of model to run')

	parser.add_argument('-r', dest='real_song', default='', 
						type=str, help='Sample from a real song')
	parser.add_argument('-t', dest='temperature', default=1.0, 
						type=float, help='Temperature')
	parser.add_argument('-w', dest='warm_len', default=10, 
						type=int, help='Warm start length')
	parser.add_argument('-n', dest='ckpt_num', default=-1, 
						type=int, help='Checkpoint Number')

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parseCommandLineSong()

	generateSong(args)