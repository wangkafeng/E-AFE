import logging
from Agents import Agents, Agents_sequence, Agents_pure
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
import argparse
from multiprocessing import Pool, cpu_count, Process
import multiprocessing
#from utils import mod_column, evaluate, init_name_and_log, save_result
from utils import mod_column, init_name_and_log, save_result
from collections import ChainMap
from subprocess import Popen, PIPE
from time import time, sleep
import os
from Java_service import start_service_pool, stop_service_pool, find_free_port
import rpyc
import random
import tensorflow as tf

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn import svm

import pickle
from WeightedMinHashToolbox.WeightedMinHash import WeightedMinHash  # weiwu
# from sklearn.externals import joblib  # kafeng modify
import joblib

import datetime
import time

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def parse_args():
	parser = argparse.ArgumentParser(description="Run.")
	parser.add_argument('--num_op_unary', type=int,
		default=4, help='unary operation num')
	parser.add_argument('--num_op_binary', type=int,
		default=5, help='binary operation num')
	parser.add_argument('--max_order', type=int,
		default=5, help='max order of feature operation')
	parser.add_argument('--num_batch', type=int,
		default=32, help='batch num')
	parser.add_argument('--optimizer', nargs='?',
		default='adam', help='choose an optimizer')
	parser.add_argument('--lr', type=float,
		default=0.01, help='set learning rate')
	parser.add_argument('--epochs', type=int,
		default=100, help='training epochs')
	parser.add_argument('--evaluate', nargs='?',
		default='1-rae', help='choose evaluation method')
	parser.add_argument('--task', nargs='?',
		default='regression', help='choose between classification and regression')
	parser.add_argument('--dataset', nargs='?',
		default='PimaIndian', help='choose dataset to run')
	parser.add_argument('--model', nargs='?',
		default='RF', help='choose a model')
	parser.add_argument('--alpha', type=float,
		default=0.99, help='set discount factor')
	parser.add_argument('--lr_value', type=float,
		default=1e-3, help='value network learning rate')
	parser.add_argument('--RL_model', nargs='?',
		default='PG', help='choose RL model, PG or AC')
	parser.add_argument('--reg', type=float,
		default=1e-5, help='regularization')
	parser.add_argument('--controller', nargs='?',
		default='rnn', help='choose a controller')
	parser.add_argument('--num_random_sample', type=int,
		default=5, help='sample num of random baseline')
	parser.add_argument('--lambd', type=float,
		default=0.4, help='TD lambd')
	#parser.add_argument('--multiprocessing', type=bool,
	parser.add_argument('--multiprocessing', type=boolean_string,
		default=True, help='whether get reward using multiprocess')
	parser.add_argument('--package', nargs='?',
		default='weka', help='choose sklearn or weka to evaluate')
	#add num process here
	parser.add_argument('--num_process',nargs='?',
		default=64, help='process number used in parallel')
	
	parser.add_argument('--log_dir', nargs='?',
		default='log_tb', help='tensorboard log dir')

	parser.add_argument('--feature_extract_alg', nargs='?',
		default='CCWS', help='meta feature extract algorithm, such as minhash algorithm PCWS, statistic. CCWS 0.01 48 0.729, CCWS 0.01 52 0.90 ')
	parser.add_argument('--minhash', type=boolean_string, 
		default=False, help='whether get reward using multiprocess True or False')
	parser.add_argument('--dimension_pcws', type=int,
		default=52, help='pcws output length for feature vector. 32: 0.63, 48: 0.709, 52: 0.709 , 56: 0.63,  64:0.665  128: 0.7339  256: ')
	return parser.parse_args()


def get_reword(actions):
	global path, args, method, origin_result
	X = pd.read_csv(path)
	num_feature = X.shape[1] - 1
	action_per_feature = int(len(actions) / num_feature)
	copies, copies_run, rewards = {}, [], []

	for feature_count in range(num_feature):
		feature_name = X.columns[feature_count]
		feature_actions = actions[feature_count*action_per_feature: \
			(feature_count+1)*action_per_feature]
		copies[feature_count] = []
		if feature_actions[0] == 0:
			continue
		else:
			copy = np.array(X[feature_name].values)		

		for action in feature_actions:
			if action == 0:
				# EOF
				break
			elif action > 0 and action <= args.num_op_unary:
				# unary
				action_unary = action - 1
				if action_unary == 0:
					copy = np.squeeze(np.sqrt(abs(copy)))
				elif action_unary == 1:
					scaler = MinMaxScaler()
					copy = np.squeeze(scaler.fit_transform(np.reshape(copy,[-1,1])))
				elif action_unary == 2:
					while (np.any(copy == 0)):
						copy = copy + 1e-5
					copy = np.squeeze(np.log(abs(np.array(copy)))) ###   
				elif action_unary == 3:
					while (np.any(copy == 0)):
						copy = copy + 1e-5
					copy = np.squeeze(1 / (np.array(copy)))   ###  
			
			else:
				# binary
				action_binary = (action-args.num_op_unary-1) // (num_feature-1)
				rank = np.mod(action-args.num_op_unary-1, num_feature-1)

				if rank >= feature_count:
					rank += 1
				target_feature_name = X.columns[rank]

				target = np.array(X[target_feature_name].values) #  

				# if action_binary == 0:
				# 	copy = np.squeeze(copy + X[target_feature_name].values)
				# elif action_binary == 1:
				# 	copy = np.squeeze(copy - X[target_feature_name].values)
				# elif action_binary == 2:
				# 	copy = np.squeeze(copy * X[target_feature_name].values)
				# elif action_binary == 3:
				# 	copy = np.squeeze(copy / (X[target_feature_name].values+1e-3)) ###  
				# elif action_binary == 4:
				# 	copy = np.squeeze(mod_column(copy, X[target_feature_name].values))

				if action_binary == 0:
					copy = np.squeeze(copy + target)
				elif action_binary == 1:
					copy = np.squeeze(copy - target)
				elif action_binary == 2:
					copy = np.squeeze(copy * target)
				elif action_binary == 3:
					while (np.any(target == 0)):
						target = target + 1e-5
					copy = np.squeeze(copy / target) ###  
				elif action_binary == 4:
					copy = np.squeeze(mod_column(copy, X[target_feature_name].values))

			if copy.max() != copy.min():  # kafeng add for minhash
				copies[feature_count].append(copy)

		copies_run.append(copy)
	# print(copies)

	# kafeng add minhash
	transformed_feturs = copies
	#print('transformed_feturs = ', transformed_feturs)
	probs = []
	if args.minhash:
		norm_data = pd.DataFrame()
		count = 0
		#print('len(transformed_feturs) = ', len(transformed_feturs))
		for i in range(len(transformed_feturs)):  #
			#print('len(transformed_feturs[i]) = ', len(transformed_feturs[i]))  # kafeng max_order = 1 ,  len = 0 
			for j in range(len(transformed_feturs[i])):
				#print('transformed_feturs[i][j] = ', transformed_feturs[i][j])
				norm_new = (transformed_feturs[i][j]-transformed_feturs[i][j].min())/(transformed_feturs[i][j].max()-transformed_feturs[i][j].min())
				norm_data.insert(count, '%d'%(count), norm_new)  # title is str
				count = count + 1

		#print('len(norm_data) = ', len(norm_data))
		#print('norm_data = ', norm_data)
		weighted_set = norm_data.values
		#print('weighted_set = ', weighted_set)
		wmh = WeightedMinHash(weighted_set, args.dimension_pcws, seed=0)
		if args.feature_extract_alg == 'PCWS':
			k, y, e = wmh.pcws_pytorch()
			k = k.numpy()
		elif args.feature_extract_alg == 'ICWS':
			k, y, e = wmh.icws_pytorch()
			k = k.numpy()
		elif args.feature_extract_alg == 'LICWS':
			k, e = wmh.licws_pytorch()
			k = k.numpy()
		elif args.feature_extract_alg == 'CCWS':
			device = 'cpu'
			if device == 'cpu':
				k, y, e = wmh.ccws_pytorch()
				k = k.numpy()
			elif device == 'cuda':
				k, y, e = wmh.ccws_gpu()
				k = k.cpu().numpy()
			else:
				k, y, e = wmh.ccws()
		indexs = np.transpose(k.astype(np.int32))
		#print('indexs = ', indexs)
		pcws = pd.DataFrame()
		for col in range(norm_data.shape[1]):
			indexs_values = norm_data['%d'%(col)][indexs[:,col]].reset_index(drop=True)
			pcws = pd.concat([pcws, indexs_values], axis=1, ignore_index=True )

		test_x = pd.DataFrame(np.transpose(pcws.values))  # = meta_features
		#print('len(test_x) = ', len(test_x))
		pre_prob = opengl_rf.predict_proba(test_x)
		# pre_prob = opengl_mlp.predict_proba(test_x)   # kafeng predict all the same proba ????
		#print('pre_prob = ', pre_prob)
		# paser prob
		#probs = []
		num = 0
		for i in range(len(transformed_feturs)):
			#print('i = ', i)
			#print('len(probs) = ', len(probs))
			if len(transformed_feturs[i]) == 0:  # kafeng  for max_order = 1
				#continue
				probs.append([])
			else:
				probs.append(pre_prob[num : num+ len(transformed_feturs[i])])   # kafeng max_order = 1
				num = num + 1
		#print('len(probs) = ', len(probs))
		#print('probs = ', probs)
		#print('len(probs[0]) = ', len(probs[0]))

	if method == 'train':
		former_result = origin_result
		former_copys = [None]
		#print('copies.keys() = ', copies.keys())
		for key in sorted(copies.keys()):
			#print('key = ', key)
			reward, former_result, return_copy = get_reward_per_feature( 
				#copies[key], action_per_feature, former_result, former_copys)  # kafeng for minhash
				copies[key], action_per_feature, former_result, probs[key], former_copys)
			former_copys.append(return_copy)
			rewards += reward
		return rewards
	
	elif method == 'test':
		for i in range(len(copies_run)):
			X.insert(0, 'new%d'%i, copies_run[i])
		if args.package == 'weka':
			result = get_weka_result(X)
		elif args.package == 'sklearn':
			y = X[X.columns[-1]]
			del X[X.columns[-1]]
			result = evaluate(X, y, args)
			# print("result: ", result)
		return result
		

#def get_reward_per_feature(copies, count, former_result, former_copys=[None]):
def get_reward_per_feature(copies, count, former_result, probs, former_copys=[None]):
	global path, args
	
	X = pd.read_csv(path)
	if args.package == 'sklearn':
		y = X[X.columns[-1]]
		del X[X.columns[-1]]

	reward = []
	previous_result = former_result
	for i,former_copy in enumerate(former_copys):
		if not former_copy is None:
			X.insert(0, 'former%d'%i, former_copy)

	if args.minhash:
		# give a random dropout
		#print('len(copies) = ', len(copies))  # kafeng for max_order = 1  ,  len(copies) = 0
		#print('copies = ', copies)
		feature_live = np.random.uniform(0, 1, len(copies))
		#print('feature_live = ', feature_live)
		#print('probs = ', probs)

		i = 0
		for copy in copies:  #  5   new transform features diffent orders
			#if random.random() > 0.5:
			#print('feature_live[i] = ', feature_live[i])
			#print('probs[i][1] = ', probs[i][1])
			#print('i = ', i)
			if feature_live[i] > probs[i][1]:  # negtive feature
				X.insert(0, 'new', copy)
				if args.package == 'weka':
					current_result = get_weka_result(X)
				elif args.package == 'sklearn':
					current_result = evaluate(X, y, args)

				reward.append(current_result - previous_result)
				previous_result = current_result
				del X['new']
			i = i+1
	else:
		for copy in copies:
			X.insert(0, 'new', copy)
			if args.package == 'weka':
				current_result = get_weka_result(X)
			elif args.package == 'sklearn':
				current_result = evaluate(X, y, args)

			reward.append(current_result - previous_result)
			previous_result = current_result
			del X['new']

	reward_till_now = len(reward)
	for _ in range(count - reward_till_now):
		reward.append(0)
	if len(copies) == 0:
		return_copy = None
	else:
		return_copy = copies[-1]

	return reward, previous_result, return_copy

def random_run(num_random_sample, model, l=None, p=None):
	global args, num_process
	samples = []
	for i in range(num_random_sample):
		sample = []
		for _ in range(model.num_action):
			sample.append(np.random.randint(model.num_op))
		samples.append(sample)

	if args.multiprocessing:	
		if args.package == 'weka':
			pool = Pool(num_process, initializer=init, initargs=(l, p)) # num_proess // 2  
		elif args.package == 'sklearn':
			pool = Pool(num_process)    # num_proess // 2  
		res = list(pool.map(get_reword, samples))
		pool.close()
		pool.join()
	else:
		res = []
		for sample in samples:
			res.append(get_reword(sample))

	random_result = max(res)
	random_sample = samples[res.index(random_result)]

	return random_result, random_sample


def train(model, l=None, p=None):
	global path, args, infos, method, origin_result, num_process

	X = pd.read_csv(path)
	if args.package == 'weka':
		origin_result = get_weka_result(X)
	elif args.package == 'sklearn':
		y = X[X.columns[-1]]
		del X[X.columns[-1]]
		print(X.shape)
		origin_result = evaluate(X, y, args)	
	best_result = origin_result
	all_best_result = best_result
	print(origin_result)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		# kafeng add for tensorboard
		# nfs_best_result = tf.Variable(best_result, name="best_result")
		# our_best_result = tf.Variable(all_best_result, name="best_result")
		# tf.summary.scalar(args.dataset + "_pure_nfs_best_result", nfs_best_result)
		# tf.summary.scalar(args.dataset + "_pure_all_best_result", our_best_result)
		# summary_merge_op = tf.summary.merge_all()

		init_op = tf.group(tf.global_variables_initializer(), 
			tf.local_variables_initializer())
		# kafeng add for tensorboard
		#log_fold = os.path.join(args.log_dir, args.controller)  # minhash False
		log_fold = os.path.join(args.log_dir, args.controller + '_' + str(args.minhash) + '_' + args.package)
		#log_fold = './' + args.log_dir + '/' + args.controller + '_' + str(args.minhash) + '_' + args.cache_method
		if not os.path.exists(log_fold):
			os.makedirs(log_fold)
		writer = tf.summary.FileWriter(log_fold, sess.graph)
		sess.run(init_op)
		
		model_result = -10000.0 #   model_result = 0
		train_set, values = [], []
		for epoch_count in range(args.epochs):
			concat_action = []
			probs_action = sess.run(tf.nn.softmax(model.concat_output))

			# sample actions
			for batch_count in range(args.num_batch):
				batch_action = []
				for i in range(probs_action.shape[0]):
					sample_action = np.random.choice(len(probs_action[i]), p=probs_action[i])
					batch_action.append(sample_action)
				concat_action.append(batch_action)
			# print(concat_action)
				
			# get rewards			
			method = 'train'
			if args.multiprocessing:
				if args.package == 'weka':
					pool = Pool(num_process, initializer=init, initargs=(l, p))  # num_proess // 2  
				elif args.package == 'sklearn':
					pool = Pool(num_process)           # num_proess // 2  
				rewards = np.array(pool.map(get_reword, concat_action))
				pool.close()
				pool.join()
			else:
				rewards = []
				for action in concat_action:
					rewards.append(get_reword(action))
				rewards = np.array(rewards)

			method = 'test'
			if args.multiprocessing:
				if args.package == 'weka':
					pool = Pool(num_process, initializer=init, initargs=(l, p))  # num_proess // 2  
				elif args.package == 'sklearn':
					pool = Pool(num_process)           # num_proess // 2  
				results = pool.map(get_reword, concat_action)
				pool.close()
				pool.join()
			else:
				results = []
				for action in concat_action:
					results.append(get_reword(action))
			model_result = max(model_result, max(results))


			if args.RL_model == 'AC':
				# using actor critic
				target_set = []
				for batch_count in range(args.num_batch):
					action = concat_action[batch_count]
					for i in range(model.num_action):
						train_tmp = list(np.zeros(model.num_action, dtype=int))
						target_tmp = list(np.zeros(model.num_action, dtype=int))
						
						train_tmp[0:i] = list(action[0:i])
						target_tmp[0:i+1] = list(action[0:i+1])

						train_set.append(train_tmp)
						target_set.append(target_tmp)

				state = np.reshape(train_set, [-1,model.num_action])
				next_state = np.reshape(target_set, [-1,model.num_action])

				value = model.predict_value(next_state) * args.alpha + rewards.flatten()
				values += list(value)
				model.update_value(state, values)

				# compute estimate reward
				rewards_predict = model.predict_value(next_state) * args.alpha - \
					model.predict_value(state[-np.shape(next_state)[0]:]) + rewards.flatten()
				rewards = np.reshape(rewards_predict, [args.num_batch,-1])

			
			elif args.RL_model == 'PG':
				for i in range(model.num_action):
					base = rewards[:,i:]
					rewards_order = np.zeros_like(rewards[:,i])
					for j in range(base.shape[1]):
						order = j + 1
						base_order = base[:,0:order]
						alphas = []
						for o in range(order):
							alphas.append(pow(args.alpha, o))
						base_order = np.sum(base_order*alphas, axis=1)
						# base_order = base_order * pow(args.lambd, j)
						base_order = base_order * np.power(args.lambd, j) # 
						# rewards_order += base_order
						rewards_order = rewards_order.astype(float)  # 
						rewards_order += base_order.astype(float)  # 
					rewards[:,i] = (1-args.lambd) * rewards_order
				

			# update policy params

			feed_dict = {model.concat_action: np.reshape(concat_action, [args.num_batch,-1]), \
				model.rewards: np.reshape(rewards,[args.num_batch,-1])}
			loss_epoch = model.update_policy(feed_dict, sess)


			# test
			method = 'test'
			probs_action = sess.run(tf.nn.softmax(model.concat_output))
			best_action = probs_action.argmax(axis=1)
			model_result = max(model_result, get_reword(best_action))

			# random result
			random_result, random_sample = random_run(args.num_random_sample, model, l, p)

			# update best_result
			best_result = max(best_result, model_result)

			print('Epoch %d: loss = %.4f, origin_result = %.4f, lr = %.3f, \n model_result = %.4f, best_action = %s, \n best_result = %.4f, random_result = %.4f, random_sample = %s' 
				% (epoch_count+1, loss_epoch, origin_result, args.lr, model_result, str(best_action), best_result, random_result, str(random_sample)))
			logging.info('Epoch %d: loss = %.4f, origin_result = %.4f, lr = %.3f, \n model_result = %.4f, best_action = %s, \n best_result = %.4f, random_result = %.4f, random_sample = %s' 
				% (epoch_count+1, loss_epoch, origin_result, args.lr, model_result, str(best_action), best_result, random_result, str(random_sample)))

			info = [epoch_count, loss_epoch, origin_result, model_result, random_result]
			infos.append(info)

			# kafeng add
			all_best_result = max(all_best_result, max(all_f1_score))
			all_f1_score[:] = []  # ListProxy
			print('all_best_result = ', all_best_result)

			# Write logs at every iteration
			# sess.run(tf.assign(nfs_best_result, best_result))
			# sess.run(tf.assign(our_best_result, all_best_result))
			# summary_info = sess.run(summary_merge_op)
			# writer.add_summary(summary_info, epoch_count+1)
			# writer.flush()
		
def get_port():
	global lock, ports
	with lock:
		avail_port = [port for port in ports.keys() if ports[port]]
		# print("number of ports: " + str(len(avail_port)))
		port = random.choice(avail_port)
		ports[port] = False
	return port

def return_port(port):
	global lock, ports
	with lock:
		ports[port] = True

def init(l, p):
	global lock, ports
	lock = l
	ports = p

def get_weka_result(X):
	d = X.shape[0]
	X_tmp = str(X.values.tolist())
	port = get_port()
	conn = conns[port]
	result = conn.root.evaluate(X_tmp, d, args.task, args.model, args.evaluate)
	all_f1_score.append(result)  # kafeng 
	return_port(port)
	return result

def evaluate(X, y, args):
	# my_scorer = make_scorer(my_custom_loss_func, greater_is_better=True)
	if args.task == 'regression':
		if args.model == 'LR':
			# model = LinearRegression()
			model = Lasso()
			# model = Ridge()
		elif args.model == 'RF':
			model = RandomForestRegressor(n_estimators=10, random_state=0)
		if args.evaluate == 'mae':
			r_mae = cross_val_score(model, X, y, cv=5, 
				scoring='neg_mean_absolute_error').mean()
			return r_mae
		elif args.evaluate == 'mse':
			r_mse = cross_val_score(model, X, y, cv=5, 
				scoring='neg_mean_squared_error').mean()
			return r_mse
		elif args.evaluate == 'r2':
			r_r2 = cross_val_score(model, X, y, cv=5).mean()
			return r_r2

	elif args.task == 'classification':
		le = LabelEncoder()
		y = le.fit_transform(y)
		# print(np.isnan(X).any())
		# print(np.any(np.isnan(X)), np.all(np.isfinite(X)))
		# if np.any(np.isnan(X)) == True or np.all(np.isfinite(X)) == False:
		# 	print(np.where(np.isinf(X)))
		if args.model == 'RF':
			model = RandomForestClassifier(n_estimators=20, random_state=0) # n = 10
		elif args.model == 'LR':
			model = LogisticRegression(multi_class='ovr')
			# model = LogisticRegression()
		elif args.model == 'SVM':
			model = svm.SVC()
		if args.evaluate == 'f_score':
			# s = cross_val_score(model, X, y, scoring='f1', cv=5).mean() # cv = 5
			s = cross_val_score(model, X, y, scoring='f1_micro', cv=5).mean()
		elif args.evaluate == 'auc':
			model = RandomForestClassifier(max_depth=10, random_state=0)
			split_pos = X.shape[0] // 10
			X_train, X_test = X[:9*split_pos], X[9*split_pos:]
			y_train, y_test = y[:9*split_pos], y[9*split_pos:]
			model.fit(X_train, y_train)
			y_pred = model.predict_proba(X_test)
			s = evaluate_(y_test, y_pred)

		all_f1_score.append(s)
		return s

if __name__ == '__main__':
	start_time = time.time()
	openml_model = 'openml_model.md'
	# openml_model = 'mlp_model_CCWS_48_0.01.m'
	have_opengl_model = os.path.exists(openml_model)

	if have_opengl_model:
		opengl_rf = pickle.load(open(openml_model, 'rb'))
		# opengl_mlp = joblib.load(openml_model)

	args = parse_args()
	print('args = ', args)
	all_f1_score=multiprocessing.Manager().list()
	origin_result, method, name = None, None, None
	#num_process, infos = 64, []
	#num_process, infos = int(args.num_process), []
	print('cpu_count = ',multiprocessing.cpu_count())
	num_process, infos = multiprocessing.cpu_count() - 2, []
	num_weka_process = num_process
	# name = init_name_and_log(args)
	print(name)

	if args.package == 'weka':
		ports = [find_free_port() for _ in range(num_weka_process)]
		port_state = [True for _ in range(num_weka_process)]
		pids = start_service_pool(ports)
		sleep(5)
		conns = {}
		for port in ports:
			conns[port] = rpyc.connect('localhost', port)
		m = multiprocessing.Manager()
		lock = m.Lock()
		ports = m.dict(zip(ports, port_state))


	path = 'data/' + args.dataset + '.csv'
	
	num_feature = pd.read_csv(path).shape[1] - 1
	if args.controller == 'rnn':
		controller = Agents(args, num_feature)
	elif args.controller == 'pure':
		controller = Agents_pure(args, num_feature)
	controller.build_graph()

	if args.package == 'weka':
		train(controller, lock, ports)
	elif args.package == 'sklearn':
		train(controller)

	save_result(infos, name)
	
	if args.package == 'weka':
		stop_service_pool(pids)
		for port in conns.keys():
			conns[port].close()

	duration = time.time() - start_time
	print('%s  duration = %.5f seconds' %(datetime.datetime.now(), duration))