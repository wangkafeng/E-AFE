#import logging  # kafeng
from torch import rand
from Agents import Agents, Agents_sequence,Agents_pure
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool, cpu_count, Process
import multiprocessing
from utils_sklearn import mod_column, evaluate
from collections import ChainMap
from subprocess import Popen, PIPE
from time import time, sleep
import os
import random
import tensorflow as tf

import datetime
import time
import utils_sklearn
import copy
from gp import args, num_feature, orig_features, target_label, orig_data, infos, num_process
from WeightedMinHashToolbox.WeightedMinHash import WeightedMinHash  # weiwu
import pickle


def transformation_search_space(actions):
	action_per_feature = int(len(actions) / num_feature)   # 40/8 = 5
	transformed_feturs, max_order_fetrs = {}, []

	for feature_count in range(num_feature):      # 8
		feature_name = orig_data.columns[feature_count]
		feature_orders_actions = actions[feature_count*action_per_feature: (feature_count+1)*action_per_feature]  # 
		transformed_feturs[feature_count] = []     # kafeng 2 demension  8 * 5

		#print('feature_orders_actions = ', feature_orders_actions)
		if feature_orders_actions[0] == 0:
			continue
		else:
			#print('feature_name = ', feature_name)
			fetr = np.array(orig_data[feature_name].values)  # 0-7 columns 

		for action in feature_orders_actions:  # 5
			if action == 0:  # EOF 1 # kafeng no transform max_orders, stop
				break
			elif action > 0 and action <= args.num_op_unary:   # 0-4 = 5 types
				# unary
				action_unary = action - 1
				if action_unary == 0:
					fetr = np.squeeze(np.sqrt(abs(fetr)))
				elif action_unary == 1:
					scaler = MinMaxScaler()
					fetr = np.squeeze(scaler.fit_transform(np.reshape(fetr,[-1,1])))
				elif action_unary == 2:
					while (np.any(fetr == 0)):
						fetr = fetr + 1e-5
					fetr = np.squeeze(np.log(abs(np.array(fetr)))) 
				elif action_unary == 3:
					while (np.any(fetr == 0)):
						fetr = fetr + 1e-5
					fetr = np.squeeze(1 / (np.array(fetr))) 
			else:   # 5-39 = 35 types    =5*7 
				# binary
				action_binary = (action-args.num_op_unary-1) // (num_feature-1)
				#print('action_binary = ', action_binary)
				rank = np.mod(action-args.num_op_unary-1, num_feature-1)
				#print('rank = ', rank)

				if rank >= feature_count:  #  kafeng ????
					rank += 1
				target_feature_name = orig_data.columns[rank]
				#print('target_feature_name = ', target_feature_name)
				target = np.array(orig_data[target_feature_name].values)

				if action_binary == 0:
					fetr = np.squeeze(fetr + target)  # 0-8  ,  actions decode column
				elif action_binary == 1:
					fetr = np.squeeze(fetr - target)
				elif action_binary == 2:
					fetr = np.squeeze(fetr * target)
				elif action_binary == 3:
					while (np.any(target == 0)):
						target = target + 1e-5
					fetr = np.squeeze(fetr / target)  
				elif action_binary == 4:
					fetr = np.squeeze(mod_column(fetr, orig_data[target_feature_name].values))  # kafeng this will generate all 0 column ?
			
			#if fetr.sum() == 0.0:
				#print('fetr = ', fetr)
				#print('action_binary = %d, rank = %d , feature_count = %d'%(action_binary, rank, feature_count))
				#print('feature_orders_actions 0 transfer = ', feature_orders_actions)

			transformed_feturs[feature_count].append(fetr)  # append 5 ,  1-5 transform orders 
		max_order_fetrs.append(fetr)  # append 8 , use the last action , just for test ????  no thing about train ???
	#print(transformed_feturs)

	return transformed_feturs, max_order_fetrs


def transformation_search_space_attention(actions):
	action_per_feature = int(len(actions) / num_feature)   # 40/8 = 5
	transformed_feturs, max_order_fetrs = {}, []

	for feature_count in range(num_feature):      # 8
		feature_name = orig_data.columns[feature_count]
		feature_orders_actions = actions[feature_count*action_per_feature: (feature_count+1)*action_per_feature]  # 
		transformed_feturs[feature_count] = []     # kafeng 2 demension  8 * 5

		#print('feature_orders_actions = ', feature_orders_actions)
		if feature_orders_actions[0] == 0:
			continue
		else:
			#print('feature_name = ', feature_name)
			fetr = np.array(orig_data[feature_name].values)  # 0-7 columns 

		for action in feature_orders_actions:  # 5
			if action == 0:  # EOF 1 # kafeng no transform max_orders, stop
				break
			elif action > 0 and action <= args.num_op_unary:   # 0-4 = 5 types
				# unary
				action_unary = action - 1
				if action_unary == 0:
					fetr = np.squeeze(np.sqrt(abs(fetr)))
				elif action_unary == 1:
					scaler = MinMaxScaler()
					fetr = np.squeeze(scaler.fit_transform(np.reshape(fetr,[-1,1])))
				elif action_unary == 2:
					while (np.any(fetr == 0)):
						fetr = fetr + 1e-5
					fetr = np.squeeze(np.log(abs(np.array(fetr)))) 
				elif action_unary == 3:
					while (np.any(fetr == 0)):
						fetr = fetr + 1e-5
					fetr = np.squeeze(1 / (np.array(fetr))) 
			else:   # 5-39 = 35 types    =5*7 
				# binary
				action_binary = (action-args.num_op_unary-1) // (num_feature-1)
				#print('action_binary = ', action_binary)
				rank = np.mod(action-args.num_op_unary-1, num_feature-1)
				#print('rank = ', rank)

				if rank >= feature_count:  #  kafeng ????
					rank += 1
				target_feature_name = orig_data.columns[rank]
				#print('target_feature_name = ', target_feature_name)
				target = np.array(orig_data[target_feature_name].values)

				if action_binary == 0:
					fetr = np.squeeze(fetr + target)  # 0-8  ,  actions decode column
				elif action_binary == 1:
					fetr = np.squeeze(fetr - target)
				elif action_binary == 2:
					fetr = np.squeeze(fetr * target)
				elif action_binary == 3:
					while (np.any(target == 0)):
						target = target + 1e-5
					fetr = np.squeeze(fetr / target)  
				elif action_binary == 4:
					fetr = np.squeeze(mod_column(fetr, orig_data[target_feature_name].values))  # kafeng this will generate all 0 column ?
			
			#if fetr.sum() == 0.0:
				#print('fetr = ', fetr)
				#print('action_binary = %d, rank = %d , feature_count = %d'%(action_binary, rank, feature_count))
				#print('feature_orders_actions 0 transfer = ', feature_orders_actions)
			if fetr.max() != fetr.min():
				transformed_feturs[feature_count].append(fetr)  # append 5 ,  1-5 transform orders 
			else:
				continue

		if fetr.max() != fetr.min():
			max_order_fetrs.append(fetr)  # append 8 , use the last action , just for test ????  no thing about train ???
		else:
			continue
	#print(transformed_feturs)

	return transformed_feturs, max_order_fetrs


def get_reword_train(actions):
	action_per_feature = int(len(actions) / num_feature)   # 40/8 = 5
	rewards = []

	#start_time = time.time()

	#print('get_reword_train actions = ', actions)	
	#transformed_feturs, max_order_fetrs = transformation_search_space(actions)  # minhash empty seqence.
	transformed_feturs, max_order_fetrs = transformation_search_space_attention(actions)

	# add for minhash
	if args.minhash:
		norm_data = pd.DataFrame()
		count = 0
		for i in range(len(transformed_feturs)):  #
			for j in range(len(transformed_feturs[i])):
				norm_new = (transformed_feturs[i][j]-transformed_feturs[i][j].min())/(transformed_feturs[i][j].max()-transformed_feturs[i][j].min())
				norm_data.insert(count, '%d'%(count), norm_new)  # title is str
				count = count + 1

		#print(norm_data)
		#print('norm_data.shape = ', norm_data.shape)
		weighted_set = norm_data.values
		wmh = WeightedMinHash(weighted_set, args.dimension_pcws, seed=0)
		if args.feature_extract_alg == 'PCWS':
			k, y, e = wmh.pcws()
		elif args.feature_extract_alg == 'ICWS':
			k, y, e = wmh.icws()
		elif args.feature_extract_alg == 'LICWS':
			k, e = wmh.licws()
		elif args.feature_extract_alg == 'CCWS':
			#k, y, e = wmh.ccws()
			k, y, e = wmh.ccws_pytorch()
			k = k.numpy()
		indexs = np.transpose(k.astype(np.int32))
		#print('indexs = ', indexs)
		pcws = pd.DataFrame()
		for col in range(norm_data.shape[1]):
			indexs_values = norm_data['%d'%(col)][indexs[:,col]].reset_index(drop=True)
			pcws = pd.concat([pcws, indexs_values], axis=1, ignore_index=True )

		test_x = pd.DataFrame(np.transpose(pcws.values))  # = meta_features
		pre_prob = opengl_rf.predict_proba(test_x)
		#print('pre_prob = ', pre_prob)
		# paser prob
		probs = []
		for i in range(len(transformed_feturs)):
			probs.append(pre_prob[i : i+ len(transformed_feturs[i])])

		former_result = origin_result
		former_max_order_fetrs = []  # original  # delete None  ValueError: Length of values does not match length of index ??

		#print('transformed_feturs.keys() = ', transformed_feturs.keys())  # continue
		for key in sorted(transformed_feturs.keys()):   # 0-7   len(transformed_feturs) = 8   # transformed_feturs is 2 demension  8 * 5  # 8 
			reward, former_result, return_fetr = get_reward_per_feature(transformed_feturs[key], action_per_feature, former_result, former_max_order_fetrs, probs[key])
			former_max_order_fetrs.append(return_fetr)   # kafeng  append the last columns of transformed_feturs[key] .  max_order 
			rewards += reward
		# print("rewards: {.3f}".format(rewards))  # list error
		# print("rewards: {}".format(rewards))
	
	#duration = time.time() - start_time
	#print('get_reword_train %s  duration = %.5f seconds' %(datetime.datetime.now(), duration))

	utils_sklearn.constructed_features_list.clear()  # kafeng add
	utils_sklearn.constructed_trees_list.clear()
	utils_sklearn.selected_features_list.clear()
	utils_sklearn.selected_fscore_list.clear()
	return rewards


def get_reword_init(actions):
	# global synthesis_time
	# kafeng add for test
	#actions = [6, 10, 12, 21, 24, 20, 37, 32, 10, 20, 2, 1, 32, 21, 6, 17, 22, 15, 21, 31, 34, 11, 11, 20, 19, 39, 11, 17, 32, 36, 20, 9, 4, 35, 24, 30, 34, 18, 4, 39]
	
	action_per_feature = int(len(actions) / num_feature)   # 40/8 = 5
	rewards = []

	#start_time = time.time()

	#print('get_reword_init actions = ', actions)	
	# syn_time = time.time()
	# transformed_feturs, max_order_fetrs = transformation_search_space(actions)  # minhash empty seqence.
	transformed_feturs, max_order_fetrs = transformation_search_space_attention(actions)
	# synthesis_time = synthesis_time + time.time() - syn_time

	# args.minhash:
	norm_data = pd.DataFrame()
	count = 0
	for i in range(len(transformed_feturs)):  #
		for j in range(len(transformed_feturs[i])):
			norm_new = (transformed_feturs[i][j]-transformed_feturs[i][j].min())/(transformed_feturs[i][j].max()-transformed_feturs[i][j].min())
			norm_data.insert(count, '%d'%(count), norm_new)  # title is str
			count = count + 1

	#print(norm_data)
	#print('norm_data.shape = ', norm_data.shape)
	weighted_set = norm_data.values
	wmh = WeightedMinHash(weighted_set, args.dimension_pcws, seed=0)
	if args.feature_extract_alg == 'PCWS':
		k, y, e = wmh.pcws()
	elif args.feature_extract_alg == 'ICWS':
		k, y, e = wmh.icws()
	elif args.feature_extract_alg == 'LICWS':
		k, e = wmh.licws()
	elif args.feature_extract_alg == 'CCWS':
		#k, y, e = wmh.ccws()
		k, y, e = wmh.ccws_pytorch()
		k = k.numpy()
	indexs = np.transpose(k.astype(np.int32))
	#print('indexs = ', indexs)
	pcws = pd.DataFrame()
	for col in range(norm_data.shape[1]):
		indexs_values = norm_data['%d'%(col)][indexs[:,col]].reset_index(drop=True)
		pcws = pd.concat([pcws, indexs_values], axis=1, ignore_index=True )

	test_x = pd.DataFrame(np.transpose(pcws.values))  # = meta_features
	pre_prob = opengl_rf.predict_proba(test_x)
	# print('pre_prob = ', pre_prob)
	# paser prob
	probs = []
	probs_positive = []
	for i in range(len(transformed_feturs)):
		probs.append(pre_prob[i : i+ len(transformed_feturs[i])])
		prob_pre = pre_prob[i : i+ len(transformed_feturs[i])]
		# print('prob_pre = ', prob_pre)
		# if prob_pre == []:
		# 	probs_positive.append([]) 
		# else:
			# prob_sum = 0
		for i in range(len(prob_pre)):
			# print(type(prob_pre[i]))
			# prob_sum += prob_pre[i][0]
			probs_positive.append(prob_pre[i][1])  # prob 0, 1
			
		for _ in range(action_per_feature - len(prob_pre)):   #  padded to 5
			probs_positive.append(0)
			# print('prob_sum = ', prob_sum)
			# probs_positive.append(prob_sum)  # only select positive  for RL init rewards

	# print('probs = ', probs)
	# print('len(probs_positive) = ', len(probs_positive))
	# print('probs_positive = ', probs_positive)
	return probs_positive

	former_result = origin_result
	former_max_order_fetrs = []  # original  # delete None  ValueError: Length of values does not match length of index ??

	for key in sorted(transformed_feturs.keys()):   # 0-7   len(transformed_feturs) = 8   # transformed_feturs is 2 demension  8 * 5  # 8 
		reward, former_result, return_fetr = get_reward_init_feature(transformed_feturs[key], action_per_feature, former_result, former_max_order_fetrs, probs[key])
		former_max_order_fetrs.append(return_fetr)   # kafeng  append the last columns of transformed_feturs[key] .  max_order 
		rewards += reward
	print("len(rewards) : ", len(rewards))
	print("rewards: ", rewards)

	#duration = time.time() - start_time
	#print('get_reword_init %s  duration = %.5f seconds' %(datetime.datetime.now(), duration))

	#print('utils_sklearn.constructed_features_list = ', utils_sklearn.constructed_features_list)  # save idx 
	#print('utils_sklearn.constructed_trees_list = ', utils_sklearn.constructed_trees_list)  # save trees 
	#print('len(utils_sklearn.constructed_features_list) = ', len(utils_sklearn.constructed_features_list))
	#print('len(utils_sklearn.constructed_trees_list) = ', len(utils_sklearn.constructed_trees_list))
	# utils_sklearn.constructed_features_list.clear()  # kafeng add
	# utils_sklearn.constructed_trees_list.clear()
	# utils_sklearn.selected_features_list.clear()
	# utils_sklearn.selected_fscore_list.clear()
	return rewards
	



def get_reword_test(actions):
	X = orig_features.copy()

	# kafeng add for test
	#actions = [6, 10, 12, 21, 24, 20, 37, 32, 10, 20, 2, 1, 32, 21, 6, 17, 22, 15, 21, 31, 34, 11, 11, 20, 19, 39, 11, 17, 32, 36, 20, 9, 4, 35, 24, 30, 34, 18, 4, 39]
	
	#print('get_reword_test actions = ', actions)
	transformed_feturs, max_order_fetrs = transformation_search_space(actions)
	
	# print('len(max_order_fetrs) = ', len(max_order_fetrs))  # 7, 8 
	for i in range(len(max_order_fetrs)):  # 
		X.insert(len(X.columns), '%d'%(len(X.columns)+1), max_order_fetrs[i])
	
	result = evaluate(X, target_label, args, origin_result)

	return result


def get_reward_per_feature(transformed_feturs, action_per_feature, former_result, former_max_order_fetrs, probs):
	X = orig_features.copy()

	# print('former_result = ', former_result)  # up and down update
	#print('len(transformed_feturs) = ', len(transformed_feturs))  # 5  or < 5
	#print('transformed_feturs = ', transformed_feturs)
	#print('len(former_max_order_fetrs) = ', len(former_max_order_fetrs)) # 0-7

	reward = []
	previous_result = former_result

	for i, former_fetr in enumerate(former_max_order_fetrs):     # old transform features
		if former_fetr != []:  # prevent former []
			X.insert(len(X.columns), '%d'%(len(X.columns)+1), former_fetr)  # 

	if args.minhash:
		# give a random dropout
		feature_live = np.random.uniform(0, 1, len(transformed_feturs))

		i = 0
		for fetr in transformed_feturs:  #  5   new transform features diffent orders
			#print('X = ', X)
			#print('X.columns.values.tolist() = ', X.columns.values.tolist())
			#if random.random() > 0.5:
			if feature_live[i] > probs[i][1]:  # negtive feature , don't dropout
				X.insert(len(X.columns), '%d'%(len(X.columns)+1), fetr)  # 

				current_result = evaluate(X, target_label, args, origin_result)

				# if current_result - previous_result < 0:   # scale to [0, +++]
				# 	reward.append(0)
				# else:
				reward.append(current_result - previous_result)
				previous_result = current_result
				del X['%d'%len(X.columns)]  #  X.shape[1]  #  delete use other order
			else:  # dropout
				# X.insert(len(X.columns), '%d'%(len(X.columns)+1), fetr)  # 
				# current_result = evaluate(X, target_label, args, origin_result)
				# reward.append(current_result - previous_result)
				# previous_result = current_result
				# print('dropout')
				reward.append(0)
			i = i+1

	for _ in range(action_per_feature - len(reward)):   # kafeng padded to 5
		# print('padded')
		reward.append(0)

	if len(transformed_feturs) == 0:
		return_fetr = []    # no feature be transformed
	else:
		return_fetr = transformed_feturs[-1]  # use the highest order transform, reserved
	return reward, previous_result, return_fetr


def get_reward_init_feature(transformed_feturs, action_per_feature, former_result, former_max_order_fetrs, probs):
	X = orig_features.copy()

	#print('len(transformed_feturs) = ', len(transformed_feturs))  # 5  or < 5
	#print('transformed_feturs = ', transformed_feturs)
	#print('len(former_max_order_fetrs) = ', len(former_max_order_fetrs)) # 0-7

	reward = []
	previous_result = former_result

	for i, former_fetr in enumerate(former_max_order_fetrs):     # old transform features
		if former_fetr != []:  # prevent former []
			X.insert(len(X.columns), '%d'%(len(X.columns)+1), former_fetr)  # 

	if args.minhash:
		# give a random dropout
		feature_live = np.random.uniform(0, 1, len(transformed_feturs))

		i = 0
		for fetr in transformed_feturs:  #  5   new transform features diffent orders
			#print('X = ', X)
			#print('X.columns.values.tolist() = ', X.columns.values.tolist())
			#if random.random() > 0.5:
			if feature_live[i] > probs[i][1]:  # negtive feature
				X.insert(len(X.columns), '%d'%(len(X.columns)+1), fetr)  # 

				current_result = evaluate(X, target_label, args, origin_result)

				reward.append(current_result - previous_result) 
				previous_result = current_result
				del X['%d'%len(X.columns)]  #  X.shape[1]  #  delete use other order
			i = i+1
	else:
		for fetr in transformed_feturs:  #  5   new transform features diffent orders
			#print('X = ', X)
			#print('X.columns.values.tolist() = ', X.columns.values.tolist())
			X.insert(len(X.columns), '%d'%(len(X.columns)+1), fetr)  # 

			current_result = evaluate(X, target_label, args, origin_result)

			reward.append(current_result - previous_result) 
			previous_result = current_result
			del X['%d'%len(X.columns)]  #  X.shape[1]  #  delete use other order

	for _ in range(action_per_feature - len(reward)):   # kafeng padded to 5
		reward.append(0)

	if len(transformed_feturs) == 0:
		return_fetr = []    # no feature be transformed
	else:
		return_fetr = transformed_feturs[-1]  # use the highest order transform, reserved
	return reward, previous_result, return_fetr



def train(model, l=None, p=None):
	global origin_result , evaluate_time
	
	origin_result = evaluate(orig_features, target_label, args, origin_result=0)

	best_result = origin_result
	print('origin_result  = ', origin_result)

	#return  # for less log from cpython
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init_op)
		
		model_result = -10000.0
		train_set, values = [], []  # for AC

		for epoch_count in range(args.epochs):
			#utils_sklearn.all_entropy_mean, utils_sklearn.all_f1_score = [], []
			# init
			concat_action = []
			probs_action = sess.run(tf.nn.softmax(model.concat_output))  # kafeng probs_action connect RL controller and features transfrom
			# print('probs_action init = ', probs_action)

			# sample actions           # kafeng how the controller have some thing to the sample actions ??
			for batch_count in range(args.num_batch):  # 32
				batch_action = []
				for i in range(probs_action.shape[0]):  # 40
					sample_action = np.random.choice(len(probs_action[i]), p=probs_action[i])  # 0-39  probs_action update epochs
					batch_action.append(sample_action)
				#print('batch_action = ', batch_action)
				concat_action.append(batch_action)
			# print('concat_action = ', concat_action)

			#start_time = time.time()	
			# get rewards
			if args.init_rl:
				rewards = []
				# print('len(concat_action) = ', len(concat_action))
				for action in concat_action:   # kafeng series actions
					rewards.append(get_reword_init(action))    # kafeng use minhash evaluation ......
					# print('len(action) = ', len(action))
					# sample_results = random.sample([i for i in range(0, 10)], 10)
					# sample_results = [random.randint(0, 1000) for i in range(len(action))]
					# sample_results = [random.random() for i in range(len(action))]
					# print('sample_results = ', sample_results)
					# rewards.append(sample_results)
				rewards = np.array(rewards)
			# print('rewards init = ', rewards)  # fertility 
			#duration = time.time() - start_time
			#print('get_reword_train %s  duration = %.5f seconds' %(datetime.datetime.now(), duration))

			if args.multiprocessing:
				pool = Pool(num_process)
				results = pool.map(get_reword_test, concat_action)
				pool.close()
				pool.join()
			else:
				results = []
				for action in concat_action:
					results.append(get_reword_test(action))         # 
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
				for i in range(model.num_action):   # 40   kafeng \lamda-return  ...
					base = rewards[:,i:]
					rewards_order = np.zeros_like(rewards[:,i])
					for j in range(base.shape[1]):
						order = j + 1
						base_order = base[:,0:order]
						alphas = []
						for o in range(order):
							alphas.append(pow(args.alpha, o))
						base_order = np.sum(base_order*alphas, axis=1)
						base_order = base_order * np.power(args.lambd, j)
						rewards_order = rewards_order.astype(float)
						rewards_order += base_order.astype(float)  # G t k
					rewards[:,i] = (1-args.lambd) * rewards_order  # kafeng  G t lambd   ,  TD

			# kafeng this concat_action is not sample by the controller ?  just ramdom sample ???
			# update policy params
			feed_dict = {model.concat_action: np.reshape(concat_action, [args.num_batch,-1]), model.rewards: np.reshape(rewards,[args.num_batch,-1])}
			loss_epoch = model.update_policy(feed_dict, sess)   # kafeng train controller ..


			# train
			concat_action = []
			probs_action = sess.run(tf.nn.softmax(model.concat_output))
			# print('probs_action  trian = ', probs_action)

			# sample actions           # kafeng how the controller have some thing to the sample actions ??
			for batch_count in range(args.num_batch):  # 32
				batch_action = []
				for i in range(probs_action.shape[0]):  # 40
					sample_action = np.random.choice(len(probs_action[i]), p=probs_action[i])  # 0-39  probs_action update epochs
					batch_action.append(sample_action)
				#print('batch_action = ', batch_action)
				concat_action.append(batch_action)
			# print('concat_action = ', concat_action)

			if args.multiprocessing:
				pool = Pool(num_process)
				rewards = np.array(pool.map(get_reword_train, concat_action))  # kafeng parallel actions
				pool.close()
				pool.join()
			else:
				rewards = []
				for action in concat_action:   # kafeng series actions
					rewards.append(get_reword_train(action))    # 
				rewards = np.array(rewards)
			# print('rewards train = ', rewards)  # fertility 

			if args.multiprocessing:
				pool = Pool(num_process)
				results = pool.map(get_reword_test, concat_action)
				pool.close()
				pool.join()
			else:
				results = []
				for action in concat_action:
					results.append(get_reword_test(action))         # 
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
				for i in range(model.num_action):   # 40   kafeng \lamda-return  ...
					base = rewards[:,i:]
					rewards_order = np.zeros_like(rewards[:,i])
					for j in range(base.shape[1]):
						order = j + 1
						base_order = base[:,0:order]
						alphas = []
						for o in range(order):
							alphas.append(pow(args.alpha, o))
						base_order = np.sum(base_order*alphas, axis=1)
						base_order = base_order * np.power(args.lambd, j)
						rewards_order = rewards_order.astype(float)
						rewards_order += base_order.astype(float)  # G t k
					rewards[:,i] = (1-args.lambd) * rewards_order  # kafeng  G t lambd

			# kafeng this concat_action is not sample by the controller ?  just ramdom sample ???
			# update policy params
			feed_dict = {model.concat_action: np.reshape(concat_action, [args.num_batch,-1]), model.rewards: np.reshape(rewards,[args.num_batch,-1])}
			loss_epoch = model.update_policy(feed_dict, sess)   # kafeng train controller ..

			# test
			probs_action = sess.run(tf.nn.softmax(model.concat_output))
			best_action = probs_action.argmax(axis=1)
			#print('len(best_action) = ', len(best_action))
			model_result = max(model_result, get_reword_test(best_action))

			# update best_result
			best_result = max(best_result, model_result)

			print('Epoch %d: loss = %.4f, origin_result = %.4f, lr = %.3f, \n model_result = %.4f, best_action = %s, \n best_result = %.4f' 
				% (epoch_count+1, loss_epoch, origin_result, args.lr, model_result, str(best_action), best_result))



def random_run(num_random_sample, model, l=None, p=None):
	global args, num_process
	samples = []

	for i in range(num_random_sample):  # 5
		sample = []
		for _ in range(model.num_action):  # 8 * 5
			sample.append(np.random.randint(model.num_op))  # 4 + (8-1)*5  + 1 = 40
		samples.append(sample)

	if args.multiprocessing:	
		pool = Pool(num_process)    
		#res = list(pool.map(get_reword, samples))
		#res = list(pool.map(get_reword_train, samples))
		res = list(pool.map(get_reword_test, samples))
		pool.close()
		pool.join()
	else:
		res = []
		for sample in samples:
			#res.append(get_reword(sample))
			#res.append(get_reword_train(sample))
			res.append(get_reword_test(sample))

	random_result = max(res)  # 5 or 32   * 8
	random_sample = samples[res.index(random_result)]

	return random_result, random_sample

random_max_result = 0.0 # save result
random_max_samples = []  # save samples 
def train_random(model, l=None, p=None):
	global origin_result, random_max_result, random_max_samples
	
	origin_result = evaluate(orig_features, target_label, args, origin_result=0)

	best_result = origin_result
	print('origin_result  = ', origin_result)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init_op)
		
		model_result = -10000.0

		for epoch_count in range(args.epochs):
			concat_action = []
			probs_action = sess.run(tf.nn.softmax(model.concat_output))  # kafeng probs_action connect RL controller and features transfrom
			# sample actions           # kafeng how the controller have some thing to the sample actions ??
			for batch_count in range(args.num_batch):  # 32
				batch_action = []
				for i in range(probs_action.shape[0]):  # 40
					sample_action = np.random.choice(len(probs_action[i]), p=probs_action[i])  # 0-39  probs_action update epochs
					batch_action.append(sample_action)
				concat_action.append(batch_action)
			
			# random result
			#for batch_count in range(args.num_batch):  # 32
			# train 32 + test 33
			#args.num_random_sample = len(concat_action) + len(concat_action) + 1
			args.num_random_sample = len(concat_action) * model.num_action
			#args.num_random_sample = 1
			random_result, random_sample = random_run(args.num_random_sample, model, l, p)  #  eval 188
	
			# save new random_result, random_sample 
			if random_result > random_max_result:
				random_max_result = random_result
				random_max_samples = random_sample

			print('Epoch %d: origin_result = %.4f, random_max_result = %.4f, random_max_samples = %s' 
				% (epoch_count+1, origin_result, random_max_result, str(random_max_samples)))



if __name__ == '__main__':
	start_time = time.time()

	synthesis_time = 0
	evaluate_time = 0
	evaluate_count = 0

	openml_model = 'openml_model.txt'   # 
	have_opengl_model = os.path.exists(openml_model)

	if have_opengl_model:
		opengl_rf = pickle.load(open(openml_model, 'rb'))
	
	if args.controller == 'random':
		#controller = Controller_random(args, num_feature) # random
		controller = Agents(args, num_feature)  # just use para
		controller.build_graph()
		train_random(controller)
	elif args.controller == 'transfer':
		controller = Agents(args, num_feature)  # just use para
		controller.build_graph()
		# train_transfer(controller)
	else:
		if args.controller == 'rnn':
			controller = Agents(args, num_feature)
		elif args.controller == 'pure':
			controller = Agents_pure(args, num_feature)  # pure
		elif args.controller == 'attention':
			controller = Agents_attention(args, num_feature)  # attention
		controller.build_graph()
		train(controller)  # sklearn

	print('synthesis_time = ', synthesis_time)
	print('evaluate_time = ', evaluate_time)
	print('evaluate_count = ', evaluate_count)

	duration = time.time() - start_time
	print('%s  duration = %.5f seconds' %(datetime.datetime.now(), duration))

