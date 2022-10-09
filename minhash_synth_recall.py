#import logging  # kafeng
from Controller import Controller, Controller_sequence, Controller_pure, Controller_attention, Controller_random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool, cpu_count, Process
import multiprocessing
from collections import ChainMap
from subprocess import Popen, PIPE
from time import time, sleep
import os
import random
import tensorflow as tf


# from gp
import datetime
import time
import copy
import argparse
import pandas as pd
import numpy as np
from xlutils.copy import copy
import shutil 
from sklearn.decomposition import KernelPCA
import math

#from utils_sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.feature_selection import SelectFromModel
from sklearn import svm

from WeightedMinHashToolbox.WeightedMinHash import WeightedMinHash  # weiwu

#import xgboost as xgb
#import lightgbm as lgb
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split,cross_val_score

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import precision_score,roc_auc_score, recall_score
from sklearn.preprocessing import LabelEncoder


import autosklearn.classification

#from joblib import dump, load
# from sklearn.externals import joblib 
import pickle

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

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
	default=1, help='training epochs')
parser.add_argument('--evaluate', nargs='?',
	default='f_score', help='choose evaluation method')  # 1-rae
parser.add_argument('--task', nargs='?',
	default='classification', help='choose between classification and regression')
parser.add_argument('--dataset', nargs='?',
	default='PimaIndian', help='choose dataset to run  PimaIndian')
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
	default='rnn', help='choose a controller, random, transfer, rnn, pure, attention')
parser.add_argument('--num_random_sample', type=int,
	default=5, help='sample num of random baseline')
parser.add_argument('--lambd', type=float,
	default=0.4, help='TD lambd')
#parser.add_argument('--multiprocessing', type=bool,
parser.add_argument('--multiprocessing', type=boolean_string,  # kafeng modify
	default=False, help='whether get reward using multiprocess True or False')
parser.add_argument('--package', nargs='?',
	default='sklearn', help='choose sklearn or weka to evaluate')
parser.add_argument('--num_process', type=int,
	default=48, help='process num')	 # kafeng add

parser.add_argument('--cache_method', nargs='?',
	default='no_cache', help='choose cache method, no_cache, selection ,or trees ')

parser.add_argument('--data', nargs='?',
	default='paperData/', help='datasests   openml/ ,  paperData/')
parser.add_argument('--train_data', nargs='?',
	default='openml/', help='datasests   openml/ ,  paperData/')
parser.add_argument('--test_data', nargs='?',
	default='paperData/', help='datasests   ../openml/ ,  paperData/')
parser.add_argument('--process_data', nargs='?',
	default='.preprocess_data', help='  .preprocess_data ,  .train_synth_data')
parser.add_argument('--synth_meta_feat', nargs='?',
	default='synth_meta_feat', help='save synth meta features  ')

parser.add_argument('--threshold', type=float,
	default=0.01, help='meta label threshold. f1 0.01, 0.015, 0.02,   recall 0.0')
parser.add_argument('--dimension_pcws', type=int,
	default=48, help='pcws output length for feature vector. 32: 0.63, 48: 0.709, 52: 0.709 , 56: 0.63,  64:0.665  128: 0.7339  256: ')
parser.add_argument('--PtoN', type=int,
	default=1, help='openml train posive to negtive')
parser.add_argument('--feature_extract_alg', nargs='?',
	default='CCWS', help='meta feature extract algorithm, such as minhash algorithm PCWS, statistic. CCWS 0.01 48 0.729 ')

parser.add_argument('--auto_skl_min', type=int,
	default=2, help='openml train posive to negtive')

args = parser.parse_args()
print('args = ', args)

num_process = args.num_process


def f1_eval(y_pred, dtrain):
	y_true = dtrain.get_label()
	#err = 1-f1_score(y_true, np.round(y_pred))
	#return 'f1_err', err
	return f1_score(y_true, np.round(y_pred))


def xgb_evaluate(X, y, num_class, metric='acc'):

	dtrain = xgb.DMatrix(X, label=y)

	if metric == 'f1':
		params = {
			# Parameters that we are going to tune.
			# Other parameters
			'objective': 'multi:softmax',
			'num_class': num_class,
			#'eval_metric': xgb_eval_f1,
			#'eval_metric': F1_eval,
			#'eval_metric': xgb.max_f1,
			'eval_metric': f1_eval,
			'verbosity': 0 # 0=slient、1=warning、2=info、3=debug
		}
	else:
		params = {
			# Parameters that we are going to tune.
			# Other parameters
			'objective': 'multi:softmax',
			'num_class': num_class,
			#'eval_metric': xgb_eval_f1,
			#'eval_metric': F1_eval,
			'verbosity': 0 # 0=slient、1=warning、2=info、3=debug
		}
	
	num_boost_round = 999

	cv_results = xgb.cv(
		params,
		dtrain,
		num_boost_round=num_boost_round,
		nfold=5,
		#metrics={'mae'},
		early_stopping_rounds=10,
		seed=0
	)
	
	#print(cv_results)

	s = 1- cv_results['test-merror-mean'].min()
	return s


def f1_error(preds,dtrain):
    label=dtrain.get_label()
    preds = 1.0/(1.0+np.exp(-preds))
    pred = [int(i >= 0.5) for i in preds]
    tp = sum([int(i == 1 and j == 1) for i,j in zip(pred,label)])
    precision=float(tp)/sum(pred)
    recall=float(tp)/sum(label)
    return 'f1-score',2 * ( precision*recall/(precision+recall) )

def xgb_evaluate2(X, y, num_class, metric='acc'):
	
	xgb_model = xgb.XGBClassifier()
	#print(xgb_model)

	cv = 5
	recall = cross_val_score(xgb_model, X, y, cv=cv, scoring='recall').mean()
	print('recall = ', recall)

	return recall


def lgb_f1_score(y_hat, data):
	y_true = data.get_label()
	y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
	#return 'f1', f1_score(y_true, y_hat), True
	print(y_true)  # len = 4
	print(y_hat)   # len = 8 ???
	return f1_score(y_true, y_hat)


def lgb_evaluate(X, y, num_class, metric='acc'):

	data_train = lgb.Dataset(X, y)
	
	params = {
		'objective': 'binary',
		#'metric' : "recall",  # error ??  custom_eval ???
		'verbose': -1
	}
	cv_results = lgb.cv(params, data_train, num_boost_round=1000, nfold=5, early_stopping_rounds=50, seed=0)
	#print(cv_results)
	s = 1- pd.Series(cv_results['binary_logloss-mean']).min()
	print('recall s = ', s)
	
	return s

def rf_evaluate(X, y):
	
	train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.5, random_state=0)
	rf = RandomForestClassifier()
	rf.fit(train_x,train_y)
	pre_test = rf.predict(test_x)
	test_y = np.array(test_y)
	#print('test_y = ', test_y)  
	
	#auc_score = roc_auc_score(test_y,pre_test)
	pre_score = precision_score(test_y,pre_test,labels={0,1})
	rec_score = recall_score(test_y,pre_test,labels={0,1})

	#print("auc_score = ", auc_score, 'pre_score = ', pre_score, 'rec_score = ', rec_score)
	print('pre_score = ', pre_score, 'rec_score = ', rec_score)


def paper_eval(train_x, test_x, train_y, test_y):
	# rf
	
	rf = RandomForestClassifier()
	rf.fit(train_x,train_y)
	pre_test = rf.predict(test_x)
	test_y = np.array(test_y)
	#print('test_y = ', test_y)  
	
	from sklearn.metrics import precision_score,roc_auc_score, recall_score
	#rec_score = recall_score(test_y,pre_test,labels={0,1})
	rec_score = recall_score(test_y,pre_test)  # 0.68

	print('rec_score = ', rec_score)
	
	# xgb 

	# svm 

	# auto sklearn
	
	#automl = autosklearn.classification.AutoSklearnClassifier()
	#automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120)
	automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60*args.auto_skl_min)
	# automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60*args.auto_skl_min, ml_memory_limit=12000) # MB
	automl.fit(train_x,train_y)
	pre_test = automl.predict(test_x)
	rec_score = recall_score(test_y,pre_test)
	print('autosklearn rec_score = ', rec_score)
	
	#automl.show_models()

	# mlp
	
	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler()
	scaler.fit(train_x)
	train_x = scaler.transform(train_x)
	test_x = scaler.transform(test_x)

	from sklearn.neural_network import MLPClassifier
	#mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)  # 0.66  0.47
	#mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 2), max_iter=500)   # 0.64  0.56 
	#mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 2), max_iter=2000)  # 0.51 1.0  0.246  0.5 32 CCWS, 0.51  1.0  0.246  0.5  48  CCWS , 0.015  0.53  1.0  0.15  0.5  48  CCWS, 0.02 
	mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 2), max_iter=2000)  # 0.51  1.0  0.246  0.5  52  CCWS , 0.51 1.0  0.246 0.5  89 CCWS, 0.51 1.0 0.246 0.5 128 CCWS,  0.51 1.0  0.246  0.5  192 CCWS, 
	# 0.015  0.67  0.58  0.16  0.53  52  CCWS, 0.02 
	#mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 2), max_iter=1900)  # 0.67  0.59  0.25  0.5  48  CCWS
	#mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 2), max_iter=1800)  # 0.63  0.57  0.25  0.5  48  CCWS
	#mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 2), max_iter=2000)  # 0.68  0.53  0.25  0.51  56 CCWS
	#mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 4), max_iter=2000)  # 0.67  0.51  0.254  0.51 52 CCWS 
	#mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 2), max_iter=2000)  # 0.51  1.0  0.24  0.5  48 LICWS, 0.51  1.0  0.246 0.5 128 LICWS , 0.015 0.63  0.53  0.15 0.50 
	#mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 2), max_iter=2000)  # 0.51  1.0  0.246  0.5  52 LICWS, 0.015  0.66  0.52  0.17  0.54 
	#mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 2), max_iter=2000)  # 0.66  0.43  0.22 0.47  56 LICWS
	#mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 2), max_iter=2000)  # 0.51 1.0 0.246 0.5 32 PCWS,  0.64  0.55  0.265 0.52  48 PCWS
	#mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 2), max_iter=2000)  # 0.69  0.47  025 0.50  64  CCWS 
	#mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 3), max_iter=2000)  # 0.67  0.53  0.24 0.49 
	#mlp = MLPClassifier(hidden_layer_sizes=(13, 14, 2), max_iter=2000)  # 0.66  0.52  0.25  0.50
	#mlp = MLPClassifier(hidden_layer_sizes=(14, 13, 2), max_iter=2000)  # 0.66  0.53  0.24  0.49 
	#mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 2), max_iter=3000)  # 0.66  0.52  0.24  0.49
	#mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 2), max_iter=5000)  # 0.66  0.52  0.24  0.49
	#mlp = MLPClassifier(hidden_layer_sizes=(32, 16, 8), max_iter=2000)  # 0.70  0.51  0.24  0.49 
	#mlp = MLPClassifier(hidden_layer_sizes=(64, 32, 16, 8), max_iter=2000)  # 0.73 0.39  0.24  0.50
	#mlp = MLPClassifier(hidden_layer_sizes=(32, 64, 16, 2), max_iter=2000)  # 0.71  0.52  0.24  0.49
	#mlp = MLPClassifier(hidden_layer_sizes=(13, 39, 2), max_iter=2000)  # 0.65 0.51 0.24 0.50
	#mlp = MLPClassifier(hidden_layer_sizes=(13, 20, 2), max_iter=2000)  # 0.62 0.60  0.24 0.49 
	#mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 2), max_iter=2000, activation='tanh')  # 0.67 0.56  0.24 0.49 
	#mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 2), max_iter=2000, activation='logistic')  # 0.59  0.68  0.26  0.52 
	#mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 2), max_iter=2000, activation='identity')  # 0.58  0.45  0.24  0.49   # long time
	#mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 2), max_iter=2000, solver='sgd')  # 0.64  0.44  0.25  0.50 
	#mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 2), max_iter=1500)  # 0.66  0.47  0.249  0.50 
	#mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 2), max_iter=1000)  # 0.66  0.47 0.25 0.50
	#mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 2), max_iter=2000) # 0.75  0.39    64 minhash   the first must < dimension_size ??
	#mlp = MLPClassifier(hidden_layer_sizes=(24, 24, 2), max_iter=2000)  # 0.69  0.53
	mlp.fit(train_x,train_y)

	#joblib.dump(mlp, "mlp_model.m")  # 0.01  CCWS  48 
	#mlp = joblib.load("mlp_model.m")

	predictions = mlp.predict(test_x)
	#pre_prob = mlp.predict_proba(test_x)
	#print('pre_prob = ', pre_prob)
	#print('mlp.classes_ = ', mlp.classes_)
	#from sklearn.metrics import classification_report, confusion_matrix
	#print(confusion_matrix(y_test, predictions))
	#print(classification_report(y_test, predictions))
	#print(mlp.score(train_x, train_y))  # accuracy   ????
	from sklearn.metrics import precision_score,roc_auc_score,recall_score, accuracy_score
	acc_score = accuracy_score(test_y, predictions)
	print('mlp acc_score = ', acc_score)

	rec_score = recall_score(test_y, predictions) # recall  
	print('mlp rec_score = ', rec_score)

	prec_score = precision_score(test_y, predictions) # precision  
	print('mlp prec_score = ', prec_score)

	ra_score = roc_auc_score(test_y, predictions) # roc_auc  
	print('mlp ra_score = ', ra_score)
	
	# for RF model
	# opengl_rf = pickle.load(open('openml_model.txt', 'rb'))  # 0.01  CCWS  52
	# predictions = opengl_rf.predict(test_x)
	#pre_prob = opengl_rf.predict_proba(test_x)

	#print(opengl_rf.score(train_x, train_y))  # accuracy  very slow ???
	acc_score = accuracy_score(test_y, predictions)
	print('rf acc_score = ', acc_score)

	rec_score = recall_score(test_y, predictions) # recall  
	print('rf rec_score = ', rec_score)

	prec_score = precision_score(test_y, predictions) # precision  
	print('rf prec_score = ', prec_score)

	ra_score = roc_auc_score(test_y, predictions) # roc_auc  
	print('rf ra_score = ', ra_score)

# from utils_sklearn.py
evaluate_count = 0 # kafeng add
evaluate_time = 0.0
constructed_features_list = [] # save indices
constructed_trees_list = []  # save trees for less time

selected_features_list = []  # for feature selection
selected_fscore_list = [] 
def evaluate(X, y, args, origin_result=0):
	#global evaluate_count, evaluate_time, model
	global evaluate_count, evaluate_time, all_entropy_mean, all_f1_score #, model
	global constructed_features_list, constructed_trees_list, selected_features_list
	
	s = 0

	evaluate_count += 1  # kafeng add
	#print('evaluate_count = ', evaluate_count)
	if args.task == 'regression':
		if args.model == 'LR':
			model = Lasso()
		elif args.model == 'RF':
			model = RandomForestRegressor(n_estimators=10, random_state=0)
			# add for cache
			select_rate = 0.5  # kafeng pay attention to featrures
			#print('int(np.ceil(X.shape[1]*select_rate)) = ', int(np.ceil(X.shape[1]*select_rate)))
			sfm = SelectFromModel(model, max_features=int(np.ceil(X.shape[1]*select_rate)))

		if args.evaluate == 'mae':
			s = cross_val_score(model, X, y, cv=5, 
				scoring='neg_mean_absolute_error').mean()
		elif args.evaluate == 'mse':
			s = cross_val_score(model, X, y, cv=5, 
				scoring='neg_mean_squared_error').mean()
		elif args.evaluate == 'r2':
			if args.cache_method == 'no_cache':
				s = cross_val_score(model, X, y, cv=5).mean()  # adaptive_valid
			elif args.cache_method == 'selection':
				# use feature selection
				sfm.fit(X, y)   # 
				#print('sfm.estimator_ = ', sfm.estimator_)  # RandomForestRegressor
				mask = sfm._get_support_mask()
				if mask.tolist() in selected_features_list:
					same_index = selected_features_list.index(mask.tolist())
					#print('same_index = ', same_index)
					return selected_fscore_list[same_index]

				selected_features_list.append(mask.tolist())
				#print(selected_features_list)
				
				X_transform = sfm.transform(X)  # 
				#print('X_transform.shape = ', X_transform.shape)
				s = cross_val_score(model, X=X_transform, y=y, cv=5).mean()  # auto run use f1_score  adaptive_valid

				selected_fscore_list.append(s)
				#print(selected_fscore_list)

	elif args.task == 'classification':
		le = LabelEncoder()
		y = le.fit_transform(y)    # kafeng pandas.core.series.Series to numpy.ndarray

		if args.model == 'RF':
			model = RandomForestClassifier(n_estimators=20, random_state=0)
			select_rate = 0.5  # kafeng pay attention to featrures
			sfm = SelectFromModel(model, max_features=int(np.ceil(X.shape[1]*select_rate)))

		elif args.model == 'LR':
			model = LogisticRegression(multi_class='ovr')
		elif args.model == 'SVM':
			model = svm.SVC()
		
		cv = 5
		if args.evaluate == 'f_score':
			if args.cache_method == 'no_cache':
				s = cross_val_score(model, X, y, scoring='f1_micro', cv=cv).mean()
			elif args.cache_method == 'selection':
				# use feature selection
				sfm.fit(X, y)   # 0.03706 seconds   0.06638 seconds
				mask = sfm._get_support_mask()
				if mask.tolist() in selected_features_list:
					same_index = selected_features_list.index(mask.tolist())
					#print('same_index = ', same_index)
					return selected_fscore_list[same_index]

				selected_features_list.append(mask.tolist())
				#print(selected_features_list)
				
				X_transform = sfm.transform(X)  # 0.00307 seconds   0.00289 seconds

				#print('X_transform.shape = ', X_transform.shape)
				s = cross_val_score(model, X=X_transform, y=y, scoring='f1_micro', cv=cv).mean()  # 0.24863 seconds

				selected_fscore_list.append(s)
				#print(selected_fscore_list)
		elif args.evaluate == 'f1_macro':
			s = cross_val_score(model, X, y, scoring='f1_macro', cv=cv).mean()
		elif args.evaluate == 'roc_auc':
			s = cross_val_score(model, X, y, scoring='roc_auc', cv=cv).mean()
		elif args.evaluate == 'recall':
			#print(cross_val_score(model, X, y, scoring='recall', cv=cv))
			s = cross_val_score(model, X, y, scoring='recall', cv=cv).mean()
		elif args.evaluate == 'precision':
			#print(cross_val_score(model, X, y, scoring='precision', cv=cv))
			s = cross_val_score(model, X, y, scoring='precision', cv=cv).mean()
	#print('s = ', s)
	return s


def train(dataset, orig_features, target_label):

	origin_result = evaluate(orig_features, target_label, args, origin_result=0)
	print('origin_result  = ', origin_result)
	
	f1_results = []
	
	for i in range(orig_features.shape[1]):
		reduce_features = orig_features.drop(['%d'%(i)], axis=1) # column
		#print(reduce_features)
		reduce_result = evaluate(reduce_features, target_label, args, origin_result=0)
		f1_results.append(reduce_result)
		
	return f1_results, origin_result

def get_meta_label(f1_results, origin_result):
	pn_results = []
	f1_changes = []

	for i in range(len(f1_results)):
		reduce_change = origin_result - f1_results[i]
		f1_changes.append(reduce_change) 
		if reduce_change >= args.threshold:
			pn_results.append(1)  # add this feature is positive 
		else:
			pn_results.append(0)  # add this feature is negtive or no use 
		
	return pn_results, f1_changes


def preprocess_data():
	start_time = time.time()

	for ds in datasets:
		print('ds = ', ds)
		#path = dataPath + ds + '.csv'
		path = os.path.join(dataPath, ds + '.csv')
		orig_data = pd.read_csv(path)  #  pandas.core.frame.DataFrame

		# preprocess feature name. cloumn name  
		new_col = []
		for col in range(orig_data.shape[1]):
			new_col.append('%d'%(col))
		orig_data.columns = new_col

		# preprocess special feature, all the same
		for col in range(orig_data.shape[1]-1):
			feature = orig_data['%d'%(col)]
			if feature.max() == feature.min():
				del orig_data['%d'%(col)]

		# reset name
		new_col = []
		for col in range(orig_data.shape[1]):
			new_col.append('%d'%(col))
		orig_data.columns = new_col

		target_label = orig_data[orig_data.columns[-1]]  # pandas.core.frame.Series

		
		orig_features = orig_data.copy()  # 
		del orig_features[orig_features.columns[-1]]

		# save to temp data
		new_data = pd.concat([orig_features, pd.Series(target_label)], axis=1)
		#print(new_data)
		#new_data.to_csv(dataPath + '.preprocess_data/'+ds+'.csv', sep=',', header=True, index=True)
		new_data.to_csv(os.path.join(dataPath, '.preprocess_data/', ds+'.csv'), sep=',', header=True, index=True)

		f1_results, origin_result = train(ds, orig_features, target_label)  # waste lots of time, save to new_data ??

		results = []
		results.append(origin_result)
		results = results + f1_results
		results_save = pd.DataFrame({'results': results})
		#results_save.to_csv(dataPath + '.preprocess_data/'+ds+'_results.csv', sep=',', header=True, index=True)
		results_save.to_csv(os.path.join(dataPath, '.preprocess_data/', ds+'_results.csv'), sep=',', header=True, index=True)
		

	duration = time.time() - start_time
	print('%s  duration = %.5f seconds' %(datetime.datetime.now(), duration))


def load_results():
	meta_label_all = [] 
	f1_changes_all = []

	for ds in datasets:
		path = dataPath + '.preprocess_data/' + ds + '_results.csv'
		# path = os.path.join(dataPath, args.process_data, ds + '_results.csv')
		results = pd.read_csv(path, header=0, index_col=0)
		results_list = results['results'].tolist()
		origin_result = results_list[0]
		f1_results = results_list[1:]
		pn_results, f1_changes = get_meta_label(f1_results, origin_result)
		meta_label_all = meta_label_all + pn_results
		f1_changes_all = f1_changes_all + f1_changes

	f1_change_label = pd.DataFrame({'meta_label_all': meta_label_all, 'f1_changes_all': f1_changes_all})
	#f1_change_label.to_csv('f1_change_label.csv', sep=',', header=True, index=True)

	return meta_label_all, f1_changes_all

def load_train_results():
	meta_label_all = [] 
	f1_changes_all = []

	for ds in datasets:
		path = dataPath + '.preprocess_data/' + ds + '_results.csv'
		# path = os.path.join(dataPath, args.process_data, ds + '_results.csv')
		results = pd.read_csv(path, header=0, index_col=0)
		results_list = results['results'].tolist()
		origin_result = results_list[0]
		f1_results = results_list[1:]
		pn_results, f1_changes = get_meta_label(f1_results, origin_result)
		meta_label_all = meta_label_all + pn_results
		f1_changes_all = f1_changes_all + f1_changes

	f1_change_label = pd.DataFrame({'meta_label_all': meta_label_all, 'f1_changes_all': f1_changes_all})
	#f1_change_label.to_csv('f1_change_label.csv', sep=',', header=True, index=True)

	#f1_change_label.where(, )
	negtive_samples = f1_change_label.loc[f1_change_label['meta_label_all'] == 0]
	positive_samples = f1_change_label.loc[f1_change_label['meta_label_all'] == 1]
	step = int(np.ceil(negtive_samples.shape[0] / (positive_samples.shape[0]*args.PtoN)))  # args.PtoN = 1, PtoN < step
	print('step = ', step)
	#print(negtive_samples)
	print('len(negtive_samples) = ', len(negtive_samples))
	sorted_samples = negtive_samples.sort_values(by='f1_changes_all')
	#print(sorted_samples)
	compress_samples = sorted_samples.iloc[range(0, negtive_samples.shape[0], step)]
	#print(compress_samples)
	print('len(compress_samples) = ', len(compress_samples))

	f1_change_label = pd.concat([compress_samples, positive_samples], axis=0) # ignore_index=True  must save
	#f1_change_label.to_csv('f1_change_label_compress.csv', sep=',', header=True, index=True)
	
	#return meta_label_all, f1_changes_all
	#return f1_change_label['meta_label_all'], f1_change_label['f1_changes_all']
	#return meta_label_all, f1_changes_all, f1_change_label['meta_label_all'], f1_change_label['f1_changes_all']
	return meta_label_all, f1_changes_all, f1_change_label

def meta_feature_eval(meta_features, meta_label_all):

	args.evaluate = 'recall'
	recall_result = evaluate(meta_features, meta_label_all, args)  # recall
	print('recall_result = ', recall_result)
	

def data_process():

	# preprocess_data()  # 1481 s  in train 
	
	meta_label_all, f1_changes_all = load_results()
	
	norm_mean = []
	norm_var = []
	all_mean = []
	all_var = []

	all_pcws = pd.DataFrame()
	ds_count = 0
	feature_count = 0
	
	for ds in datasets:
		#print('ds = ', ds)
		path = dataPath + ds + '.csv'
		# path = os.path.join(dataPath, args.process_data , ds + '_synth.csv')  # kafeng for synth
		orig_data = pd.read_csv(path)  #  pandas.core.frame.DataFrame

		# preprocess feature name. cloumn name  
		new_col = []
		for col in range(orig_data.shape[1]):
			new_col.append('%d'%(col))
		orig_data.columns = new_col

		# preprocess special feature, all the same
		for col in range(orig_data.shape[1]-1):
			feature = orig_data['%d'%(col)]
			if feature.max() == feature.min():
				del orig_data['%d'%(col)]

		# reset name
		new_col = []
		for col in range(orig_data.shape[1]):
			new_col.append('%d'%(col))
		orig_data.columns = new_col

		target_label = orig_data[orig_data.columns[-1]]  # pandas.core.frame.Series

		orig_features = orig_data.copy()  # 
		del orig_features[orig_features.columns[-1]]
	
		num_feature = orig_data.shape[1] - 1   # 9-1
		
		norm_data = pd.DataFrame()
		for col in range(num_feature):
			feature = orig_features['%d'%(col)]
			norm = (feature-feature.min())/(feature.max()-feature.min())
			
			norm_data.insert(col, '%d'%(col), norm)
			norm_mean.append(norm.mean())
			norm_var.append(norm.var())

		# weiwu pcws
		weighted_set = norm_data.values
		wmh = WeightedMinHash(weighted_set, args.dimension_pcws, seed=0)
		if args.feature_extract_alg == 'PCWS':
			k, y, e = wmh.pcws()
		elif args.feature_extract_alg == 'ICWS':
			k, y, e = wmh.icws()
		elif args.feature_extract_alg == 'LICWS':
			k, e = wmh.licws()
		elif args.feature_extract_alg == 'CCWS':
			k, y, e = wmh.ccws()
		indexs = np.transpose(k.astype(np.int32))
		pcws = pd.DataFrame()
		for col in range(norm_data.shape[1]):
			indexs_values = norm_data['%d'%(col)][indexs[:,col]].reset_index(drop=True)
			pcws = pd.concat([pcws, indexs_values], axis=1, ignore_index=True )

		norm_sum_mean = norm_data.mean(axis=1)

		all_mean = all_mean + [norm_sum_mean.mean()] * num_feature
		all_var = all_var + [norm_sum_mean.var()] * num_feature

		all_pcws = pd.concat([all_pcws, pcws], axis=1, ignore_index=True)

		ds_count = ds_count + 1
		
		meta_features = pd.DataFrame(np.transpose(pcws.values))
		#print('meta_features = ', meta_features)
		
		feature_count = feature_count + num_feature
	
	print('ds_count = ', ds_count, ' feature_count = ', feature_count)
	meta_features = pd.DataFrame(np.transpose(all_pcws.values))  # 0.71
	#meta_feature_eval(meta_features, meta_label_all)  # 

	return meta_features, meta_label_all


def train_data_process():

	# preprocess_data()  # 1481 s  in train 
	
	meta_label_all, f1_changes_all, f1_change_compress = load_train_results()
	# print('f1_changes_all = ',  f1_changes_all)
	# print('len(f1_changes_all) = ',  len(f1_changes_all))
	# pd.DataFrame( data=f1_changes_all).to_csv('f1_changes_all.csv')

	
	norm_mean = []
	norm_var = []
	all_mean = []
	all_var = []

	all_pcws = pd.DataFrame()
	ds_count = 0
	feature_count = 0
	
	for ds in datasets:
		#print('ds = ', ds)
		path = dataPath + ds + '.csv'
		# path = os.path.join(dataPath, args.process_data, ds + '_synth.csv')  # kafeng for synth
		# print('path = ', path)
		orig_data = pd.read_csv(path)  #  pandas.core.frame.DataFrame

		# preprocess feature name. cloumn name  
		new_col = []
		for col in range(orig_data.shape[1]):
			new_col.append('%d'%(col))
		orig_data.columns = new_col

		# preprocess special feature, all the same
		for col in range(orig_data.shape[1]-1):
			feature = orig_data['%d'%(col)]
			if feature.max() == feature.min():
				del orig_data['%d'%(col)]

		# reset name
		new_col = []
		for col in range(orig_data.shape[1]):
			new_col.append('%d'%(col))
		orig_data.columns = new_col

		target_label = orig_data[orig_data.columns[-1]]  # pandas.core.frame.Series

		orig_features = orig_data.copy()  # 
		del orig_features[orig_features.columns[-1]]
	
		num_feature = orig_data.shape[1] - 1   # 9-1
		
		norm_data = pd.DataFrame()
		for col in range(num_feature):
			feature = orig_features['%d'%(col)]
			norm = (feature-feature.min())/(feature.max()-feature.min())
			
			norm_data.insert(col, '%d'%(col), norm)
			norm_mean.append(norm.mean())
			norm_var.append(norm.var())

		# weiwu pcws
		weighted_set = norm_data.values
		wmh = WeightedMinHash(weighted_set, args.dimension_pcws, seed=0)
		if args.feature_extract_alg == 'PCWS':
			#k, y, e = wmh.pcws()
			k, y, e = wmh.pcws_pytorch()
			k = k.numpy()
		elif args.feature_extract_alg == 'ICWS':
			#k, y, e = wmh.icws()
			k, y, e = wmh.icws_pytorch()
			k = k.numpy()
		elif args.feature_extract_alg == 'LICWS':
			#k, e = wmh.licws()
			k, e = wmh.licws_pytorch()
			k = k.numpy()
		elif args.feature_extract_alg == 'CCWS':
			#k, y, e = wmh.ccws()
			k, y, e = wmh.ccws_pytorch()
			k = k.numpy()
		indexs = np.transpose(k.astype(np.int32))
		pcws = pd.DataFrame()
		for col in range(norm_data.shape[1]):
			indexs_values = norm_data['%d'%(col)][indexs[:,col]].reset_index(drop=True)
			pcws = pd.concat([pcws, indexs_values], axis=1, ignore_index=True )

		norm_sum_mean = norm_data.mean(axis=1)

		all_mean = all_mean + [norm_sum_mean.mean()] * num_feature
		all_var = all_var + [norm_sum_mean.var()] * num_feature

		all_pcws = pd.concat([all_pcws, pcws], axis=1, ignore_index=True)

		ds_count = ds_count + 1
		
		meta_features = pd.DataFrame(np.transpose(pcws.values))
		#print('meta_features = ', meta_features)
		
		feature_count = feature_count + num_feature
	
	print('ds_count = ', ds_count, ' feature_count = ', feature_count)
	meta_features = pd.DataFrame(np.transpose(all_pcws.values))  # 0.71

	print('len(meta_features) = ', len(meta_features))
	print('len(f1_change_compress) = ', len(f1_change_compress))
	print('len(f1_change_compress.index) = ', len(f1_change_compress.index))
	#meta_feature_eval(meta_features.iloc[f1_change_compress.index], f1_change_compress['meta_label_all'])

	#return meta_features, meta_label_all
	return meta_features.iloc[f1_change_compress.index], f1_change_compress['meta_label_all']

if __name__ == '__main__':
	start_time = time.time()
	
	# for openml dataset
	dataPath = args.train_data
	fileList = os.listdir(dataPath)
	#print('fileList = ', fileList)
	datasets = []
	for data in fileList:
		fileName = data.split('.')[0]
		if len(fileName) > 1:
			datasets.append(fileName)
	
	print('len(datasets) = ', len(datasets))
	# print(datasets)

	train_x, train_y = train_data_process()
	
	# for paper dataset
	dataPath = args.test_data
	fileList = os.listdir(dataPath)
	#print('fileList = ', fileList)
	datasets = []
	for data in fileList:
		fileName = data.split('.')[0]
		if len(fileName) > 1:
			datasets.append(fileName)
	
	print('len(datasets) = ', len(datasets))
	print(datasets)

	test_x, test_y = data_process()
	
	# save data for dnn 
	print(type(train_x))
	print(type(train_y))
	print(type(test_x))
	print(type(test_y))
	if not os.path.exists(os.path.join(args.synth_meta_feat, args.train_data)):
		os.makedirs(os.path.join(args.synth_meta_feat, args.train_data))
	train_x.to_csv(os.path.join(args.synth_meta_feat, args.train_data, 'train_x_' +str(args.threshold)+ '_' +args.feature_extract_alg+ '_' +str(args.dimension_pcws)+ '.csv'), sep=',', header=False, index=False)
	train_y.to_csv(os.path.join(args.synth_meta_feat, args.train_data, 'train_y_' +str(args.threshold)+ '_' +args.feature_extract_alg+ '_' +str(args.dimension_pcws)+ '.csv'), sep=',', header=False, index=False)
	if not os.path.exists(os.path.join(args.synth_meta_feat, args.test_data)):
		os.makedirs(os.path.join(args.synth_meta_feat, args.test_data))
	test_x.to_csv(os.path.join(args.synth_meta_feat, args.test_data, 'test_x_' +str(args.threshold)+ '_' +args.feature_extract_alg+ '_' +str(args.dimension_pcws)+ '.csv'), sep=',', header=False, index=False)
	pd.DataFrame(test_y).to_csv(os.path.join(args.synth_meta_feat, args.test_data, 'test_y_' +str(args.threshold)+ '_' +args.feature_extract_alg+ '_' +str(args.dimension_pcws)+ '.csv'), sep=',', header=False, index=False)
	


	# load data for dnn, auto-sklearn
	train_x = pd.read_csv(os.path.join(args.synth_meta_feat, args.train_data, 'train_x_' +str(args.threshold)+ '_' +args.feature_extract_alg+ '_' +str(args.dimension_pcws)+ '.csv'), header=None)
	train_y = pd.read_csv(os.path.join(args.synth_meta_feat, args.train_data, 'train_y_' +str(args.threshold)+ '_' +args.feature_extract_alg+ '_' +str(args.dimension_pcws)+ '.csv'), header=None)
	test_x = pd.read_csv(os.path.join(args.synth_meta_feat, args.test_data, 'test_x_' +str(args.threshold)+ '_' +args.feature_extract_alg+ '_' +str(args.dimension_pcws)+ '.csv'), header=None)
	test_y = pd.read_csv(os.path.join(args.synth_meta_feat, args.test_data, 'test_y_' +str(args.threshold)+ '_' +args.feature_extract_alg+ '_' +str(args.dimension_pcws)+ '.csv'), header=None)

	print('train_x.shape = ', train_x.shape)
	print('train_y.shape = ', train_y.shape)
	print('test_x.shape = ', test_x.shape)
	print('test_y.shape = ', test_y.shape)

	
	#for i in range(1,9):
	# for i in range(1,4):
	for i in range(1,1):   # for xiaomi
		print('i = ', i)
		train_x_i = pd.read_csv(os.path.join(args.synth_meta_feat + '_' + str(i), args.train_data, 'train_x_' +str(args.threshold)+ '_' +args.feature_extract_alg+ '_' +str(args.dimension_pcws)+ '.csv'), header=None)
		print('train_x_i.shape = ', train_x_i.shape)
		train_x = pd.concat([train_x, train_x_i],axis=0,ignore_index=True)
		print('train_x.shape = ', train_x.shape)
		train_y_i = pd.read_csv(os.path.join(args.synth_meta_feat + '_' + str(i), args.train_data, 'train_y_' +str(args.threshold)+ '_' +args.feature_extract_alg+ '_' +str(args.dimension_pcws)+ '.csv'), header=None)
		train_y = train_y.append(train_y_i)
		test_x_i = pd.read_csv(os.path.join(args.synth_meta_feat + '_' + str(i), args.test_data, 'test_x_' +str(args.threshold)+ '_' +args.feature_extract_alg+ '_' +str(args.dimension_pcws)+ '.csv'), header=None)
		test_x = pd.concat([test_x, test_x_i],axis=0,ignore_index=True)
		test_y_i = pd.read_csv(os.path.join(args.synth_meta_feat + '_' + str(i), args.test_data, 'test_y_' +str(args.threshold)+ '_' +args.feature_extract_alg+ '_' +str(args.dimension_pcws)+ '.csv'), header=None)
		test_y = test_y.append(test_y_i)

		
		print('train_y_i.shape = ', train_y_i.shape)
		print('test_x_i.shape = ', test_x_i.shape)
		print('test_y_i.shape = ', test_y_i.shape)

		
		print('train_y.shape = ', train_y.shape)
		print('test_x.shape = ', test_x.shape)
		print('test_y.shape = ', test_y.shape)
	

	print('train_x.shape = ', train_x.shape)
	print('train_y.shape = ', train_y.shape)
	print('test_x.shape = ', test_x.shape)
	print('test_y.shape = ', test_y.shape)
	#meta_feature_eval(train_x, train_y)
	#meta_feature_eval(test_x, test_y)
	paper_eval(train_x, test_x, train_y, test_y)
	
	duration = time.time() - start_time
	print('%s  duration = %.5f seconds' %(datetime.datetime.now(), duration))


