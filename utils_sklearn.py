from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
import numpy as np
import pandas as pd
import sys
import os
#import logging  

import datetime
import time
import numpy as np

import gp
import xlwt
from xlutils.copy import copy
import xlrd
from gp import num_feature, mum_columns
from sklearn import metrics   # kafeng modify
from sklearn.metrics import f1_score  # kafeng modify


#import xgboost as xgb
from scipy import stats
#from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold


from sklearn.feature_selection import SelectFromModel  # kafeng add

def mod_column(c1, c2):
	r = []
	for i in range(c2.shape[0]):
		if c2[i] == 0:
			r.append(0)
		else:
			r.append(np.mod(c1[i],c2[i]))
	return r


# xgboost parameters
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 3,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

def GetCVScore(estimator,X,y):
    nested_score = cross_val_score(estimator, X=X, y=y, cv=5).mean()
    return nested_score


evaluate_count = 0 # kafeng add
evaluate_time = 0.0
#model = None  # n_estimators default 10  %94
all_entropy_mean, all_f1_score = [], []

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
			sfm = SelectFromModel(model, max_features=int(np.ceil(X.shape[1]*select_rate)))   # kafeng minhash is to replace this.  Meta-transformer for selecting features based on importance weights.

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
				#print('X.shape = ', X.shape)
				mask = sfm._get_support_mask()
				if mask.tolist() in selected_features_list:
					same_index = selected_features_list.index(mask.tolist())
					#print('same_index = ', same_index)
					return selected_fscore_list[same_index]

				selected_features_list.append(mask.tolist())
				#print('selected_features_list = ', selected_features_list)
				#print('sfm.threshold_ = ', sfm.threshold_)
				
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
	# jaccard
	return s


def rf_evaluate(X, y):
    le = LabelEncoder()
    y = le.fit_transform(y)    # kafeng pandas.core.series.Series to numpy.ndarray

    model = RandomForestClassifier(n_estimators=20, random_state=0)
    #model = RandomForestClassifier()
    
    s = cross_val_score(model, X, y, scoring='f1_micro', cv=5).mean()
    return s
