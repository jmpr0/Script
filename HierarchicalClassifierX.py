#!/usr/bin/python3

import arff
import numpy as np
import sys, gc
import getopt
import os, time
from copy import copy
import pickle as pk

import networkx as nx
from scipy.spatial.distance import hamming
from scipy.sparse import issparse, csr_matrix

from importlib import import_module

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

import sklearn

from threading import Thread

# eps = np.finfo(float).eps
eps = sys.float_info.epsilon

def warn(*args, **kwargs):
	pass
import warnings
warnings.warn = warn

def import_sklearn():

	global RandomForestClassifier,DecisionTreeClassifier,SelectKBest,mutual_info_classif,\
		StratifiedKFold,train_test_split,OrdinalEncoder,OneHotEncoder,MyLabelEncoder,MinMaxScaler,KernelDensity

	global OneClassSVMAnomalyDetector,IsolationForestAnomalyDetector,LocalOutlierFactorAnomalyDetector,\
		EllipticEnvelopeAnomalyDetector,FactorAnalysis,OutputCodeClassifier,MinMaxScalerExpPenalty,QuantileTransformer
		# ,\
		# BaseEstimator,ClassifierMixin,GridSearchCV

	RandomForestClassifier = import_module('sklearn.ensemble').RandomForestClassifier
	DecisionTreeClassifier = import_module('sklearn.tree').DecisionTreeClassifier
	OneClassSVM = import_module('sklearn.svm').OneClassSVM
	IsolationForest = import_module('sklearn.ensemble').IsolationForest
	LocalOutlierFactor = import_module('sklearn.neighbors').LocalOutlierFactor
	EllipticEnvelopeAnomalyDetector = import_module('sklearn.covariance').EllipticEnvelope
	SelectKBest = import_module('sklearn.feature_selection').SelectKBest
	mutual_info_classif = import_module('sklearn.feature_selection').mutual_info_classif
	StratifiedKFold = import_module('sklearn.model_selection').StratifiedKFold
	train_test_split = import_module('sklearn.model_selection').train_test_split
	OrdinalEncoder = import_module('sklearn.preprocessing').OrdinalEncoder
	OneHotEncoder = import_module('sklearn.preprocessing').OneHotEncoder
	MinMaxScaler = import_module('sklearn.preprocessing').MinMaxScaler
	QuantileTransformer = import_module('sklearn.preprocessing').QuantileTransformer
	KernelDensity = import_module('sklearn.neighbors').KernelDensity
	FactorAnalysis = import_module('sklearn.decomposition').FactorAnalysis
	OutputCodeClassifier = import_module('sklearn.multiclass').OutputCodeClassifier
	# LabelEncoder = import_module('sklearn.preprocessing').LabelEncoder
	# BaseEstimator = import_module('sklearn.base').BaseEstimator
	# ClassifierMixin = import_module('sklearn.base').ClassifierMixin
	# GridSearchCV = import_module('sklearn.model_selection').GridSearchCV

	class MinMaxScalerExpPenalty(MinMaxScaler):
		def __init__(self, feature_range=(0, 1), copy=True):
			super(MinMaxScalerExpPenalty, self).__init__(
				feature_range=feature_range, copy=copy
				)
		def transform_exppenalty(self,X):
			X=self.transform(X)
			X[np.where(X<0)] = X[np.where(X<0)]-np.exp(-X[np.where(X<0)])+1
			x[np.where(X>1)] = X[np.where(X>1)]+np.exp(X[np.where(X>1)]-1)-1
			return X

	class OneClassSVMAnomalyDetector(OneClassSVM):
		def __init__(self, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1, random_state=None):
			super(OneClassSVMAnomalyDetector, self).__init__(
				kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, nu=nu,
				shrinking=shrinking, cache_size=cache_size, verbose=verbose,
				max_iter=max_iter, random_state=random_state
				)
		def score(self,X,y=None):
			'''
			OneClassSVM decision function returns signed distance to the separating hyperplane.
			This distance results positive for inlier and negative for outlier.
			OneClassSVM learns the hyperplane that define soft boundaries of training samples,
			and returns the outlierness score for each sample.
			In this work we optimize OneClassSVM hyperparameters with an heuripstic that borns
			from the passing of solely benign examples, i.e. we want the hyperparameters combination
			that maximize the score (higher the less abnormal) and minimize the negative scores.
			This can be obtained by maximizing the mean of positive scores,
			and minimizing (maximizing) the ratio of negative (positives) ones.
			'''
			# decisions = self.decision_function(X)
			# true_decisions = decisions[ np.where(decisions >= 0) ]
			# if len(true_decisions) > 0:
			# 	true_decisions = MinMaxScaler().fit_transform(true_decisions.reshape(-1,1))
			# 	factor = len(true_decisions)/len(decisions)
			# 	return np.mean(true_decisions)*factor
			# else:
			# 	return 0
			decisions = self.decision_function(X)
			# print(np.isinf(decisions).any())
			if np.isinf(decisions).all():
				return 0
			decisions = MinMaxScaler().fit_transform(decisions.reshape(-1,1))
			return np.mean(decisions)

	class IsolationForestAnomalyDetector(IsolationForest):
		def __init__(self, n_estimators=100, max_samples='auto', contamination='legacy', max_features=1.0, bootstrap=False, n_jobs=None, behaviour='old', random_state=None, verbose=0):
			super(IsolationForestAnomalyDetector, self).__init__(
				n_estimators=n_estimators, max_samples=max_samples, contamination=contamination,
				max_features=max_features, bootstrap=bootstrap, n_jobs=n_jobs, behaviour=behaviour,
				random_state=random_state, verbose=verbose
				)
		def score(self,X,y=None):
			'''
			IsolationForest decision function returns the anomaly score for each sample, negative for outliers and positive for inliers. This score is the measure of
			normality of an observation and is equivalent to the number of the split of the tree to isolate the particular
			sample. During optimization phase we set the contamination factor (portion of outliers in the dataset) to zero,
			because we are training only over benign samples. To found optimal hyperparameters configuration we had to
			maximize the mean of positive scores and minimize the negatives ones. So the approach is the same we use to optimize OC-SVM.
			'''
			# decisions = self.decision_function(X)
			# true_decisions = decisions[ np.where(decisions >= 0) ]
			# if len(true_decisions) > 0:
			# 	true_decisions = MinMaxScaler().fit_transform(true_decisions.reshape(-1,1))
			# 	factor = len(true_decisions)/len(decisions)
			# 	return np.mean(true_decisions)*factor
			# else:
			# 	return 0
			decisions = self.decision_function(X)
			if np.isinf(decisions).all():
				# decisions[np.where(np.isinf(decisions))] = np.nanmax(decisions[np.where(np.isfinite(decisions))])
				return 0
			decisions = MinMaxScaler().fit_transform(decisions.reshape(-1,1))
			return np.mean(decisions)

	class LocalOutlierFactorAnomalyDetector(LocalOutlierFactor):
		def __init__(self, n_neighbors=20, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, contamination='legacy', novelty=False, n_jobs=None):
			super(LocalOutlierFactorAnomalyDetector, self).__init__(
				n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size, metric=metric,
				p=p, metric_params=metric_params, contamination=contamination, novelty=novelty, n_jobs=n_jobs
				)
			self.n_neighbors=n_neighbors
		def score(self,X,y=None):
			'''
			LOF decision function return the shifted opposite of the local outlier factor of samples. The greater the more normal.
			To train this algorithm we cannot set contamination to 0, because zero is the -inf asymptote. So we set contamination
			to the minimum permitted value (e.g. 1e-4).
			The optimization of hyperparameters for LOF is slightly diverse w.r.t. the used in previous models.
			In particular, we only maximize mean of scores, after min max normalization.
			'''
			# decisions = self.decision_function(X)
			# true_decisions = decisions[ np.where(decisions >= 0) ]
			# if len(true_decisions) > 0:
			# 	true_decisions = MinMaxScaler().fit_transform(true_decisions.reshape(-1,1))
			# 	factor = len(true_decisions)/len(decisions)
			# 	return 1/np.std(true_decisions)*factor
			# else:
			# 	return 0
			decisions = self.decision_function(X)
			if np.isinf(decisions).all():
				return 0
			decisions = MinMaxScaler().fit_transform(decisions.reshape(-1,1))
			return np.mean(decisions)

def import_spark():

	global SparkContext,SparkConf,SparkSession,udf,Spark_RandomForestClassifier,Spark_NaiveBayesClassifier,\
		Spark_MultilayerPerceptronClassifier,Spark_GBTClassifier,Spark_DecisionTreeClassifier,Vectors,SparseVector,VectorUDT,\
		Spark_MinMaxScaler,pandas,VectorAssembler

	SparkContext = import_module('pyspark').SparkContext
	SparkConf = import_module('pyspark').SparkConf
	SparkSession = import_module('pyspark.sql.session').SparkSession
	udf = import_module('pyspark.sql.functions').udf
	Spark_RandomForestClassifier = import_module('pyspark.ml.classification').RandomForestClassifier
	Spark_NaiveBayesClassifier = import_module('pyspark.ml.classification').NaiveBayes
	Spark_MultilayerPerceptronClassifier = import_module('pyspark.ml.classification').MultilayerPerceptronClassifier
	Spark_GBTClassifier = import_module('pyspark.ml.classification').GBTClassifier
	Spark_DecisionTreeClassifier = import_module('pyspark.ml.classification').DecisionTreeClassifier
	Vectors = import_module('pyspark.ml.linalg').Vectors
	SparseVector = import_module('pyspark.ml.linalg').SparseVector
	VectorUDT = import_module('pyspark.ml.linalg').VectorUDT
	Spark_MinMaxScaler = import_module('pyspark.ml.feature').MinMaxScaler
	pandas = import_module('pandas')
	VectorAssembler = import_module('pyspark.ml.feature').VectorAssembler

	# import_distkeras
	global ADAG,ModelPredictor,OneHotTransformer

	ADAG = import_module('distkeras.trainers').ADAG
	ModelPredictor = import_module('distkeras.predictors').ModelPredictor
	OneHotTransformer = import_module('distkeras.transformers').OneHotTransformer

def import_weka():

	global jvm,ndarray_to_instances,Attribute,Classifier

	jvm = import_module('weka.core.jvm')
	ndarray_to_instances = import_module('weka.core.converters').ndarray_to_instances
	Attribute = import_module('weka.core.dataset').Attribute
	Classifier = import_module('weka.classifiers').Classifier

def import_keras():
	global keras,cython,optimizers,Sequential,Model,Dense,Input,Conv1D,Conv2D,MaxPooling2D,Flatten,Dropout,\
		Reshape,BatchNormalization,LSTM,AveragePooling1D,MaxPooling1D,UpSampling1D,Activation,relu,elu,sigmoid,tanh,softmax,hard_sigmoid,\
		EarlyStopping,CSVLogger,mean_squared_error,categorical_crossentropy,K,SGD,Adadelta,concatenate,plot_model,\
		regularizers,Lambda,ModelCheckpoint,clone_model
	
	keras = import_module('keras')
	cython = import_module('cython')
	optimizers = import_module('keras').optimizers
	Sequential = import_module('keras.models').Sequential
	Model = import_module('keras.models').Model
	Dense = import_module('keras.layers').Dense
	Input = import_module('keras.layers').Input
	Conv1D = import_module('keras.layers').Conv1D
	Conv2D = import_module('keras.layers').Conv2D
	MaxPooling2D = import_module('keras.layers').MaxPooling2D
	Flatten = import_module('keras.layers').Flatten
	Dropout = import_module('keras.layers').Dropout
	Reshape = import_module('keras.layers').Reshape
	BatchNormalization = import_module('keras.layers').BatchNormalization
	LSTM = import_module('keras.layers').LSTM
	AveragePooling1D = import_module('keras.layers').AveragePooling1D
	MaxPooling1D = import_module('keras.layers').MaxPooling1D
	UpSampling1D = import_module('keras.layers').UpSampling1D
	Activation = import_module('keras.layers').Activation
	relu = import_module('keras.activations').relu
	elu = import_module('keras.activations').elu
	sigmoid = import_module('keras.activations').sigmoid
	hard_sigmoid = import_module('keras.activations').hard_sigmoid
	tanh = import_module('keras.activations').tanh
	softmax = import_module('keras.activations').softmax
	EarlyStopping = import_module('keras.callbacks').EarlyStopping
	CSVLogger = import_module('keras.callbacks').CSVLogger
	ModelCheckpoint  = import_module('keras.callbacks').ModelCheckpoint
	mean_squared_error = import_module('keras.losses').mean_squared_error
	categorical_crossentropy = import_module('keras.losses').categorical_crossentropy
	K = import_module('keras').backend
	SGD = import_module('keras.optimizers').SGD
	Adadelta = import_module('keras.optimizers').Adadelta
	concatenate = import_module('keras.layers').concatenate
	plot_model = import_module('keras.utils.vis_utils').plot_model
	regularizers = import_module('keras').regularizers
	Lambda = import_module('keras.layers').Lambda
	clone_model = import_module('keras.models').clone_model

def equal_partitioner(nb,ps):
	if list(ps) != list(reversed(sorted(ps))):
		print('Warning: ps vector had to be alrealy sorted in ascendent way')
		return []
	buckets_i = []
	buckets_p = []
	for _ in range(nb):
		buckets_i.append([])
		buckets_p.append(0)
	for i,p in enumerate(ps):
		min_bucket_i = buckets_p.index(np.min(buckets_p))
		buckets_i[min_bucket_i].append(i)
		buckets_p[min_bucket_i] += p
	return buckets_i

def online_equal_partitioner(nb,ps,ts):
	if list(ps) != list(reversed(sorted(ps))):
		print('Warning: ps vector had to be alrealy sorted in ascendent way. Moreover, elements of ps and ts have to be aligned.')
		return []
	buckets_i = []
	buckets_t = []
	for i,t in enumerate(ts):
		if i < nb:
			buckets_i.append([i])
			buckets_t.append(t)
		else:
			min_bucket_i = buckets_t.index(np.min(buckets_t))
			buckets_i[min_bucket_i].append(i)
			buckets_t[min_bucket_i] += t
	return buckets_i

def classist_partitioner(nb,nc):
	combos, k= [], 0
	nb_max_complete = nc%nb
	nb_dim_complete = np.ceil(nc/nb)
	nb_max_partial = nb-nc%nb
	nb_dim_partial = np.floor(nc/nb)
	for _ in range(nb):
		combos.append([])
	k=0
	nb_count_complete=0
	nb_count_partial=0
	for j in range(nc):
		combos[k].append(j)
		if k < nb_max_complete and len(combos[k]) == nb_dim_complete\
			or nb_max_complete <= k < nb_max_complete+nb_max_partial and len(combos[k]) == nb_dim_partial:
			if k%2 != 0:
				combos[k] = list(reversed(combos[k]))
			k += 1
	return combos

class TreeNode(object):
	def __init__(self):

		# Position info
		self.parent = None
		self.children = {}

		# Node info
		self.tag = 'ROOT'
		self.level = 0				# 0-based
		self.fold = 0
		self.children_tags = []
		self.children_number = 0

		# Classification info
		self.features_index = []
		self.train_index = []
		self.test_index = []

		self.test_index_all = []

		# Configuration info
		self.features_number = 0
		# self.encoded_features_number = 0
		self.packets_number = 0
		self.classifier_name = 'srf'

		self.test_duration = 0.0
		self.complexity = 0.0

		self.label_encoder = None

class MyLabelEncoder(object):

	def __init__(self):

		self.nom_to_cat = {}
		self.cat_to_nom = {}

		self.base_type = None

	def fit(self, y_nom):

		self.nom_to_cat = {}
		self.cat_to_nom = {}
		cat = 0

		self.base_type = type(y_nom[0])

		max_n = len(list(set(y_nom)))

		for nom in y_nom:
			if nom not in self.nom_to_cat:
				self.nom_to_cat[nom] = cat
				self.cat_to_nom[cat] = nom
				cat += 1

				if cat == max_n:
					break

	def transform(self, y_nom):

		y_cat = [ self.nom_to_cat[nom] for nom in y_nom ]
		return np.asarray(y_cat, dtype=int)

	def inverse_transform(self, y_cat):

		y_nom = [ self.cat_to_nom[cat] for cat in y_cat ]
		return np.asarray(y_nom, dtype=self.base_type)

	def fit_transform(self, y_nom):

		self.fit(y_nom)
		return self.transform(y_nom)

class SklearnWekaWrapper(object):

	def __init__(self, classifier_name):

		if classifier_name == 'wrf':
			class_name='weka.classifiers.trees.RandomForest'
			options=None
		elif classifier_name == 'wj48':
			class_name='weka.classifiers.trees.J48'
			options=None
		elif classifier_name == 'wnb':
			class_name='weka.classifiers.bayes.NaiveBayes'
			options='-D'
		elif classifier_name == 'wbn':
			class_name='weka.classifiers.bayes.BayesNet'
			options='-D -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5'
		elif classifier_name == 'wsv':
			# Implementation of one-class SVM used in Anomaly Detection mode
			class_name='weka.classifiers.functions.LibSVM'
			options='-S 2'

		if options is not None:
			self._classifier = Classifier(classname=class_name, options=[option for option in options.split()])
		else:
			self._classifier = Classifier(classname=class_name)

	def fit(self, training_set, ground_truth):

		print(np.unique(ground_truth))

		self.ground_truth = ground_truth

		training_set = self._sklearn2weka(training_set, self.ground_truth)
		training_set.class_is_last()

		t = time.time()
		self._classifier.build_classifier(training_set)
		t = time.time() - t

		self.t_ = t

		return self

	def predict(self, testing_set):

		testing_set = self._sklearn2weka(testing_set, self.ground_truth)
		testing_set.class_is_last()

		preds = []
		for index, inst in enumerate(testing_set):
			pred = self._classifier.classify_instance(inst)
			preds.append(pred)

		preds = np.vectorize(self._dict.get)(preds)

		return np.array(preds)

	def predict_proba(self, testing_set):

		testing_set = self._sklearn2weka(testing_set, self.ground_truth)
		testing_set.class_is_last()

		dists = []
		for index, inst in enumerate(testing_set):
			dist = self._classifier.distribution_for_instance(inst)
			dists.append(dist)

		return np.array(dists)

	def set_oracle(self, oracle):

		pass

	def _sklearn2weka(self, features, labels=None):

		labels_encoder = OrdinalEncoder()
		labels_nominal = labels_encoder.fit_transform(np.array(labels).reshape(-1, 1))

		if not hasattr(self, 'dict') and labels is not None:

			dict = {}

			for label, nominal in zip(labels, labels_nominal):
				if nominal.item(0) not in dict:
					dict[nominal.item(0)] = label

			self._dict = dict

		labels_column = np.reshape(labels_nominal,[labels_nominal.shape[0], 1])

		weka_dataset = ndarray_to_instances(np.ascontiguousarray(features, dtype=np.float_), 'weka_dataset')
		weka_dataset.insert_attribute(Attribute.create_nominal('tag', [str(float(i)) for i in range(len(self._dict))]), features.shape[1])

		if labels is not None:
			for index, inst in enumerate(weka_dataset):
				inst.set_value(features.shape[1], labels_column[index])
				weka_dataset.set_instance(index,inst)

		return weka_dataset

class SklearnSparkWrapper(object):

	def __init__(self, spark_session, classifier_class, num_classes=None,
		numerical_features_length=None, nominal_features_lengths=None, numerical_features_index=None,
		nominal_features_index=None, fine_nominal_features_index=None,
		classifier_opts=None, epochs_number=None, level=None, fold=None, classify=None, workers_number=None,
		arbitrary_discr=''):

		self.spark_session = spark_session
		self.scale = False
		self.probas_ = None
		self.is_keras = False
		self.workers_number = workers_number

		if classifier_class == 'drf':
			self._classifier = Spark_RandomForestClassifier(featuresCol='features', labelCol='categorical_label',\
			predictionCol='prediction', probabilityCol='probability', rawPredictionCol='rawPrediction',\
			maxDepth=20, maxBins=128, minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=1024, cacheNodeIds=False, checkpointInterval=10,\
			impurity='gini', numTrees=100, featureSubsetStrategy='sqrt', seed=None, subsamplingRate=1.0)
		elif classifier_class == 'dnb':
			self._classifier = Spark_NaiveBayesClassifier(featuresCol='scaled_features', labelCol='categorical_label',\
			predictionCol='prediction', probabilityCol='probability', rawPredictionCol='rawPrediction', smoothing=1.0,\
			modelType='multinomial', thresholds=None, weightCol=None)
			self.scale = True
		elif classifier_class == 'dmp':
			layers = []
			input_dim = numerical_features_length+np.sum(nominal_features_lengths)
			depth = classifier_opts[0]
			if input_dim is not None and num_classes is not None:
				layers = [ input_dim ] + [ 100 for _ in range(int(depth)) ] + [ num_classes ]
			else:
				raise Exception('Both input_dim and num_classes must be declared.')
			self._classifier = Spark_MultilayerPerceptronClassifier(featuresCol='scaled_features', labelCol='categorical_label',\
			predictionCol='prediction', maxIter=100, tol=1e-06, seed=0, layers=layers, blockSize=32, stepSize=0.03, solver='l-bfgs',\
			initialWeights=None, probabilityCol='probability', rawPredictionCol='rawPrediction')
			self.scale = True
		elif classifier_class == 'dgb':
			self._classifier = Spark_GBTClassifier(featuresCol='features', labelCol='categorical_label',\
			predictionCol='prediction', maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256,\
			cacheNodeIds=False, checkpointInterval=10, lossType='logistic', maxIter=20, stepSize=0.1, seed=None,\
			subsamplingRate=1.0, featureSubsetStrategy='all')
		elif classifier_class == 'ddt':
			self._classifier = Spark_DecisionTreeClassifier(featuresCol='features', labelCol='categorical_label',\
			predictionCol='prediction', probabilityCol='probability', rawPredictionCol='rawPrediction', maxDepth=5, maxBins=32,\
			minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, impurity='gini', seed=None)
		elif classifier_class.startswith('dk'):
			depth = classifier_opts[0]
			self.keras_wrapper = SklearnKerasWrapper(*classifier_opts,model_class=classifier_class[1:],
				epochs_number=epochs_number,
				numerical_features_length=numerical_features_length+np.sum(nominal_features_lengths),
				nominal_features_lengths=[],num_classes=num_classes,
				nominal_features_index=[], fine_nominal_features_index=[],
				numerical_features_index=numerical_features_index+fine_nominal_features_index+nominal_features_index,
				level=level, fold=fold, classify=classify, weight_features=weight_features, arbitrary_discr=arbitrary_discr)
			self._classifier = self.keras_wrapper.init_model()[2]
			self.is_keras = True

	def fit(self, training_set, ground_truth):

		self.ground_truth = ground_truth

		if self.is_keras:
			nom_training_set, num_training_set = self.keras_wrapper.split_nom_num_features(training_set)
			training_set = np.array([ num_training_set ] + nom_training_set)
			training_set = self._sklearn2spark(training_set, self.ground_truth, True)
			self._classifier = ADAG(keras_model=self._classifier, worker_optimizer=Adadelta(lr=1.), loss='categorical_crossentropy', num_workers=workers_number,
				batch_size=32, communication_window=12, num_epoch=epochs_number, features_col='features', label_col='categorical_label', metrics=['categorical_accuracy'])
		else:
			training_set = self._sklearn2spark(training_set, self.ground_truth)
		# print(self.ground_truth)
		# input()
		if self.scale:
			scaler = Spark_MinMaxScaler(inputCol='features', outputCol='scaled_features')
			self.scaler_model_ = scaler.fit(training_set)
			scaled_training_set = self.scaler_model_.transform(training_set)
		else:
			scaled_training_set = training_set

		# Maybe keras model train is returned
		if not self.is_keras:
			t = time.time()
			self.model = self._classifier.fit(scaled_training_set)
			t = time.time() - t
		else:
			t = time.time()
			self.model = self._classifier.train(scaled_training_set)
			t = time.time() - t
			self.model = ModelPredictor(keras_model=self.model, features_col='features')

		self.t_ = t

		return self

	def predict(self, testing_set):

		if self.is_keras:
			nom_testing_set, num_testing_set = self.keras_wrapper.split_nom_num_features(testing_set)
			testing_set = [ num_testing_set ] + nom_testing_set

		testing_set = self._sklearn2spark(testing_set)
		if self.scale:
			scaled_testing_set = self.scaler_model_.transform(testing_set)
		else:
			scaled_testing_set = testing_set

		if not self.is_keras:
			self.results = self.model.transform(scaled_testing_set)
			preds = np.array([int(float(row.prediction)) for row in self.results.collect()])
			self.probas_ = np.array([row.probability.toArray() for row in self.results.collect()])
		else:
			preds = self.model.predict(testing_set)
	 
		return preds

	def predict_proba(self, testing_set):

		if self.probas_ is None:
			self.predict(testing_set)
		return np.array(self.probas_)

	def set_oracle(self, oracle):

		self.oracle = oracle

	def _sklearn2spark(self, features, labels=None, multi_input=False):

		if multi_input:
			features_names = []
			labels_names = []
			dataset = pandas.DataFrame()
			c=0
			for i,feature in enumerate(features):
				feature = feature.toarray().T.tolist() if issparse(feature) else feature.T.tolist()
				for f in feature:
					dataset['features_%s'%c] = f
					features_names.append('features_%s'%c)
					c+=1
			# c=0
			# labels = labels.toarray().T.tolist() if issparse(labels) else labels.T.tolist()
			# for label in labels:
			dataset['categorical_label'] = labels if labels is not None else ['']*features.shape[0]
				# labels_names.append('categorical_label_%s'%c)
				# c+=1
		else:
			dataset = pandas.DataFrame(
				{'features': features.tolist(),
				 'categorical_label': labels if labels is not None else ['']*features.shape[0]
				})

		spark_dataset_with_list = self.spark_session.createDataFrame(dataset)
		if multi_input:
			# Join all features columns
			assembler = VectorAssembler(inputCols=features_names, outputCol='features')
			assembler.setHandleInvalid('skip')
			spark_dataset_with_list = assembler.transform(spark_dataset_with_list)
			# Join all labels columns
			onehotencoder = OneHotTransformer(output_dim=len(np.unique(labels)),input_col='categorical_label', output_col='ohe_label')
			spark_dataset_with_list = onehotencoder.transform(spark_dataset_with_list)
		
		list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
		
		# spark_dataset = spark_dataset_with_list.select(
		# 	*[ list_to_vector_udf(spark_dataset_with_list[features_name]).alias(features_name) for features_name in features_names ],
		# 	spark_dataset_with_list['categorical_label']
		# )
		# return spark_dataset, features_names

		if multi_input:
			spark_dataset = spark_dataset_with_list.select(
				list_to_vector_udf(spark_dataset_with_list['features']).alias('features'),
				spark_dataset_with_list['ohe_label'].alias('categorical_label')
			)
		else:
			spark_dataset = spark_dataset_with_list.select(
				list_to_vector_udf(spark_dataset_with_list['features']).alias('features'),
				spark_dataset_with_list['categorical_label']
			)

		# if spark_dataset.rdd.getNumPartitions() != self.workers_number:
		# 	spark_dataset = spark_dataset.coalesce(numPartitions=self.workers_number)

		return spark_dataset

class SklearnKerasWrapper(BaseEstimator, ClassifierMixin):

	def __init__(self, depth='3', hidden_activation_function_name='elu', mode='n', sharing_ray='3', grafting_depth='0', compression_ratio='.1',
		model_class='kdae', epochs_number=10, numerical_features_length=None, nominal_features_lengths=None, num_classes=1,
		nominal_features_index=None, fine_nominal_features_index=None, numerical_features_index=None, fold=0, level=0, classify=False, weight_features=False,
		arbitrary_discr=''):

		# print(depth, hidden_activation_function, mode, sharing_ray, grafting_depth, compression_ratio,
		# model_class, epochs_number, num_classes, fold, level, classify)
		# input()
		model_discr = model_class + '_' + '_'.join(
			[str(c) for c in [ depth, hidden_activation_function_name, mode, sharing_ray, grafting_depth, compression_ratio ]]
		)

		self.sharing_ray = int(sharing_ray)
		self.grafting_depth = int(grafting_depth)
		self.depth = int(depth)
		self.compression_ratio = float(compression_ratio)
		self.sparse = False
		self.variational = False
		self.denoising = False
		self.propagational = False
		self.auto = False
		self.mode = mode
		self.fold = fold
		self.level = level
		if 'n' not in mode:
			if 'v' in mode:
				self.variational = True
				self.patience = 10
			if 'd' in mode:
				self.denoising = True
				self.patience = 10
			if 's' in mode:
				self.sparse = True
				self.patience = 10
			if 'p' in mode:
				self.propagational = True
			if 'a' in mode:
				self.auto = True
				model_discr = model_class + '_' + '_'.join(
					[str(c) for c in [ depth, hidden_activation_function_name, mode ]]
				)
		self.hidden_activation_function_name = hidden_activation_function_name
		self.hidden_activation_function = globals()[hidden_activation_function_name]
		self.model_class = model_class
		self.epochs_number = epochs_number
		self.numerical_features_length = numerical_features_length
		self.nominal_features_lengths = nominal_features_lengths
		self.num_classes = num_classes
		self.nominal_features_index = nominal_features_index
		self.fine_nominal_features_index = fine_nominal_features_index
		self.numerical_features_index = numerical_features_index
		self.classify = classify
		self.weight_features = weight_features
		log_folder = './data_%s/material/log' % model_discr
		plot_folder = './data_%s/material/plot' % model_discr
		summary_folder = './data_%s/material/summary' % model_discr
		tmp_folder = './data_%s/tmp' % model_discr
		if not os.path.exists(log_folder):
			os.makedirs(log_folder)
		if not os.path.exists(plot_folder):
			os.makedirs(plot_folder)
		if not os.path.exists(summary_folder):
			os.makedirs(summary_folder)
		if not os.path.exists(tmp_folder):
			os.makedirs(tmp_folder)
		self.log_file = '%s/%s_%s_model_%s_l_%s_log.csv' % (log_folder, arbitrary_discr, fold, model_discr, level+1)
		self.plot_file = '%s/%s_%s_model_%s_l_%s_plot' % (plot_folder, arbitrary_discr, fold, model_discr, level+1)
		self.summary_file = '%s/%s_%s_model_%s_l_%s_summary' % (summary_folder, arbitrary_discr, fold, model_discr, level+1)
		self.losses_file = '%s/%s_%s_model_%s_l_%s_losses.dat' % (summary_folder, arbitrary_discr, fold, model_discr, level+1)
		self.checkpoint_file = "%s/%s_weights.hdf5" % (tmp_folder,arbitrary_discr)
		self.scaler = QuantileTransformer()
		self.label_encoder = OneHotEncoder()
		self.patience = 1
		self.min_delta = 1e-4
		self.proba = None
		# For internal optimization
		self.k_internal_fold = 3
		self.complexity_red = 2

		self.numerical_output_activation_function = 'sigmoid'
		self.numerical_loss_function = 'mean_squared_error'
		self.nominal_output_activation_functions = None
		self.nominal_loss_functions = None

		if self.fine_nominal_features_index != None:
			self.nominal_output_activation_functions = []
			self.nominal_loss_functions = []
			for index in sorted(self.nominal_features_index + self.fine_nominal_features_index):
				if index in self.nominal_features_index:
					self.nominal_output_activation_functions.append('softmax')
					self.nominal_loss_functions.append('categorical_crossentropy')
				elif index in self.fine_nominal_features_index:
					self.nominal_output_activation_functions.append('sigmoid')
					self.nominal_loss_functions.append('mean_squared_error')

	def fit(self, X, y=None):

		if not -1 <= self.sharing_ray <= self.depth or not 0 <= self.grafting_depth <= self.depth+1:
			raise Exception('Invalid value for Shared Ray or for Grafting Depth.\n\
				Permitted values are: S_r in [-1, Depth] and G_d in [0, Depth+1].')

		nom_X, num_X = self.split_nom_num_features(X)
		
		# If current parameters are not previously initialized, classifier would infer them from training set
		if self.nominal_features_lengths == None:
			self.nominal_features_lengths = [ v[0][0].shape[1] for v in nom_X]
		if self.numerical_features_length == None:
			self.numerical_features_length = num_X.shape[1]
		if self.nominal_output_activation_functions == None:
			self.nominal_output_activation_functions = [ 'softmax' if np.array([ np.sum(r) == 1 and np.max(r) == 1 for r in v ]).all() else 'sigmoid' for v in nom_X ]
		if self.nominal_loss_functions == None:
			self.nominal_loss_functions = [ 'categorical_crossentropy' if np.array([ np.sum(r) == 1 and np.max(r) == 1 for r in v ]).all() else 'mean_squared_error' for v in nom_X ]
		
		# Define weights for loss average
		if self.weight_features:
			self.numerical_weight = self.numerical_features_length
		else:
			self.numerical_weight = 1
		self.nominal_weights = [1] * len(self.nominal_features_lengths)

		reconstruction_models, _, classification_models, models_names = self.init_models(self.model_class, self.num_classes, self.classify)
		# Scaling training set for Autoencoder
		scaled_num_X = self.scaler.fit_transform(num_X)
		if self.classify:
			one_hot_y = self.label_encoder.fit_transform(y.reshape(-1,1))
		normal_losses_per_fold = []
		times_per_fold = []
		if len(reconstruction_models) > 1:
			# Iterating over all the models to find the best
			for index in range(len(reconstruction_models)):
				print('\nStarting StratifiedKFold for %s\n' % models_names[index])
				skf = StratifiedKFold(n_splits=self.k_internal_fold, shuffle=True)
				normal_losses_per_fold.append([])
				times_per_fold.append([])
				f=0
				reconstruction_model = reconstruction_models[index]
				classification_model = classification_models[index]
				for train_index, validation_index in skf.split(
					np.zeros(shape=(num_X.shape[0], self.numerical_features_length + len(self.nominal_features_lengths))),
					y):
					# To reduce amount of similar models to built, we save initial weights at first fold, than we will use this for subsequent folds.
					if f == 0:
						reconstruction_model.save_weights(self.checkpoint_file)
					else:
						reconstruction_model.load_weights(self.checkpoint_file)
					f += 1
					print('\nFold %s\n' % f)
					# for _model in self._models:
					if not self.denoising:
						times_per_fold[-1].append(time.time())
						# Train Autoencoder
						train_history =\
							reconstruction_model.fit(
								[ scaled_num_X[train_index,:] ] + [ nom_training[train_index,:] for nom_training in nom_X ],
								[ scaled_num_X[train_index,:] ] + [ nom_training[train_index,:] for nom_training in nom_X ],
								epochs=self.epochs_number,
								batch_size=32,
								shuffle=True,
								verbose=2,
								# validation_split = 0.1,
								validation_data=(
									[ scaled_num_X[validation_index,:] ] + [ nom_training[validation_index,:] for nom_training in nom_X ],
									[ scaled_num_X[validation_index,:] ] + [ nom_training[validation_index,:] for nom_training in nom_X ]
									),
								callbacks = [
									EarlyStopping(monitor='val_loss', patience=self.patience, min_delta=self.min_delta),
									CSVLogger(self.log_file),
									# ModelCheckpoint(filepath=self.checkpoint_file, verbose=1, save_best_only=True)
									]
								)
						if self.classify:
							for layer in reconstruction_model:
								layer.trainable = False
							train_history =\
								classification_model.fit(
									[ scaled_num_X[train_index,:] ] + [ nom_training[train_index,:] for nom_training in nom_X ],
									one_hot_y[train_index,:],
									epochs=self.epochs_number,
									batch_size=32,
									shuffle=True,
									verbose=2,
									# validation_split = 0.1,
									validation_data=(
										[ scaled_num_X[validation_index,:] ] + [ nom_training[validation_index,:] for nom_training in nom_X ],
										one_hot_y[validation_index,:]
										),
									callbacks = [
										EarlyStopping(monitor='val_loss', patience=self.patience, min_delta=self.min_delta),
										CSVLogger(self.log_file),
										# ModelCheckpoint(filepath=self.checkpoint_file, verbose=1, save_best_only=True)
										]
									)
					else:
						noise_factor = .5
						noisy_scaled_num_X = scaled_num_X[train_index,:] + noise_factor * np.random.normal(loc=.0, scale=1., size=scaled_num_X[train_index,:].shape)
						times_per_fold[-1].append(time.time())
						# Train Autoencoder
						train_history =\
							reconstruction_model.fit(
								[ noisy_scaled_num_X ] + [ nom_training[train_index,:] for nom_training in nom_X ],
								[ scaled_num_X[train_index,:] ] + [ nom_training[train_index,:] for nom_training in nom_X ],
								epochs=self.epochs_number,
								batch_size=32,
								shuffle=True,
								verbose=2,
								# validation_split = 0.1,
								validation_data=(
									[ scaled_num_X[validation_index,:] ] + [ nom_training[validation_index,:] for nom_training in nom_X ],
									[ scaled_num_X[validation_index,:] ] + [ nom_training[validation_index,:] for nom_training in nom_X ]
									),
								callbacks = [
									EarlyStopping(monitor='val_loss', patience=self.patience, min_delta=self.min_delta),
									CSVLogger(self.log_file),
									# ModelCheckpoint(filepath=self.checkpoint_file, verbose=1, save_best_only=True)
									]
								)
					# num_losses, nom_losses, clf_losses = self.get_losses(train_history.history,discr='val')
					# # This order (i.e. num+nom) must be respected to compute weighthed average
					# if self.classify or (len(num_losses) == 0 and len(nom_losses) == 0):
					# 	losses = clf_losses
					# 	normal_losses_per_fold[-1].append(losses)
					# else:
					# 	losses = num_losses + nom_losses
					# 	# print(losses)
					# 	# print(num_weight + nom_weights)
					# 	normal_losses_per_fold[-1].append(np.mean(losses, weights=[ self.numerical_weight ]+self.nominal_weights))
					normal_losses_per_fold[-1].append(self.get_loss(train_history.history,'val_loss'))
					times_per_fold[-1][-1] = time.time() - times_per_fold[-1][-1]
					# train_history = losses = nom_losses = num_losses = clf_losses = None
					# del train_history, losses, nom_losses, num_losses, clf_losses
					# gc.collect()
				normal_losses_per_fold[-1] = np.asarray(normal_losses_per_fold[-1])
				times_per_fold[-1] = np.asarray(times_per_fold[-1])
			mean_normal_losses = np.mean(normal_losses_per_fold,axis=1)
			std_normal_losses = np.std(normal_losses_per_fold,axis=1)
			# We use the sum of mean and std deviation to choose the best model
			meanPstd_normal_losses = mean_normal_losses+std_normal_losses
			mean_times = np.mean(times_per_fold,axis=1)
			# Saving losses per model
			with open(self.losses_file,'a') as f:
				for mean_normal_loss,std_normal_loss,model_name in zip(mean_normal_losses,std_normal_losses,models_names):
					f.write('%s %s %s\n' % (mean_normal_loss,std_normal_loss,model_name))
			min_loss = np.min(meanPstd_normal_losses)
			print(meanPstd_normal_losses)
			best_model_index = list(meanPstd_normal_losses).index(min_loss)
			best_model_name = models_names[best_model_index]
			t = np.max(mean_times)
			print(t)
		else:
			best_model_name = models_names[0]
		t=0
		reconstruction_models = None
		classification_models = None
		del reconstruction_models
		del classification_models
		gc.collect()
		sharing_ray = int(best_model_name.split('_')[1])
		grafting_depth = int(best_model_name.split('_')[2])
		self.reconstruction_model_, _, self.classification_model_, _ = self.init_model(self.model_class, sharing_ray, grafting_depth, self.num_classes, self.classify)
		if not self.denoising:
			t = time.time() - t
			if self.reconstruction_model_ is not None:
				# Train Autoencoder
				train_history =\
					self.reconstruction_model_.fit(
						[ scaled_num_X ] + nom_X,
						[ scaled_num_X ] + nom_X,
						epochs=self.epochs_number,
						batch_size=32,
						shuffle=True,
						verbose=2,
						callbacks = [
							EarlyStopping(monitor='loss', patience=self.patience, min_delta=self.min_delta),
							CSVLogger(self.log_file),
							# ModelCheckpoint(filepath=self.checkpoint_file, verbose=1, save_best_only=True)
							]
						)
				if self.classify:
					for layer in self.reconstruction_model_.layers:
						layer.trainable = False
			if self.classify:
				train_history =\
					self.classification_model_.fit(
						[ scaled_num_X ] + nom_X,
						one_hot_y,
						epochs=self.epochs_number,
						batch_size=32,
						shuffle=True,
						verbose=2,
						callbacks = [
							EarlyStopping(monitor='loss', patience=self.patience, min_delta=self.min_delta),
							CSVLogger(self.log_file),
							# ModelCheckpoint(filepath=self.checkpoint_file, verbose=1, save_best_only=True)
							]
						)
		else:
			noise_factor = .5
			noisy_scaled_num_X = scaled_num_X[train_index,:] + noise_factor * np.random.normal(loc=.0, scale=1., size=scaled_num_X[train_index,:].shape)
			t = time.time() - t
			# Train Autoencoder
			train_history =\
				self.reconstruction_model_.fit(
					[ noisy_scaled_num_X ] + nom_X,
					[ scaled_num_X ] + nom_X,
					epochs=self.epochs_number,
					batch_size=32,
					shuffle=True,
					verbose=2,
					callbacks = [
						EarlyStopping(monitor='loss', patience=self.patience, min_delta=self.min_delta),
						CSVLogger(self.log_file),
						# ModelCheckpoint(filepath=self.checkpoint_file, verbose=1, save_best_only=True)
						]
					)
		# num_losses, nom_losses, clf_losses = self.get_losses(train_history.history)
		# # This order (i.e. num+nom) must be respected to compute weighthed average
		# if self.classify or (len(num_losses) == 0 and len(nom_losses) == 0):
		# 	losses = clf_losses
		# 	# self.normal_loss_ = np.mean(losses)
		# else:
		# 	losses = num_losses + nom_losses
			# self.normal_loss_ = np.average(losses, weights = num_weight + nom_weights)
		# self.normal_loss_ = np.mean(losses)
		self.normal_loss_ = self.get_loss(train_history.history,'loss')
		t = time.time() - t
		if self.reconstruction_model_ is not None:
			with open(self.summary_file+'_best_reconstruction_model_'+best_model_name+'.dat','w') as f:
				self.reconstruction_model_.summary(print_fn=lambda x: f.write(x + '\n'))
		if self.classify:
			with open(self.summary_file+'_best_classification_model_'+best_model_name+'.dat','w') as f:
				self.classification_model_.summary(print_fn=lambda x: f.write(x + '\n'))
		print('loss',self.normal_loss_)
		print('time',t)
		self.t_ = t
		return self

	def get_losses(self, history, discr=''):
		num_losses = []
		nom_losses = []
		clf_losses = []
		for loss_tag in history:
			if discr in loss_tag:
				if 'num' in loss_tag:
					for num_loss in history[loss_tag][::-1]:
						if not np.isnan(num_loss):
							num_losses.append(num_loss)
							break
				elif 'nom' in loss_tag:
					for nom_loss in history[loss_tag][::-1]:
						if not np.isnan(nom_loss):
							nom_losses.append(nom_loss)
							break
				elif loss_tag == 'loss':
					for clf_loss in history[loss_tag][::-1]:
						if not np.isnan(clf_loss):
							clf_losses.append(clf_loss)
							break

		return num_losses, nom_losses, clf_losses

	def get_loss(self, history, loss_tag):
		for loss in history[loss_tag][::-1]:
			if not np.isnan(loss):
				return loss
		return -1

	def get_weights(self,mod='r'):
		if 'r' in mod:
			return self.reconstruction_model_.get_weights()
		elif 'c' in mod:
			return self.classification_model_.get_weights()
		elif 'k' in mod:
			return self.compression_model.get_weights()
		return None

	def set_weights(self,weights,mod='r'):
		if 'r' in mod:
			self.reconstruction_model_.set_weights(weights)
			return self.reconstruction_model_
		elif 'c' in mod:
			self.classification_model_.set_weights(weights)
			return self.classification_model_
		elif 'k' in mod:
			self.compression_model.set_weights(weights)
			return self.compression_model
		return None

	def get_models(self,mod='rck'):
		models = []
		if 'r' in mod:
			models.append(self.reconstruction_model_)
		if 'c' in mod:
			models.append(self.classification_model_)
		if 'k' in mod:
			models.append(self.compression_model)
		if len(models) == 1:
			return models[0]
		return models

	def init_models(self, model_class=None, num_classes=None, classify=None):

		if model_class is None:
			model_class = self.model_class
		if num_classes is None:
			num_classes = self.num_classes
		if classify is None:
			classify = self.classify

		reconstruction_models = []
		compression_models = []
		classification_models = []
		models_n = []

		if self.auto:
			range_sr = range(-1,self.depth+1,self.complexity_red)
			range_gd = range(0,self.depth+2,self.complexity_red)
		else:
			range_sr = [ self.sharing_ray ]
			range_gd = [ self.grafting_depth ]

		for sharing_ray in range_sr:
			for grafting_depth in range_gd:
				reconstruction_model, compression_model, classification_model, model_n = self.init_model(model_class, sharing_ray, grafting_depth, num_classes, classify)
				reconstruction_models.append(reconstruction_model)
				compression_models.append(compression_model)
				classification_models.append(classification_model)
				models_n.append(model_n)

		return reconstruction_models, compression_models, classification_models, models_n

	def init_model(self, model_class=None, sharing_ray=None, grafting_depth=None, num_classes=None, classify=None):

		if None in [self.nominal_features_lengths, self.numerical_features_length, self.nominal_output_activation_functions, self.nominal_loss_functions]:
			raise Exception('Cannot call init_models or init_model methods of SklearnSparkWrapper without setting\
				nominal_features_lengths, numerical_features_length, nominal_output_activation_functions, and nominal_loss_functions\
				attributes. If you want to automatically initialize them, use SklearnSparkWrapper fit method.')

		if model_class is None:
			model_class = self.model_class
		if sharing_ray is None:
			sharing_ray = self.sharing_ray
		if grafting_depth is None:
			grafting_depth = self.grafting_depth
		if num_classes is None:
			num_classes = self.num_classes
		if classify is None:
			classify = self.classify

		reduction_coeff = (1 - self.compression_ratio) / (self.depth + 1) 
		reduction_factors = [ 1 - reduction_coeff * i for i in range(1,self.depth+1) ] + [self.compression_ratio]
		latent_ratio = .10
		# This parameter is used to set the number of e/d layers nearest the middle layer to join
		# If it's more or equal than zero, the middle layer is automatically joined
		# The parameter take values in [-1,self.depth], where -1 states no join, zero states only middle layer join
		# If it's more than zero, it indicates the couple of e/d layers to join, starting from nearest middle layer
		# sharing_ray = -1
		# This parameter specify at which level num model grafts on nom one
		# It takes values in [0,self.depth+1], where self.depth+1 state for no graft, and others value refers to the 0,1,2,..self.depth-1 i-e/d layer
		# E.g. if it is equal to self.depth, the graft is done on middle layer, if to 0, on the first encoder layer
		# grafting_depth = self.depth-1
		num_hidden_activation_function = self.hidden_activation_function
		nom_hidden_activation_function = self.hidden_activation_function
		hyb_activation_function = self.hidden_activation_function
		reconstruction_model = None
		compression_model = None
		classification_model = None
		model_name = '%s_%s_%s' % (model_class, sharing_ray, grafting_depth)
		# Defining input tensors, one for numerical and one for each categorical
		input_datas = []
		input_datas.append(Input(shape=(self.numerical_features_length,), name='num_input'))
		for i,nom_length in enumerate(self.nominal_features_lengths):
			input_datas.append(Input(shape=(nom_length,), name='nom_input%s' % i, sparse=True))
		# TODO - StackedAutoencoder per-packet anomaly detection.
		if model_class == 'ksae':
			reconstruction_models = []
			compression_models = []
			x = None
			for _ in range(32):
				reconstruction_model, compression_model, _, _ = self.init_model(model_class='kdae')
				if len(compression_model.outputs) > 1:
					x = concatenate(compression_model.outputs)
				else:
					x = compression_model.output
				reconstruction_models.append(reconstruction_model)
				compression_models.append(compression_model)

		if model_class == 'kdae' or model_class == 'kc2dae':
			join_factors = [reduction_factors[i] if i < 0 else reduction_factors[-i-1] for i in range(-(sharing_ray+1),(sharing_ray+1)) if i != 0]
			encodeds = []
			decodeds = []
			xs = []
			z_means = []
			z_log_var = []
			# reparameterization trick
			# instead of sampling from Q(z|X), sample epsilon = N(0,I)
			# z = z_mean + sqrt(var) * epsilon
			def sampling(args):
				"""Reparameterization trick by sampling from an isotropic unit Gaussian.
				# Arguments
					args (tensor): mean and log of variance of Q(z|X)
				# Returns
					z (tensor): sampled latent vector
				"""
				z_mean, z_log_var = args
				batch = K.shape(z_mean)[0]
				dim = K.int_shape(z_mean)[1]
				# by default, random_normal has mean = 0 and std = 1.0
				epsilon = K.random_normal(shape=(batch, dim))
				return z_mean + K.exp(0.5 * z_log_var) * epsilon
			propagateds = {}
			# Building model for numerical features
			x = input_datas[0]
			for i in range(np.min([self.depth+1, grafting_depth])):
				if i < self.depth: # Adding encoding layers
					x = Dense(int(np.ceil(reduction_factors[i]*self.numerical_features_length)), activation=num_hidden_activation_function, name='num_enc%s' % i)(x)
					if self.propagational:
						if 'num' not in propagateds:
							propagateds['num'] = []
						propagateds['num'].append(x)
				if i == self.depth: # Adding middle layer
					intermediate_dim = int(np.ceil(reduction_factors[i]*self.numerical_features_length))
					x = Dense(intermediate_dim, activation=num_hidden_activation_function, name='num_mid')(x)
					if self.variational: # self.variational middle layer
						latent_dim = int(np.ceil(latent_ratio*self.numerical_features_length))
						z_mean = Dense(latent_dim, name='num_z_mean%s' % (i))(x)
						z_log_var = Dense(latent_dim, name='num_z_log_var%s' % (i))(x)
						z = Lambda(sampling, output_shape=(latent_dim,), name='num_z%s' % (i))([z_mean, z_log_var])
						# z = Lambda(lambda z_mean, z_log_var: self.sampling([z_mean, z_log_var]), output_shape=(latent_dim,), name='nom_z%s' % (i))
						x = Dense(intermediate_dim, activation=num_hidden_activation_function, name='num_z_mid%s' % (i))(z)
					# Applying sparsifycation only on output self.variational or as in input as in output? <----------------------#####
					if self.sparse: # Sparsifying middle output
						x.W_regularizer = regularizers.l1(1e-4)
					encodeds.append(x)
			graft_in = x
			graft_out_bk = x
			graft_out = []
			# Building model for nominal features
			for i,nom_length in enumerate(self.nominal_features_lengths):
				# Instantiate separated inputs
				x = input_datas[i+1]
				for j in range(self.depth - sharing_ray):
					if j == grafting_depth: # Saving layer to be grafted as graft_in
						concat_ix = concatenate([ graft_in, x ])
						x = concat_ix
					if j != len(reduction_factors)-1: # Adding encoding layers
						x = Dense(int(np.ceil(reduction_factors[j]*nom_length)), activation=nom_hidden_activation_function, name='nom_enc%s%s' % (i,j))(x)
						if self.propagational:
							if 'nom%s' % i not in propagateds:
								propagateds['nom%s' % i] = []
							propagateds['nom%s' % i].append(x)
					if j == len(reduction_factors)-1: # Adding middle layer
						intermediate_dim = int(np.ceil(reduction_factors[j]*nom_length))
						x = Dense(intermediate_dim, activation=nom_hidden_activation_function, name='nom_mid%s' % (i))(x)
						if self.variational: # self.variational middle layer
							latent_dim = int(np.ceil(latent_ratio*nom_length))
							z_mean = Dense(latent_dim, name='nom_z_mean%s' % (i))(x)
							z_log_var = Dense(latent_dim, name='nom_z_log_var%s' % (i))(x)
							z = Lambda(sampling, output_shape=(latent_dim,), name='nom_z%s' % (i))([z_mean, z_log_var])
							# z = Lambda(lambda z_mean, z_log_var: self.sampling([z_mean, z_log_var]), output_shape=(latent_dim,), name='nom_z%s' % (i))
							x = Dense(intermediate_dim, activation=nom_hidden_activation_function, name='nom_z_mid%s' % (i))(z)
						# Applying sparsifycation only on output self.variational or as in input as in output? <----------------------#####
						if self.sparse: # Sparsifying middle output
							x.W_regularizer = regularizers.l1(1e-4)
						encodeds.append(x)
						if j == grafting_depth: # If middle layer if the graft layer, we save it also as graft_out
							graft_out.append(x)
				xs.append(x)
			if len(xs) > 0:
				concat_xs = concatenate(xs)
				x = concat_xs
				for i,join_factor in enumerate(join_factors):
					shared_length = np.sum(self.nominal_features_lengths)
					if i < np.floor(len(join_factors)/2.): # Adding encoding shared layers
						if i == (grafting_depth - self.depth + sharing_ray): # Saving graft_in
							concat_ix = concatenate([ graft_in, x ])
							x = concat_ix
						x = Dense(int(np.ceil(join_factor*shared_length)), activation=hyb_activation_function, name='nom_jenc%s' % (i))(x)
						if self.propagational:
							if 'hyb' not in propagateds:
								propagateds['hyb'] = []
							propagateds['hyb'].append(x)
					if i > np.floor(len(join_factors)/2.): # Adding decoding shared layers
						output_shape = int(np.ceil(join_factor*shared_length))
						x = Dense(output_shape, activation=hyb_activation_function, name='nom_jdec%s' % (i-sharing_ray-1))(x)
						if self.propagational:
							x = concatenate([ x, propagateds['hyb'][-1] ])
							x = Dense(output_shape, activation=hyb_activation_function, name='nom_pjdec%s' % (i-sharing_ray-1))(x)
							del propagateds['hyb'][-1]
						if i == (-grafting_depth + self.depth + sharing_ray): # Saving graft_out
							graft_out.append(x)
					if i == np.floor(len(join_factors)/2.): # Adding middle shared layer
						if i == (grafting_depth - self.depth + sharing_ray): # Saving graft_in
							concat_ix = concatenate([ graft_in, x ])
							x = concat_ix
						intermediate_dim = int(np.ceil(join_factor*shared_length))
						x = Dense(intermediate_dim, activation=hyb_activation_function, name='nom_jmid')(x)
						if self.variational: # self.variational middle layer
							latent_dim = int(np.ceil(latent_ratio*shared_length))
							z_mean = Dense(latent_dim, name='nom_z_mean%s' % (i))(x)
							z_log_var = Dense(latent_dim, name='nom_z_log_var%s' % (i))(x)
							z = Lambda(sampling, output_shape=(latent_dim,), name='nom_z%s' % (i))([z_mean, z_log_var])
							# z = Lambda(lambda z_mean, z_log_var: self.sampling([z_mean, z_log_var]), output_shape=(latent_dim,), name='nom_z%s' % (i))
							x = Dense(intermediate_dim, activation=hyb_activation_function, name='nom_z_mid%s' % (i))(z)
						# Applying sparsifycation only on output self.variational or as in input as in output? <----------------------#####
						if self.sparse: # Sparsifying middle output
							x.W_regularizer = regularizers.l1(1e-4)
						encodeds.append(x)
						if i == (grafting_depth - self.depth + sharing_ray): # And also graft_out, if middle shared layer is graft layer
							graft_out.append(x)
					xs = [x] * len(self.nominal_features_lengths)
			for i,nom_length in enumerate(self.nominal_features_lengths):
				x = xs[i]
				for j in reversed(range(self.depth - np.max([0, sharing_ray]))): # Adding decoder layers
					output_shape = int(np.ceil(reduction_factors[j]*nom_length))
					x = Dense(output_shape, activation=nom_hidden_activation_function, name='nom_dec%s%s' % (i,self.depth-j-1))(x)
					if self.propagational:
						x = concatenate([ x, propagateds['nom%s' % i][-1] ])
						x = Dense(output_shape, activation=hyb_activation_function, name='nom_pdec%s%s' % (i,self.depth-j-1))(x)
						del propagateds['nom%s' % i][-1]
					if j == grafting_depth: # If layer is graft layer, save graft_out
						graft_out.append(x)
				# Instantiate separated outputs
				decodeds.append(Dense(nom_length, activation=self.nominal_output_activation_functions[i], name='nom_out%s' % (i))(x))
			if len(graft_out) > 1: # More than one graft layer, if graft is on non shared layers
				x = concatenate(graft_out)
			elif len(graft_out) > 0: # Only one graft layer, if graft is on a shared layer
				x = graft_out[0]
			else: # When no graft, we use previously saved numerical output
				x = graft_out_bk
			for i in reversed(range(np.min([self.depth, grafting_depth]))): # Adding decoder layers
				output_shape = int(np.ceil(reduction_factors[i]*self.numerical_features_length))
				x = Dense(output_shape, activation=self.numerical_output_activation_function, name='num_dec%s' % (self.depth-i-1))(x)
				if self.propagational:
					x = concatenate([ x, propagateds['num'][-1] ])
					x = Dense(output_shape, activation=hyb_activation_function, name='num_pdec%s' % (self.depth-i-1))(x)
					del propagateds['num'][-1]
			decodeds = [ Dense(self.numerical_features_length, activation=self.numerical_output_activation_function, name='num_out')(x) ] + decodeds
			# For fitting
			reconstruction_model = Model(input_datas, decodeds)
			# For feature extraction
			compression_model = Model(input_datas, encodeds)
			plot_model(reconstruction_model, to_file=self.plot_file+'_reconstruction_jr%s_il%s' % (sharing_ray, grafting_depth)+'.png', show_shapes=True, show_layer_names=True)
			with open(self.summary_file+'_reconstruction_jr%s_il%s' % (sharing_ray, grafting_depth)+'.dat','w') as f:
				reconstruction_model.summary(print_fn=lambda x: f.write(x + '\n'))
			reconstruction_model.compile(optimizer=Adadelta(lr=1.),
				loss=[ self.numerical_loss_function ]+self.nominal_loss_functions,
				loss_weights=[ self.numerical_weight ]+self.nominal_weights)
			if classify:
				reconstruction_model, compression_model, _, _ = self.init_model(model_class, sharing_ray, grafting_depth)
				if len(compression_model.outputs) > 1:
					x = concatenate(compression_model.outputs)
				else:
					x = compression_model.output
				initial_size = int(x.shape[1])
				final_size = num_classes
				step = (initial_size - final_size) / self.depth
				mid_sizes = [ int(initial_size-step*i) for i in range(1, self.depth) ]
				for i,mid_size in enumerate(mid_sizes):
					x = Dense(mid_size, activation=self.hidden_activation_function, name='clf_hid%s' % i)(x)
				x = Dropout(.1)(x)
				x = Dense(final_size, activation='softmax', name='clf_out')(x)
				classification_model = Model(compression_model.inputs, x)
				plot_model(classification_model, to_file=self.plot_file+'_classification_jr%s_il%s' % (sharing_ray, grafting_depth)+'.png', show_shapes=True, show_layer_names=True)
				with open(self.summary_file+'_classification_jr%s_il%s' % (sharing_ray, grafting_depth)+'.dat','w') as f:
					classification_model.summary(print_fn=lambda x: f.write(x + '\n'))
				classification_model.compile(optimizer=Adadelta(lr=1.), loss='categorical_crossentropy')
		if model_class == 'kmlp':
			final_size = num_classes
			mid_size = 100
			if len(input_datas) == 1:
				x = input_datas[0]
			else:
				x = concatenate(input_datas)
			for i in range(self.depth):
				x = Dense(mid_size, activation=self.hidden_activation_function, name='clf_hid%s' % i)(x)
			x = Dropout(.1)(x)
			x = Dense(final_size, activation='softmax', name='clf_out')(x)
			classification_model = Model(input_datas, x)
			plot_model(classification_model, to_file=self.plot_file+'_d_%s_classification' % (self.depth)+'.png', show_shapes=True, show_layer_names=True)
			with open(self.summary_file+'_d_%s_classification' % (self.depth)+'.dat','w') as f:
				classification_model.summary(print_fn=lambda x: f.write(x + '\n'))
			classification_model.compile(optimizer=Adadelta(lr=1.), loss='categorical_crossentropy')
		# TODO: convolutional layer for mlp to flatten matricial input of weights from NeuralWeightsDecisor
		if model_class == 'kcmlp':
			final_size = num_classes
			mid_size = 100
			if len(input_datas) == 1:
				x = input_datas[0]
			else:
				x = concatenate(input_datas)
			x = Conv2D()
			for i in range(self.depth):
				x = Dense(mid_size, activation=self.hidden_activation_function, name='clf_hid%s' % i)(x)
			x = Dropout(.1)(x)
			x = Dense(final_size, activation='softmax', name='clf_out')(x)
			classification_model = Model(input_datas, x)
			plot_model(classification_model, to_file=self.plot_file+'_d_%s_classification' % (self.depth)+'.png', show_shapes=True, show_layer_names=True)
			with open(self.summary_file+'_d_%s_classification' % (self.depth)+'.dat','w') as f:
				classification_model.summary(print_fn=lambda x: f.write(x + '\n'))
			classification_model.compile(optimizer=Adadelta(lr=1.), loss='categorical_crossentropy')
		return reconstruction_model, compression_model, classification_model, model_name

	def predict(self, X, y=None):

		# X, nominal_features_index, numerical_features_index = self.expand_onehot_features(X)
		nom_X, num_X = self.split_nom_num_features(X)
		scaled_num_X = self.scaler.transform(num_X)
		if self.classify:
			pred = self.classification_model_.predict([ scaled_num_X ] + nom_X)
			self.proba = copy(pred)
			for i,sp in enumerate(pred):
				sp_max = np.max(sp)
				n_max = len(pred[pred == sp_max])
				pred[i] = [ 1 if p == sp_max else 0 for p in sp ]
				indexes = np.where(sp == 1)[0]
				if len(indexes) > 1:
					rand_sp = np.random.choice(indexes)
					pred[i] = [ 1 if j == rand_sp else 0 for j in range(len(sp)) ]
			pred = self.label_encoder.inverse_transform(pred)
		else:
			reconstr_X = self.reconstruction_model_.predict([ scaled_num_X ] + nom_X)
			losses = []
			losses.append(K.eval(globals()[self.numerical_loss_function](scaled_num_X,reconstr_X[0])))
			for nts, rnts, nominal_loss_function in zip(nom_X, reconstr_X[1:], self.nominal_loss_functions):
				losses.append(K.eval(globals()[nominal_loss_function](nts.toarray(),K.constant(rnts))))
			# first losses refers to numerical features, so we weighted them
			# num_weight = [ self.numerical_features_length ]
			# nom_weights = [1] * len(self.nominal_features_lengths)
			# loss = np.average(losses, axis = 0, weights = num_weight + nom_weights)
			# Weighted sum of losses for each sample
			loss = np.dot(np.array(losses).T, [self.numerical_weight]+self.nominal_weights)
			# loss = [ (l-self._loss_min)/(self._loss_max-self._loss_min) + eps for l in loss ]
			pred = np.asarray([
				-1 if l > self.normal_loss_
				else 1 for l in loss
				])
			self.proba = np.asarray(loss)
			self.proba = np.reshape(self.proba,self.proba.shape[0])
		pred = np.reshape(pred,pred.shape[0])
		return pred

	def predict_proba(self, X):
		
		if self.proba is None:
			self.predict(X)
		return self.proba

	def score(self, X, y=None):

		return 1/np.mean(self.predict_proba(X))

	def set_oracle(self, oracle):

		self.oracle = oracle

	def sklearn2keras(self, features, labels=None):

		pass

	def expand_onehot_features(self, set):

		# nominal_features_index = [0,1,2,5]
		nominal_features_index = [ i for i,v in enumerate(set[0,:]) if isinstance(v, np.ndarray) ]
		nominal_features_lengths = [ len(set[0,i]) for i in nominal_features_index ]

		nom_len = [0]
		nom_base = [0]
		for i,nfi in enumerate(nominal_features_index):
			nom_base.append(nfi + nom_len[-1])
			nom_len.append(nom_len[-1] + self.nominal_features_lengths[i] - 1)
			set = np.c_[ set[:,:nom_base[-1]], np.asarray(list(zip(*set[:,nom_base[-1]]))).T, set[:,nom_base[-1]+1:] ]

		nom_index = nom_base[1:]
		nom_length = self.nominal_features_lengths
		features_number = set.shape[1]

		nominal_features_index = []
		caught_features_index = []
		for nom_i,nom_l in zip(nom_index,nom_length):
			nominal_features_index.append([ i for i in range(nom_i,nom_i+nom_l) ])
			caught_features_index += range(nom_i,nom_i+nom_l)

		numerical_features_index = [ i for i in range(features_number) if i not in caught_features_index ]

		return np.asarray(set,dtype=float),\
			np.asarray(nominal_features_index,dtype=object),\
			np.asarray(numerical_features_index,dtype=int)

	def split_nom_num_features(self,set):
		nom_index = []
		num_index = []
		
		for i,v in enumerate(set[0,:]):
			# if isinstance(v, np.ndarray):
			if issparse(v):
				nom_index.append(i)
			else:
				num_index.append(i)

		nom_set = set[:, nom_index]
		num_set = set[:, num_index]

		temp_set = []
		for i in range(nom_set.shape[1]):
			temp_set.append(nom_set[:,i])

		# Fare un nom_set list di 4 ndarray ognuno con tutte le colonne dentro
		nom_set = [ csr_matrix([ sp[0].toarray()[0] for sp in np.vstack(v) ]) for v in temp_set ]
		# nom_set = [ csr_matrix(v) for v in temp_set ]

		return nom_set, np.asarray(num_set, dtype=float)

class SuperLearnerClassifier(object):
	def __init__(self, first_level_learners, num_classes, anomaly_class=None, features_number=0):

		self.num_classes = num_classes
		self.first_level_classifiers = []

		for first_level_learner in first_level_learners:
			if first_level_learner.startswith('s') or first_level_learner.startswith('k'):
				self.first_level_classifiers.append(AnomalyDetector(first_level_learner, anomaly_class, features_number))

		self.anomaly_class = anomaly_class

		if self.anomaly_class == 1:
			self.normal_class = 0
		else:
			self.normal_class = 1

		self.first_level_classifiers_number = len(self.first_level_classifiers)

		self.proba = None

	def fit(self, dataset, labels):
		
		t = time.time()
		# First: split train set of node of hyerarchy in 20% train and 80% testing
		# with respective ground truth NB: for first training of first level models
		# SuperLearner use only training set
		# training_set, _, ground_truth, _ = train_test_split(dataset, labels, train_size=0.2)

		# Second: fit classifiers of first level through train set
		# proba, true = self._first_level_fit(training_set, ground_truth)

		# Three: fit combiner passing it obtained probabilities and respective ground truth
		# this phase return weights, i.e. decisions template, per each classifier
		# self.weights = self._meta_learner_fit(proba, true)

		# Four: final fit of first level classifier consist in fitting all models
		# with all the dataset, that is the train set of the node of hierarchy
		self._first_level_final_fit(dataset, labels)

		t = time.time() - t

		self.t_ = t

		return self

	def _first_level_fit(self, dataset, labels):

		# First level fit consist in a stratified ten-fold validation
		# to retrieve probabilities associated at the ground truth
		# NB: in input there is the train set, in output probabilities associated to this set
		# NB2: we don't need the predictions
		skf = StratifiedKFold(n_splits=10, shuffle=True)

		oracles_per_fold = []
		# predictions_per_fold = []
		probabilities_per_fold = []

		for train_index, test_index in skf.split(dataset, labels):

			training_set = dataset[train_index]
			testing_set = dataset[test_index]
			ground_truth = labels[train_index]
			oracle = labels[test_index]

			# prediction = np.ndarray(shape=[len(test_index), self.first_level_classifiers_number],dtype=int)
			probability = np.ndarray(shape=[len(test_index), self.first_level_classifiers_number],dtype=object)

			for i, first_level_classifier in enumerate(self.first_level_classifiers):

				first_level_classifier.fit(training_set, ground_truth)

				first_level_classifier.set_oracle(oracle)
				# prediction[:, i] = first_level_classifier.predict(testing_set).tolist()
				probability[:, i] = first_level_classifier.predict_proba(testing_set).tolist()

			oracles_per_fold.append(oracle)
			# predictions_per_fold.append(prediction)
			probabilities_per_fold.append(probability)

		probabilities = np.concatenate(probabilities_per_fold)
		oracles = np.concatenate(oracles_per_fold)

		return probabilities, oracles

	def _meta_learner_fit(self, dataset, labels):

		# TODO: fit the aggregation algorithm

		# if len(dataset.shape) == 1: # Hard combiner

		# elif len(dataset.shape) == 2: # Soft combiner

		# We use Decision_Template as combiner
		decision_template = self._fit_Decision_Template(np.transpose(dataset), labels)

		return decision_template

		# else: # Error
		# 	raise Exception

		# pass

	def _fit_Decision_Template(self, decision_profile_list, validation_labels):

		vect_cont = [0] * self.num_classes
		decision_template = np.zeros((self.num_classes, len(decision_profile_list) ,self.num_classes)) #Decision Template [LxKxL]
		
		dp=[] #lista dei profili di decisione corrispondenti ad ogni biflusso
		dp_temp=[] #lista temporanea usata per costruire un decision profile a partire dai sotf output dei classificatori
		
		#costruzione lista decision profile
		for i in range(len(decision_profile_list[0])): #ciclo sulle predizioni dei classificatori
			for j in range(len(decision_profile_list)): #ciclo sui classificatori
				dp_temp.append(decision_profile_list[j][i])
			dp.append(dp_temp)
			dp_temp=[]
			
		#Calcolo Decision Template(Decision Profile medi)
		for i in range(len(validation_labels)): #ciclo sulla ground truth del validation set
			# try:

			decision_template[validation_labels[i]]= np.add(decision_template[validation_labels[i]], dp[i])
			vect_cont[validation_labels[i]] +=1
			# except:
			# 	print(set(validation_labels))
			# 	print(decision_template.shape)
			# 	exit(1)
		
		for i in range(decision_template.shape[0]): #Ciclo sul numero di classi
			decision_template[i]= decision_template[i]/vect_cont[i] 
			
		return decision_template

	def _first_level_final_fit(self, dataset, labels):

		for first_level_classifier in self.first_level_classifiers:
			first_level_classifier.fit(dataset, labels)

	def predict(self, testing_set):
		
		# First: we need prediction of first level models
		# first_proba = self._first_level_predict_proba(testing_set)
		first_pred = self._first_level_predict(testing_set)
		# Second: we pass first level prediction to combiner
		pred = self._meta_learner_predict(first_pred)
		# Third: for consistency, predict returns only predictions
		# NB: probabilities are not wasted, but saved in order to return
		# in case of calling predict_proba function
		# NB2: if you measure testing time, you have to comment next row
		# self.proba = proba

		return pred

	def _first_level_predict(self, testing_set):

		predictions = []

		for first_level_classifier in self.first_level_classifiers:

			first_level_classifier.set_oracle(self.oracle)
			pred = first_level_classifier.predict(testing_set)

			predictions.append(pred)
		
		return predictions

	def _first_level_predict_proba(self, testing_set):

		probabilities = []

		for first_level_classifier in self.first_level_classifiers:

			first_level_classifier.set_oracle(self.oracle)
			proba = first_level_classifier.predict_proba(testing_set)

			probabilities.append(proba)
		
		return probabilities

	def _meta_learner_predict(self, testing_set):

		# predictions = self._predict_Decision_Template(testing_set, self.weights)

		predictions = self._MV(testing_set)

		return predictions

	def _predict_Decision_Template (self, decision_profile_list, decision_template):
		
		mu_DT_SE=[0]* self.num_classes #vettore delle "distanze" tra  il Decision Profile e il Decision Template
		decision = []
		soft_comb_values=[]
		
		dp_predict=[]
		dp_temp_predict=[]
		
		for i in range(len(decision_profile_list[0])):#ciclo sulle predizioni dei classificatori
			for j in range(len(decision_profile_list)): #ciclo sui classificatori
				dp_temp_predict.append(decision_profile_list[j][i])
			dp_predict.append(dp_temp_predict)
			dp_temp_predict=[]
		
		dp_predict=np.asarray(dp_predict, dtype=int)
		
		#Distanza Euclidea quadratica
		for i in range(len(dp_predict)):
			for j in range(self.num_classes): #ciclo sulle classi
				mu_DT_SE[j]= 1 - (1/(float(self.num_classes)*(len(dp_predict[0])))) * (np.linalg.norm(decision_template[j] - dp_predict[i])**2)
			
			soft_comb_values.append(mu_DT_SE)
			
			#tie-Braking
			if mu_DT_SE.count(max(mu_DT_SE))<2:
				decision_class=mu_DT_SE.index(max(mu_DT_SE))
			else:
				decision_class=random.choice([i for i,x in enumerate(mu_DT_SE) if x==max(mu_DT_SE)])
			
			mu_DT_SE=[0]*self.num_classes
			decision.append(decision_class)
		
		
		#soft value per le filtered performance
		soft_comb_values=np.asarray(soft_comb_values)
		
		x_min=soft_comb_values.min()
		x_max=soft_comb_values.max()
					
		for i in range(soft_comb_values.shape[0]):
			for j in range(soft_comb_values.shape[1]):
				soft_comb_values[i][j]=(soft_comb_values[i][j] - x_min) /(x_max - x_min)
		
		return decision, soft_comb_values

	def _mode(self, array):
		'''
		Returns the (multi-)mode of an array
		'''
		
		most = max(list(map(array.count, array)))
		return list(set(filter(lambda x: array.count(x) == most, array)))		

	def _MV(self, predictions_list):
		'''
		Implements majority voting combination rule.
		:param predictions_list: i-th list contains the predictions of i-th classifier
		'''

		import random
		
		sample_predictions_list = list(zip(*predictions_list))
		
		MV_predictions = []
		for sample_predictions in sample_predictions_list:
			modes = self._mode(sample_predictions)
			MV_predictions.append(random.choice(modes))
	
		return np.asarray(MV_predictions, dtype=int)

	def predict_proba(self, testing_set):
		
		if self.proba is None:
			self.predict(testing_set)

		return self.proba

	def set_oracle(self, oracle):

		self.oracle = oracle

		print(np.unique(self.oracle, return_counts=True))

class AnomalyDetector(object):

	def __init__(self, detector_class, detector_opts, anomaly_class=None, features_number=0, epochs_number=0,
		level=0, fold=0, n_clusters=1, optimize=False, weight_features=False, workers_number=1, unsupervised=False,arbitrary_discr=''):
		
		self.anomaly_class = anomaly_class
		if self.anomaly_class == 1:
			self.normal_class = 0
		else:
			self.normal_class = 1
		if detector_class.startswith('k'):
			self.dl = True
		else:
			self.dl = False
		self.n_clusters = n_clusters
		self.optimize = optimize
		self.unsupervised = unsupervised
		self.anomaly_detectors = []
		for _ in range(self.n_clusters):
			if detector_class == 'ssv':
				if not self.optimize:
					self.anomaly_detector = OneClassSVMAnomalyDetector(kernel='rbf')
				else:
					# Parameters for Grid Search
					anomaly_detector = OneClassSVMAnomalyDetector()
					param_grid = {
							'kernel': [ 'rbf', 'sigmoid' ],
							'gamma': np.linspace(.1,1,5),
							'nu': np.linspace(.1,1,5)
						}
					# [
						# ,
						# {
						# 	'kernel': ['poly'],
						# 	'degree': [3,5,7,9],
						# 	'gamma': np.linspace(.1,1,10),
						# 	'nu': np.linspace(.1,1,10)
						# },
						# {
						# 	'kernel': ['linear'],
						# 	'nu': np.linspace(.1,1,10)
						# }
					# ]
			elif detector_class == 'sif':
				if not self.optimize:
					self.anomaly_detector = IsolationForestAnomalyDetector(n_estimators=100, contamination=0.)
				else:
					# Parameters for Grid Search
					anomaly_detector = IsolationForestAnomalyDetector()
					param_grid = {
						'n_estimators': [ int(i) for i in np.linspace(10,300,5) ],
						'contamination': [0.]
					}
					# self.grid_search_anomaly_detector = GridSearchCV(anomaly_detector, param_grid)
			elif detector_class == 'slo':
				if not self.optimize:
					self.anomaly_detector = LocalOutlierFactorAnomalyDetector(
						algorithm='ball_tree',
						novelty=True,
						n_neighbors=2,
						leaf_size=100,
						metric='canberra'
						)
				else:
					# Parameters for Grid Search
					anomaly_detector = LocalOutlierFactorAnomalyDetector()
					param_grid = [
						{
							'n_neighbors': [ int(i) for i in np.linspace(2,100,5) ],
							'algorithm': ['kd_tree'],
							'leaf_size': [ int(i) for i in np.linspace(3,150,5) ],
							# 'metric': ['chebyshev', 'cityblock', 'euclidean', 'infinity', 'l1', 'l2', 'manhattan', 'minkowski', 'p'],
							'contamination': [1e-4],
							'novelty': [True]
						},
						{
							'n_neighbors': [ int(i) for i in np.linspace(2,100,5) ],
							'algorithm': ['ball_tree'],
							'leaf_size': [ int(i) for i in np.linspace(3,150,5) ],
							# 'metric': ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'dice', 'euclidean', 'hamming',\
							# 'infinity', 'jaccard', 'kulsinski', 'l1', 'l2', 'manhattan', 'matching', 'minkowski', 'p',\
							# 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath'],
							'contamination': [1e-4],
							'novelty': [True]
						}
						# ,
						# {
						# 	'n_neighbors': [ int(i) for i in np.linspace(2,100,10) ],
						# 	'algorithm': ['brute'],
						# 	'metric': ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'cosine', 'dice',\
						# 	'euclidean', 'hamming', 'jaccard', 'kulsinski', 'l1', 'l2', 'manhattan', 'matching', 'minkowski',\
						# 	'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'],
						# 	'contamination': [1e-4],
						# 	'novelty': [True]
						# }
					]
					# self.grid_search_anomaly_detector = GridSearchCV(anomaly_detector, param_grid)
			elif self.dl:
				if not self.optimize:
					self.anomaly_detector = SklearnKerasWrapper(*detector_opts,model_class=detector_class,
					num_classes=2, epochs_number=epochs_number, level=level, fold=fold, weight_features=weight_features,
					arbitrary_discr=arbitrary_discr)
				else:
					# Parameters for Grid Search
					anomaly_detector = SklearnKerasWrapper()
					param_grid = [ {
						'sharing_ray': [ str(i) for i in range(-1,d+1) ],
						'grafting_depth': [ str(i) for i in range(0,d+2) ],
						'depth': [str(d)],
						# 'compression_ratio': ['.1','.15','.2','.25'],
						'compression_ratio': ['.1'],
						# 'mode': ['n','v','s','p','sv','sp','vp','svp'],
						'mode': ['n'],
						# 'hidden_activation_function': ['elu','tanh','sigmoid'],
						'hidden_activation_function_name': ['elu'],
						'model_class': [detector_class],
						'epochs_number': [epochs_number],
						'num_classes': [2],
						'fold': [fold], 
						'level': [level]
					# } for d in range(1,6) ]
					} for d in range(3,4) ]
					# self.grid_search_anomaly_detector = GridSearchCV(anomaly_detector, param_grid)
			if not self.optimize:
				self.anomaly_detectors.append(self.anomaly_detector)
		self.detector_class = detector_class

		if self.optimize:
			self.grid_search_anomaly_detector = GridSearchCV(anomaly_detector, param_grid, verbose=1, n_jobs=workers_number)
			# self.grid_search_anomaly_detector.evaluate_candidates(ParameterGrid(param_grid))
			# input()
			self.file_out = 'data_%s/material/optimization_results_fold_%s.csv' % (detector_class,fold)

			# print(self.grid_search_anomaly_detector.get_params())
			# input()

	def fit(self, training_set, ground_truth):

		if not self.unsupervised:
			normal_index = [ i for i,gt in enumerate(ground_truth) if gt == self.normal_class ]
		else:
			normal_index = [ i for i in range(len(ground_truth)) ]

		if self.dl:
			local_training_set = training_set[normal_index]
			clustering_training_set, self.train_scaler = self.scale_n_flatten(training_set[normal_index], return_scaler=True)
		else:
			local_training_set, self.train_scaler = self.scale_n_flatten(training_set[normal_index], return_scaler=True)
			clustering_training_set = local_training_set
		
		t = time.time()
		if self.n_clusters < 2:
			# if self.dl:
			if self.optimize:
				for row in local_training_set:
					for elem in row:
						try:
							float(elem)
						except:
							print(row)
							print(elem)
							input()
				self.grid_search_anomaly_detector.fit(local_training_set, ground_truth[normal_index])
				self.anomaly_detector = self.grid_search_anomaly_detector.best_estimator_
				import pandas as pd
				df = pd.DataFrame(list(self.grid_search_anomaly_detector.cv_results_['params']))
				ranking = self.grid_search_anomaly_detector.cv_results_['rank_test_score']
				# The sorting is done based on the test_score of the models.
				sorting = np.argsort(self.grid_search_anomaly_detector.cv_results_['rank_test_score'])
				# Sort the lines based on the ranking of the models
				df_final = df.iloc[sorting]
				# The first line contains the best model and its parameters
				df_final.to_csv(self.file_out)
			else:
				self.anomaly_detector.fit(local_training_set, ground_truth[normal_index])
			# else:
			# 	self.grid_search_anomaly_detector.fit(local_training_set, ground_truth[normal_index])
			# 	self.anomaly_detector = self.grid_search_anomaly_detector.best_estimator_
		else:
			from sklearn.cluster import KMeans
			self.clustering = KMeans(n_clusters=self.n_clusters)
			# We compose training set to feed clustering, in particular categorical (nominal) features are converted from
			# array of 0 and 1 to string of 0 and 1, then numerical are treated as float and then minmax scaled
			# clustering_training_set, self.clustering_scaler = self.ndarray2str(training_set[normal_index], return_scaler=True)
			print('\nStarting clustering on training set\n')
			cluster_index = self.clustering.fit_predict(clustering_training_set)
			for i in range(self.n_clusters):
				print('\nTraining Model of Cluster n.%s\n' % (i+1))
				red_normal_index = [ j for j,v in enumerate(cluster_index) if v == i ]
				self.anomaly_detectors[i].fit(local_training_set[red_normal_index], ground_truth[normal_index])

		t = time.time() - t

		self.t_ = t

		return self

	def predict(self, testing_set):

		# # Only for debugging
		# self.anomaly_detector.set_oracle(self.oracle)

		#self.clustering.transform(X) to get distances for each sample to each centers
		#self.clustering.cluster_centers_ to get centers

		if self.dl:
			local_testing_set = testing_set
			clustering_testing_set = self.scale_n_flatten(testing_set, self.train_scaler)
		else:
			local_testing_set = self.scale_n_flatten(testing_set, self.train_scaler)
			clustering_testing_set = local_testing_set

		if self.n_clusters < 2:
			pred = self.anomaly_detector.predict(local_testing_set)
			pred = np.asarray([ self.anomaly_class if p < 0 else self.normal_class for p in pred ])
		else:
			cluster_index = self.clustering.predict(clustering_testing_set)
			pred = np.ndarray(shape=[len(cluster_index),])
			for i,_anomaly_detector in enumerate(self.anomaly_detectors):
				red_index = [ j for j,v in enumerate(cluster_index) if v == i ]
				pred[red_index] = _anomaly_detector.predict(local_testing_set[red_index])
				pred[red_index] = np.asarray([ self.anomaly_class if p < 0 else self.normal_class for p in pred[red_index] ])

		return pred

	def predict_proba(self, testing_set):

		if self.dl:
			local_testing_set = testing_set
			clustering_testing_set = self.scale_n_flatten(testing_set, self.train_scaler)
			score = self.anomaly_detector.predict_proba
			scores = [ad.predict_proba for ad in self.anomaly_detectors]
		else:
			local_testing_set = self.scale_n_flatten(testing_set, self.train_scaler)
			clustering_testing_set = local_testing_set
			score = self.anomaly_detector.decision_function
			scores = [ad.decision_function for ad in self.anomaly_detectors]

		if self.n_clusters < 2:
			proba = score(local_testing_set)
		else:
			cluster_index = self.clustering.predict(clustering_testing_set)
			proba = np.ndarray(shape=[len(cluster_index),])
			for i,_anomaly_detector in enumerate(self.anomaly_detectors):
				red_index = [ j for j,v in enumerate(cluster_index) if v == i ]
				proba[red_index] = scores[i](local_testing_set[red_index])

		# We return the the opposite of decision_function because of the TPR and FPR we want to compute is related to outlier class
		# In this way, the higher is proba value, the more outlier is the sample.
		if not self.dl:
			proba = proba * -1

		return proba

	def set_oracle(self, oracle):

		self.oracle = oracle

		pass

	def get_distances(self, X, num_centers=None, nom_centers=None):
		'''
		This function take in input the matrix samples x features, where nominal are sting and numerical are float
		and centers vector, where 0-element refers numericals and 1 to nominals.
		Returns for each sample the distances from each centers, so a ndarray of shape samples x n_centroids
		'''
		num_index = [ j for j,v in enumerate(X[0,:]) if isinstance(v, float) ]
		nom_index = [ j for j,v in enumerate(X[0,:]) if isinstance(v, str) ]

		num_weight = len(num_index)
		nom_weight = len(nom_index)
		
		# In the centers list, first item refers to numerical centers, other to nominal
		num_centers = centers[0]
		nom_centers = centers[1]

		num_distances = []
		for num_centroid in num_centers:
			tmp_cen_dist = []
			for sample in X[:,num_index]:
				sample_dist = np.linalg.norm(num_centroid-sample)
				tmp_cen_dist.append(sample_dist)
			num_distances.append(tmp_cen_dist)
		num_distances = np.asarray(num_distances).T

		# np.asarray([ np.linalg.norm(X[:,num_index] - num_centroid, axis = 1) for num_centroid in num_centers ]).T

		nom_distances = []
		for nom_centroid in nom_centers:
			tmp_cen_dist = []
			for sample in X[:,nom_index]:
				sample_dist = np.sum([ self.one_hot_hamming(nc,sm) for nc,sm in zip(nom_centroid,sample) ]) / nom_weight
				tmp_cen_dist.append(sample_dist)
			nom_distances.append(tmp_cen_dist)
		nom_distances = np.asarray(nom_distances).T

		distances = np.average((num_distances,nom_distances), axis=0, weights=[num_weight, nom_weight])

		return distances

	def one_hot_hamming(self, u, v):
		'''
		Hamming distance of two string or vectors one hot encoded of equal length
		'''
		assert len(u) == len(v)
		if u.index('1') == v.index('1'):
			return 0.
		else:
			return 1.

	def ndarray2str(self, X, scaler=None, return_scaler=False):
		'''
		This function take in input a matrix samples x features, identifying nominal ones if they are ndarray, and transform them in string.
		Numerical features are minmaxscaled.
		Returns the entire X as a list of samples, where numerical features are float and nominal are str.
		'''
		if scaler is None:
			scaler = MinMaxScaler()
		samples_count = X.shape[0]
		trans_X = np.asarray([
				np.asarray([ ''.join([ str(int(e)) for e in v ]) for v in X[:,j] ], dtype=object)
				if isinstance(v, np.ndarray)
				else scaler.fit_transform(np.asarray(X[:,j], dtype=float).reshape(-1,1)).reshape((samples_count,))
				for j,v in enumerate(X[0,:])
			], dtype=object).T

		if return_scaler:
			return trans_X, scaler
		return trans_X

	def scale_n_flatten(self, X, scaler=None, return_scaler=False):
		'''
		This function take in input a matrix samples x features, identifying nominal ones if they are ndarray,
		and transform them in flatten representation.
		Numerical features are minmaxscaled.
		Returns the entire X as a list of samples, where numerical features are float and nominal are str.
		'''
		if scaler is None:
			scaler = MinMaxScaler()
		samples_count = X.shape[0]

		trans_X = np.asarray([
				X[:,j]
				if issparse(v)
				else scaler.fit_transform(np.asarray(X[:,j], dtype=float).reshape(-1,1)).reshape((samples_count,))
				for j,v in enumerate(X[0,:])
			], dtype=object).T

		new_trans_X = []
		for i in range(trans_X.shape[0]):
			new_trans_X.append(np.hstack(trans_X[i]))
		trans_X = np.asarray(new_trans_X)

		if return_scaler:
			return trans_X, scaler
		return trans_X

	def fit_NeuralWeightsDecisor(self, training_set, ground_truth):

		decisor = AnomalyDetector(self.detector_class, self.detector_opts, self.anomaly_class, self.features_number, self.epochs_number,
		self.level, self.fold, self.n_clusters, self.optimize, self.weight_features, self.workers_number)

		self.fit(training_set, ground_truth)

		weights = self.decisor.get_weights()

class HierarchicalClassifier(object):

	def __init__(self,input_file,levels_number,max_level_target,level_target,features_number,packets_number,
		classifier_class,classifier_opts,detector_class,detector_opts,workers_number,anomaly_classes,
		epochs_number,arbitrary_discr,n_clusters,anomaly,deep,hidden_classes,benign_class,nominal_features_index,
		optimize,weight_features,parallelize,buckets_number,unsupervised):

		self.input_file = input_file
		self.levels_number = levels_number
		self.max_level_target = max_level_target
		self.level_target = level_target
		self.features_number = features_number
		self.packets_number = packets_number
		self.classifier_class = classifier_class
		self.classifier_opts = classifier_opts
		self.detector_class = detector_class
		self.detector_opts = detector_opts
		self.workers_number = workers_number
		self.has_config = False
		self.anomaly_classes = anomaly_classes
		self.epochs_number = epochs_number
		self.arbitrary_discr = arbitrary_discr
		self.n_clusters = n_clusters
		self.buckets_number = buckets_number

		self.anomaly = anomaly
		self.deep = deep
		self.hidden_classes = hidden_classes
		self.benign_class = benign_class

		self.nominal_features_index = nominal_features_index
		self.optimize = optimize
		self.weight_features = weight_features
		self.parallelize = parallelize
		self.unsupervised = unsupervised

		self.super_learner_default=['ssv','sif']

	def set_config(self, config_name, config):
		self.has_config = True
		self.config_name = config_name
		self.config = config

	def init_output_files(self):

		if self.anomaly:
			folder_discr = detector_name
		else:
			folder_discr = classifier_name

		if self.has_config:
			folder_discr = self.config_name
		if len(self.anomaly_classes) > 0:
			folder_discr = self.detector_class + ''.join([ '_'+str(opt) for opt in self.detector_opts ])

		# File discriminator, modify params_discr to change dipendence variable
		self.type_discr = 'flow'
		feat_discr = '_f_' + str(self.features_number)
		if not self.has_config and self.packets_number != 0:
			self.type_discr = 'early'
			feat_discr = '_p_' + str(self.packets_number)
		elif self.has_config:
			if 'p' in self.config:
				self.type_discr = 'early'
			feat_discr = '_c_' + self.config_name
		if self.has_config and self.classifier_class:
			if self.features_number != 0:
				feat_discr = '_f_' + str(self.features_number) + feat_discr + '_' + self.classifier_class
			if self.packets_number != 0:
				feat_discr = '_p_' + str(self.packets_number) + feat_discr + '_' + self.classifier_class
		work_discr = '_w_' + str(self.workers_number)
		model_discr = '_e_' + str(self.epochs_number)
		buck_discr = '_b_' + str(self.buckets_number)

		self.params_discr = feat_discr
		if self.workers_number > 0:
			self.params_discr = work_discr + self.params_discr
			if self.buckets_number > 0:
				self.params_discr = buck_discr + self.params_discr
		elif self.epochs_number > 0:
			self.params_discr = model_discr + self.params_discr

		self.material_folder = './data_'+folder_discr+'/material/'
		self.material_proba_folder = './data_'+folder_discr+'/material_proba/'

		if not os.path.exists('./data_'+folder_discr):
			os.makedirs('./data_'+folder_discr)
			os.makedirs(self.material_folder)
			os.makedirs(self.material_proba_folder)
		else:
			if not os.path.exists(self.material_folder):
				os.makedirs(self.material_folder)
			if not os.path.exists(self.material_proba_folder):
				os.makedirs(self.material_proba_folder)

		self.material_features_folder = './data_'+folder_discr+'/material/features/'
		self.material_train_durations_folder = './data_'+folder_discr+'/material/train_durations/'

		if not os.path.exists(self.material_folder):
			os.makedirs(self.material_folder)
			os.makedirs(self.material_features_folder)
			os.makedirs(self.material_train_durations_folder)
		if not os.path.exists(self.material_features_folder):
			os.makedirs(self.material_features_folder)
		if not os.path.exists(self.material_train_durations_folder):
			os.makedirs(self.material_train_durations_folder)

		self.graph_folder = './data_'+folder_discr+'/graph/'

		if not os.path.exists('./data_'+folder_discr):
			os.makedirs('./data_'+folder_discr)
			os.makedirs(self.graph_folder)
		elif not os.path.exists(self.graph_folder):
			os.makedirs(self.graph_folder)

	def write_fold(self, level, tag, labels, _all=False, arbitrary_discr=None):

		if arbitrary_discr is None:
			arbitrary_discr = self.arbitrary_discr

		if _all:
			global_fouts = [
				'%s%s_multi_%s_level_%s%s_all.dat' % (self.material_folder,arbitrary_discr,self.type_discr,level,self.params_discr),
				'%s%s_multi_%s_level_%s%s_all.dat' % (self.material_proba_folder,arbitrary_discr,self.type_discr,level,self.params_discr)
				]
			local_fouts = [
				'%s%s_multi_%s_level_%s%s_tag_%s_all.dat' % (self.material_folder,arbitrary_discr,self.type_discr,level,self.params_discr,tag),
				'%s%s_multi_%s_level_%s%s_tag_%s_all.dat' % (self.material_proba_folder,arbitrary_discr,self.type_discr,level,self.params_discr,tag)
				]
			fouts = local_fouts
			if not self.global_folds_writed_all[level]:
				self.global_folds_writed_all[level] = True
				fouts += global_fouts
		else:
			global_fouts = [
				'%s%s_multi_%s_level_%s%s.dat' % (self.material_folder,arbitrary_discr,self.type_discr,level,self.params_discr),
				'%s%s_multi_%s_level_%s%s.dat' % (self.material_proba_folder,arbitrary_discr,self.type_discr,level,self.params_discr)
				]
			local_fouts = [
				'%s%s_multi_%s_level_%s%s_tag_%s.dat' % (self.material_folder,arbitrary_discr,self.type_discr,level,self.params_discr,tag),
				'%s%s_multi_%s_level_%s%s_tag_%s.dat' % (self.material_proba_folder,arbitrary_discr,self.type_discr,level,self.params_discr,tag),
				]
			fouts = local_fouts
			if not self.global_folds_writed[level]:
				self.global_folds_writed[level] = True
				fouts += global_fouts
		
		for fout in fouts:
			with open(fout, 'a') as file:
				if 'proba' in fout:
					file.write('@fold %s\n' % ' '.join(['Actual'] + labels))
				else:
					file.write('@fold Actual Pred\n')

	def write_pred(self, oracle, pred, level, tag, _all=False, arbitrary_discr=None):

		if arbitrary_discr is None:
			arbitrary_discr = self.arbitrary_discr

		if not _all:
			fouts = ['%s%s_multi_%s_level_%s%s.dat' % (self.material_folder,arbitrary_discr,self.type_discr,level,self.params_discr),
				'%s%s_multi_%s_level_%s%s_tag_%s.dat' % (self.material_folder,arbitrary_discr,self.type_discr,level,self.params_discr,tag)]
		else:
			fouts = ['%s%s_multi_%s_level_%s%s_all.dat' % (self.material_folder,arbitrary_discr,self.type_discr,level,self.params_discr),
				'%s%s_multi_%s_level_%s%s_tag_%s_all.dat' % (self.material_folder,arbitrary_discr,self.type_discr,level,self.params_discr,tag)]
		
		for fout in fouts:
			with open(fout, 'a') as file:
				for o, p in zip(oracle, pred):
					file.write('%s %s\n' % (o,p))

	def write_bucket_times(self, bucket_times, arbitrary_discr=None):
		if arbitrary_discr is None:
			arbitrary_discr = self.arbitrary_discr

		with open(self.material_train_durations_folder + arbitrary_discr + 'multi_' + self.type_discr + self.params_discr + '_buckets_train_durations.dat', 'a') as file:
			file.write('%.6f\n' % (np.max(bucket_times)))
			for i,bucket_time in enumerate(bucket_times):
				with open(self.material_train_durations_folder + arbitrary_discr + 'multi_' + self.type_discr + self.params_discr + '_bucket_' + str(i) + '_train_durations.dat', 'a') as file:
					file.write('%.6f\n' % (bucket_time))

	def write_time(self, time, level, tag, arbitrary_discr=None):
		if arbitrary_discr is None:
			arbitrary_discr = self.arbitrary_discr

		with open('%s%s_multi_%s_level_%s%s_tag_%s_train_durations.dat'%(self.material_train_durations_folder,arbitrary_discr,self.type_discr,level,self.params_discr,tag), 'a') as file:
			file.write('%.6f\n' % (np.max(time)))

	def write_proba(self, oracle, proba, level, tag, _all=False, arbitrary_discr=None):

		if arbitrary_discr is None:
			arbitrary_discr = self.arbitrary_discr

		if not _all:
			fouts = ['%s%s_multi_%s_level_%s%s.dat' % (self.material_proba_folder,arbitrary_discr,self.type_discr,level,self.params_discr),
				'%s%s_multi_%s_level_%s%s_tag_%s.dat' % (self.material_proba_folder,arbitrary_discr,self.type_discr,level,self.params_discr,tag)]
		else:
			fouts = ['%s%s_multi_%s_level_%s%s_all.dat' % (self.material_proba_folder,arbitrary_discr,self.type_discr,level,self.params_discr),
				'%s%s_multi_%s_level_%s%s_tag_%s_all.dat' % (self.material_proba_folder,arbitrary_discr,self.type_discr,level,self.params_discr,tag)]
		
		for fout in fouts:
			with open(fout, 'a') as file:
				for o, p in zip(oracle, proba):
					try:
						file.write('%s %s\n' % (o,' '.join([str(v) for v in p])))
					except:
						file.write('%s %s\n' % (o,p))

	def load_image(self):
		
		print('\nCaricando '+self.input_file+' con opts -f'+str(self.features_number)+' -c'+self.classifier_class+'\n')
		
		with open(self.input_file,'rb') as dataset_pk:
			dataset = pk.load(dataset_pk)

		data = np.array(dataset['data'], dtype=object)

		# After dataset preprocessing, we could save informations about dimensions
		self.attributes_number = data.shape[1]
		self.dataset_features_number = self.attributes_number - self.levels_number
		self.features = data[:, :self.dataset_features_number]
		self.labels = data[:, self.dataset_features_number:]

	def load_dataset(self, memoryless):

		print('\nCaricando '+self.input_file+' con opts -f'+str(self.features_number)+' -c'+self.classifier_class+'\n')

		if not memoryless and os.path.exists('%s.yetpreprocessed.pickle'%self.input_file):
			self.features_names, self.nominal_features_index, self.numerical_features_index, self.fine_nominal_features_index,\
			self.attributes_number, self.dataset_features_number, self.numerical_features_length, self.nominal_features_lengths,\
			self.features_number, self.anomaly_class, self.features, self.labels = pk.load(open('%s.yetpreprocessed.pickle'%self.input_file,'rb'))
		else:
			# load .arff file
			with open(self.input_file, 'r') as fi:

				if self.input_file.lower().endswith('.arff'):
					# load .arff file
					dataset = arff.load(fi)
				elif self.input_file.lower().endswith('.csv'):
					# TODO: to correct bugs
					dataset = {}
					delimiter = ','
					attributes = fi.readline()
					samples = fi.readlines()
					first_row = samples[0].split(delimiter)
					attrs_type = []
					# Inferring attribute type by first row, this could led to error due to non analysis of all the column
					for value in first_row:
						try:
							float(value)
							attrs_type.append(u'NUMERIC')
						except:
							attrs_type.append(u'STRING')
					##__##
					dataset['attributes'] = [ (attr, attr_type) for attr,attr_type in zip(attributes.strip().split(delimiter),attrs_type) ]
					dataset['data'] = [ sample.strip().split(delimiter) for sample in samples ] # TODO gestire i tipi degli attributi
				elif self.input_file.lower().endswith('.pickle'):
					with open(self.input_file,'rb') as dataset_pk:
						dataset = pk.load(dataset_pk)

			data = np.array(dataset['data'], dtype=object)

			# Next portion will merge all categorical features into one
				###
				# nominal_features_index = [i for i in range(len(dataset['attributes'][:-self.levels_number])) if dataset['attributes'][i][1] != u'NUMERIC']

				# data_nominals = np.ndarray(shape=(self.data.shape[0],len(nominal_features_index)),dtype=object)
				# for i,nfi in enumerate(nominal_features_index):
				# 	data_nominals[:,i] = self.data[:,nfi]
				# data_nominal = np.ndarray(shape=(self.data.shape[0]),dtype=object)
				# for i,datas in enumerate(data_nominals):
				# 	data_nominal[i] = ''.join([ d for d in datas ])

				# self.data[:,nominal_features_index[0]] = data_nominal

				# self.data = np.delete(self.data, nominal_features_index[1:], axis=1).reshape((self.data.shape[0],self.data.shape[1]-len(nominal_features_index[1:])))
				###

			self.features_names = [x[0] for x in dataset['attributes']]

			# If is passed a detector that works woth keras, perform a OneHotEncoding, else proceed with OrdinalEncoder
			if self.nominal_features_index is None:
				self.nominal_features_index = [ i for i in range(len(dataset['attributes'][:-self.levels_number])) if dataset['attributes'][i][1] not in [ u'NUMERIC', u'REAL', u'SPARRAY' ] ]
			self.numerical_features_index = [ i for i in range(len(dataset['attributes'][:-self.levels_number])) if dataset['attributes'][i][1] in [ u'NUMERIC', u'REAL' ] ]
			self.fine_nominal_features_index = [ i for i in range(len(dataset['attributes'][:-self.levels_number])) if dataset['attributes'][i][1] in [ u'SPARRAY' ] ]

			# if self.deep:
			# if self.detector_class.startswith('k'):

			# Nominal features index should contains only string features that need to be processed with a one hot encoding.
			# These kind of features will be treated as sparse array, if deep learning models are used.
			features_encoder = OneHotEncoder()
			for i in self.nominal_features_index:
				# data[:, i] = [ sp.toarray()[0] for sp in features_encoder.fit_transform(data[:, i].reshape(-1,1)) ]
				data[:, i] = [ sp for sp in features_encoder.fit_transform(data[:, i].reshape(-1,1)) ]

			# If model is not deep, we need to flatten sparsed features, to permit an unbiased treatment of categorical featurees w.r.t numerical one.
			if not self.deep:
				new_data = []
				for col in data.T:
					if issparse(col[0]):
						for col0 in np.array([ c.todense() for c in col ]).T:
							new_data.append(col0[0])
					else:
						new_data.append(col)
				data = np.array(new_data).T

			# After dataset preprocessing, we could save informations about dimensions
			self.attributes_number = data.shape[1]
			self.dataset_features_number = self.attributes_number - self.levels_number
			# Moreover, we provide a count of nominal and numerical features, to correctly initialize models.
			self.numerical_features_length = 0
			self.nominal_features_lengths = []
			for obj in data[0,:self.dataset_features_number]:
				if issparse(obj):
					self.nominal_features_lengths.append(obj.shape[1])
				else:
					self.numerical_features_length += 1
			# Bypassing the features selection
			self.features_number = self.numerical_features_length+len(self.nominal_features_lengths)
				# else:
					# If classifier/detector is not deep, the framework categorize nominal features.
					# if len(self.nominal_features_index) > 0:
					# 	features_encoder = OrdinalEncoder()
					# 	data[:, self.nominal_features_index] = features_encoder.fit_transform(data[:, self.nominal_features_index])

				# self.distribution_features_index = [ i for i in range(len(dataset['attributes'][:-self.levels_number])) if dataset['attributes'][i][1] in [ u'SPARRAY' ] ]
				# for i in self.distribution_features_index:
				# 	data[:, i] = [ sp.toarray()[0] for sp in data[:, i] ]

			# Multiclass AD - OneVSAll
			# TODO: now it works only on one level dataset for AD, where when multiclass and anomaly is specified, anomaly goes to '1' and others to '0'
			# if self.anomaly and np.unique(data[:,self.attributes_number-1]).shape[0] > 2: <---- BUG??
			print(self.anomaly,self.anomaly_classes)
			if self.anomaly and len(self.anomaly_classes) > 0:
					for anomaly_class in self.anomaly_classes:
						data[np.where(data[:,self.dataset_features_number+self.level_target] == anomaly_class),self.dataset_features_number+self.level_target] = '1'
					data[np.where(data[:,self.dataset_features_number+self.level_target] != '1'),self.dataset_features_number+self.level_target] = '0'
					self.anomaly_class = '1'
			#TODO: manage hidden classes in presence of benign declared and array of hidden classes
			if self.benign_class != '':
				if len(self.hidden_classes) > 0:
					data[np.where((data[:,self.dataset_features_number+self.level_target] != self.benign_class) & (data[:,self.dataset_features_number+self.level_target] != self.hidden_classes)),self.attributes_number-1] = '1'
				else:
					data[np.where(data[:,self.dataset_features_number+self.level_target] != self.benign_class),self.dataset_features_number+self.level_target] = '1'
				data[np.where(data[:,self.dataset_features_number+self.level_target] == self.benign_class),self.dataset_features_number+self.level_target] = '0'
				self.anomaly_class = '1'

			self.features = data[:, :self.dataset_features_number]
			self.labels = data[:, self.dataset_features_number:]

			if not memoryless:
				pk.dump(
					[
						self.features_names, self.nominal_features_index, self.numerical_features_index, self.fine_nominal_features_index,
						self.attributes_number, self.dataset_features_number, self.numerical_features_length, self.nominal_features_lengths,
						self.features_number, self.anomaly_class, self.features, self.labels
					]
					,open('%s.yetpreprocessed.pickle'%self.input_file,'wb'),pk.HIGHEST_PROTOCOL)

	def load_early_dataset(self):

		print('\nCaricando '+self.input_file+' con opts -f'+str(self.features_number)+' -c'+self.classifier_class+'\n')
		# load .arff file
		if self.input_file.lower().endswith('.pickle'):
			with open(self.input_file,'rb') as dataset_pk:
				dataset = pk.load(dataset_pk)
		else:
			raise(Exception('Early classification supports only pickle input.'))

		data = np.array(dataset['data'], dtype=object)

		# Next portion will merge all categorical features into one
			###
			# nominal_features_index = [i for i in range(len(dataset['attributes'][:-self.levels_number])) if dataset['attributes'][i][1] != u'NUMERIC']

			# data_nominals = np.ndarray(shape=(self.data.shape[0],len(nominal_features_index)),dtype=object)
			# for i,nfi in enumerate(nominal_features_index):
			# 	data_nominals[:,i] = self.data[:,nfi]
			# data_nominal = np.ndarray(shape=(self.data.shape[0]),dtype=object)
			# for i,datas in enumerate(data_nominals):
			# 	data_nominal[i] = ''.join([ d for d in datas ])

			# self.data[:,nominal_features_index[0]] = data_nominal

			# self.data = np.delete(self.data, nominal_features_index[1:], axis=1).reshape((self.data.shape[0],self.data.shape[1]-len(nominal_features_index[1:])))
			###

		self.features_names = [x[0] for x in dataset['attributes']]

		# If is passed a detector that works woth keras, perform a OneHotEncoding, else proceed with OrdinalEncoder
		if self.nominal_features_index is None:
			self.nominal_features_index = [ i for i in range(len(dataset['attributes'][:-self.levels_number])) if dataset['attributes'][i][1] not in [ u'NUMERIC', u'REAL', u'SPARRAY' ] ]
		self.numerical_features_index = [ i for i in range(len(dataset['attributes'][:-self.levels_number])) if dataset['attributes'][i][1] in [ u'NUMERIC', u'REAL' ] ]
		self.fine_nominal_features_index = [ i for i in range(len(dataset['attributes'][:-self.levels_number])) if dataset['attributes'][i][1] in [ u'SPARRAY' ] ]

		# if self.deep:
		# if self.detector_class.startswith('k'):

		# Nominal features index should contains only string features that need to be processed with a one hot encoding.
		# These kind of features will be treated as sparse array, if deep learning models are used.
		features_encoder = OneHotEncoder()
		for i in self.nominal_features_index:
			# data[:, i] = [ sp.toarray()[0] for sp in features_encoder.fit_transform(data[:, i].reshape(-1,1)) ]
			features_encoder.fit(np.array([ v for f in data[:, i] for v in f ]).reshape(-1,1))
			for j,f in enumerate(data[:, i]):
				data[:, i][j] = [ sp for sp in features_encoder.transform(np.array(data[:, i][j]).reshape(-1,1)) ]

		# After dataset preprocessing, we could save informations about dimensions
		self.attributes_number = data.shape[1]
		self.dataset_features_number = self.attributes_number - self.levels_number
		# Moreover, we provide a count of nominal and numerical features, to correctly initialize models.
		self.numerical_features_length = 0
		self.nominal_features_lengths = []
		for obj in data[0,:self.dataset_features_number]:
			if issparse(obj[0]):
				self.nominal_features_lengths.append(obj[0].shape[1])
			else:
				self.numerical_features_length += 1
		# Bypassing the features selection
		self.features_number = self.numerical_features_length+len(self.nominal_features_lengths)
			# else:
				# If classifier/detector is not deep, the framework categorize nominal features.
				# if len(self.nominal_features_index) > 0:
				# 	features_encoder = OrdinalEncoder()
				# 	data[:, self.nominal_features_index] = features_encoder.fit_transform(data[:, self.nominal_features_index])

			# self.distribution_features_index = [ i for i in range(len(dataset['attributes'][:-self.levels_number])) if dataset['attributes'][i][1] in [ u'SPARRAY' ] ]
			# for i in self.distribution_features_index:
			# 	data[:, i] = [ sp.toarray()[0] for sp in data[:, i] ]

		# Multiclass AD - OneVSAll
		# TODO: now it works only on one level dataset for AD, where when multiclass and anomaly is specified, anomaly goes to '1' and others to '0'
		if self.anomaly and np.unique(data[:,self.attributes_number-1]).shape[0] > 2:
			if len(self.anomaly_classes) > 0:
				for anomaly_class in self.anomaly_classes:
					data[np.where(data[:,self.dataset_features_number+self.level_target] == anomaly_class),self.dataset_features_number+self.level_target] = '1'
				data[np.where(data[:,self.dataset_features_number+self.level_target] != '1'),self.dataset_features_number+self.level_target] = '0'
				self.anomaly_class = '1'
		#TODO: manage hidden classes in presence of benign declared
		if self.benign_class != '':
			if len(self.hidden_classes) > 0:
				data[np.where((data[:,self.attributes_number-1] != self.benign_class) & (data[:,self.attributes_number-1] != self.hidden_classes)),self.attributes_number-1] = '1'
			else:
				data[np.where(data[:,self.attributes_number-1] != self.benign_class),self.attributes_number-1] = '1'
			data[np.where(data[:,self.attributes_number-1] == self.benign_class),self.attributes_number-1] = '0'
			self.anomaly_class = '1'

		self.features = data[:, :self.dataset_features_number]
		self.labels = data[:, self.dataset_features_number:]

	def initialize_nodes(self, fold_cnt, train_index, test_index):

		print('\nInitializing nodes of fold n.%s\n' % fold_cnt)
		nodes = []

		root = TreeNode()
		nodes.append(root)

		root.fold = fold_cnt
		root.train_index = train_index
		root.test_index = test_index
		root.test_index_all = test_index
		root.level = self.level_target
		root.children_tags = [ tag for tag in np.unique((self.labels[:,root.level])) if tag not in self.hidden_classes ]
		root.children_number = len(root.children_tags)
		root.label_encoder = MyLabelEncoder()
		root.label_encoder.fit(root.children_tags)

		# Apply config to set number of features
		if self.has_config and root.tag + '_' + str(root.level + 1) in self.config:
			if 'f' in self.config[root.tag + '_' + str(root.level + 1)]:
				root.features_number = self.config[root.tag + '_' + str(root.level + 1)]['f']
			elif 'p' in self.config[root.tag + '_' + str(root.level + 1)]:
				root.packets_number = self.config[root.tag + '_' + str(root.level + 1)]['p']
			root.classifier_class = self.config[root.tag + '_' + str(root.level + 1)]['c']

			print('\nconfig','tag',root.tag,'level',root.level,'f',root.features_number,\
				'c',root.classifier_class,'train_test_len',len(root.train_index),len(root.test_index))
		else:
			# Assign encoded features number to work with onehotencoder
			root.features_number = self.features_number
			# root.encoded_features_number = self.encoded_dataset_features_number
			root.packets_number = self.packets_number

		# If self.anomaly_class is set and it has two children (i.e. is a binary classification problem)
		# ROOT had to be an Anomaly Detector
		# In this experimentation we force the setting of AD to be a SklearnAnomalyDetector
		if self.anomaly and root.children_number == 2:
			root.detector_class = self.detector_class
			root.detector_opts = self.detector_opts
			print('\nconfig','tag',root.tag,'level',root.level,'f',root.features_number,\
			'd',root.detector_class,'train_test_len',len(root.train_index),len(root.test_index))
			classifier_to_call = self._AnomalyDetector
		else:
			root.classifier_class = self.classifier_class
			root.classifier_opts = self.classifier_opts
			print('\nconfig','tag',root.tag,'level',root.level,'f',root.features_number,\
			'c',root.classifier_class,'train_test_len',len(root.train_index),len(root.test_index))
			classifier_to_call = getattr(self, supported_classifiers[root.classifier_class])
		
		root.classifier = classifier_to_call(node=root)

		# Creating hierarchy recursively, if max level target is set, classification continue while reaches it
		# If it's not set, it is equal to levels number. If level target is set, we have only the analysis at selected level.
		if root.level < self.max_level_target-1 and root.children_number > 0 and self.level_target < 0:
			self.initialize_nodes_recursive(root, nodes)

		return nodes

	def initialize_nodes_recursive(self,parent,nodes):

		for i in range(parent.children_number):
			child = TreeNode()
			nodes.append(child)

			child.level = parent.level+1
			child.fold = parent.fold
			child.tag = parent.children_tags[i]
			child.parent = parent

			child.train_index = [index for index in parent.train_index if self.labels[index, parent.level] == child.tag]
			child.test_index_all = [index for index in parent.test_index_all if self.labels[index, parent.level] == child.tag]
			child.children_tags = [ tag for tag in np.unique((self.labels[child.train_index,child.level])) if tag not in self.hidden_classes ]
			child.children_number = len(child.children_tags)
			# other_childs_children_tags correspond to the possible labels that in testing phase could arrive to the node
			other_childs_children_tags = [ tag for tag in np.unique(self.labels[:, child.level]) if tag not in self.hidden_classes ]
			child.label_encoder = MyLabelEncoder()
			child.label_encoder.fit(np.concatenate((child.children_tags, other_childs_children_tags)))

			if self.has_config and child.tag + '_' + str(child.level + 1) in self.config:
				if 'f' in self.config[child.tag + '_' + str(child.level + 1)]:
					child.features_number = self.config[child.tag + '_' + str(child.level + 1)]['f']
				elif 'p' in self.config[child.tag + '_' + str(child.level + 1)]:
					child.packets_number = self.config[child.tag + '_' + str(child.level + 1)]['p']
				child.classifier_class = self.config[child.tag + '_' + str(child.level + 1)]['c']
				print('config','tag',child.tag,'level',child.level,'f',child.features_number,\
					'c',child.classifier_class,'d',child.detector_class,'o',child.detector_opts,'train_test_len',len(child.train_index),len(child.test_index))
			else:
				child.features_number = self.features_number
				# child.encoded_features_number = self.encoded_dataset_features_number
				child.packets_number = self.packets_number

			if self.anomaly and child.children_number == 2:
				child.detector_class = self.detector_class
				child.detector_opts = self.detector_opts
			else:
				child.classifier_class = self.classifier_class
				child.classifier_opts = self.classifier_opts
			print('config','tag',child.tag,'level',child.level,'f',child.features_number,\
				'c',child.classifier_class,'train_test_len',len(child.train_index),len(child.test_index))

			if self.anomaly and child.children_number == 2:
				classifier_to_call = self._AnomalyDetector
			else:
				classifier_to_call = getattr(self, supported_classifiers[child.classifier_class])
			child.classifier = classifier_to_call(node=child)

			nodes[nodes.index(parent)].children[child.tag]=child

			if child.level < self.max_level_target-1 and child.children_number > 0:
				self.initialize_nodes_recursive(child, nodes)

	def kfold_validation(self, k=10, starting_fold=0, ending_fold=10):

		self.k = k
		classifiers_per_fold = []
		# bucket_times_per_fold = []

		print('\n***\nStart testing with '+str(self.k)+'Fold cross-validation -f'+str(self.features_number)+' -c'+self.classifier_class+'\n***\n')

		skf = StratifiedKFold(n_splits=self.k, shuffle=True)

		# Testing set of each fold are not overlapping, we use only one array to maintain all the predictions over the folds
		self.predictions = np.ndarray(shape=self.labels.shape, dtype=object)

		# for fold_cnt, (train_index, test_index) in enumerate(skf.split(self.features, self.labels[:,-1])):
		# 	with open('labels.dat','a') as fout:
		# 		fout.write('@fold\n')
		# 		for l in self.labels[test_index,1]:
		# 			fout.write('%s\n'%l)

		for fold_cnt, (train_index, test_index) in enumerate(skf.split(self.features, self.labels[:,-1])):

			if fold_cnt < starting_fold-1 or fold_cnt > ending_fold-1:
				continue

			print('\nStarting fold n.%s\n' % fold_cnt)

			self.global_folds_writed = [ False ] * self.levels_number
			self.global_folds_writed_all = [ False ] * self.levels_number

			nodes = self.initialize_nodes(fold_cnt, train_index, test_index)

			# Defining complexity for each node
			for node in [ n for n in nodes if n.children_number > 1 ]:
				node.complexity = (len(node.train_index)*node.children_number)/np.sum([len(n.train_index)*n.children_number for n in nodes if n.children_number > 1])

			# Index of sorted (per complexity) nodes
			sorting_index = list(reversed(np.argsort([ n.complexity for n in nodes if n.children_number > 1 ])))
			# Indexes of sorted nodes per bucket
			bucket_indexes = equal_partitioner(self.buckets_number,[[n for n in nodes if n.children_number > 1][i].complexity for i in sorting_index])

			if parallelize:
				# Iterative training, it is parallelizable leveraging bucketization:
				# buckets are launched in parallel, nodes in a bucket are trained sequentially
				def parallel_train(bucket_nodes):
					for node in bucket_nodes:
						print('Train node %s %s_%s'%(node,node.tag,node.level))
						self.train(node)
				threads = []
				for bucket_index in bucket_indexes:
					bucket_nodes = [ [ n for n in nodes if n.children_number > 1 ][i] for i in [ sorting_index[j] for j in bucket_index ] ]
					threads.append(Thread(target=parallel_train,args=[ bucket_nodes ]))
				for thread in threads:
					thread.start()
				for thread in threads:
					thread.join()
			else:
				for bucket_index in bucket_indexes:
					bucket_nodes = [ [ n for n in nodes if n.children_number > 1 ][i] for i in [ sorting_index[j] for j in bucket_index ] ]
					for node in bucket_nodes:
						if node.children_number > 1:
							print('Train node %s %s_%s'%(node,node.tag,node.level))
							self.train(node)

			bucket_times = []
			for bucket_index in bucket_indexes:
				bucket_times.append(np.sum([ nodes[i].test_duration for i in [ sorting_index[j] for j in bucket_index ] ]))
			self.write_bucket_times(bucket_times)
			# bucket_times_per_fold.append(bucket_times)

			# Iterative testing_all, it is parallelizable in various ways
			for node in nodes:
				if node.children_number > 1:
					self.write_fold(node.level, node.tag, node.children_tags, _all=True)
					self.test_all(node)

			# Recursive testing, each level is parallelizable and depends on previous level predictions
			root = nodes[0]
			self.write_fold(root.level, root.tag, root.children_tags)
			self.test(root)
			if root.level < self.max_level_target-1 and root.children_number > 1 and self.level_target < 0:
				self.test_recursive(root)

			# Writing unary class predictions
			for node in nodes:
				if node.children_number == 1:
					self.unary_class_results_inferring(node)
			classifiers_per_fold.append(nodes)

		if self.anomaly and self.deep:

			weights_per_fold = []
			# Weights information
			for classifiers in classifiers_per_fold:
				for node in classifiers:
					weights_per_fold.append(node.classifier.anomaly_detector.get_weights('r'))

			weights = np.mean(weights_per_fold, axis=0)

			for classifiers in classifiers_per_fold:
				for node in classifiers:
					node.classifier.anomaly_detector.set_weights(weights, 'r')
					# To distinguish among 10-fold realted and weighted ones
					self.params_discr = self.params_discr + '_MEAN'

			for classifiers in classifiers_per_fold:
				for node in classifiers:
					self.train(node)
					self.test(node)

		# for classifier in classifiers_per_fold[0]:

		# 	with open(self.material_features_folder + self.arbitrary_discr + 'multi_' + self.type_discr + '_level_' + str(classifier.level+1) + self.params_discr + '_tag_' + str(classifier.tag) + '_features.dat', 'w+'):
		# 		pass
		# 	with open(self.material_train_durations_folder + self.arbitrary_discr + 'multi_' + self.type_discr + '_level_' + str(classifier.level+1) + self.params_discr + '_tag_' + str(classifier.tag) + '_train_durations.dat', 'w+'):
		# 		pass

		# # for bucket_times in bucket_times_per_fold:
		# # 	with open(self.material_train_durations_folder + self.arbitrary_discr + 'multi_' + self.type_discr + '_level_' + str(classifier.level+1) + self.params_discr + '_buckets_train_durations.dat', 'w+'):
		# # 		pass
		# # 	for i,_ in enumerate(bucket_times):
		# # 		with open(self.material_train_durations_folder + self.arbitrary_discr + 'multi_' + self.type_discr + '_level_' + str(classifier.level+1) + self.params_discr + '_bucket_' + str(i) + '_train_durations.dat', 'w+'):
		# # 			pass
			
		# with open(self.material_train_durations_folder + self.arbitrary_discr + 'multi_' + self.type_discr + self.params_discr + '_train_durations.dat', 'w+'):
		# 	pass

		# for fold_n, classifiers in enumerate(classifiers_per_fold):

		# 	for classifier in classifiers:

		# 		# with open(self.material_features_folder + self.arbitrary_discr + 'multi_' + self.type_discr + '_level_' + str(classifier.level+1) + self.params_discr + '_tag_' + str(classifier.tag) + '_features.dat', 'a') as file:

		# 		# 	file.write('@fold\n')
		# 		# 	file.write(self.features_names[classifier.features_index[0]])

		# 		# 	for feature_index in classifier.features_index[1:]:
		# 		# 		file.write(','+self.features_names[feature_index])

		# 		# 	file.write('\n')
				
		# 		with open(self.material_train_durations_folder + self.arbitrary_discr + 'multi_' + self.type_discr + '_level_' + str(classifier.level+1) + self.params_discr + '_tag_' + str(classifier.tag) + '_train_durations.dat', 'a') as file:

		# 			file.write('%.6f\n' % (classifier.test_duration))

		# # Retrieve train_durations for each classifier
		# test_durations_per_fold = []
		
		# for classifiers in classifiers_per_fold:
		# 	test_durations_per_fold.append([])
		# 	for classifier in classifiers:
		# 		test_durations_per_fold[-1].append(classifier.test_duration)

		# with open(self.material_train_durations_folder + self.arbitrary_discr + 'multi_' + self.type_discr + self.params_discr + '_train_durations.dat', 'w+') as file:

		# 	mean_parallel_test_duration = np.mean(np.max(test_durations_per_fold, axis=1))
		# 	std_parallel_test_duration = np.std(np.max(test_durations_per_fold, axis=1))

		# 	mean_sequential_test_duration = np.mean(np.sum(test_durations_per_fold, axis=1))
		# 	std_sequential_test_duration = np.std(np.sum(test_durations_per_fold, axis=1))
		 
		# 	file.write('mean_par,std_par,mean_seq,std_seq\n')
		# 	file.write('%.6f,%.6f,%.6f,%.6f\n' % (mean_parallel_test_duration,std_parallel_test_duration,mean_sequential_test_duration,std_sequential_test_duration))

		# Graph plot
		G = nx.DiGraph()
		for info in classifiers_per_fold[0]:
			G.add_node(str(info.level)+' '+info.tag, level=info.level,
						 tag=info.tag, children_tags=info.children_tags)
		for node_parent, data_parent in G.nodes.items():
			for node_child, data_child in G.nodes.items():
				if data_child['level']-data_parent['level'] == 1 and any(data_child['tag'] in s for s in data_parent['children_tags']):
					G.add_edge(node_parent, node_child)
		nx.write_gpickle(G, self.graph_folder+self.arbitrary_discr + 'multi_' + self.type_discr + self.params_discr +'_graph.gml')

	def test_recursive(self,parent):

		for child_tag in parent.children:
			child = parent.children[child_tag]
			child.test_index = [index for index in parent.test_index if self.predictions[index, parent.level] == child_tag]
			print('Test node %s %s_%s'%(child,child.tag,child.level))
			if child.children_number > 1:
				self.write_fold(child.level, child.tag, child.children_tags)
				self.test(child)
				if child.level < self.max_level_target-1:
					self.test_recursive(child)

	def unary_class_results_inferring(self, node):

		node.test_duration = 0.
		node.features_index = node.parent.features_index

		self.write_fold(node.level, node.tag, node.children_tags)
		self.write_fold(node.level, node.tag, node.children_tags, _all=True)
		self.write_pred(self.labels[node.test_index, node.level], [node.tag] * len(node.test_index), node.level, node.tag)
		self.write_pred(self.labels[node.test_index_all, node.level], [node.tag] * len(node.test_index_all), node.level, node.tag, True)

		if len(self.anomaly_classes) > 0:
			proba_base = (1.,1.,1.)
			for index in node.test_index:
				self.predictions[index, node.level] = node.tag
			# 	self.probability[index, node.level] = (1., 1., 1.)
			# for index in node.test_index_all:
			# 	self.prediction_all[index, node.level] = node.tag
			# 	self.probability_all[index, node.level] = (1., 1., 1.)
		else:
			proba_base = 0.
			for index in node.test_index:
				self.predictions[index, node.level] = node.tag
			# 	self.probability[index, node.level] = 0.
			# for index in node.test_index_all:
			# 	self.prediction_all[index, node.level] = node.tag
			# 	self.probability_all[index, node.level] = 0.

		self.write_proba(self.labels[node.test_index, node.level], [proba_base] * len(node.test_index), node.level, node.tag)
		self.write_proba(self.labels[node.test_index_all, node.level], [proba_base] * len(node.test_index_all), node.level, node.tag, True)

	# NON APPLICARE LA CAZZO DI FEATURES SELECTION SUPERVISIONATA QUANDO FAI ANOMALY DETECTION
	# PORCODDIO
	def features_selection(self,node):
		features_index = []

		#TODO: does not work with onehotencoded

		if node.features_number != 0 and node.features_number < self.dataset_features_number:

			selector = SelectKBest(mutual_info_classif, k=node.features_number)

			selector.fit(
				self.training_set[node.train_index, :self.dataset_features_number],
				node.label_encoder.transform(self.ground_truth[node.train_index, node.level])
				)

			support = selector.get_support()

			features_index = [ i for i,v in enumerate(support) if v ]
		else:
			if node.packets_number == 0:
				features_index = range(self.dataset_features_number)
			else:
				features_index = np.r_[0:node.packets_number, self.dataset_features_number/2:self.dataset_features_number/2+node.packets_number]

		return features_index

	def Sklearn_RandomForest(self, node):

		# Instantation
		classifier = OutputCodeClassifier(RandomForestClassifier(n_estimators=100, n_jobs=-1),
			code_size=np.ceil(np.log2(node.children_number)/node.children_number))
		# Features selection
		node.features_index = self.features_selection(node)
		return classifier
		self.train(node, classifier)
		self.test(node, classifier)
		self.test_all(node, classifier)

	def Sklearn_CART(self, node):

		# Instantation
		classifier = DecisionTreeClassifier()
		# Features selection
		node.features_index = self.features_selection(node)
		return classifier
		self.train(node, classifier)
		self.test(node, classifier)
		self.test_all(node, classifier)

	def Spark_Classifier(self, node):

		# Instantation
		classifier = SklearnSparkWrapper(spark_session=spark, classifier_class=node.classifier_class, num_classes=node.children_number,
			numerical_features_length=self.numerical_features_length, nominal_features_lengths=self.nominal_features_lengths,
			numerical_features_index=self.numerical_features_index,nominal_features_index=self.nominal_features_index,
			fine_nominal_features_index=self.fine_nominal_features_index,classifier_opts=node.classifier_opts,
			epochs_number=self.epochs_number,level=node.level, fold=node.fold, classify=True, workers_number=self.workers_number,
			arbitrary_discr=self.arbitrary_discr)
		# Features selection
		node.features_index = self.features_selection(node)
		classifier.set_oracle(
			node.label_encoder.transform(self.labels[node.test_index, node.level])
			)
		return classifier
		self.train(node, classifier)
		self.test(node, classifier)

	def Spark_RandomForest(self, node):
		return self.Spark_Classifier(node)
	
	def Spark_NaiveBayes(self, node):
		return self.Spark_Classifier(node)
	
	def Spark_Keras_StackedDeepAutoencoderClassifier(self, node):
		return self.Spark_Classifier(node)

	def Spark_Keras_MultiLayerPerceptronClassifier(self, node):
		return self.Spark_Classifier(node)

	def Spark_MultilayerPerceptron(self, node):
		return self.Spark_Classifier(node)
	
	def Spark_GBT(self, node):
		return self.Spark_Classifier(node)
	
	def Spark_DecisionTree(self, node):
		return self.Spark_Classifier(node)

	def Weka_Classifier(self, node):

		# Instantation
		classifier = SklearnWekaWrapper(node.classifier_class)
		# Features selection
		node.features_index = self.features_selection(node)
		return classifier
		self.train(node, classifier)
		self.test(node, classifier)

		self.test_all(node, classifier)

	def Weka_NaiveBayes(self, node):
		return self.Weka_Classifier(node)
	
	def Weka_BayesNetwork(self, node):
		return self.Weka_Classifier(node)
	
	def Weka_RandomForest(self, node):
		return self.Weka_Classifier(node)
	
	def Weka_J48(self, node):
		return self.Weka_Classifier(node)
	
	def Weka_SuperLearner(self, node):

		# Instantation
		# classifier = SuperLearnerClassifier(['ssv', 'sif', 'slo', 'slo1', 'slo2', 'ssv1', 'ssv2', 'ssv3', 'ssv4'], node.children_number, node.label_encoder.transform(self.anomaly_class)[0])
		classifier = SuperLearnerClassifier(self.super_learner_default, node.children_number, node.label_encoder.transform([ self.anomaly_class ])[0], node.features_number)
		# Features selection
		node.features_index = self.features_selection(node)
		classifier.set_oracle(
			node.label_encoder.transform(self.labels[node.test_index, node.level])
			)
		return classifier
		self.train(node, classifier)		
		self.test(node, classifier)
		self.test_all(node, classifier)

	def Keras_Classifier(self, node):

		# Instantation
		classifier = SklearnKerasWrapper(*node.classifier_opts,model_class=node.classifier_class,
			epochs_number=self.epochs_number, numerical_features_length=self.numerical_features_length,
			nominal_features_lengths=self.nominal_features_lengths,num_classes=node.children_number,
			nominal_features_index=self.nominal_features_index, fine_nominal_features_index=self.fine_nominal_features_index,
			numerical_features_index=self.numerical_features_index,level=node.level, fold=node.fold, classify=True,
			weight_features=self.weight_features,arbitrary_discr=self.arbitrary_discr)
		# Features selection
		node.features_index = self.features_selection(node)
		return classifier
		self.train(node, classifier)
		self.test(node, classifier)

		self.test_all(node, classifier)

	def Keras_StackedDeepAutoencoderClassifier(self, node):
		return self.Keras_Classifier(node)

	def Keras_MultiLayerPerceptronClassifier(self, node):
		return self.Keras_Classifier(node)

	def Keras_Convolutional2DAutoencoder(self, node):
		return self.Keras_Classifier(node)

	def _AnomalyDetector(self, node):

		# Instantation
		classifier = AnomalyDetector(node.detector_class, node.detector_opts,
			node.label_encoder.transform([ self.anomaly_class ])[0], node.features_number,
			self.epochs_number, node.level, node.fold, self.n_clusters, self.optimize,
			self.weight_features, self.workers_number, self.unsupervised)
		# Features selection
		node.features_index = self.features_selection(node)
		return classifier
		self.train(node, classifier)
		# classifier.set_oracle(
		# 	node.label_encoder.transform(self.oracle[node.test_index][:, node.level])
		# 	)
		self.test(node, classifier)
		# classifier.set_oracle(
		# 	node.label_encoder.transform(self.oracle[node.test_index_all][:, node.level])
		# 	)
		self.test_all(node, classifier)

	# @profile
	def train(self, node):

		# print(node.label_encoder.nom_to_cat)

		if self.hidden_classes != '':
			not_hidden_index = [ i for i in node.train_index if self.labels[i,node.level] not in self.hidden_classes ]
		else:
			not_hidden_index = node.train_index

		node.test_duration = node.classifier.fit(
			self.features[not_hidden_index][:, node.features_index],
			node.label_encoder.transform(self.labels[not_hidden_index, node.level])
			).t_

		print(node.test_duration)
		self.write_time(node.test_duration, node.level, node.tag)

	# @profile
	def test(self, node):

		pred = node.label_encoder.inverse_transform(
			node.classifier.predict(self.features[node.test_index][:, node.features_index]))
		try:
			proba = node.classifier.predict_proba(self.features[node.test_index][:, node.features_index])
		except:
			proba = node.classifier.score(self.features[node.test_index][:, node.features_index])

		for p, b, i in zip(pred, proba, node.test_index):
			self.predictions[i, node.level] = p
		# 	self.probability[i, node.level] = b

		self.write_pred(self.labels[node.test_index, node.level], pred, node.level, node.tag)
		self.write_proba(self.labels[node.test_index, node.level], proba, node.level, node.tag)

		# exit(1)

	def test_all(self, node):

		pred = node.label_encoder.inverse_transform(
			node.classifier.predict(self.features[node.test_index_all][:, node.features_index]))
		try:
			proba = node.classifier.predict_proba(self.features[node.test_index_all][:, node.features_index])
		except:
			proba = node.classifier.score(self.features[node.test_index_all][:, node.features_index])

		# for p, b, i in zip(pred, proba, node.test_index_all):
		# 	self.prediction_all[i, node.level] = p
		# 	self.probability_all[i, node.level] = b

		self.write_pred(self.labels[node.test_index_all, node.level], pred, node.level, node.tag, True)
		self.write_proba(self.labels[node.test_index_all, node.level], proba, node.level, node.tag, True)

if __name__ == "__main__":
	os.environ["DISPLAY"] = ":0"	# used to show xming display

	supported_classifiers = {
		'srf': 'Sklearn_RandomForest',
		'scr': 'Sklearn_CART',		# C4.5
		'drf': 'Spark_RandomForest',
		'dnb': 'Spark_NaiveBayes',
		'dmp': 'Spark_MultilayerPerceptron',
		'dgb': 'Spark_GBT',
		'ddt': 'Spark_DecisionTree',
		'wnb': 'Weka_NaiveBayes',
		'wbn': 'Weka_BayesNetwork',
		'wrf': 'Weka_RandomForest',
		'wj48': 'Weka_J48',			# C4.5
		'wsl': 'Weka_SuperLearner',
		'kdae': 'Keras_StackedDeepAutoencoderClassifier',
		'kmlp': 'Keras_MultiLayerPerceptronClassifier',
		'dkdae': 'Spark_Keras_StackedDeepAutoencoderClassifier',
		'dkmlp': 'Spark_Keras_MultiLayerPerceptronClassifier'
	}

	supported_detectors = {
		'ssv': 'Sklearn_OC-SVM',
		'sif': 'Sklearn_IsolationForest',		# C4.5
		'slo': 'Sklearn_LocalOutlierFactor',
		'ssv1': 'Sklearn_OC-SVM',
		'sif1': 'Sklearn_IsolationForest',		# C4.5
		'slo1': 'Sklearn_LocalOutlierFactor',
		'ssv2': 'Sklearn_OC-SVM',
		'slo2': 'Sklearn_LocalOutlierFactor',
		'ksae': 'Keras_StackedAutoencoder',
		'kdae': 'Keras_DeepAutoencoder',
		'kc2dae' : 'Keras_Convolutional2DAutoencoder'
	}

	# Number of fold to test
	k=10

	input_file = ''
	levels_number = 0
	max_level_target = 0
	level_target = -1
	features_number = 0
	packets_number = 0
	classifier_name = ''
	detector_name = ''
	workers_number = 0
	executors_number = 1
	epochs_number = 10
	anomaly_classes = ''
	weight_features = False
	parallelize = False
	starting_fold = 1
	ending_fold = k

	anomaly = False
	deep = False
	early = False

	config_file = ''
	config = ''
	arbitrary_discr = ''
	n_clusters = 1
	buckets_number = 1

	hidden_classes = ''
	benign_class = ''

	nominal_features_index = None
	optimize = False
	unsupervised = False
	input_is_image = False
	memoryless = False

	try:
		opts, args = getopt.getopt(
			sys.argv[1:], "hi:n:t:f:p:c:o:w:x:a:e:d:s:C:H:b:N:OT:WPB:UF:G:EM",
				"[input_file=,levels_number=,max_level_target=,features_number=,packets_number=,\
				classifier_name=,configuration=,workers_number=,executors_number=,anomaly_class=,\
				epochs_number=,detector_name=,arbitrary_discr=,n_clusters=,hidden_classes=,nominal_features_index=,\
				level_target=,buckets_number=]"
			)
	except getopt.GetoptError:
		print(sys.argv[0],'-i <input_file> -n <levels_number> \
			(-f <features_number>|-p <packets_number>) -c <classifier_name> (-o <configuration_file>) \
			(-w <workers_number> -x <executors_number>) (-a <anomaly_class> -e epochs_number -d detector_name) (-s arbitrary_discr) -O')
		print(sys.argv[0],'-h (or --help) for a carefuler help')
		sys.exit(2)
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print(sys.argv[0],'-i <input_file> -n <levels_number> \
			(-f <features_number>|-p <packets_number>) -c <classifier_name> (-o <configuration_file>) \
			(-w <workers_number> -x <executors_number>) (-a <anomaly_class> -e epochs_number -d detector_name)')
			print('Options:\n\t-i: dataset file, must be in arff or in csv format\n\t-n: number of levels (number of labels\' columns)')
			print('\t-f or -p: former refers features number, latter refers packets number\n\t-c: classifier name choose from following list:')
			for sc in supported_classifiers:
				print('\t\t-c '+sc+'\t--->\t'+supported_classifiers[sc].split('_')[1]+'\t\timplemented in '+supported_classifiers[sc].split('_')[0])
			print('\t-o: configuration file in JSON format')
			print('\t-w: number of workers when used with spark.ml classifiers')
			print('\t-x: number of executors per worker when used with spark.ml classifiers')
			print('\t-a: hierarchical classifier starts with a one-class classification at each level if set and classification is binary (i.e. Anomaly Detection)\n\
				\tmust provide label of baseline class')
			print('\t-e: number of epochs in caso of DL models')
			print('\t-s: arbitrary discriminator, a string to put at the start of saved files')
			print('\t-O: optimization of models')
			for sd in supported_detectors:
				print('\t\t-c '+sd+'\t--->\t'+supported_detectors[sd].split('_')[1]+'\t\timplemented in '+supported_detectors[sd].split('_')[0])
			sys.exit()
		if opt in ("-i", "--input_file"):
			input_file = arg
		if opt in ("-n", "--levels_number"):
			levels_number = int(arg)
		if opt in ("-t", "--max_level_target"):
			max_level_target = int(arg)
		if opt in ("-f", "--nfeat"):
			features_number = int(arg)
		if opt in ("-p", "--npacket"):
			packets_number = int(arg)
		if opt in ("-c", "--clf"):
			classifier_name = arg
		if opt in ("-o", "--configuration"):
			config_file = arg
		if opt in ("-w", "--workers_number"):
			workers_number = int(arg)
		if opt in ("-x", "--executors_number"):
			executors_number = int(arg)
		if opt in ("-a", "--anomaly_class"):
			anomaly_classes = arg
		if opt in ("-e", "--epochs_number"):
			epochs_number = int(arg)
		if opt in ("-d", "--detector_name"):
			detector_name = arg
		if opt in ("-s", "--arbitrary_discr"):
			arbitrary_discr = arg
		if opt in ("-C", "--n_clusters"):
			n_clusters = int(arg)
		if opt in ("-H", "--hidden_classes"):
			hidden_classes = arg
		if opt in ("-b", "--benign_class"):
			benign_class = arg
		if opt in ("-N", "--nominal_features_index"):
			nominal_features_index = [ int(i) for i in arg.split(',') ]
		if opt in ("-O", "--optimize"):
			optimize = True
		if opt in ("-T", "--level_target"):
			level_target = int(arg)
		if opt in ("-W", "--weight_features"):
			weight_features = True
		if opt in ("-P", "--parallelize"):
			parallelize = True
		if opt in ("-B", "--buckets_number"):
			buckets_number = int(arg)
		if opt in ("-U", "--unsupervised"):
			unsupervised = True
		if opt in ("-F", "--starting_fold"):
			starting_fold = int(arg)
		if opt in ("-G", "--ending_fold"):
			ending_fold = int(arg)
		if opt in ("-E", "--early"):
			early = True
		if opt in ("-M", "--memoryless"):
			memoryless = True

	if anomaly_classes != '' or (benign_class != '' and hidden_classes == ''):
		anomaly = True

	# import_str variable contains initial character of name of used classifiers
	# it is used to import specific module
	import_str = ''

	if config_file:
		if not config_file.endswith('.json'):
			print('config file must have .json extention')
			sys.exit()

		import json

		with open(config_file) as f:
			config = json.load(f)

		for node in config:
			if config[node] not in supported_classifiers:
				print('Classifier not supported in configuration file\nList of available classifiers:\n')
				for sc in np.sort(supported_classifiers.keys()):
					print('-c '+sc+'\t--->\t'+supported_classifiers[sc].split('_')[1]+'\t\timplemented in '+supported_classifiers[sc].split('_')[0])
				print('Configuration inserted:\n', config)
				sys.exit()
			if config[node][0] not in import_str:
				import_str += config[node][0]
				if config[node][1] == 'k':
					import_str += config[node][1]

	else:

		if packets_number != 0 and features_number != 0 or packets_number == features_number:
			print('-f and -p option should not be used together')
			sys.exit()

		if len(classifier_name.split('_')) > 1:
			classifier_class = classifier_name.split('_')[0]
			classifier_opts = classifier_name.split('_')[1:]
		else:
			classifier_class = classifier_name
			classifier_opts = []

		if len(detector_name.split('_')) > 1:
			detector_class = detector_name.split('_')[0]
			detector_opts = detector_name.split('_')[1:]
		else:
			detector_class = detector_name
			detector_opts = []

		anomaly_classes = [ v for v in anomaly_classes.split(',') if v!='' ]
		hidden_classes = [ v for v in hidden_classes.split(',') if v!='' ]

		if classifier_class != '' and classifier_class not in supported_classifiers:
			print('Classifier not supported\nList of available classifiers:\n')
			for sc in supported_classifiers:
				print('-c '+sc+'\t--->\t'+supported_classifiers[sc].split('_')[1]+'\t\timplemented in '+supported_classifiers[sc].split('_')[0])
			sys.exit()

		if len(anomaly_classes) > 0 and detector_class not in supported_detectors:
			print('Detector not supported\nList of available detectors:\n')
			for sd in supported_detectors:
				print('\t\t-c '+sd+'\t--->\t'+supported_detectors[sd].split('_')[1]+'\t\timplemented in '+supported_detectors[sd].split('_')[0])
			sys.exit()

		if classifier_name != '':
			import_str = classifier_name[0]
			if classifier_name[1] == 'k':
				import_str = classifier_name[:2]
		if detector_name != '':
			import_str = detector_name[0]

	if levels_number == 0:
		print('Number of level must be positive and non zero')
		sys.exit()

	if max_level_target == 0 or max_level_target > levels_number:
		max_level_target = levels_number

	if level_target > levels_number:
		level_target = levels_number
		print('Warning: level target was greater than levels number, it will be treated as equal.')

	if not input_file.endswith('.arff') and not input_file.endswith('.csv') and not input_file.endswith('.pickle'):
		print('input files must be or .arff or .csv or .pickle')
		sys.exit()

	import_sklearn()
	
	if 'k' in import_str:
		# For reproducibility
		from tensorflow import set_random_seed
		set_random_seed(0)
		import_keras()
		deep = True

	if 'd' in import_str:
		import_spark()
		conf = SparkConf().setAll([
			('spark.app.name', '_'.join(sys.argv)),
			('spark.driver.cores', 3),
			('spark.driver.memory','7g'),
			('spark.executor.memory', '2900m'),
			('spark.master', 'spark://192.168.200.45:7077'),
			('spark.executor.cores', str(executors_number)),
			('spark.cores.max', str(executors_number * workers_number)),
			('spark.rpc.message.maxSize', '512')
		])
		sc = SparkContext(conf=conf)
		sc.setLogLevel("ERROR")
		spark = SparkSession(sc)

	# if 'k' in import_str and 'd' in import_str:
	# 	import_distkeras()

	# Momentarily SuperLearner works with weka models
	if 'w' in import_str:
		import_weka()
		jvm.start()

	if classifier_class == 'kc2dae':
		input_is_image = True

	# For reproducibility
	np.random.seed(0)

	hierarchical_classifier = HierarchicalClassifier(
		input_file=input_file,
		levels_number=levels_number,
		max_level_target=max_level_target,
		level_target=level_target,
		features_number=features_number,
		packets_number=packets_number,
		classifier_class=classifier_class,
		classifier_opts=classifier_opts,
		detector_class=detector_class,
		detector_opts=detector_opts,
		workers_number=workers_number,
		anomaly_classes=anomaly_classes,
		epochs_number=epochs_number,
		arbitrary_discr=arbitrary_discr,
		n_clusters=n_clusters,
		anomaly=anomaly,
		deep=deep,
		hidden_classes=hidden_classes,
		benign_class=benign_class,
		nominal_features_index=nominal_features_index,
		optimize=optimize,
		weight_features=weight_features,
		parallelize=parallelize,
		buckets_number=buckets_number,
		unsupervised=unsupervised,
		)
	
	if config:
		config_name = config_file.split('.')[0]
		hierarchical_classifier.set_config(config_name, config)

	hierarchical_classifier.init_output_files()
	if input_is_image:
		hierarchical_classifier.load_image()
	elif early:
		hierarchical_classifier.load_early_dataset()
	else:
		hierarchical_classifier.load_dataset(memoryless)
	hierarchical_classifier.kfold_validation(k=k, starting_fold=starting_fold, ending_fold=ending_fold)

	if 'w' in import_str:
		jvm.stop()
