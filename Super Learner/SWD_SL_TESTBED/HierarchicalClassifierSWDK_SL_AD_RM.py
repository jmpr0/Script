#!/usr/bin/python2

import arff
import numpy as np
import sys, gc
import getopt
import os, time
from copy import copy

import networkx as nx
from scipy.stats.mstats import hmean

from importlib import import_module

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
		EllipticEnvelopeAnomalyDetector,FactorAnalysis

	RandomForestClassifier = import_module('sklearn.ensemble').RandomForestClassifier
	DecisionTreeClassifier = import_module('sklearn.tree').DecisionTreeClassifier
	OneClassSVMAnomalyDetector = import_module('sklearn.svm').OneClassSVM
	IsolationForestAnomalyDetector = import_module('sklearn.ensemble').IsolationForest
	LocalOutlierFactorAnomalyDetector = import_module('sklearn.neighbors').LocalOutlierFactor
	EllipticEnvelopeAnomalyDetector = import_module('sklearn.covariance').EllipticEnvelope
	SelectKBest = import_module('sklearn.feature_selection').SelectKBest
	mutual_info_classif = import_module('sklearn.feature_selection').mutual_info_classif
	StratifiedKFold = import_module('sklearn.model_selection').StratifiedKFold
	train_test_split = import_module('sklearn.model_selection').train_test_split
	OrdinalEncoder = import_module('sklearn.preprocessing').OrdinalEncoder
	OneHotEncoder = import_module('sklearn.preprocessing').OneHotEncoder
	MinMaxScaler = import_module('sklearn.preprocessing').MinMaxScaler
	KernelDensity = import_module('sklearn.neighbors').KernelDensity
	FactorAnalysis = import_module('sklearn.decomposition').FactorAnalysis
	# LabelEncoder = import_module('sklearn.preprocessing').LabelEncoder

def import_spark():

	global SparkContext,SparkConf,SparkSession,udf,Spark_NaiveBayesClassifier,Spark_MultilayerPerceptronClassifier,\
		Spark_GBTClassifier,Spark_DecisionTreeClassifier,VectorUDT

	global pandas

	SparkContext = import_module('pyspark').SparkContext
	SparkConf = import_module('pyspark').SparkConf
	SparkSession = import_module('pyspark.sql.session').SparkSession
	udf = import_module('pyspark.sql.functions.udf')
	Spark_RandomForestClassifier = import_module('pyspark.ml.classification').RandomForestClassifier
	Spark_NaiveBayesClassifier = import_module('pyspark.ml.classification').NaiveBayes
	Spark_MultilayerPerceptronClassifier = import_module('pyspark.ml.classification').MultilayerPerceptronClassifier
	Spark_GBTClassifier = import_module('pyspark.ml.classification').GBTClassifier
	Spark_DecisionTreeClassifier = import_module('pyspark.ml.classification').DecisionTreeClassifier
	Vectors = import_module('pyspark.ml.linalg').Vectors
	VectorUDT = import_module('pyspark.ml.linalg').VectorUDT

	pandas = import_module('pandas')

def import_weka():

	global jvm,ndarray_to_instances,Attribute,Classifier

	jvm = import_module('weka.core.jvm')
	ndarray_to_instances = import_module('weka.core.converters').ndarray_to_instances
	Attribute = import_module('weka.core.dataset').Attribute
	Classifier = import_module('weka.classifiers').Classifier

def import_keras():
	global keras,cython,optimizers,Sequential,Model,Dense,Input,Conv1D,Conv2D,MaxPooling2D,Flatten,Dropout,\
		Reshape,BatchNormalization,LSTM,AveragePooling1D,MaxPooling1D,UpSampling1D,Activation,relu,elu,sigmoid,softmax,hard_sigmoid,\
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

		return t

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

		features_encoder = OrdinalEncoder()
		labels_nominal = features_encoder.fit_transform(np.array(labels).reshape(-1, 1))

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

	def __init__(self, classifier_name):

		if classifier_name == 'drf':
			self._classifier = Spark_RandomForestClassifier(featuresCol='features', labelCol='categorical_label',\
			predictionCol='prediction', probabilityCol='probability', rawPredictionCol='rawPrediction',\
			maxDepth=20, maxBins=128, minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=1024, cacheNodeIds=False, checkpointInterval=10,\
			impurity='gini', numTrees=100, featureSubsetStrategy='sqrt', seed=None, subsamplingRate=1.0)
		elif classifier_name == 'dnb':
			self._classifier = Spark_NaiveBayesClassifier(featuresCol='features', labelCol='categorical_label',\
			predictionCol='prediction', probabilityCol='probability', rawPredictionCol='rawPrediction', smoothing=1.0,\
			modelType='multinomial', thresholds=None, weightCol=None)
		elif classifier_name == 'dmp':
			self._classifier = Spark_MultilayerPerceptronClassifier(featuresCol='features', labelCol='categorical_label',\
			predictionCol='prediction', maxIter=100, tol=1e-06, seed=None, layers=None, blockSize=128, stepSize=0.03, solver='l-bfgs',\
			initialWeights=None, probabilityCol='probability', rawPredictionCol='rawPrediction')
		elif classifier_name == 'dgb':
			self._classifier = Spark_GBTClassifier(featuresCol='features', labelCol='categorical_label',\
			predictionCol='prediction', maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256,\
			cacheNodeIds=False, checkpointInterval=10, lossType='logistic', maxIter=20, stepSize=0.1, seed=None,\
			subsamplingRate=1.0, featureSubsetStrategy='all')
		elif classifier_name == 'ddt':
			self._classifier = Spark_DecisionTreeClassifier(featuresCol='features', labelCol='categorical_label',\
			predictionCol='prediction', probabilityCol='probability', rawPredictionCol='rawPrediction', maxDepth=5, maxBins=32,\
			minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, impurity='gini', seed=None)

	def fit(self, training_set, ground_truth):

		self.ground_truth = ground_truth

		training_set = self._sklearn2spark(training_set, self.ground_truth)

		t = time.time()
		self._model = self._classifier.fit(training_set)
		t = time.time() - t

		return t

	def predict(self, testing_set):

		testing_set = self._sklearn2spark(testing_set, self.oracle)

		self.results = self._model.transform(testing_set)

		preds = [int(float(row.prediction)) for row in self.results.collect()]
	 
		return np.array(preds)

	def predict_proba(self, testing_set):

		testing_set = self._sklearn2spark(testing_set, self.oracle)

		self.results = self._model.transform(testing_set)

		probas = [row.probability.toArray() for row in self.results.collect()]

		return np.array(probas)

	def set_oracle(self, oracle):

		self.oracle = oracle

	def _sklearn2spark(self, features, labels=None):

		dataset = pandas.DataFrame(
			{'features': features.tolist(),
			 'categorical_label': labels
			})
		
		spark_dataset_with_list = spark.createDataFrame(dataset)
		
		list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
		
		spark_dataset = spark_dataset_with_list.select(
			list_to_vector_udf(spark_dataset_with_list['features']).alias('features'),
			spark_dataset_with_list['categorical_label']
		)
		
		return spark_dataset

class SklearnKerasWrapper(object):

	def __init__(self, classifier_name, features_number, num_classes, epochs_number=0, level=0, fold=0):

		self.classifier_name = classifier_name
		self._scaler = MinMaxScaler()
		self.epochs_number = epochs_number
		self.level = level
		self.proba = None
		self._nom_loss_min = 0
		self._nom_loss_max = 0
		self._num_loss_min = 0
		self._num_loss_max = 0

		log_folder = './data_%s/material/log' % self.classifier_name
		plot_folder = './data_%s/material/plot' % self.classifier_name
		summary_folder = './data_%s/material/summary' % self.classifier_name
		tmp_folder = './data_%s/tmp' % self.classifier_name

		if not os.path.exists(log_folder):
			os.makedirs(log_folder)
		if not os.path.exists(plot_folder):
			os.makedirs(plot_folder)
		if not os.path.exists(summary_folder):
			os.makedirs(summary_folder)
		if not os.path.exists(tmp_folder):
			os.makedirs(tmp_folder)

		self.log_file = '%s/%s_model_%s_l_%s_log.csv' % (log_folder, fold, self.classifier_name, self.level+1)
		self.plot_file = '%s/%s_model_%s_l_%s_plot.png' % (plot_folder, fold, self.classifier_name, self.level+1)
		self.summary_file = '%s/%s_model_%s_l_%s_summary.dat' % (summary_folder, fold, self.classifier_name, self.level+1)
		self.checkpoint_file = "%s/weights.hdf5" % (tmp_folder)

		self.fold = fold

		# EarlyStopping(monitor='val_loss', self.patience=3, self.min_delta=1e-4)
		# CSVLogger(self.log_file)
		# ModelCheckpoint(filepath=self.checkpoint_file, verbose=1, save_best_only=True)
		self.patience = 3
		self.min_delta = 1e-4

		self._models = []
		self._encoders = []

		self.sparse = False
		self.variational = False
		self.denoising = False

	# @profile
	def fit(self, training_set, ground_truth, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1):

		# if self.fold != 4:
		# 	return

		nom_training_set, num_training_set = self._split_nom_num_features(training_set)

		self.nominal_features_lengths = [ v.shape[1] for v in nom_training_set]
		self.numerical_features_length = num_training_set.shape[1]

		k=3

		self._models, _, self._models_n =  self._init_models(3, self.nominal_features_lengths, self.numerical_features_length, k)

		# Define weights for loss average
		num_weight = [ self.numerical_features_length ]
		nom_weights = [1] * len(self.nominal_features_lengths)

		# Scaling training set for Autoencoder
		scaled_num_training_set = self._scaler.fit_transform(num_training_set)

		normal_losses_per_fold = []
		times_per_fold = []

		# Iterating over all the models to find the best
		for index in range(len(self._models)):

			print('\nStarting StratifiedKFold for %s\n' % self._models_n[index])

			skf = StratifiedKFold(n_splits=k, shuffle=True)

			normal_losses_per_fold.append([])
			times_per_fold.append([])

			f=0

			for train_index, validation_index in skf.split(
				np.zeros(shape=(num_training_set.shape[0], self.numerical_features_length + len(self.nominal_features_lengths))),
				ground_truth):

				model = self._models[index][f]

				f += 1
				print('\nFold %s\n' % f)

				# for _model in self._models:
				if not self.denoising:

					times_per_fold[-1].append(time.time())
					# Train Autoencoder
					train_history =\
						model.fit(
							[ scaled_num_training_set[train_index,:] ] + [ nom_training[train_index,:] for nom_training in nom_training_set ],
							[ scaled_num_training_set[train_index,:] ] + [ nom_training[train_index,:] for nom_training in nom_training_set ],
							epochs=self.epochs_number,
							batch_size=32,
							shuffle=True,
							verbose=2,
							# validation_split = 0.1,
							validation_data=(
								[ scaled_num_training_set[validation_index,:] ] + [ nom_training[validation_index,:] for nom_training in nom_training_set ],
								[ scaled_num_training_set[validation_index,:] ] + [ nom_training[validation_index,:] for nom_training in nom_training_set ]
								),
							callbacks = [
								EarlyStopping(monitor='val_loss', patience=self.patience, min_delta=self.min_delta),
								CSVLogger(self.log_file),
								# ModelCheckpoint(filepath=self.checkpoint_file, verbose=1, save_best_only=True)
								]
							)

				else:

					noise_factor = .5
					noisy_scaled_num_training_set = scaled_num_training_set[train_index,:] + noise_factor * np.random.normal(loc=.0, scale=1., size=scaled_num_training_set[train_index,:].shape)

					times_per_fold[-1].append(time.time())
					# Train Autoencoder
					train_history =\
						model.fit(
							[ noisy_scaled_num_training_set ] + [ nom_training[train_index,:] for nom_training in nom_training_set ],
							[ scaled_num_training_set[train_index,:] ] + [ nom_training[train_index,:] for nom_training in nom_training_set ],
							epochs=self.epochs_number,
							batch_size=32,
							shuffle=True,
							verbose=2,
							# validation_split = 0.1,
							validation_data=(
								[ scaled_num_training_set[validation_index,:] ] + [ nom_training[validation_index,:] for nom_training in nom_training_set ],
								[ scaled_num_training_set[validation_index,:] ] + [ nom_training[validation_index,:] for nom_training in nom_training_set ]
								),
							callbacks = [
								EarlyStopping(monitor='val_loss', patience=self.patience, min_delta=self.min_delta),
								CSVLogger(self.log_file),
								# ModelCheckpoint(filepath=self.checkpoint_file, verbose=1, save_best_only=True)
								]
							)

				num_losses = []
				nom_losses = []
				for loss_tag in train_history.history:
					if 'val' in loss_tag:
						if 'num' in loss_tag:
							for num_loss in train_history.history[loss_tag][::-1]:
								if not np.isnan(num_loss):
									num_losses.append(num_loss)
									break
						if 'nom' in loss_tag:
							for nom_loss in train_history.history[loss_tag][::-1]:
								if not np.isnan(nom_loss):
									nom_losses.append(nom_loss)
									break

				# This order (i.e. num+nom) must be respected to compute weighthed average
				losses = num_losses + nom_losses

				normal_losses_per_fold[-1].append(np.average(losses, weights = num_weight + nom_weights))

				times_per_fold[-1][-1] = time.time() - times_per_fold[-1][-1]

				self._models[index][f-1] = None
				model = train_history = losses = nom_losses = num_losses = None
				del model, train_history, losses, nom_losses, num_losses
				gc.collect()
				
			normal_losses_per_fold[-1] = np.asarray(normal_losses_per_fold[-1])
			times_per_fold[-1] = np.asarray(times_per_fold[-1])

			# if np.isnan(losses[0][0]):
			# 	for layer in self._model.layers[1:-1]:
			# 		w=layer.get_weights()
			# 		print(w)
			# 		input()

			# In case of "model exploding" we use last best network configuration
			# if np.isnan(losses[0][0]):

			# reconstr_training_set = self._model.predict([ scaled_num_training_set ] + nom_training_set)
			# losses = []
			# losses.append(K.eval(mean_squared_error(scaled_num_training_set,reconstr_training_set[0])))
			# for nts, rnts in zip(nom_training_set, reconstr_training_set[1:]):
			# 	losses.append(K.eval(categorical_crossentropy(nts,K.constant(rnts))))

			# first losses refers to numerical features, so we weighted them

			# self.normal_loss_stat = np.percentile(self.loss,[95,97.5,99])

		self._models = None
		del self._models
		gc.collect()


		mean_normal_losses = np.mean(normal_losses_per_fold,axis=1)
		mean_times = np.mean(times_per_fold,axis=1)
		print(times_per_fold)
		print(mean_times)
		# std_normal_losses = np.std(normal_losses_per_fold,axis=1)
		# std_times = np.std(times_per_fold,axis=1)

		min_loss = np.min(mean_normal_losses)
		best_model_index = list(mean_normal_losses).index(min_loss)
		best_model_n = self._models_n[best_model_index]
		t = np.max(mean_times)

		print(t)

		join_ray = int(best_model_n.split('_')[1])
		grafting_layer = int(best_model_n.split('_')[2])

		self._model,_,_ = self._init_model(3, self.nominal_features_lengths, self.numerical_features_length, join_ray, grafting_layer, k=1)

		if not self.denoising:

			t = time.time() - t
			# Train Autoencoder
			train_history =\
				self._model.fit(
					[ scaled_num_training_set ] + nom_training_set,
					[ scaled_num_training_set ] + nom_training_set,
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
			noisy_scaled_num_training_set = scaled_num_training_set[train_index,:] + noise_factor * np.random.normal(loc=.0, scale=1., size=scaled_num_training_set[train_index,:].shape)

			t = time.time() - t
			# Train Autoencoder
			train_history =\
				self._model.fit(
					[ noisy_scaled_num_training_set ] + nom_training_set,
					[ scaled_num_training_set ] + nom_training_set,
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

		# self._model.load_weights(self.checkpoint_file)

		num_losses = []
		nom_losses = []
		for loss_tag in train_history.history:
			# if 'val' in loss_tag:
			if 'num' in loss_tag:
				for num_loss in train_history.history[loss_tag][::-1]:
					if not np.isnan(num_loss):
						num_losses.append(num_loss)
						break
			if 'nom' in loss_tag:
				for nom_loss in train_history.history[loss_tag][::-1]:
					if not np.isnan(nom_loss):
						nom_losses.append(nom_loss)
						break

		# This order (i.e. num+nom) must be respected to compute weighthed average
		losses = num_losses + nom_losses
		self.normal_loss = np.average(losses, weights = num_weight + nom_weights)

		t = time.time() - t

		with open('.' + self.summary_file.split('.')[1]+'_best_model_'+best_model_n+'.dat','w') as f:
					self._model.summary(print_fn=lambda x: f.write(x + '\n'))

		print('loss',self.normal_loss)
		print('time',t)

		return t

	def _init_models(self, depth, nom_lengths, num_length, k=1):

		_models = []
		_encoders = []
		_models_n = []

		for join_ray in range(-1,depth+1,2):
			for grafting_layer in range(0,depth+2,2):
				models, encoders, model_n = self._init_model(depth, nom_lengths, num_length, join_ray, grafting_layer, k)
				_models.append(models)
				_encoders.append(encoders)
				_models_n.append(model_n)

		return _models, _encoders, _models_n

	def _init_model(self, depth, nom_lengths, num_length, join_ray, grafting_layer, k=1):

		compressed_ratio = .25
		reduction_coeff = (1 - compressed_ratio) / (depth + 1) 
		reduction_factors = [ 1 - reduction_coeff * i for i in range(1,depth+1) ] + [compressed_ratio]

		latent_ratio = .10

		# This parameter is used to set the number of e/d layers nearest the middle layer to join
		# If it's more or equal than zero, the middle layer is automatically joined
		# The parameter take values in [-1,depth], where -1 states no join, zero states only middle layer join
		# If it's more than zero, it indicates the couple of e/d layers to join, starting from nearest middle layer
		# join_ray = -1
		# This parameter specify at which level num model grafts on nom one
		# It takes values in [0,depth+1], where depth+1 state for no graft, and others value refers to the 0,1,2,..depth-1 i-e/d layer
		# E.g. if it is equal to depth, the graft is done on middle layer, if to 0, on the first encoder layer
		# grafting_layer = depth-1

		num_hidden_activation_function = elu
		nom_hidden_activation_function = elu

		num_output_activation_function = sigmoid
		nom_output_activation_function = softmax

		hyb_activation_function = elu

		# if self.classifier_name.startswith('kdae'):
		# 	if True:
		_model_n = 'kdae_%s_%s' % (join_ray, grafting_layer)

		join_factors = [reduction_factors[i] if i < 0 else reduction_factors[-i-1] for i in range(-(join_ray+1),(join_ray+1)) if i != 0]

		input_datas = []
		encodeds =[]
		decodeds =[]
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

		# Building model for numerical features
		input_datas.append(Input(shape=(num_length,), name='num_input'))
		x = input_datas[-1]
		for i in range(np.min([depth+1, grafting_layer])):
			if i < depth: # Adding encoding layers
				x = Dense(int(np.ceil(reduction_factors[i]*num_length)), activation=num_hidden_activation_function, name='num_enc%s' % i)(x)
			if i == depth: # Adding middle layer
				intermediate_dim = int(np.ceil(reduction_factors[i]*num_length))
				x = Dense(intermediate_dim, activation=num_hidden_activation_function, name='num_mid')(x)
				if self.sparse: # Sparsifying middle output
					x.W_regularizer = regularizers.l1(1e-4)
				if self.variational: # self.variational middle layer
					latent_dim = int(np.ceil(latent_ratio*num_length))
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
		for i,nom_length in enumerate(nom_lengths):
			# Instantiate separated inputs
			input_datas.append(Input(shape=(nom_length,), name='nom_input%s' % i))
			x = input_datas[-1]
			for j in range(depth - join_ray):
				if j == grafting_layer: # Saving layer to be grafted as graft_in
					concat_ix = concatenate([ graft_in, x ])
					x = concat_ix
				if j != len(reduction_factors)-1: # Adding encoding layers
					x = Dense(int(np.ceil(reduction_factors[j]*nom_length)), activation=nom_hidden_activation_function, name='nom_enc%s%s' % (i,j))(x)
				if j == len(reduction_factors)-1: # Adding middle layer
					intermediate_dim = int(np.ceil(reduction_factors[j]*nom_length))
					x = Dense(intermediate_dim, activation=nom_hidden_activation_function, name='nom_mid%s' % (i))(x)
					if self.sparse: # Sparsifying middle output
						x.W_regularizer = regularizers.l1(1e-4)
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
					if j == grafting_layer: # If middle layer if the graft layer, we save it also as graft_out
						graft_out.append(x)
			xs.append(x)
		concat_xs = concatenate(xs)
		x = concat_xs
		for i,join_factor in enumerate(join_factors):
			shared_length = np.sum(nom_lengths)
			if i < np.floor(len(join_factors)/2.): # Adding encoding shared layers
				if i == (grafting_layer - depth + join_ray): # Saving graft_in
					concat_ix = concatenate([ graft_in, x ])
					x = concat_ix
				x = Dense(int(np.ceil(join_factor*shared_length)), activation=hyb_activation_function, name='nom_jenc%s' % (i))(x)
			if i > np.floor(len(join_factors)/2.): # Adding decoding shared layers
				x = Dense(int(np.ceil(join_factor*shared_length)), activation=hyb_activation_function, name='nom_jdec%s' % (i-join_ray-1))(x)
				if i == (-grafting_layer + depth + join_ray): # Saving graft_out
					graft_out.append(x)
			if i == np.floor(len(join_factors)/2.): # Adding middle shared layer
				if i == (grafting_layer - depth + join_ray): # Saving graft_in
					concat_ix = concatenate([ graft_in, x ])
					x = concat_ix
				intermediate_dim = int(np.ceil(join_factor*shared_length))
				x = Dense(intermediate_dim, activation=hyb_activation_function, name='nom_jmid')(x)
				if self.sparse: # Sparsifying middle output
					x.W_regularizer = regularizers.l1(1e-4)
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
				if i == (grafting_layer - depth + join_ray): # And also graft_out, if middle shared layer is graft layer
					graft_out.append(x)
			xs = [x] * len(nom_lengths)
		for i,nom_length in enumerate(nom_lengths):
			x = xs[i]
			for j in reversed(range(depth - np.max([0, join_ray]))): # Adding decoder layers
				x = Dense(int(np.ceil(reduction_factors[j]*nom_length)), activation=nom_hidden_activation_function, name='nom_dec%s%s' % (i,depth-j-1))(x)
				if j == grafting_layer: # If layer is graft layer, save graft_out
					graft_out.append(x)
			# Instantiate separated outputs
			decodeds.append(Dense(nom_length, activation=nom_output_activation_function, name='nom_out%s' % (i))(x))

		if len(graft_out) > 1: # More than one graft layer, if graft is on non shared layers
			x = concatenate(graft_out)
		elif len(graft_out) > 0: # Only one graft layer, if graft is on a shared layer
			x = graft_out[0]
		else: # When no graft, we use previously saved numerical output
			x = graft_out_bk
		for i in reversed(range(np.min([depth, grafting_layer]))): # Adding decoder layers
			x = Dense(int(np.ceil(reduction_factors[i]*num_length)), activation=num_output_activation_function, name='num_dec%s' % (depth-i-1))(x)
		decodeds = [ Dense(num_length, activation='sigmoid', name='num_out')(x) ] + decodeds

		# For fitting
		_model = Model(input_datas, decodeds)
		# For feature extraction
		_encoder = Model(input_datas, encodeds)

		plot_model(_model, to_file='.' + self.plot_file.split('.')[1]+'_jr%s_il%s' % (join_ray, grafting_layer)+'.png', show_shapes=True, show_layer_names=True)
		with open('.' + self.summary_file.split('.')[1]+'_jr%s_il%s' % (join_ray, grafting_layer)+'.dat','w') as f:
			_model.summary(print_fn=lambda x: f.write(x + '\n'))

		if k > 1:
			_models = []
			_encoders = []

			for _ in range(k):
				_models.append(clone_model(_model))
				_encoders.append(clone_model(_encoder))
				_models[-1].compile(optimizer=Adadelta(lr=1.), loss=[ 'mean_squared_error' ] + [ 'categorical_crossentropy' ] * len(nom_lengths))

			return _models, _encoders, _model_n
		else:
			_model.compile(optimizer=Adadelta(lr=1.), loss=[ 'mean_squared_error' ] + [ 'categorical_crossentropy' ] * len(nom_lengths))

			return _model, _encoder, _model_n

	def predict(self, testing_set):

		# if self.fold != 4:
		# 	self.proba = self.oracle
		# 	return self.oracle

		# testing_set, nominal_features_index, numerical_features_index = self._expand_onehot_features(testing_set)
		nom_testing_set, num_testing_set = self._split_nom_num_features(testing_set)

		scaled_num_testing_set = self._scaler.transform(num_testing_set)
		reconstr_testing_set = self._model.predict([ scaled_num_testing_set ] + nom_testing_set)

		losses = []
		losses.append(K.eval(mean_squared_error(scaled_num_testing_set,reconstr_testing_set[0])))
		for nts, rnts in zip(nom_testing_set, reconstr_testing_set[1:]):
			losses.append(K.eval(categorical_crossentropy(nts,K.constant(rnts))))

		# first losses refers to numerical features, so we weighted them
		num_weight = [ self.numerical_features_length ]
		nom_weights = [1] * len(self.nominal_features_lengths)
		loss = np.average(losses, axis = 0, weights = num_weight + nom_weights)

		# loss = [ (l-self._loss_min)/(self._loss_max-self._loss_min) + eps for l in loss ]

		pred = np.asarray([
			-1 if l > self.normal_loss
			else 1 for l in loss
			])
		self.proba = np.asarray(loss)

		pred = np.reshape(pred,pred.shape[0])
		self.proba = np.reshape(self.proba,self.proba.shape[0])

		return pred

	def predict_proba(self, testing_set):

		if self.proba is None:
			self.predict(testing_set)

		return self.proba

	def set_oracle(self, oracle):

		self.oracle = oracle

	def _sklearn2keras(self, features, labels=None):

		pass

	def _expand_onehot_features(self, set):

		# nominal_features_index = [0,1,2,5]
		nominal_features_index = [ i for i,v in enumerate(set[0,:]) if isinstance(v, np.ndarray) ]
		nominal_features_length = [ len(set[0,i]) for i in nominal_features_index ]

		nom_len = [0]
		nom_base = [0]
		for i,nfi in enumerate(nominal_features_index):
			nom_base.append(nfi + nom_len[-1])
			nom_len.append(nom_len[-1] + nominal_features_length[i] - 1)
			set = np.c_[ set[:,:nom_base[-1]], np.asarray(list(zip(*set[:,nom_base[-1]]))).T, set[:,nom_base[-1]+1:] ]

		nom_index = nom_base[1:]
		nom_length = nominal_features_length
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

	def _split_nom_num_features(self,set):
		nom_index = []
		num_index = []
		
		for i,v in enumerate(set[0,:]):
			if isinstance(v, np.ndarray):
				nom_index.append(i)
			else:
				num_index.append(i)

		nom_set = set[:, nom_index]
		num_set = set[:, num_index]

		temp_set = []
		for i in range(nom_set.shape[1]):
			temp_set.append(np.asarray(nom_set[:,i]))

		# Fare un nom_set list di 4 ndarray ognuno con tutte le colonne dentro
		nom_set = [ np.vstack(v) for v in temp_set ]

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

		return t

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

	def __init__(self, anomaly_detector, anomaly_class = None, features_number=0, epochs_number=0, level=0, fold=0):
		
		self.anomaly_class = anomaly_class

		if self.anomaly_class == 1:
			self.normal_class = 0
		else:
			self.normal_class = 1

		if anomaly_detector == 'ssv':
			self._anomaly_detector = OneClassSVMAnomalyDetector(kernel='rbf')
		elif anomaly_detector == 'sif':
			self._anomaly_detector = IsolationForestAnomalyDetector(n_estimators=150)
			# self._anomaly_detector = IsolationForestAnomalyDetector(bootstrap=True, behaviour='new', contamination=0.03)
		elif anomaly_detector == 'slo':
			self._anomaly_detector = LocalOutlierFactorAnomalyDetector(algorithm='ball_tree',novelty=True, n_neighbors=100)
			# self._anomaly_detector = LocalOutlierFactorAnomalyDetector(algorithm='ball_tree',n_neighbors=100, novelty=True, contamination=0.03)
		elif anomaly_detector == 'slo1':
			# self._anomaly_detector = LocalOutlierFactorAnomalyDetector(novelty=True, contamination=np.finfo(float).eps)
			self._anomaly_detector = LocalOutlierFactorAnomalyDetector(algorithm='kd_tree',n_neighbors=100, novelty=True)
		elif anomaly_detector == 'slo2':
			# self._anomaly_detector = LocalOutlierFactorAnomalyDetector(novelty=True, contamination=np.finfo(float).eps)
			self._anomaly_detector = LocalOutlierFactorAnomalyDetector(algorithm='brute',n_neighbors=100, novelty=True)
		elif anomaly_detector == 'ssv1':
			self._anomaly_detector = OneClassSVMAnomalyDetector(kernel='linear')
		elif anomaly_detector == 'ssv2':
			self._anomaly_detector = OneClassSVMAnomalyDetector(kernel='poly',degree=3)
		elif anomaly_detector == 'sif1':
			self._anomaly_detector = IsolationForestAnomalyDetector(n_estimators=300)
		# elif anomaly_detector == 'ssv3':
		# 	self._anomaly_detector = OneClassSVMAnomalyDetector(kernel='rbf', gamma=.5)
		# elif anomaly_detector == 'ssv4':
		# 	self._anomaly_detector = OneClassSVMAnomalyDetector(kernel='sigmoid', gamma=.5)
		# elif anomaly_detector == 'smlp':
		# 	from sklearn.neural_network import MLPClassifier
			# self._anomaly_detector = MLPClassifier(activation='relu',hidden_layer_sizes=(100,2))
		elif anomaly_detector.startswith('k'):
			self._anomaly_detector = SklearnKerasWrapper(anomaly_detector, features_number, 2, epochs_number, level, fold)

		self.anomaly_detector = anomaly_detector

	def fit(self, training_set, ground_truth):

		normal_index = [ i for i,gt in enumerate(ground_truth) if gt == self.normal_class ]
		
		# OneHotEncoder does not work with s* anomaly detectors
		t = time.time()
		self._anomaly_detector.fit(training_set[normal_index], ground_truth[normal_index])
		t = time.time() - t
		
		return t

	def predict(self, testing_set):

		# # Only for debugging
		# self._anomaly_detector.set_oracle(self.oracle)

		pred = self._anomaly_detector.predict(testing_set)
		if not self.anomaly_detector.startswith('k') or 'ae' in self.anomaly_detector:
			pred = np.asarray([ self.anomaly_class if p < 0 else self.normal_class for p in pred ])

		return pred

	def predict_proba(self, testing_set):

		if self.anomaly_detector.startswith('k'):
			proba = self._anomaly_detector.predict_proba(testing_set)
		else:
			proba = self._anomaly_detector.decision_function(testing_set)

		return proba

	def set_oracle(self, oracle):

		self.oracle = oracle

		pass

class HierarchicalClassifier(object):
	def __init__(self,input_file,levels_number,features_number,packets_number,classifier_name,detector_name,workers_number,anomaly_class,epochs_number,arbitrary_discr):

		self.input_file = input_file
		self.levels_number = levels_number
		self.features_number = features_number
		self.packets_number = packets_number
		self.classifier_name = classifier_name
		self.detector_name = detector_name
		self.workers_number = workers_number
		self.has_config = False
		self.anomaly_class = anomaly_class
		self.epochs_number = epochs_number
		self.arbitrary_discr = arbitrary_discr

		self.super_learner_default=['ssv','sif']

	def set_config(self, config_name, config):
		self.has_config = True
		self.config_name = config_name
		self.config = config

	def kfold_validation(self, k=10):

		print('\nCaricando '+self.input_file+' con opts -f'+str(self.features_number)+' -c'+self.classifier_name+'\n')
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
				# We had to treat dport as categorical, we can force it ##__##
				attrs_type[2] = u'STRING'
				##__##
				dataset['attributes'] = [ (attr, attr_type) for attr,attr_type in zip(attributes.strip().split(delimiter),attrs_type) ]
				dataset['data'] = [ sample.strip().split(delimiter) for sample in samples ] # TODO gestire i tipi degli attributi
			elif self.input_file.lower().endswith('.pickle'):
				# TODO: provide a method to load dataset from pickle
				pass

		data = np.array(dataset['data'], dtype=object)

		# Next portion will merge all categorical features into one
		###
		# nominal_features_index = [i for i in range(len(dataset['attributes'][:-self.levels_number])) if dataset['attributes'][i][1] != u'NUMERIC']

		# data_nominals = np.ndarray(shape=(data.shape[0],len(nominal_features_index)),dtype=object)
		# for i,nfi in enumerate(nominal_features_index):
		# 	data_nominals[:,i] = data[:,nfi]
		# data_nominal = np.ndarray(shape=(data.shape[0]),dtype=object)
		# for i,datas in enumerate(data_nominals):
		# 	data_nominal[i] = ''.join([ d for d in datas ])

		# data[:,nominal_features_index[0]] = data_nominal

		# data = np.delete(data, nominal_features_index[1:], axis=1).reshape((data.shape[0],data.shape[1]-len(nominal_features_index[1:])))
		###

		self.features_names = [x[0] for x in dataset['attributes']]

		self.attributes_number = data.shape[1]
		self.dataset_features_number = self.attributes_number - self.levels_number

		# If is passed a detector that works woth keras, perform a OneHotEncoding, else proceed with OrdinalEncoder
		nominal_features_index = [i for i in range(len(dataset['attributes'][:-self.levels_number])) if dataset['attributes'][i][1] != u'NUMERIC']
		if self.detector_name.startswith('k'):
			features_encoder = OneHotEncoder()
			for i in nominal_features_index:
				data[:, i] = [ sp.toarray()[0] for sp in features_encoder.fit_transform(data[:, i].reshape(-1,1)) ]
		else:
			features_encoder = OrdinalEncoder()
			if len(nominal_features_index) > 0:
				data[:, nominal_features_index] = features_encoder.fit_transform(data[:, nominal_features_index])


		features_encoder = nominal_features_index = dataset = samples = attributes = None
		del features_encoder, nominal_features_index, dataset, samples, attributes
		gc.collect()

		classifiers_per_fold = []
		# oracles_per_fold = []
		# predictions_per_fold = []
		# probabilities_per_fold = []
		# predictions_per_fold_all = []
		# probabilities_per_fold_all = []

		print('\n***\nStart testing with '+str(k)+'Fold cross-validation -f'+str(self.features_number)+' -c'+self.classifier_name+'\n***\n')

		skf = StratifiedKFold(n_splits=k, shuffle=True)
		fold_cnt = 1

		for train_index, test_index in skf.split(data, data[:,self.attributes_number-1]):
			print(fold_cnt)
			fold_cnt += 1
			self.classifiers = []

			self.training_set = data[train_index, :self.dataset_features_number]
			self.testing_set = data[test_index, :self.dataset_features_number]
			self.ground_truth = data[train_index, self.dataset_features_number:]
			self.oracle = data[test_index, self.dataset_features_number:]
			
			self.prediction = np.ndarray(shape=[len(test_index),self.levels_number],dtype=object)		# Hard Output
			# self.probability = np.ndarray(shape=[len(test_index),self.levels_number],dtype=object)		# Soft Output
			# self.prediction_all = np.ndarray(shape=[len(test_index),self.levels_number],dtype='<U24')
			# self.probability_all = np.ndarray(shape=[len(test_index),self.levels_number],dtype=object)

			root = TreeNode()

			root.fold = fold_cnt-1
			root.train_index = [i for i in range(self.training_set.shape[0])]
			root.test_index = [i for i in range(self.testing_set.shape[0])]
			root.test_index_all = root.test_index
			root.children_tags = list(set(self.ground_truth[root.train_index, root.level]))
			root.children_number = len(root.children_tags)

			self.write_fold(root.level, root.tag)

			root.label_encoder = MyLabelEncoder()
			root.label_encoder.fit(self.ground_truth[root.train_index, root.level])

			if self.has_config and root.tag + '_' + str(root.level + 1) in self.config:
				if 'f' in self.config[root.tag + '_' + str(root.level + 1)]:
					root.features_number = self.config[root.tag + '_' + str(root.level + 1)]['f']
				elif 'p' in self.config[root.tag + '_' + str(root.level + 1)]:
					root.packets_number = self.config[root.tag + '_' + str(root.level + 1)]['p']
				root.classifier_name = self.config[root.tag + '_' + str(root.level + 1)]['c']

				print('\nconfig','tag',root.tag,'level',root.level,'f',root.features_number,\
					'c',root.classifier_name,'train_test_len',len(root.train_index),len(root.test_index))
			else:
				# Assign encoded features number to work with onehotencoder
				root.features_number = self.features_number
				# root.encoded_features_number = self.encoded_dataset_features_number
				root.packets_number = self.packets_number

				# If self.anomaly_class is set and it has two children (i.e. is a binary classification problem)
				# ROOT had to be an Anomaly Detector
				# In this experimentation we force the setting of AD to be a SklearnAnomalyDetector
				if self.anomaly_class != '' and root.children_number == 2:
					root.detector_name = self.detector_name
				else:
					root.classifier_name = self.classifier_name

				print('\nconfig','tag',root.tag,'level',root.level,'f',root.features_number,\
					'c',root.classifier_name,'train_test_len',len(root.train_index),len(root.test_index))

			self.classifiers.append(root)

			if root.children_number > 1:

				if self.anomaly_class != '' and root.children_number == 2:
					classifier_to_call = self._AnomalyDetector
				else:
					classifier_to_call = getattr(self, supported_classifiers[root.classifier_name])
				classifier_to_call(node=root)
				
			else:

				self.unary_class_results_inferring(root)

			# Creating hierarchy recursively
			if root.level < self.levels_number-1 and root.children_number > 0:
				self.recursive(root)

			classifiers_per_fold.append(self.classifiers)

			# oracles_per_fold.append(self.oracle)
			# predictions_per_fold.append(self.prediction)
			# probabilities_per_fold.append(self.probability)
			# predictions_per_fold_all.append(self.prediction_all)
			# probabilities_per_fold_all.append(self.probability_all)

	# for i in range(self.levels_number):

	# 	self.pred_files_per_level.append(material_folder + self.arbitrary_discr + 'multi_' + type_discr + '_level_' + str(i+1) + params_discr + '.dat')
	# 	with open(self.pred_files_per_level[-1], 'w+'):
	# 		pass
	# 	self.proba_files_per_level.append(material_proba_folder + self.arbitrary_discr + 'multi_' + type_discr + '_level_' + str(i+1) + params_discr + '_proba.dat')
	# 	with open(self.proba_files_per_level[-1], 'w+'):
	# 		pass

		for classifier in classifiers_per_fold[0]:

			with open(self.material_features_folder + self.arbitrary_discr + 'multi_' + self.type_discr + '_level_' + str(classifier.level+1) + self.params_discr + '_tag_' + str(classifier.tag) + '_features.dat', 'w+'):
				pass
			with open(self.material_train_durations_folder + self.arbitrary_discr + 'multi_' + self.type_discr + '_level_' + str(classifier.level+1) + self.params_discr + '_tag_' + str(classifier.tag) + '_test_durations.dat', 'w+'):
				pass
			
		with open(self.material_train_durations_folder + self.arbitrary_discr + 'multi_' + self.type_discr + self.params_discr + '_test_durations.dat', 'w+'):
			pass

		for fold_n, classifiers in enumerate(classifiers_per_fold):

			for classifier in classifiers:

				with open(self.material_features_folder + self.arbitrary_discr + 'multi_' + self.type_discr + '_level_' + str(classifier.level+1) + self.params_discr + '_tag_' + str(classifier.tag) + '_features.dat', 'a') as file:

					file.write('@fold\n')
					file.write(self.features_names[classifier.features_index[0]])

					for feature_index in classifier.features_index[1:]:
						file.write(','+self.features_names[feature_index])

					file.write('\n')
				
				with open(self.material_train_durations_folder + self.arbitrary_discr + 'multi_' + self.type_discr + '_level_' + str(classifier.level+1) + self.params_discr + '_tag_' + str(classifier.tag) + '_test_durations.dat', 'a') as file:

					file.write('%.6f\n' % (classifier.test_duration))

		# Retrieve train_durations for each classifier
		test_durations_per_fold = []
		
		for classifiers in classifiers_per_fold:
			test_durations_per_fold.append([])
			for classifier in classifiers:
				test_durations_per_fold[-1].append(classifier.test_duration)

		with open(self.material_train_durations_folder + self.arbitrary_discr + 'multi_' + self.type_discr + self.params_discr + '_test_durations.dat', 'w+') as file:

			mean_parallel_test_duration = np.mean(np.max(test_durations_per_fold, axis=1))
			std_parallel_test_duration = np.std(np.max(test_durations_per_fold, axis=1))

			mean_sequential_test_duration = np.mean(np.sum(test_durations_per_fold, axis=1))
			std_sequential_test_duration = np.std(np.sum(test_durations_per_fold, axis=1))
		 
			file.write('mean_par,std_par,mean_seq,std_seq\n')
			file.write('%.6f,%.6f,%.6f,%.6f\n' % (mean_parallel_test_duration,std_parallel_test_duration,mean_sequential_test_duration,std_sequential_test_duration))

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

	def init_output_files(self):

		folder_discr = self.classifier_name

		if self.has_config:
			folder_discr = self.config_name
		if self.anomaly_class != '':
			folder_discr = self.detector_name

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
		if self.has_config and self.classifier_name:
			if self.features_number != 0:
				feat_discr = '_f_' + str(self.features_number) + feat_discr + '_' + self.classifier_name
			if self.packets_number != 0:
				feat_discr = '_p_' + str(self.packets_number) + feat_discr + '_' + self.classifier_name
		work_discr = '_w_' + str(self.workers_number)
		model_discr = '_e_' + str(self.epochs_number)

		if self.workers_number > 0:
			self.params_discr = work_discr + feat_discr
		if self.epochs_number > 0:
			self.params_discr = model_discr + feat_discr
		else:
			self.params_discr = feat_discr

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

	def write_fold(self, level, tag):

		fouts = ['%s%smulti_%s_level_%s%s.dat' % (self.material_folder,self.arbitrary_discr,self.type_discr,level,self.params_discr),
			'%s%smulti_%s_level_%s%s.dat' % (self.material_proba_folder,self.arbitrary_discr,self.type_discr,level,self.params_discr),
			'%s%smulti_%s_level_%s%s_all.dat' % (self.material_folder,self.arbitrary_discr,self.type_discr,level,self.params_discr),
			'%s%smulti_%s_level_%s%s_all.dat' % (self.material_proba_folder,self.arbitrary_discr,self.type_discr,level,self.params_discr),
			'%s%smulti_%s_level_%s%s_tag_%s.dat' % (self.material_folder,self.arbitrary_discr,self.type_discr,level,self.params_discr,tag),
			'%s%smulti_%s_level_%s%s_tag_%s.dat' % (self.material_proba_folder,self.arbitrary_discr,self.type_discr,level,self.params_discr,tag),
			'%s%smulti_%s_level_%s%s_tag_%s_all.dat' % (self.material_folder,self.arbitrary_discr,self.type_discr,level,self.params_discr,tag),
			'%s%smulti_%s_level_%s%s_tag_%s_all.dat' % (self.material_proba_folder,self.arbitrary_discr,self.type_discr,level,self.params_discr,tag)]
		
		for fout in fouts:
			with open(fout, 'a') as file:
				file.write('@fold\n')

	def write_pred(self, oracle, pred, level, tag, _all=False):

		if not _all:
			fouts = ['%s%smulti_%s_level_%s%s.dat' % (self.material_folder,self.arbitrary_discr,self.type_discr,level,self.params_discr),
				'%s%smulti_%s_level_%s%s_tag_%s.dat' % (self.material_folder,self.arbitrary_discr,self.type_discr,level,self.params_discr,tag)]
		else:
			fouts = ['%s%smulti_%s_level_%s%s_all.dat' % (self.material_folder,self.arbitrary_discr,self.type_discr,level,self.params_discr),
				'%s%smulti_%s_level_%s%s_tag_%s_all.dat' % (self.material_folder,self.arbitrary_discr,self.type_discr,level,self.params_discr,tag)]
		
		for fout in fouts:
			with open(fout, 'a') as file:
				for o, p in zip(oracle, pred):
					file.write('%s %s\n' % (o,p))

	def write_proba(self, oracle, proba, level, tag, _all=False):

		if not _all:
			fouts = ['%s%smulti_%s_level_%s%s.dat' % (self.material_proba_folder,self.arbitrary_discr,self.type_discr,level,self.params_discr),
				'%s%smulti_%s_level_%s%s_tag_%s.dat' % (self.material_proba_folder,self.arbitrary_discr,self.type_discr,level,self.params_discr,tag)]
		else:
			fouts = ['%s%smulti_%s_level_%s%s_all.dat' % (self.material_proba_folder,self.arbitrary_discr,self.type_discr,level,self.params_discr),
				'%s%smulti_%s_level_%s%s_tag_%s_all.dat' % (self.material_proba_folder,self.arbitrary_discr,self.type_discr,level,self.params_discr,tag)]
		
		for fout in fouts:
			with open(fout, 'a') as file:
				for o, p in zip(oracle, proba):
					try:
						file.write('%s %s\n' % (o,' '.join([str(v) for v in p])))
					except:
						file.write('%s %s\n' % (o,p))

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

	def recursive(self,parent):

		for i in range(parent.children_number):
			child = TreeNode()
			self.classifiers.append(child)

			child.level = parent.level+1
			child.fold = parent.fold
			child.tag = parent.children_tags[i]
			child.parent = parent

			child.train_index = [index for index in parent.train_index if self.ground_truth[index, parent.level] == child.tag]
			child.test_index = [index for index in parent.test_index if self.prediction[index, parent.level] == child.tag]

			child.test_index_all = [index for index in parent.test_index_all if self.oracle[index, parent.level] == child.tag]
			child.children_tags = list(set(self.ground_truth[child.train_index, child.level]))

			child.children_number = len(child.children_tags)

			# other_childs_children_tags correspond to the possible labels that in testing phase could arrive to the node
			# potentially_used_labels_index = [index for index in parent.train_index if index not in child.train_index]
			other_childs_children_tags = np.asarray(list(set(self.ground_truth[:, child.level])))

			self.write_fold(child.level, child.tag)

			child.label_encoder = MyLabelEncoder()
			child.label_encoder.fit(np.concatenate((child.children_tags, other_childs_children_tags)))

			if self.has_config and child.tag + '_' + str(child.level + 1) in self.config:
				if 'f' in self.config[child.tag + '_' + str(child.level + 1)]:
					child.features_number = self.config[child.tag + '_' + str(child.level + 1)]['f']
				elif 'p' in self.config[child.tag + '_' + str(child.level + 1)]:
					child.packets_number = self.config[child.tag + '_' + str(child.level + 1)]['p']
				child.classifier_name = self.config[child.tag + '_' + str(child.level + 1)]['c']
				print('config','tag',child.tag,'level',child.level,'f',child.features_number,\
					'c',child.classifier_name,'d',child.detector_name,'train_test_len',len(child.train_index),len(child.test_index))
			else:
				child.features_number = self.features_number
				# child.encoded_features_number = self.encoded_dataset_features_number
				child.packets_number = self.packets_number

				if self.anomaly_class != '' and child.children_number == 2:
					child.detector_name = self.detector_name
				else:
					child.classifier_name = self.classifier_name
				print('config','tag',child.tag,'level',child.level,'f',child.features_number,\
					'c',child.classifier_name,'train_test_len',len(child.train_index),len(child.test_index))

			self.classifiers[self.classifiers.index(parent)].children[child.tag]=child

			if child.children_number > 1 and len(child.test_index) > 0:

				if self.anomaly_class != '' and child.children_number == 2:
					classifier_to_call = self._AnomalyDetector
				else:
					classifier_to_call = getattr(self, supported_classifiers[child.classifier_name])
				classifier_to_call(node=child)

			else:

				self.unary_class_results_inferring(child)

			if child.level < self.levels_number-1 and child.children_number > 0:
				self.recursive(child)

	def unary_class_results_inferring(self, node):

		node.test_duration = 0.
		node.features_index = node.parent.features_index

		self.write_pred(self.oracle[node.test_index, node.level], [node.tag] * len(node.test_index), node.level, node.tag)
		self.write_pred(self.oracle[node.test_index_all, node.level], [node.tag] * len(node.test_index_all), node.level, node.tag, True)

		if self.anomaly_class == '':
			proba_base = (1.,1.,1.)
			for index in node.test_index:
				self.prediction[index, node.level] = node.tag
			# 	self.probability[index, node.level] = (1., 1., 1.)
			# for index in node.test_index_all:
			# 	self.prediction_all[index, node.level] = node.tag
			# 	self.probability_all[index, node.level] = (1., 1., 1.)
		else:
			proba_base = 0.
			for index in node.test_index:
				self.prediction[index, node.level] = node.tag
			# 	self.probability[index, node.level] = 0.
			# for index in node.test_index_all:
			# 	self.prediction_all[index, node.level] = node.tag
			# 	self.probability_all[index, node.level] = 0.

		self.write_proba(self.oracle[node.test_index, node.level], [proba_base] * len(node.test_index), node.level, node.tag)
		self.write_proba(self.oracle[node.test_index_all, node.level], [proba_base] * len(node.test_index_all), node.level, node.tag, True)

	def Sklearn_RandomForest(self, node):

		# Instantation
		classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)
		# Features selection
		node.features_index = self.features_selection(node)
		self.train(node, classifier)
		self.test(node, classifier)
		self.test_all(node, classifier)

	def Sklearn_CART(self, node):

		# Instantation
		classifier = DecisionTreeClassifier()
		# Features selection
		node.features_index = self.features_selection(node)
		self.train(node, classifier)
		self.test(node, classifier)
		self.test_all(node, classifier)

	def Spark_Classifier(self, node):

		# Instantation
		classifier = SklearnSparkWrapper(self.classifier_name)
		# Features selection
		node.features_index = self.features_selection(node)
		self.train(node, classifier)
		classifier.set_oracle(
			node.label_encoder.transform(self.oracle[node.test_index][:, node.level])
			)
		self.test(node, classifier)

	def Spark_RandomForest(self, node):
		return self.Spark_Classifier(node)
	
	def Spark_NaiveBayes(self, node):
		return self.Spark_Classifier(node)
	
	def Spark_MultilayerPerceptron(self, node):
		return self.Spark_Classifier(node)
	
	def Spark_GBT(self, node):
		return self.Spark_Classifier(node)
	
	def Spark_DecisionTree(self, node):
		return self.Spark_Classifier(node)

	def Weka_Classifier(self, node):

		# Instantation
		classifier = SklearnWekaWrapper(self.classifier_name)
		# Features selection
		node.features_index = self.features_selection(node)
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
		classifier = SuperLearnerClassifier(self.super_learner_default, node.children_number, node.label_encoder.transform(self.anomaly_class)[0], node.features_number)
		# Features selection
		node.features_index = self.features_selection(node)
		self.train(node, classifier)
		classifier.set_oracle(
			node.label_encoder.transform(self.oracle[node.test_index][:, node.level])
			)
		self.test(node, classifier)

		self.test_all(node, classifier)

	def _AnomalyDetector(self, node):

		# Instantation
		classifier = AnomalyDetector(node.detector_name, node.label_encoder.transform(self.anomaly_class)[0], node.features_number, epochs_number, node.level, node.fold)
		# Features selection
		node.features_index = self.features_selection(node)
		self.train(node, classifier)
		# classifier.set_oracle(
		# 	node.label_encoder.transform(self.oracle[node.test_index][:, node.level])
		# 	)
		self.test(node, classifier)
		# classifier.set_oracle(
		# 	node.label_encoder.transform(self.oracle[node.test_index_all][:, node.level])
		# 	)
		self.test_all(node, classifier)

		classifier = None
		del classifier
		gc.collect()

	# @profile
	def train(self, node, classifier):

		print(node.label_encoder.nom_to_cat)

		node.test_duration = classifier.fit(
			self.training_set[node.train_index][:, node.features_index],
			node.label_encoder.transform(self.ground_truth[node.train_index][:, node.level])
			)

	# @profile
	def test(self, node, classifier):

		pred = node.label_encoder.inverse_transform(
			classifier.predict(self.testing_set[node.test_index][:, node.features_index]))
		proba = classifier.predict_proba(self.testing_set[node.test_index][:, node.features_index])

		for p, b, i in zip(pred, proba, node.test_index):
			self.prediction[i, node.level] = p
		# 	self.probability[i, node.level] = b

		self.write_pred(self.oracle[node.test_index, node.level], pred, node.level, node.tag)
		self.write_proba(self.oracle[node.test_index, node.level], proba, node.level, node.tag)

		# exit(1)

	def test_all(self, node, classifier):

		pred = node.label_encoder.inverse_transform(
			classifier.predict(self.testing_set[node.test_index_all][:, node.features_index]))
		proba = classifier.predict_proba(self.testing_set[node.test_index_all][:, node.features_index])

		# for p, b, i in zip(pred, proba, node.test_index_all):
		# 	self.prediction_all[i, node.level] = p
		# 	self.probability_all[i, node.level] = b

		self.write_pred(self.oracle[node.test_index_all, node.level], pred, node.level, node.tag, True)
		self.write_proba(self.oracle[node.test_index_all, node.level], proba, node.level, node.tag, True)

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
		'wsl': 'Weka_SuperLearner'
	}

	kdaes = []
	kdae_ns = []
	for i in range(5):
		for j in range(5):
			kdaes.append('kdae_%s_%s' % (i,j))
			kdae_ns.append('Keras_DeepAutoencoder_%s_%s' % (i,j))

	supported_detectors = {
		'ssv': 'Sklearn_OC-SVM',
		'sif': 'Sklearn_IsolationForest',		# C4.5
		'slo': 'Sklearn_LocalOutlierFactor',
		'ssv1': 'Sklearn_OC-SVM',
		'sif1': 'Sklearn_IsolationForest',		# C4.5
		'slo1': 'Sklearn_LocalOutlierFactor',
		'ssv2': 'Sklearn_OC-SVM',
		'slo2': 'Sklearn_LocalOutlierFactor',
		kdaes[0]: kdae_ns[0],
		kdaes[1]: kdae_ns[1],
		kdaes[2]: kdae_ns[2],
		kdaes[3]: kdae_ns[3],
		kdaes[4]: kdae_ns[4],
		kdaes[5]: kdae_ns[5],
		kdaes[6]: kdae_ns[6],
		kdaes[7]: kdae_ns[7],
		kdaes[8]: kdae_ns[8],
		kdaes[9]: kdae_ns[9],
		kdaes[10]: kdae_ns[10],
		kdaes[11]: kdae_ns[11],
		kdaes[12]: kdae_ns[12],
		kdaes[13]: kdae_ns[13],
		kdaes[14]: kdae_ns[14],
		kdaes[15]: kdae_ns[15],
		kdaes[16]: kdae_ns[16],
		kdaes[17]: kdae_ns[17],
		kdaes[18]: kdae_ns[18],
		kdaes[19]: kdae_ns[19],
		kdaes[20]: kdae_ns[20],
		kdaes[21]: kdae_ns[21],
		kdaes[22]: kdae_ns[22],
		kdaes[23]: kdae_ns[23],
		kdaes[24]: kdae_ns[24]
	}

	input_file = ''
	levels_number = 0
	features_number = 0
	packets_number = 0
	classifier_name = 'srf'
	detector_name = 'ssv'
	workers_number = 0
	executors_number = 0
	epochs_number = 0
	anomaly_class = ''

	config_file = ''
	config = ''
	arbitrary_discr = ''

	try:
		opts, args = getopt.getopt(
			sys.argv[1:], "hi:n:f:p:c:o:w:x:a:e:d:s:",
				"[input_file=,levels_number=,features_number=,packets_number=,\
				classifier_name=,configuration=,workers_number=,executors_number=,anomaly_class=,\
				epochs_number=,detector_name=,arbitrary_discr=]"
			)
	except getopt.GetoptError:
		print(sys.argv[0],'-i <input_file> -n <levels_number> \
			(-f <features_number>|-p <packets_number>) -c <classifier_name> (-o <configuration_file>) \
			(-w <workers_number> -x <executors_number>) (-a <anomaly_class> -e epochs_number -d detector_name) (-s arbitrary_discr)')
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
			for sd in supported_detectors:
				print('\t\t-c '+sd+'\t--->\t'+supported_detectors[sd].split('_')[1]+'\t\timplemented in '+supported_detectors[sd].split('_')[0])
			sys.exit()
		if opt in ("-i", "--input_file"):
			input_file = arg
		if opt in ("-n", "--levels_number"):
			levels_number = int(arg)
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
			anomaly_class = arg
		if opt in ("-e", "--epochs_number"):
			epochs_number = int(arg)
		if opt in ("-d", "--detector_name"):
			detector_name = arg
		if opt in ("-s", "--arbitrary_discr"):
			arbitrary_discr = arg

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

	else:

		if packets_number != 0 and features_number != 0 or packets_number == features_number:
			print('-f and -p option should not be used together')
			sys.exit()

		if classifier_name not in supported_classifiers:
			print('Classifier not supported\nList of available classifiers:\n')
			for sc in supported_classifiers:
				print('-c '+sc+'\t--->\t'+supported_classifiers[sc].split('_')[1]+'\t\timplemented in '+supported_classifiers[sc].split('_')[0])
			sys.exit()

		if anomaly_class != '' and detector_name not in supported_detectors:
			print('Detector not supported\nList of available detectors:\n')
			for sd in supported_detectors:
				print('\t\t-c '+sd+'\t--->\t'+supported_detectors[sd].split('_')[1]+'\t\timplemented in '+supported_detectors[sd].split('_')[0])
			sys.exit()

		import_str = classifier_name[0]

	if levels_number == 0:
		print('Number of level must be positive and non zero')
		sys.exit()

	if not input_file.endswith('.arff') and not input_file.endswith('.csv'):
		print('input files must be .arff or .csv')
		sys.exit()

	import_sklearn()
	import_keras()

	if 'd' in import_str:
		import_spark()

		conf = SparkConf().setAll([
			('spark.app.name', '_'.join(sys.argv)),
			('spark.driver.cores', 30),
			('spark.driver.memory','30g'),
			('spark.executor.memory', '3000m'),
			('spark.master', 'spark://192.168.200.24:7077'),
			('spark.executor.cores', str(executors_number)),
			('spark.cores.max', str(executors_number * workers_number))
		])
		sc = SparkContext(conf=conf)
		sc.setLogLevel("ERROR")
		spark = SparkSession(sc)

	# Momentarily SuperLearner works with weka models
	if 'w' in import_str:
		import_weka()

		jvm.start()

	# For reproducibility
	from tensorflow import set_random_seed
	set_random_seed(0)
	np.random.seed(0)

	hierarchical_classifier = HierarchicalClassifier(
		input_file=input_file,
		levels_number=levels_number,
		features_number=features_number,
		packets_number=packets_number,
		classifier_name=classifier_name,
		detector_name=detector_name,
		workers_number=workers_number,
		anomaly_class=anomaly_class,
		epochs_number=epochs_number,
		arbitrary_discr=arbitrary_discr
		)
	
	if config:
		config_name = config_file.split('.')[0]
		hierarchical_classifier.set_config(config_name, config)

	hierarchical_classifier.init_output_files()
	hierarchical_classifier.kfold_validation(k=10)

	if 'w' in import_str:
		jvm.stop()
