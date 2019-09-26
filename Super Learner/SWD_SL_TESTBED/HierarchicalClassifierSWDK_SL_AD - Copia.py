#!/usr/bin/python2

import arff
import numpy as np
import sys
import getopt
import os, time
from copy import copy

import networkx as nx

from importlib import import_module

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def import_sklearn():

	global RandomForestClassifier,DecisionTreeClassifier,SelectKBest,mutual_info_classif,\
		StratifiedKFold,train_test_split,OrdinalEncoder,OneHotEncoder,MyLabelEncoder,MinMaxScaler,KernelDensity

	global OneClassSVMAnomalyDetector,IsolationForestAnomalyDetector,LocalOutlierFactorAnomalyDetector,\
		EllipticEnvelopeAnomalyDetector

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
		Reshape,BatchNormalization,LSTM,AveragePooling1D,MaxPooling1D,UpSampling1D,Activation,relu,PReLU,\
		EarlyStopping
	
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
	PReLU = import_module('keras.layers').PReLU
	EarlyStopping = import_module('keras.callbacks').EarlyStopping

class TreeNode(object):
	def __init__(self):

		# Position info
		self.parent = None
		self.children = {}

		# Node info
		self.tag = 'ROOT'
		self.level = 0				# 0-based
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

	def __init__(self, classifier_name, features_number, num_classes, epochs_number=0):

		self.classifier_name = classifier_name

		self._scaler = MinMaxScaler()

		# To use middle layer features as anomaly score
		self.ad_enc = False
		self.de_enc = False
		# Encoded input plus MSE per sample
		self.ad_enc_mse = False
		self.de_enc_mse = False
		# Using MSE as threshold
		self.global_mse = True
		self.mse_per_feat = False

		if self.ad_enc or self.ad_enc_mse:
			self._anomaly_detector = OneClassSVMAnomalyDetector(kernel='linear')
		if self.de_enc or self.de_enc_mse:
			self._density_estimator = KernelDensity()

		self.epochs_number = epochs_number

		self.proba = None

	def fit(self, training_set, ground_truth, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1):

		# training_set, features_number = self._expand_onehot_features(training_set)

		self._init_model(self.classifier_name, features_number)

		# t = time.time()
		# self._model.fit(np.expand_dims(training_set, axis=3),ground_truth,epochs=5,batch_size=50,verbose=2)
		# t = time.time() - t

		# Useless for now: no mke changes
		# training_set = training_set.reshape((len(training_set), np.prod(training_set.shape[1:])))

		# Scaling training set for Autoencoder
		scaled_training_set = self._scaler.fit_transform(training_set)
		# Divide trainint set into training and validation sets
		scaled_training_set, scaled_validation_set, _, _ = train_test_split(scaled_training_set, ground_truth, train_size=0.9)

		t = time.time()
		# Train Autoencoder
		self._model.fit(scaled_training_set, scaled_training_set,
			epochs=self.epochs_number,
			batch_size=512,
			shuffle=True,
			verbose=1,
			validation_data=(scaled_validation_set, scaled_validation_set),
			callbacks = [ self._earlystopping ]
			)
		# Reconstr input -> out of Ouput layer of AE
		reconstr_scaled_training_set = self._model.predict(scaled_training_set)
		# To use global or per feature mse statistics as anomaly score
		self.mean_normal_mse = np.mean(np.square(scaled_training_set - reconstr_scaled_training_set))
		self.p_normal_mse = np.percentile(np.square(scaled_training_set - reconstr_scaled_training_set),[50,75,90,92.5,95,97.5,98,99])
		t = time.time() - t
		# # Encoded input -> out of Middle layer of AE
		# encoded_scaled_training_set = self._encoder.predict(scaled_training_set)
		# # import matplotlib.pyplot as plt
		# # plt.clf()
		# # plt.plot(encoded_scaled_training_set[:,0],encoded_scaled_training_set[:,1],'ro')
		# # plt.savefig('normal_distr_'+str(t)+'.png')
		# # mse per sample, it could be used as feature for anomaly detector or density estimator
		# self.normal_mse = np.square(scaled_training_set - reconstr_scaled_training_set)
		# # Adding mse to encoded features
		# encoded_scaled_training_set_mse = np.asarray([ np.append(est, self.normal_mse[i]) for i,est in enumerate(encoded_scaled_training_set) ])
		
		# if self.ad_enc:
		# 	self._anomaly_detector.fit(encoded_scaled_training_set)
		# if self.de_enc:
		# 	self._density_estimator.fit(encoded_scaled_training_set)
		# if self.ad_enc_mse:
		# 	self._anomaly_detector.fit(encoded_scaled_training_set_mse)
		# if self.de_enc_mse:
		# 	self._density_estimator.fit(encoded_scaled_training_set_mse)
		# ###
		# self.p_normal_mse_per_features = np.percentile(np.square(scaled_training_set - reconstr_scaled_training_set),[75,80,90],axis=0)
		# ###

		# if self.de_enc or self.de_enc_mse:
		# 	normal_density = self._density_estimator.transform(encoded_scaled_training_set)
		# 	self.p_normal_density = np.percentile(normal_density,[50,75,90,92.5,95])

		print('time',t)

		# decoded_data = self._decoder.predict(encoded_data)

		# for e,i in zip(encoded_data,range(10)):
		# 	print(e)

		# for t,d,i in zip(training_set,decoded_data,range(1)):
		# 	print(t,d)

		return t

	def _expand_onehot_features(self, set):

		# nominal_features_index = [0,1,2,5]
		nominal_features_index = [ i for i,v in enumerate(set[0,:]) if isinstance(v, np.ndarray) ]

		bias = 0
		for i in nominal_features_index:
			# print(i+bias,data[0,i+bias])
			extend_to = len(set[0,i+bias]) - 1
			# print(len(zip(*data[:,i+bias])))
			set = np.c_[ set[:,:i+bias], np.asarray(zip(*set[:,i+bias])).T, set[:,i+bias+1:] ]
			# print(data.shape)
			bias += copy(extend_to)

		return set, set.shape[1]

	def _init_model(self, classifier_name, features_number):

		encoding_dim = 4

		if classifier_name == 'kae':

			input_data = Input(shape=(features_number,))

			encoded = Dense(encoding_dim, activation='relu')(input_data)

			decoded = Dense(features_number, activation='sigmoid')(encoded)

			# For fitting
			self._model = autoencoder = Model(input_data, decoded)
			# For feature extraction
			self._encoder = encoder = Model(input_data, encoded)

			# Decoder Model, not used
			# encoded_input = Input(shape=(encoding_dim,))
			# decoder_layer = autoencoder.layers[-1]
			# decoder = Model(encoded_input, decoder_layer(encoded_input))

			self._model.compile(optimizer='adadelta', loss='binary_crossentropy')
		if classifier_name == 'kdae':

			input_data = Input(shape=(features_number,))

			x = Dense(16, activation='relu')(input_data)
			x = Dense(8, activation='relu')(x)
			x = Dense(4, activation='relu')(x)
			encoded = Dense(encoding_dim, activation='relu')(x)

			x = Dense(4, activation='relu')(encoded)
			x = Dense(8, activation='relu')(encoded)
			x = Dense(16, activation='relu')(x)
			decoded = Dense(features_number, activation='sigmoid')(x)

			# x = Dense(16)(input_data)
			# x = PReLU()(x)
			# x = Dense(8)(x)
			# x = PReLU()(x)
			# x = Dense(4)(x)
			# x = PReLU()(x)
			# encoded = Dense(encoding_dim, activation='relu')(x)

			# x = Dense(4)(encoded)
			# x = PReLU()(x)
			# x = Dense(8)(encoded)
			# x = PReLU()(x)
			# x = Dense(16)(x)
			# x = PReLU()(x)
			# decoded = Dense(features_number, activation='sigmoid')(x)

			# For fitting
			self._model = autoencoder = Model(input_data, decoded)
			# For feature extraction
			self._encoder = encoder = Model(input_data, encoded)
			self._model.compile(optimizer='adadelta', loss='binary_crossentropy')
		if classifier_name == 'k1dcnnae':

			input_data = Input(shape=(features_number,))

			x = Reshape((1,features_number,))(input_data)
			# x = Conv1D(256, 3, activation='relu', padding='same')(x)
			# x = MaxPooling1D(encoding_dim, padding='same')(x)
			# x = Conv1D(128, 3, activation='relu', padding='same')(x)
			# x = MaxPooling1D(encoding_dim, padding='same')(x)
			x = Conv1D(64, 3, activation='relu', padding='same')(x)
			x = MaxPooling1D(encoding_dim, padding='same')(x)
			x = Conv1D(32, 3, activation='relu', padding='same')(x)
			x = MaxPooling1D(encoding_dim, padding='same')(x)
			x = Conv1D(16, 3, activation='relu', padding='same')(x)
			x = MaxPooling1D(encoding_dim, padding='same')(x)
			x = Conv1D(8, 3, activation='relu', padding='same')(x)
			x = MaxPooling1D(encoding_dim, padding='same')(x)
			x = Flatten()(x)
			encoded = Dense(encoding_dim, activation='relu')(x)

			x = Reshape((1,encoding_dim,))(encoded)
			x = Conv1D(8, 3, activation='relu', padding='same')(x)
			x = UpSampling1D(encoding_dim)(x)
			x = Conv1D(16, 3, activation='relu', padding='same')(x)
			x = UpSampling1D(encoding_dim)(x)
			x = Conv1D(32, 3, activation='relu', padding='same')(x)
			x = UpSampling1D(encoding_dim)(x)
			x = Conv1D(64, 3, activation='relu', padding='same')(x)
			x = UpSampling1D(encoding_dim)(x)
			# x = Conv1D(128, 3, activation='relu', padding='same')(x)
			# x = UpSampling1D(encoding_dim)(x)
			# x = Conv1D(256, 3, activation='relu', padding='same')(x)
			# x = UpSampling1D(encoding_dim)(x)
			x = Flatten()(x)
			x = Dense(features_number, activation='sigmoid')(x)
			decoded = Reshape((features_number,))(x)

			# For fitting
			self._model = autoencoder = Model(input_data, decoded)
			# For feature extraction
			self._encoder = encoder = Model(input_data, encoded)
			self._model.compile(optimizer='adadelta', loss='binary_crossentropy')
		if classifier_name == 'kreviewae':

			input_data = Input(shape=(features_number,))

			x = Dense(features_number, activation='tanh')(input_data)
			x = Dense(512, activation='tanh')(x)
			x = Dense(256, activation='tanh')(x)
			x = Dense(128, activation='tanh')(x)
			encoded = Dense(encoding_dim, activation='tanh')(x)

			x = Dense(encoding_dim)(encoded)
			x = Dense(512)(x)
			x = Dense(256, activation='relu')(x)
			x = Dense(128)(x)
			x = Dense(64)(x)
			decoded = Dense(features_number, activation='relu')(x)

			# For fitting
			self._model = autoencoder = Model(input_data, decoded)
			# For feature extraction
			self._encoder = encoder = Model(input_data, encoded)
			self._model.compile(optimizer='adadelta', loss='binary_crossentropy')
		if classifier_name == 'kdae+k1dcnnae':

			input_data = Input(shape=(features_number,))

			x = Dense(16, activation='relu')(input_data)
			x = Dense(8, activation='relu')(x)
			x = Reshape((1,8,))(x)
			x = Conv1D(16, 3, activation='relu', padding='same')(x)
			x = MaxPooling1D(encoding_dim, padding='same')(x)
			x = Conv1D(8, 3, activation='relu', padding='same')(x)
			x = MaxPooling1D(encoding_dim, padding='same')(x)
			x = Flatten()(x)
			encoded = Dense(encoding_dim, activation='relu')(x)

			x = Reshape((1,encoding_dim,))(encoded)
			x = Conv1D(8, 3, activation='relu', padding='same')(x)
			x = UpSampling1D(encoding_dim)(x)
			x = Conv1D(16, 3, activation='relu', padding='same')(x)
			x = UpSampling1D(encoding_dim)(x)
			x = Flatten()(x)
			x = Dense(8, activation='relu')(x)
			x = Dense(16, activation='relu')(x)
			decoded = Dense(features_number, activation='sigmoid')(x)

			# For fitting
			self._model = autoencoder = Model(input_data, decoded)
			# For feature extraction
			self._encoder = encoder = Model(input_data, encoded)
			self._model.compile(optimizer='adadelta', loss='binary_crossentropy')

		self._earlystopping = EarlyStopping(monitor='val_loss')

	def predict(self, testing_set):

		testing_set,_ = self._expand_onehot_features(testing_set)

		scaled_testing_set = self._scaler.transform(testing_set)

		# To use mse as anomaly score
		reconstr_scaled_testing_set = self._model.predict(scaled_testing_set)
		encoded_scaled_testing_set = self._encoder.predict(scaled_testing_set)

		# normal_index = [ i for i,v in enumerate(self.oracle) if v == 1 ]
		# anomaly_index = [ i for i,v in enumerate(self.oracle) if v == 0 ]

		# t=time.time()

		# import matplotlib.pyplot as plt
		# plt.clf()
		# plt.plot(encoded_scaled_testing_set[normal_index,0],encoded_scaled_testing_set[normal_index,1],'bo')
		# plt.plot(encoded_scaled_testing_set[anomaly_index,0],encoded_scaled_testing_set[anomaly_index,1],'ro')
		# plt.axis([0,np.max(encoded_scaled_testing_set[:,0]),0,np.max(encoded_scaled_testing_set[:,1])])
		# plt.savefig('enc_distr_'+str(t)+'.png')

		testing_mse = np.square(scaled_testing_set - reconstr_scaled_testing_set)

		encoded_scaled_testing_set_mse = np.asarray([ np.append(est, testing_mse[i]) for i,est in enumerate(encoded_scaled_testing_set) ])

		if self.ad_enc:
			encoded_scaled_pred = self._anomaly_detector.predict(encoded_scaled_testing_set)
		if self.de_enc:
			encoded_scaled_proba = self._density_estimator.score_samples(encoded_scaled_testing_set)
		if self.ad_enc_mse:
			encoded_scaled_pred = self._anomaly_detector.predict(encoded_scaled_testing_set_mse)
		if self.de_enc_mse:
			encoded_scaled_proba = self._density_estimator.score_samples(encoded_scaled_testing_set_mse)

		# To use anomaly detector instead density estimator
		if self.ad_enc or self.ad_enc_mse:
			pred = encoded_scaled_pred
		# To use density estimator instead anomaly detector
		if self.de_enc or self.de_enc_mse:
			pred = np.asarray([ -1 if np.exp(pb) < self.p_normal_density[4] else 1 for pb in encoded_scaled_proba ], dtype=int)

		# Single mse per each prediction
		if self.global_mse:
			self.proba = mse_per_sample = np.mean(np.square(scaled_testing_set-reconstr_scaled_testing_set), axis=1)
			pred = np.asarray([ -1 if mse > self.p_normal_mse[0] else 1 for mse in mse_per_sample ])
			print(self.p_normal_mse)
			print(np.percentile(mse_per_sample,[50,75,90,92.5,95,97.5,98,99]))

			# import matplotlib.pyplot as plt
			# plt.clf()
			# plt.plot(np.percentile(mse_per_sample,[50,75,90,92.5,95,97.5,98,99]),'ro')
			# plt.savefig('mse_per_sample.png')
			input()
		#An mse per each feature of a prediction
		if self.mse_per_feat:
			mse_per_sample_per_feature = np.square(scaled_testing_set-reconstr_scaled_testing_set)
			pred_per_feature = []
			for mse_per_feature in mse_per_sample_per_feature:
				pred_per_feature.append([ -1 if mse > self.p_normal_mse_per_features[1][i] else 1 for i,mse in enumerate(mse_per_feature) ])
			pred = np.asarray([ -1 if np.sum(p) <= 0 else 1 for p in pred_per_feature ])

		# low_ = [ np.exp(pb) for i,pb in enumerate(proba) if self.oracle[i] == 0 ]
		# high_ = [ np.exp(pb) for i,pb in enumerate(proba) if self.oracle[i] == 1 ]

		# print(np.percentile(low_,[0,5,25,50,75,95,100]),np.mean(low_))
		# print(np.percentile(high_,[0,5,25,50,75,95,100]),np.mean(high_))

		# print(np.unique(n_pred,return_counts=True))

		# permutation = np.random.permutation(pred.shape[0])

		# for p,n_p,o,i in zip(pred[permutation],n_pred[permutation],self.oracle[permutation],range(10)):
		# 	print(p,n_p,o)
		# exit(0)

		# i=0
		# oo=1

		# permutation = np.random.permutation(preds.shape[0])

		# for o,pd,pb in zip(self.oracle[permutation],preds[permutation],proba[permutation]):
		# 	if o == oo:
		# 		print(o,pd,np.exp(pb))
		# 		i+=1
		# 		if i == 10:
		# 			break

		# 		if oo == 0:
		# 			oo = 1
		# 		else:
		# 			oo = 0

		# exit(0)

		# preds = self._model.predict_classes(np.expand_dims(testing_set, axis=3))

		# from keras import backend as K

		# get_prelast_layer_output = K.function([self._model.layers[0].input],
		# 	[self._model.layers[-2].output])

		# get_last_layer_output = K.function([self._model.layers[0].input],
		# 	[self._model.layers[-1].output])

		# f=open('out','w')
		# f.write(get_prelast_layer_output([np.expand_dims(testing_set, axis=3)])[0])
		# f.close()

		# prelast_layer_output=get_prelast_layer_output([np.expand_dims(testing_set, axis=3)])[0]
		# # last_layer_output=get_last_layer_output([np.expand_dims(testing_set, axis=3)])[0]

		# print(type(prelast_layer_output),prelast_layer_output.shape,np.unique(prelast_layer_output, return_counts=True, axis=0))
		# preds = np.mean(prelast_layer_output, axis=1)

		# counter={}
		# for pred in preds:
		# 	for c in range(10):
		# 		if float(c)/10.0 <= pred < float(c+1)/10.0:
		# 			tag = '-'.join([str(float(c)/10.0),str(float(c+1)/10.0)])
		# 			if tag not in counter:
		# 				counter[tag] = 0
		# 			counter[tag] += 1
		
		# print(counter)
		# # print(type(last_layer_output),last_layer_output.shape,np.unique(last_layer_output, return_counts=True))
		# raw_input()

		pred = np.reshape(pred,pred.shape[0])

		return pred

	def predict_proba(self, testing_set):

		if self.proba is None:
			self.predict(testing_set)

		return self.proba

	def set_oracle(self, oracle):

		self.oracle = oracle

	def _sklearn2keras(self, features, labels=None):

		pass

class SuperLearnerClassifier(object):
	def __init__(self, first_level_learners, num_classes, anomaly_class=None, features_number=0):

		self.num_classes = num_classes
		self.first_level_classifiers = []

		for first_level_learner in first_level_learners:
			# if first_level_learner.startswith('w'):
			# 	self.first_level_classifiers.append(SklearnWekaWrapper(first_level_learner))
			# if first_level_learner.startswith('d'):
			# 	self.first_level_classifiers.append(SklearnSparkWrapper(first_level_learner))
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

	def __init__(self, anomaly_detector, anomaly_class = None, features_number=0, epochs_number=0):
		
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
			self._anomaly_detector = LocalOutlierFactorAnomalyDetector(novelty=True, n_neighbors=100)
			# self._anomaly_detector = LocalOutlierFactorAnomalyDetector(algorithm='ball_tree',n_neighbors=100, novelty=True, contamination=0.03)
		elif anomaly_detector == 'slo1':
			# self._anomaly_detector = LocalOutlierFactorAnomalyDetector(novelty=True, contamination=np.finfo(float).eps)
			self._anomaly_detector = LocalOutlierFactorAnomalyDetector(algorithm='kd_tree',n_neighbors=100, novelty=True, contamination=0.03)
		elif anomaly_detector == 'slo2':
			# self._anomaly_detector = LocalOutlierFactorAnomalyDetector(novelty=True, contamination=np.finfo(float).eps)
			self._anomaly_detector = LocalOutlierFactorAnomalyDetector(algorithm='brute',n_neighbors=100, novelty=True, contamination=0.03)
		elif anomaly_detector == 'ssv1':
			self._anomaly_detector = OneClassSVMAnomalyDetector(kernel='linear')
		elif anomaly_detector == 'ssv2':
			self._anomaly_detector = OneClassSVMAnomalyDetector(kernel='poly',degree=2)
		elif anomaly_detector == 'ssv3':
			self._anomaly_detector = OneClassSVMAnomalyDetector(kernel='rbf', gamma=.5)
		elif anomaly_detector == 'ssv4':
			self._anomaly_detector = OneClassSVMAnomalyDetector(kernel='sigmoid', gamma=.5)
		elif anomaly_detector == 'smlp':
			from sklearn.neural_network import MLPClassifier
			self._anomaly_detector = MLPClassifier(activation='relu',hidden_layer_sizes=(100,2))
		elif anomaly_detector.startswith('k'):
			self._anomaly_detector = SklearnKerasWrapper(anomaly_detector, features_number, 2, epochs_number)

		self.anomaly_detector = anomaly_detector

		# Decomment to apply preclustering to Normal traffic ##_##
		# Clustering does not work with OneHotEncoder
		# self._anomaly_detectors = []
		# self.n_clusters = 1
		# from copy import copy
		# for i in range(self.n_clusters):
		# 	self._anomaly_detectors.append(copy(self._anomaly_detector))
		# 	# self._anomaly_detectors.append(self._anomaly_detector)
		##_##

	def fit(self, training_set, ground_truth):

		normal_index = [ i for i,gt in enumerate(ground_truth) if gt == self.normal_class ]
		
		# Decomment to apply preclustering to Normal traffic ##_##
		# from sklearn.cluster import KMeans
		# clustering = KMeans(n_clusters=self.n_clusters)
		# cluster_index = clustering.fit_predict(training_set[normal_index])

		# t = []

		# for i in range(self.n_clusters):
		# 	red_normal_index = [ v for j,v in enumerate(normal_index) if cluster_index[j] == i ]
		# 	if self.anomaly_detector.startswith('k'):
		# 		t.append(self._anomaly_detectors[i].fit(training_set[red_normal_index], ground_truth[red_normal_index]))
		# 	else:
		# 		t.append(time.time())
		# 		self._anomaly_detectors[i].fit(training_set[red_normal_index], ground_truth[red_normal_index])
		# 		t[-1] = time.time() - t[-1]
		# return np.sum(t)
		##_##

		# OneHotEncoder does not work with s* anomaly detectors
		t = time.time()
		self._anomaly_detector.fit(training_set[normal_index], ground_truth[normal_index])
		t = time.time() - t
		
		return t

		# types={}

		# for i,t in enumerate(training_set[normal_index]):
		# 	for j,tt in enumerate(t):
		# 		# if j in [0,1,5]:
		# 		# 	print(tt)
		# 		if type(tt) not in types:
		# 			types[type(tt)] = []
		# 		types[type(tt)].append(j)

		# for t in types:
		# 	print(t, np.unique(types[t], return_counts=True))

		# print('types', types)

		# self._anomaly_detector_IF = []

		# for i in range(1):

		# 	n = 100+i*100

		# 	self._anomaly_detector_IF.append(IsolationForestAnomalyDetector(n_estimators=n, behaviour='new', contamination=0.))

		# 	t=time.time()
		# 	pred_IF = self._anomaly_detector_IF[-1].fit_predict(training_set[normal_index])
		# 	t=time.time()-t
					
		# 	pred_IF = np.asarray([ self.anomaly_class if p < 0 else self.normal_class for p in pred_IF ])

		# 	# print(len(ground_truth[normal_index]), np.unique(ground_truth[normal_index],return_counts=True))
		# 	# print(len(pred_IF), np.unique(pred_IF,return_counts=True))

		# 	print('Train IsolationForest',n,f1_score(ground_truth[normal_index],pred_IF,average='macro'))
		# # print('\n')

		# self._anomaly_detector_LOF = []

		# for i in range(1):

		# 	n = 20+i*10

		# 	self._anomaly_detector_LOF.append(LocalOutlierFactorAnomalyDetector(n_neighbors=n, novelty=True, contamination=np.finfo(float).eps))
						
		# 	t=time.time()
		# 	self._anomaly_detector_LOF[-1].fit(training_set[normal_index])
		# 	t=time.time()-t

		# 	pred_LOF = self._anomaly_detector_LOF[-1].predict(training_set[normal_index])
					
		# 	pred_LOF = np.asarray([ self.anomaly_class if p < 0 else self.normal_class for p in pred_LOF ])

		# 	print('Train LocalOutlierFactor',n,f1_score(ground_truth[normal_index],pred_LOF,average='macro'))

		# print('\n')

		# print('EllipticEnvelope')

		# self._anomaly_detector = EllipticEnvelopeAnomalyDetector()

		# print('EllipticEnvelope')

		# t=time.time()
		# self._anomaly_detector.fit(training_set[normal_index])
		# print('EllipticEnvelope')
		# pred = self._anomaly_detector.predict(training_set[normal_index])
		# t=time.time()-t
		
		# print('EllipticEnvelope')

		# if self.anomaly_class == 0:
		# 	pred = np.asarray([ 0 if p < 0 else 1 for p in pred ])
		# if self.anomaly_class == 1:
		# 	pred = np.asarray([ 1 if p < 0 else 0 for p in pred ])

		# print(t)
		# print(f1_score(ground_truth[normal_index],pred,average='macro'))
		# print('\n')

		# for kernel in ['linear','poly','rbf','sigmoid']:

		# 	if kernel != 'linear':

		# 		for gamma in range(1,10):

		# 			gamma = float(gamma) / 10.0

		# 			if kernel == 'poly':

		# 				for degree in range(1,10):

		# 					for coef0 in range(1,10):

		# 						coef0 = float(coef0) / 10.0

		# 						self._anomaly_detector = OneClassSVMAnomalyDetector(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
								
		# 						normal_index = [ i for i,gt in enumerate(ground_truth) if gt != self.anomaly_class ]

		# 						t=time.time()
		# 						pred = self._anomaly_detector.fit_predict(training_set[normal_index])
		# 						t=time.time()-t

		# 						if self.anomaly_class == 0:
		# 							pred = np.asarray([ 0 if p < 0 else 1 for p in pred ])
		# 						if self.anomaly_class == 1:
		# 							pred = np.asarray([ 1 if p < 0 else 0 for p in pred ])

		# 						print('kernel', kernel, 'degree', degree, 'gamma', gamma, 'coef0', coef0)
		# 						print(t)
		# 						print(f1_score(ground_truth[normal_index],pred,average='macro'))
		# 						print('\n')
		# 			elif kernel == 'sigmoid':
		# 				for coef0 in range(1,10):

		# 					coef0 = float(coef0) / 10.0

		# 					self._anomaly_detector = OneClassSVMAnomalyDetector(kernel=kernel, gamma=gamma, coef0=coef0)
							
		# 					normal_index = [ i for i,gt in enumerate(ground_truth) if gt != self.anomaly_class ]

		# 					t=time.time()
		# 					pred = self._anomaly_detector.fit_predict(training_set[normal_index])
		# 					t=time.time()-t

		# 					if self.anomaly_class == 0:
		# 						pred = np.asarray([ 0 if p < 0 else 1 for p in pred ])
		# 					if self.anomaly_class == 1:
		# 						pred = np.asarray([ 1 if p < 0 else 0 for p in pred ])

		# 					print('kernel', kernel, 'gamma', gamma, 'coef0', coef0)
		# 					print(t)
		# 					print(f1_score(ground_truth[normal_index],pred,average='macro'))
		# 					print('\n')
		# 			else:
		# 				self._anomaly_detector = OneClassSVMAnomalyDetector(kernel=kernel, gamma=gamma)
						
		# 				normal_index = [ i for i,gt in enumerate(ground_truth) if gt != self.anomaly_class ]

		# 				t=time.time()
		# 				pred = self._anomaly_detector.fit_predict(training_set[normal_index])
		# 				t=time.time()-t

		# 				if self.anomaly_class == 0:
		# 					pred = np.asarray([ 0 if p < 0 else 1 for p in pred ])
		# 				if self.anomaly_class == 1:
		# 					pred = np.asarray([ 1 if p < 0 else 0 for p in pred ])

		# 				print('kernel', kernel, 'gamma', gamma)
		# 				print(t)
		# 				print(f1_score(ground_truth[normal_index],pred,average='macro'))
		# 				print('\n')
		# 	else:

		# 		self._anomaly_detector = OneClassSVMAnomalyDetector(kernel=kernel)
						
		# 		normal_index = [ i for i,gt in enumerate(ground_truth) if gt != self.anomaly_class ]

		# 		t=time.time()
		# 		pred = self._anomaly_detector.fit_predict(training_set[normal_index])
		# 		t=time.time()-t
				

		# 		if self.anomaly_class == 0:
		# 			pred = np.asarray([ 0 if p < 0 else 1 for p in pred ])
		# 		if self.anomaly_class == 1:
		# 			pred = np.asarray([ 1 if p < 0 else 0 for p in pred ])

		# 		print('kernel', kernel)
		# 		print(t)
		# 		print(f1_score(ground_truth[normal_index],pred,average='macro'))
		# 		print('\n')

	def predict(self, testing_set):

		# # Only for debugging
		self._anomaly_detector.set_oracle(self.oracle)

		pred = self._anomaly_detector.predict(testing_set)
		if not self.anomaly_detector.startswith('k') or 'ae' in self.anomaly_detector:
			pred = np.asarray([ self.anomaly_class if p < 0 else self.normal_class for p in pred ])
		# Decomment to apply preclustering to Normal traffic ##_##
		# preds = np.empty(shape=(len(testing_set),self.n_clusters))
		# for i,anomaly_detector in enumerate(self._anomaly_detectors):
		# 	preds[:,i] = anomaly_detector.predict(testing_set)
		# 	# if not self.anomaly_detector.startswith('k'):
		# 	preds[:,i] = np.asarray([ self.anomaly_class if p < 0 else self.normal_class for p in preds[:,i] ])
		# pred = np.empty(shape=len(testing_set),dtype=int)
		# for i,p in enumerate(preds):
		# 	if (p == self.anomaly_class).sum() == self.n_clusters:
		# 		pred[i] = self.anomaly_class
		# 	else:
		# 		pred[i] = self.normal_class
		##_##

		# print(self.anomaly_detector)

		return pred

	def predict_proba(self, testing_set):

		# Pezzotto, prevedere modifica per piu' clusters
		# Sistema di Max Voting?? :D
		# if self.n_clusters == 1:

		# self._anomaly_detector = self._anomaly_detectors[0]
		if self.anomaly_detector.startswith('k'):
			proba = self._anomaly_detector.predict_proba(testing_set)
		else:
			proba = self._anomaly_detector.score_samples(testing_set)

		return proba

	def set_oracle(self, oracle):

		self.oracle = oracle

		pass

class HierarchicalClassifier(object):
	def __init__(self,input_file,levels_number,features_number,packets_number,classifier_name,detector_name,workers_number,anomaly_class,epochs_number):

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

		self.super_learner_default=['ssv','sif']

	def set_config(self, config_name, config):
		self.has_config = True
		self.config_name = config_name
		self.config = config

	def kfold_validation(self, k=10):

		# available_ram = psutil.virtual_memory()[1]
		# available_ram = int(int(available_ram) * .9 * 1e-9)

		# if available_ram > 5:
		# 	jvm.start(max_heap_size='5g')
		# else:
		# 	jvm.start(max_heap_size=str(available_ram)+'g')

		###

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

		#data = np.array(dataset['data'], dtype=object)
		# attributes = np.array(dataset['attributes'])
		data = np.array(dataset['data'], dtype=object)
		
		self.features_names = [x[0] for x in dataset['attributes']]

		self.attributes_number = data.shape[1]
		self.dataset_features_number = self.attributes_number - self.levels_number

		# Factorization of Nominal features_index
		# For nominal features we use an OneHotEncoder()
		# features_encoder = OrdinalEncoder()
		features_encoder = OneHotEncoder()
		nominal_features_index = [i for i in range(len(dataset['attributes'][:-self.levels_number])) if dataset['attributes'][i][1] != u'NUMERIC']
		for i in nominal_features_index:
			# print(type(data[0,nominal_features_index[2]]))
			# print(data[10:20, nominal_features_index[2]])
			# print([ type(d) for d in features_encoder.fit_transform(data[:, nominal_features_index[2]].reshape(-1,1))[10:20]])
			# input()
			data[:, i] = [ sp.toarray()[0] for sp in features_encoder.fit_transform(data[:, i].reshape(-1,1)) ]
			# print(data[0,i],data[0,i].toarray())
			# print(data[0, nominal_features_index[0]])
			# input()

		# print(data.shape)
		# bias = 0
		# for i in nominal_features_index:
		# 	# print(i+bias,data[0,i+bias])
		# 	extend_to = len(data[0,i+bias]) - 1
		# 	# print(len(zip(*data[:,i+bias])))
		# 	data = np.c_[ data[:,:i+bias], np.asarray(zip(*data[:,i+bias])).T, data[:,i+bias+1:] ]
		# 	# print(data.shape)
		# 	bias += copy(extend_to)

		# Next two variables are used to manage extended dataset
		# self.encoded_attributes_number = data.shape[1]
		# self.encoded_dataset_features_number = self.encoded_attributes_number - self.levels_number

		# print(data[0,:])
		# input()

		# Uncomment to print data types
		# types={}

		# for i,r in enumerate(data):
		# 	for j,e in enumerate(r):
		# 		t = type(e)
		# 		if t not in types:
		# 			types[t] = []
		# 		types[t].append((i,j))

		# for t in types:
		# 	print(t,len(types[t]))

		classifiers_per_fold = []
		oracles_per_fold = []
		predictions_per_fold = []
		probabilities_per_fold = []
		predictions_per_fold_all = []

		print('\n***\nStart testing with '+str(k)+'Fold cross-validation -f'+str(self.features_number)+' -c'+self.classifier_name+'\n***\n')

		skf = StratifiedKFold(n_splits=k, shuffle=True)
		fold_cnt = 1

		#for train_index, test_index in skf.split(data, np.array(data[:,self.attributes_number-1], dtype=int)):
		for train_index, test_index in skf.split(data, data[:,self.attributes_number-1]):
			print(fold_cnt)
			fold_cnt += 1
			self.classifiers = []

			self.training_set = data[train_index, :self.dataset_features_number]
			self.testing_set = data[test_index, :self.dataset_features_number]
			self.ground_truth = data[train_index, self.dataset_features_number:]
			self.oracle = data[test_index, self.dataset_features_number:]
			
			self.prediction = np.ndarray(shape=[len(test_index),self.levels_number],dtype='<U24')		# Hard Output
			self.probability = np.ndarray(shape=[len(test_index),self.levels_number],dtype=object)		# Soft Output
			self.prediction_all = np.ndarray(shape=[len(test_index),self.levels_number],dtype='<U24')

			root = TreeNode()

			root.train_index = [i for i in range(self.training_set.shape[0])]
			root.test_index = [i for i in range(self.testing_set.shape[0])]
			root.test_index_all = root.test_index
			root.children_tags = list(set(self.ground_truth[root.train_index, root.level]))
			root.children_number = len(root.children_tags)

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

			oracles_per_fold.append(self.oracle)
			predictions_per_fold.append(self.prediction)
			probabilities_per_fold.append(self.probability)
			predictions_per_fold_all.append(self.prediction_all)

		folder_discr = self.classifier_name

		if self.has_config:
			folder_discr = self.config_name
		if self.anomaly_class != '':
			folder_discr = self.detector_name

		# File discriminator, modify params_discr to change dipendence variable
		type_discr = 'flow'
		feat_discr = '_f_' + str(self.features_number)
		work_discr = '_w_' + str(self.workers_number)
		model_discr = '_e_' + str(self.epochs_number)
		if self.workers_number > 0:
			params_discr = work_discr + feat_discr
		if self.epochs_number > 0:
			params_discr = model_discr + feat_discr
		else:
			params_discr = feat_discr

		if not self.has_config and self.packets_number != 0:
			type_discr = 'early'
			feat_discr = '_p_' + str(self.packets_number)
		elif self.has_config:
			if 'p' in self.config:
				type_discr = 'early'
			feat_discr = '_c_' + self.config_name

		if self.has_config and self.classifier_name:
			if self.features_number != 0:
				feat_discr = '_f_' + str(self.features_number) + feat_discr + '_' + self.classifier_name
			if self.packets_number != 0:
				feat_discr = '_p_' + str(self.packets_number) + feat_discr + '_' + self.classifier_name

		material_folder = './data_'+folder_discr+'/material/'
		material_proba_folder = './data_'+folder_discr+'/material_proba/'

		if not os.path.exists('./data_'+folder_discr):
			os.makedirs('./data_'+folder_discr)
			os.makedirs(material_folder)
			os.makedirs(material_proba_folder)
		else:
			if not os.path.exists(material_folder):
				os.makedirs(material_folder)
			if not os.path.exists(material_proba_folder):
				os.makedirs(material_proba_folder)

		material_features_folder = './data_'+folder_discr+'/material/features/'
		material_train_durations_folder = './data_'+folder_discr+'/material/train_durations/'

		if not os.path.exists(material_folder):
			os.makedirs(material_folder)
			os.makedirs(material_features_folder)
			os.makedirs(material_train_durations_folder)
		if not os.path.exists(material_features_folder):
			os.makedirs(material_features_folder)
		if not os.path.exists(material_train_durations_folder):
			os.makedirs(material_train_durations_folder)

		for i in range(self.levels_number):

			with open(material_folder + 'multi_' + type_discr + '_level_' + str(i+1) + params_discr + '.dat', 'w+'):
				pass
			with open(material_proba_folder + 'multi_' + type_discr + '_level_' + str(i+1) + params_discr + '_proba.dat', 'w+'):
				pass

			for j in range(k):

				with open(material_folder + 'multi_' + type_discr + '_level_' + str(i+1) + params_discr + '.dat', 'a') as file: 

					file.write('@fold\n')
					for o, p in zip(oracles_per_fold[j][:,i], predictions_per_fold[j][:,i]):
						file.write(str(o)+' '+str(p)+'\n')

				with open(material_proba_folder + 'multi_' + type_discr + '_level_' + str(i+1) + params_discr + '_proba.dat', 'a') as file: 

					file.write('@fold\n')
					for o, p in zip(oracles_per_fold[j][:,i], probabilities_per_fold[j][:,i]):
						try:
							file.write(str(o)+' '+' '.join([str(v) for v in p])+'\n')
						except:
							file.write(str(o)+' '+str(p)+'\n')

		# Inferring NW metrics per classifier

		for classifier in classifiers_per_fold[0]:

			with open(material_folder + 'multi_' + type_discr + '_level_' + str(classifier.level+1) + params_discr + '_tag_' + str(classifier.tag) + '.dat', 'w+'):
				pass
			with open(material_folder + 'multi_' + type_discr + '_level_' + str(classifier.level+1) + params_discr + '_tag_' + str(classifier.tag) + '_all.dat', 'w+'):
				pass
			with open(material_features_folder + 'multi_' + type_discr + '_level_' + str(classifier.level+1) + params_discr + '_tag_' + str(classifier.tag) + '_features.dat', 'w+'):
				pass
			with open(material_train_durations_folder + 'multi_' + type_discr + '_level_' + str(classifier.level+1) + params_discr + '_tag_' + str(classifier.tag) + '_test_durations.dat', 'w+'):
				pass
			
		with open(material_train_durations_folder + 'multi_' + type_discr + params_discr + '_test_durations.dat', 'w+'):
			pass

		for fold_n, classifiers in enumerate(classifiers_per_fold):

			for classifier in classifiers:

				with open(material_folder + 'multi_' + type_discr + '_level_' + str(classifier.level+1) + params_discr + '_tag_' + str(classifier.tag) + '.dat', 'a') as file:

					if classifier.level > 0:
						index = []

						for pred_n, prediction in enumerate(predictions_per_fold[fold_n][classifier.test_index, classifier.level-1]):
							if prediction == oracles_per_fold[fold_n][classifier.test_index[pred_n], classifier.level-1]:
								index.append(classifier.test_index[pred_n])

						prediction_nw = predictions_per_fold[fold_n][index, classifier.level]
						oracle_nw = oracles_per_fold[fold_n][index, classifier.level]
					else:
						prediction_nw = predictions_per_fold[fold_n][classifier.test_index, classifier.level]
						oracle_nw = oracles_per_fold[fold_n][classifier.test_index, classifier.level]

					file.write('@fold\n')
					for o, p in zip(oracle_nw, prediction_nw):
							file.write(str(o)+' '+str(p)+'\n')

				with open(material_folder + 'multi_' + type_discr + '_level_' + str(classifier.level+1) + params_discr + '_tag_' + str(classifier.tag) + '_all.dat', 'a') as file:

					prediction_all = predictions_per_fold_all[fold_n][classifier.test_index_all, classifier.level]
					oracle_all = oracles_per_fold[fold_n][classifier.test_index_all, classifier.level]

					file.write('@fold\n')
					for o, p in zip(oracle_all, prediction_all):
							file.write(str(o)+' '+str(p)+'\n')

				with open(material_features_folder + 'multi_' + type_discr + '_level_' + str(classifier.level+1) + params_discr + '_tag_' + str(classifier.tag) + '_features.dat', 'a') as file:

					file.write('@fold\n')
					file.write(self.features_names[classifier.features_index[0]])

					for feature_index in classifier.features_index[1:]:
						file.write(','+self.features_names[feature_index])

					file.write('\n')
				
				with open(material_train_durations_folder + 'multi_' + type_discr + '_level_' + str(classifier.level+1) + params_discr + '_tag_' + str(classifier.tag) + '_test_durations.dat', 'a') as file:

					file.write('%.6f\n' % (classifier.test_duration))

		# Retrieve train_durations for each classifier
		test_durations_per_fold = []
		
		for classifiers in classifiers_per_fold:
			test_durations_per_fold.append([])
			for classifier in classifiers:
				test_durations_per_fold[-1].append(classifier.test_duration)

		with open(material_train_durations_folder + 'multi_' + type_discr + params_discr + '_test_durations.dat', 'w+') as file:

			mean_parallel_test_duration = np.mean(np.max(test_durations_per_fold, axis=1))
			std_parallel_test_duration = np.std(np.max(test_durations_per_fold, axis=1))

			mean_sequential_test_duration = np.mean(np.sum(test_durations_per_fold, axis=1))
			std_sequential_test_duration = np.std(np.sum(test_durations_per_fold, axis=1))
		 
			file.write('mean_par,std_par,mean_seq,std_seq\n')
			file.write('%.6f,%.6f,%.6f,%.6f\n' % (mean_parallel_test_duration,std_parallel_test_duration,mean_sequential_test_duration,std_sequential_test_duration))

		graph_folder = './data_'+folder_discr+'/graph/'

		if not os.path.exists('./data_'+folder_discr):
			os.makedirs('./data_'+folder_discr)
			os.makedirs(graph_folder)
		elif not os.path.exists(graph_folder):
			os.makedirs(graph_folder)

		# Graph plot
		G = nx.DiGraph()
		for info in classifiers_per_fold[0]:
			G.add_node(str(info.level)+' '+info.tag, level=info.level,
						 tag=info.tag, children_tags=info.children_tags)
		for node_parent, data_parent in G.nodes.items():
			for node_child, data_child in G.nodes.items():
				if data_child['level']-data_parent['level'] == 1 and any(data_child['tag'] in s for s in data_parent['children_tags']):
					G.add_edge(node_parent, node_child)
		nx.write_gpickle(G, graph_folder+'multi_' + type_discr + feat_discr +'_graph.gml')

		# # TODO: save the features selected at each level, per each fold ##__##
		# features_data_folder = './data_'+self.classifier_name+'/features/'

		# if not os.path.exists(features_data_folder):
		# 	os.makedirs(features_data_folder)

		# if self.features_number != 0:
		# 	file = open(features_data_folder+'flat_flow_level_'+str(self.level_target) +
		# 				'_f_'+str(self.features_number)+'_features.dat', 'w+')

		# for i in range(k):
		# 	file.write('@fold\n')
		# 	for f in feature[i][:-1]:
		# 		file.write(str(f) + ' ')
		# 	file.write(str(feature[i][-1]) + '\n')

		# file.close()
		# ##__##

		# # Uncomment when solve the probability problem ##__##
		# print('\n***\nStart testing with incremental gamma threshold\n***\n')

		# #bar = progressbar.ProgressBar(maxval=9, widgets=[progressbar.Bar(
		# #	'=', '[', ']'), ' ', progressbar.Percentage()])
		# #bar.start()

		# thresholds_number = 9

		# oracle_gamma = np.ndarray(shape=[levels_number, thresholds_number, k], dtype=object)
		# prediction_gamma = np.ndarray(shape=[levels_number, thresholds_number, k], dtype=object)
		# classified_ratio = np.ndarray(shape=[levels_number, thresholds_number, k], dtype=float)

		# for i in range(thresholds_number):
		# 	gamma = float(i+1)/10.0

		# 	for j in range(k):

		# 		indexes = []

		# 		for l in range(levels_number):

		# 			for index, p in enumerate(probabilities_per_fold[j][:, l]):
		# 				if max(p) < gamma:
		# 					indexes.append(index)

		# 			new_oracle = np.delete(oracles_per_fold[j][:, l], [indexes])
		# 			new_prediction = np.delete(predictions_per_fold[j][:, l], [indexes])

		# 			oracle_gamma[l, i, j] = new_oracle
		# 			prediction_gamma[l, i, j] = new_prediction
		# 			classified_ratio[l, i, j] = float(len(new_prediction))/float(len(predictions_per_fold[j][:, l]))

		# 	#bar.update(i)

		# #bar.finish()

		# for i in range(thresholds_number):

		# 	for l in range(levels_number):

		# 		with open(material_folder + 'multi_' + type_discr + '_level_' + str(l+1) + params_discr + '_gamma_'+str(float(i+1)/10.0)+'.dat', 'w+') as file:
					
		# 			for j in range(k):
		# 				file.write('@fold_cr\n')
		# 				file.write(str(classified_ratio[l, i, j])+'\n')
		# 				for o, p in zip(oracle_gamma[l, i, j], prediction_gamma[l, i, j]):
		# 					file.write(str(o)+' '+str(p)+'\n')

		##__##

		# jvm.stop()

	def features_selection(self,node):
		features_index = []

		#TODO: does not work with onehotencoded

		if node.features_number != 0 and node.features_number != self.dataset_features_number:

			selector = SelectKBest(mutual_info_classif, k=node.features_number)

			selector.fit(
				self.training_set[node.train_index, :self.dataset_features_number],
				node.label_encoder.transform(self.ground_truth[node.train_index, node.level])
				)

			support = selector.get_support()

			features_index = [ i for i,v in enumerate(support) if v ]

			# training_set_selected = selector.fit_transform(
			# 	self.training_set[node.train_index, :self.dataset_features_number],
			# 	node.label_encoder.transform(self.ground_truth[node.train_index, node.level], dtype=int)
			# 	)
			
			# training_set_reconstr = selector.inverse_transform(
			# 	node.label_encoder.inverse_transform(
			# 		training_set_selected
			# 	))

			# i0 = 0
			# i1 = 0
			# while i0 < node.features_number:
			# 	if np.array_equal(training_set_selected[:, i0], training_set_reconstr[:, i1]):
			# 		features_index.append(i1)
			# 		i0 += 1
			# 	i1 += 1
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

		for index, pred in zip(node.test_index, self.prediction[node.test_index, node.level]):

			node.test_duration = 0.
			self.prediction[index, node.level] = node.tag
			self.probability[index, node.level] = (1., 1., 1.)

		for index, pred in zip(node.test_index_all, self.prediction_all[node.test_index_all, node.level]):

			self.prediction_all[index, node.level] = node.tag

		node.features_index = node.parent.features_index

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
		classifier = AnomalyDetector(node.detector_name, node.label_encoder.transform(self.anomaly_class)[0], node.features_number, epochs_number)
		# Features selection
		node.features_index = self.features_selection(node)
		self.train(node, classifier)
		classifier.set_oracle(
			node.label_encoder.transform(self.oracle[node.test_index][:, node.level])
			)
		self.test(node, classifier)

		self.test_all(node, classifier)

	def train(self, node, classifier):

		print(node.label_encoder.nom_to_cat)

		node.test_duration = classifier.fit(
			self.training_set[node.train_index][:, node.features_index],
			node.label_encoder.transform(self.ground_truth[node.train_index][:, node.level])
			)

	def test(self, node, classifier):

		pred = node.label_encoder.inverse_transform(
			classifier.predict(self.testing_set[node.test_index][:, node.features_index]))
		proba = classifier.predict_proba(self.testing_set[node.test_index][:, node.features_index])

		# print(np.unique(pred,return_counts=True))

		# from sklearn.metrics import classification_report

		# tp = 0.0
		# fn = 0.0
		# fp = 0.0
		# tn = 0.0

		# for p,o in zip(pred,self.oracle[node.test_index][:, node.level]):
		# 	if p == o == '1':
		# 		tp += 1.0
		# 	if p == o == '0':
		# 		tn += 1.0
		# 	if p != o == '1':
		# 		fn += 1.0
		# 	if p != o == '0':
		# 		fp += 1.0

		# print(str(tp/(tp+fn)))
		# print(str(fp/(fp+tn))+'\n')

		# print(classification_report(self.oracle[node.test_index][:, node.level],pred))
		


		# i=0

		# with open('temp','w') as f:
		# 	f.write('d\n'+'\n'.join([str(d) for d in np.sort(df)]))

		# n_df = [ (d-np.min(df))*(.5-0.)/(0.-np.min(df))+0. if d < 0. else (d-0.)*(1.-.5)/(np.max(df)-0.)+.5 for d in df ]

		# for o, p, s, d, n in zip(self.oracle[node.test_index][:, node.level], pred, ss, df, n_df):
		# 	if o == '0':
		# 		print(o, p, s, d, -n+1.)
		# 		i += 1
		# 		if i==30:
		# 			break

		# exit(0)

		for p, b, i in zip(pred, proba, node.test_index):
			self.prediction[i, node.level] = p
			self.probability[i, node.level] = b

	def test_all(self, node, classifier):

		pred = node.label_encoder.inverse_transform(
			classifier.predict(self.testing_set[node.test_index_all][:, node.features_index]))

		for p, i in zip(pred, node.test_index_all):
			self.prediction_all[i, node.level] = p

if __name__ == "__main__":
	os.environ["DISPLAY"] = ":0"	# used to show xming display
	np.random.seed(0)

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

	supported_detectors = {
		'ssv': 'Sklearn_OC-SVM',
		'sif': 'Sklearn_IsolationForest',		# C4.5
		'slo': 'Sklearn_LocalOutlierFactor',
		'kae': 'Keras_Autoencoder',
		'kdae': 'Keras_DeepAutoencoder',
		'k1dcnnae': 'Keras_1DCNNAutoencoder'
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

	try:
		opts, args = getopt.getopt(
			sys.argv[1:], "hi:n:f:p:c:o:w:x:a:e:d:",
				"[input_file=,levels_number=,features_number=,packets_number=,\
				classifier_name=,configuration=,workers_number=,executors_number=,anomaly_class=,\
				epochs_number=,detector_name=]"
			)
	except getopt.GetoptError:
		print(sys.argv[0],'-i <input_file> -n <levels_number> \
			(-f <features_number>|-p <packets_number>) -c <classifier_name> (-o <configuration_file>) \
			(-w <workers_number> -x <executors_number>) (-a <anomaly_class> -e epochs_number -d detector_name)')
		print(sys.argv[0],'-h (or --help) for a carefuler help')
		sys.exit(2)
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print(sys.argv[0],'-i <input_file> -n <levels_number> \
			(-f <features_number>|-p <packets_number>) -c <classifier_name> (-o <configuration_file>) \
			(-w <workers_number> -x <executors_number>) (-a <anomaly_class> -e epochs_number -d detector_name)')
			print('Options:\n\t-i: dataset file, must be in arff or in csv format\n\t-n: number of levels (number of labels\' columns)')
			print('\t-f or -p: former refers features number, latter refers packets number\n\t-c: classifier name choose from following list:')
			for sc in np.sort(supported_classifiers.keys()):
				print('\t\t-c '+sc+'\t--->\t'+supported_classifiers[sc].split('_')[1]+'\t\timplemented in '+supported_classifiers[sc].split('_')[0])
			print('\t-o: configuration file in JSON format')
			print('\t-w: number of workers when used with spark.ml classifiers')
			print('\t-x: number of executors per worker when used with spark.ml classifiers')
			print('\t-a: hierarchical classifier starts with a one-class classification at each level if set and classification is binary (i.e. Anomaly Detection)\n\
				\tmust provide label of baseline class')
			print('\t-e: number of epochs in caso of DL models')
			for sd in np.sort(supported_detectors.keys()):
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
			for sd in np.sort(supported_detectors.keys()):
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

	hierarchical_classifier = HierarchicalClassifier(
		input_file=input_file,
		levels_number=levels_number,
		features_number=features_number,
		packets_number=packets_number,
		classifier_name=classifier_name,
		detector_name=detector_name,
		workers_number=workers_number,
		anomaly_class=anomaly_class,
		epochs_number=epochs_number
		)
	
	if config:
		config_name = config_file.split('.')[0]
		hierarchical_classifier.set_config(config_name, config)

	hierarchical_classifier.kfold_validation(k=10)

	if 'w' in import_str:
		jvm.stop()
