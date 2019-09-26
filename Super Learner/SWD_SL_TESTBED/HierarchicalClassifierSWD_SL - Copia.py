#!/usr/bin/python2

import arff
import numpy as np
import sys
import getopt
import os, time

import networkx as nx

from importlib import import_module

def import_sklearn():

	global RandomForestClassifier,DecisionTreeClassifier,SelectKBest,mutual_info_classif,\
		StratifiedKFold,train_test_split,OrdinalEncoder,MyLabelEncoder

	RandomForestClassifier = import_module('sklearn.ensemble').RandomForestClassifier
	DecisionTreeClassifier = import_module('sklearn.tree').DecisionTreeClassifier
	SelectKBest = import_module('sklearn.feature_selection').SelectKBest
	mutual_info_classif = import_module('sklearn.feature_selection').mutual_info_classif
	StratifiedKFold = import_module('sklearn.model_selection').StratifiedKFold
	train_test_split = import_module('sklearn.model_selection').train_test_split
	OrdinalEncoder = import_module('sklearn.preprocessing').OrdinalEncoder
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
		self.packets_number = 0
		self.classifier_name = 'srf'

		self.test_duration = 0.0

		self.encoder = None

class MyLabelEncoder(object):

	def __init__(self):

		self.nom_to_cat = {}
		self.cat_to_nom = {}

	def fit(self, y_nom):

		self.nom_to_cat = {}
		self.cat_to_nom = {}
		cat = 0

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
		return np.asarray(y_cat)

	def inverse_transform(self, y_cat):

		y_nom = [ self.cat_to_nom[cat] for cat in y_cat ]
		return np.asarray(y_nom)

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
		elif classifier_name == 'asv':
			# Implementation of one-class SVM used in Anomaly Detection mode
			class_name='weka.classifiers.functions.LibSVM'
			options='-S 2'

		if options is not None:
			self._classifier = Classifier(classname=class_name, options=[option for option in options.split()])
		else:
			self._classifier = Classifier(classname=class_name)

	def fit(self, training_set, ground_truth):

		self.ground_truth = ground_truth

		training_set = self._sklearn2weka(training_set, self.ground_truth)
		training_set.class_is_last()

		self._classifier.build_classifier(training_set)

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

class SuperLearnerClassifier(object):
	def __init__(self, first_level_learners, num_classes):

		self.num_classes = num_classes
		self.first_level_classifiers = []

		for first_level_learner in first_level_learners:
			if first_level_learner.startswith('w'):
				self.first_level_classifiers.append(SklearnWekaWrapper(first_level_learner))
			elif first_level_learner.startswith('d'):
				self.first_level_classifiers.append(SklearnSparkWrapper(first_level_learner))

		self.first_level_classifiers_number = len(self.first_level_classifiers)

		self.proba = None

	def fit(self, dataset, labels):
		
		t = time.time()
		# First: split train set of node of hyerarchy in 20% train and 80% testing
		# with respective ground truth NB: for first training of first level models
		# SuperLearner use only training set
		training_set, _, ground_truth, _ = train_test_split(dataset, labels, train_size=0.2)

		# Second: fit classifiers of first level through train set
		proba, true = self._first_level_fit(training_set, ground_truth)

		# Three: fit combiner passing it obtained probabilities and respective ground truth
		# this phase return weights, i.e. decisions template, per each classifier
		self.weights = self._meta_learner_fit(proba, true)

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
		first_proba = self._first_level_predict_proba(testing_set)
		# Second: we pass first level prediction to combiner
		pred, proba = self._meta_learner_predict(first_proba)
		# Third: for consistency, predict returns only predictions
		# NB: probabilities are not wasted, but saved in order to return
		# in case of calling predict_proba function
		# NB2: if you measure testing time, you have to comment next row
		self.proba = proba

		return pred

	def _first_level_predict_proba(self, testing_set):

		probabilities = []

		for first_level_classifier in self.first_level_classifiers:

			first_level_classifier.set_oracle(self.oracle)
			proba = first_level_classifier.predict_proba(testing_set)

			probabilities.append(proba)
		
		return probabilities

	def _meta_learner_predict(self, testing_set):

		predictions = self._predict_Decision_Template(testing_set, self.weights)

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
		
		dp_predict=np.asarray(dp_predict)
		
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

	def predict_proba(self, testing_set):
		
		if self.proba is None:
			self.predict(testing_set)

		return self.proba

	def set_oracle(self, oracle):

		self.oracle = oracle

class HierarchicalClassifier(object):
	def __init__(self,input_file,levels_number,features_number,packets_number,classifier_name,workers_number,normal_class):

		self.input_file = input_file
		self.levels_number = levels_number
		self.features_number = features_number
		self.packets_number = packets_number
		self.classifier_name = classifier_name
		self.workers_number = workers_number
		self.has_config = False
		self.normal_class = normal_class

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
				dataset['attributes'] = [ (attr, attr_type) for attr,attr_type in zip(attributes.strip().split(delimiter),attrs_type) ]
				dataset['data'] = [ sample.strip().split(delimiter) for sample in samples ] # TODO gestire i tipi degli attributi
			elif self.input_file.lower().endswith('.pickle'):
				# TODO: provide a method to load dataset from pickle
				pass

		#data = np.array(dataset['data'], dtype=object)
		# attributes = np.array(dataset['attributes'])
		data = np.array(dataset['data'])
		
		self.features_names = [x[0] for x in dataset['attributes']]

		self.attributes_number = data.shape[1]
		self.dataset_features_number = self.attributes_number - self.levels_number

		# Factorization of Nominal features_index
		features_encoder = OrdinalEncoder()
		nominal_features_index = [i for i in range(len(dataset['attributes'][:-self.levels_number])) if dataset['attributes'][i][1] != u'NUMERIC']
		if len(nominal_features_index) > 0:
			data[:, nominal_features_index] = features_encoder.fit_transform(data[:, nominal_features_index])

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

			root.encoder = MyLabelEncoder()
			root.encoder.fit(self.ground_truth[root.train_index, root.level])

			if self.has_config and root.tag + '_' + str(root.level + 1) in self.config:
				if 'f' in self.config[root.tag + '_' + str(root.level + 1)]:
					root.features_number = self.config[root.tag + '_' + str(root.level + 1)]['f']
				elif 'p' in self.config[root.tag + '_' + str(root.level + 1)]:
					root.packets_number = self.config[root.tag + '_' + str(root.level + 1)]['p']
				root.classifier_name = self.config[root.tag + '_' + str(root.level + 1)]['c']

				print('\nconfig','tag',root.tag,'level',root.level,'f',root.features_number,'c',root.classifier_name,'train_test_len',len(root.train_index),len(root.test_index))
			else:
				root.features_number = self.features_number
				root.packets_number = self.packets_number
				root.classifier_name = self.classifier_name

				print('\nconfig','tag',root.tag,'level',root.level,'f',root.features_number,'c',root.classifier_name,'train_test_len',len(root.train_index),len(root.test_index))

			self.classifiers.append(root)

			if root.children_number > 1:

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

		material_folder = './data_'+folder_discr+'/material/'

		if not os.path.exists('./data_'+folder_discr):
			os.makedirs('./data_'+folder_discr)
			os.makedirs(material_folder)
		elif not os.path.exists(material_folder):
			os.makedirs(material_folder)

		type_discr = 'flow'
		feat_discr = '_f_' + str(self.features_number)
		work_discr = '_w_' + str(self.workers_number)
		if self.workers_number > 0:
			params_discr = work_discr + feat_discr
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

			for j in range(k):

				with open(material_folder + 'multi_' + type_discr + '_level_' + str(i+1) + params_discr + '.dat', 'a') as file: 

					file.write('@fold\n')
					for o, p in zip(oracles_per_fold[j][:,i], predictions_per_fold[j][:,i]):
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

		print('\n***\nStart testing with incremental gamma threshold\n***\n')

		#bar = progressbar.ProgressBar(maxval=9, widgets=[progressbar.Bar(
		#	'=', '[', ']'), ' ', progressbar.Percentage()])
		#bar.start()

		thresholds_number = 9

		oracle_gamma = np.ndarray(shape=[levels_number, thresholds_number, k], dtype=object)
		prediction_gamma = np.ndarray(shape=[levels_number, thresholds_number, k], dtype=object)
		classified_ratio = np.ndarray(shape=[levels_number, thresholds_number, k], dtype=float)

		for i in range(thresholds_number):
			gamma = float(i+1)/10.0

			for j in range(k):

				indexes = []

				for l in range(levels_number):

					for index, p in enumerate(probabilities_per_fold[j][:, l]):
						if max(p) < gamma:
							indexes.append(index)

					new_oracle = np.delete(oracles_per_fold[j][:, l], [indexes])
					new_prediction = np.delete(predictions_per_fold[j][:, l], [indexes])

					oracle_gamma[l, i, j] = new_oracle
					prediction_gamma[l, i, j] = new_prediction
					classified_ratio[l, i, j] = float(len(new_prediction))/float(len(predictions_per_fold[j][:, l]))

			#bar.update(i)

		#bar.finish()

		for i in range(thresholds_number):

			for l in range(levels_number):

				with open(material_folder + 'multi_' + type_discr + '_level_' + str(l+1) + params_discr + '_gamma_'+str(float(i+1)/10.0)+'.dat', 'w+') as file:
					
					for j in range(k):
						file.write('@fold_cr\n')
						file.write(str(classified_ratio[l, i, j])+'\n')
						for o, p in zip(oracle_gamma[l, i, j], prediction_gamma[l, i, j]):
							file.write(str(o)+' '+str(p)+'\n')

		###

		# jvm.stop()

	def features_selection(self,node):
		features_index = []

		if node.features_number != 0 and node.features_number != self.dataset_features_number:

			selector = SelectKBest(mutual_info_classif, k=node.features_number)

			selector.fit(
				self.training_set[node.train_index, :self.dataset_features_number],
				node.encoder.transform(self.ground_truth[node.train_index, node.level])
				)

			support = selector.get_support()

			features_index = [ i for i,v in enumerate(support) if v ]

			# training_set_selected = selector.fit_transform(
			# 	self.training_set[node.train_index, :self.dataset_features_number],
			# 	node.encoder.transform(self.ground_truth[node.train_index, node.level], dtype=int)
			# 	)
			
			# training_set_reconstr = selector.inverse_transform(
			# 	node.encoder.inverse_transform(
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
				features_index = [i for i in range(self.dataset_features_number)]
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

			child.encoder = MyLabelEncoder()
			child.encoder.fit(np.concatenate((child.children_tags, other_childs_children_tags)))

			if self.has_config and child.tag + '_' + str(child.level + 1) in self.config:
				if 'f' in self.config[child.tag + '_' + str(child.level + 1)]:
					child.features_number = self.config[child.tag + '_' + str(child.level + 1)]['f']
				elif 'p' in self.config[child.tag + '_' + str(child.level + 1)]:
					child.packets_number = self.config[child.tag + '_' + str(child.level + 1)]['p']
				child.classifier_name = self.config[child.tag + '_' + str(child.level + 1)]['c']
				print('config','tag',child.tag,'level',child.level,'f',child.features_number,'c',child.classifier_name,'train_test_len',len(child.train_index),len(child.test_index))
			else:
				child.features_number = self.features_number
				child.packets_number = self.packets_number
				child.classifier_name = self.classifier_name
				print('config','tag',child.tag,'level',child.level,'f',child.features_number,'c',child.classifier_name,'train_test_len',len(child.train_index),len(child.test_index))

			self.classifiers[self.classifiers.index(parent)].children[child.tag]=child

			if child.children_number > 1 and len(child.test_index) > 0:

				classifier_to_call = getattr(self, supported_classifiers[child.classifier_name])
				classifier_to_call(node=child)

			else:

				self.unary_class_results_inferring(child)

			if child.level < self.levels_number-1 and child.children_number > 0:
				self.recursive(child)

	def unary_class_results_inferring(self, node):

		for index, pred in zip(node.test_index, self.prediction[node.test_index, node.level]):

			self.prediction[index, node.level] = node.tag
			self.probability[index, node.level] = (1.0, 1.0, 1.0)

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
			node.encoder.transform(self.oracle[node.test_index][:, node.level])
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
		classifier = SuperLearnerClassifier(['wrf', 'wnb', 'wbn', 'wj48'], node.children_number)
		# Features selection
		node.features_index = self.features_selection(node)
		self.train(node, classifier)
		classifier.set_oracle(
			node.encoder.transform(self.oracle[node.test_index][:, node.level])
			)
		self.test(node, classifier)

		self.test_all(node, classifier)

	def train(self, node, classifier):

		node.test_duration = classifier.fit(
			self.training_set[node.train_index][:, node.features_index],
			node.encoder.transform(self.ground_truth[node.train_index][:, node.level])
			)

	def test(self, node, classifier):

		pred = node.encoder.inverse_transform(
			classifier.predict(self.testing_set[node.test_index][:, node.features_index]))
		proba = classifier.predict_proba(self.testing_set[node.test_index][:, node.features_index])

		for p, b, i in zip(pred, proba, node.test_index):
			self.prediction[i, node.level] = p
			self.probability[i, node.level] = b

	def test_all(self, node, classifier):

		pred = node.encoder.inverse_transform(
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

	input_file = ''
	levels_number = 0
	features_number = 0
	packets_number = 0
	classifier_name = ''
	workers_number = 0
	executors_number = 0
	normal_class = ''

	config_file = ''
	config = ''

	try:
		opts, args = getopt.getopt(
			sys.argv[1:], "hi:n:f:p:c:o:w:e:a:",
				"[input_file=,levels_number=,features_number=,packets_number=,\
				classifier_name=,configuration=,workers_number=,executors_number=,normal_class=]"
			)
	except getopt.GetoptError:
		print(sys.argv[0],'-i <input_file> -n <levels_number> \
			(-f <features_number>|-p <packets_number>) -c <classifier_name> (-o <configuration_file>) \
			(-w <workers_number>) (-e <executors_number>) (-a <anomaly_class>)')
		print(sys.argv[0],'-h (or --help) for a carefuler help')
		sys.exit(2)
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print(sys.argv[0],'-i <input_file> -n <levels_number> \
			(-f <features_number>|-p <packets_number>) -c <classifier_name> (-o <configuration_file>) \
			(-w <workers_number>) (-e <executors_number>) (-a <normal_class>)')
			print('Options:\n\t-i: dataset file, must be in arff or in csv format\n\t-n: number of levels (number of labels\' columns)')
			print('\t-f or -p: former refers features number, latter refers packets number\n\t-c: classifier name choose from following list:')
			for sc in supported_classifiers:
				print('\t\t-c '+sc+'\t--->\t'+supported_classifiers[sc].split('_')[1]+'\t\timplemented in '+supported_classifiers[sc].split('_')[0])
			print('\t-o: configuration file in JSON format')
			print('\t-w: number of workers when used with spark.ml classifiers')
			print('\t-e: number of executors per worker when used with spark.ml classifiers')
			print('\t-a: hierarchical classifier starts with a one-class classification at first level (i.e. Anomaly Detection)\n\
				\tmust provide label of baseline class')
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
		if opt in ("-e", "--executors_number"):
			executors_number = int(arg)
		if opt in ("-a", "--normal_class"):
			normal_class = arg

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
				for sc in supported_classifiers:
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

		import_str = classifier_name[0]

	if levels_number == 0:
		print('Number of level must be positive and non zero')
		sys.exit()

	if not input_file.endswith('.arff') or not input_file.endswith('.csv'):
		print('Input files must be .arff or .csv')
		sys.exit()

	import_sklearn()

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
		workers_number=workers_number,
		normal_class=normal_class
		)
	
	if config:
		config_name = config_file.split('.')[0]
		hierarchical_classifier.set_config(config_name, config)

	hierarchical_classifier.kfold_validation(k=10)

	if 'w' in import_str:
		jvm.stop()
