#!/usr/bin/python2

import arff
import numpy as np
import sys
import getopt
import os, time
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import weka.core.jvm as jvm
from weka.core.converters import Loader, ndarray_to_instances
from weka.core.dataset import Instances, Attribute
from weka.classifiers import Classifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import progressbar
import networkx as nx

from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import udf

from pyspark.ml.classification import RandomForestClassifier as Spark_RandomForestClassifier,\
	NaiveBayes as Spark_NaiveBayesClassifier,\
	MultilayerPerceptronClassifier as Spark_MultilayerPerceptronClassifier,\
	GBTClassifier as Spark_GBTClassifier,\
	DecisionTreeClassifier as Spark_DecisionTreeClassifier
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.linalg import Vectors, VectorUDT


class Tree(object):
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

class SklearnWekaWrapper(object):

	def __init__(self, class_name, options=None):

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

	def __init__(self, classifier_name, encoder):

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

		self.encoder = encoder

	def fit(self, training_set, ground_truth):

		self.ground_truth = self.encoder.transform(ground_truth)

		training_set = self._sklearn2spark(training_set, self.ground_truth)

		t = time.time()
		self._model = self._classifier.fit(training_set)
		t = time.time() - t

		return t

	def predict(self, testing_set):

		testing_set = self._sklearn2spark(testing_set, self.oracle)

		self.results = self._model.transform(testing_set)

		preds = [int(float(row.prediction)) for row in self.results.collect()]
	 
		nominal_preds = self.encoder.inverse_transform(preds)

		return np.array(nominal_preds)

	def predict_proba(self, testing_set):

		testing_set = self._sklearn2spark(testing_set, self.oracle)

		self.results = self._model.transform(testing_set)

		probas = [row.probability.toArray() for row in self.results.collect()]

		return np.array(probas)

	# TODO: introdurre performances di training

	#def set_encoder(self, _ground_truth):

	#	self.encoder = LabelEncoder()
	#	self.encoder.fit(all_ground_truth)

	def set_oracle(self, oracle):

		self.oracle = self.encoder.transform(oracle)

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
		
		# One hot encoding procedure
		# For RF one hot encoding is not needed
		
		# one_hot_encoder = OneHotEncoder(inputCols=["categorical_label"], outputCols=["label"], dropLast=False)
		# one_hot_model = one_hot_encoder.fit(spark_dataset)
		# encoded_spark_dataset = one_hot_model.transform(spark_dataset)
		
		return spark_dataset

class HierarchicalClassifier(object):
	def __init__(self,input_file,levels_number,features_number,packets_number,classifier_name,workers_number):

		self.input_file = input_file
		self.levels_number = levels_number
		self.features_number = features_number
		self.packets_number = packets_number
		self.classifier_name = classifier_name
		self.workers_number = workers_number
		self.has_config = False

	def set_config(self, config_name, config):
		self.has_config = True
		self.config_name = config_name
		self.config = config

	def kfold_validation(self, k=10):

		available_ram = psutil.virtual_memory()[1]
		available_ram = int(int(available_ram) * .9 * 1e-9)

		if available_ram > 5:
			jvm.start(max_heap_size='5g')
		else:
			jvm.start(max_heap_size=str(available_ram)+'g')

		jvm.start()

		###

		print('\nCaricando '+self.input_file+' con opts -f'+str(self.features_number)+' -c'+self.classifier_name+'\n')
		# load .arff file
		dataset = arff.load(open(self.input_file, 'r'))

		#data = np.array(dataset['data'], dtype=object)
		data = np.array(dataset['data'])
		self.features_names = [x[0] for x in dataset['attributes']]

		self.attributes_number = data.shape[1]
		self.dataset_features_number = self.attributes_number - self.levels_number

		# Factorization of Nominal features_index
		features_encoder = OrdinalEncoder()
		nominal_features_index = [i for i in range(len(dataset['attributes'][:-self.levels_number])) if dataset['attributes'][i][1] != u'NUMERIC']
		if len(nominal_features_index) > 0:
			data[:, nominal_features_index] = features_encoder.fit_transform(data[:, nominal_features_index])

		self.labels_encoders = []

		for i in range(self.levels_number):
			self.labels_encoders.append(LabelEncoder())
			self.labels_encoders[-1].fit(data[:, self.dataset_features_number + i])

		classifiers_per_fold = []
		oracles_per_fold = []
		predictions_per_fold = []
		probabilities_per_fold = []
		predictions_per_fold_all = []

		print('\n***\nStart testing with '+str(k)+'Fold cross-validation -f'+str(self.features_number)+' -c'+self.classifier_name+'\n***\n')

		#bar = progressbar.ProgressBar(maxval=k, widgets=[progressbar.Bar(
		#	'=', '[', ']'), ' ', progressbar.Percentage()])
		#bar.start()

		skf = StratifiedKFold(n_splits=k, shuffle=True)
		#bar_cnt = 0
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

			root = Tree()

			root.train_index = [i for i in range(self.training_set.shape[0])]
			root.test_index = [i for i in range(self.testing_set.shape[0])]
			root.test_index_all = root.test_index
			root.children_tags = list(set(self.ground_truth[root.train_index, root.level]))
			root.children_number = len(root.children_tags)

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
				
				#print(root.test_duration)

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

			#bar_cnt += 1
			#bar.update(bar_cnt)

		#bar.finish()

		# for j in range(self.levels_number):

		# 	cat_labels_unique = [i for i in range(np.max(data[:, self.dataset_features_number + j]))]

		# 	labels_unique = self.labels_encoders[j].inverse_transform(cat_labels_unique)

		# 	cat_nom_dict = {}

		# 	for i, label_unique in enumerate(labels_unique):
		# 		cat_nom_dict[label_unique] = cat_labels_unique[i]

		# 	data_folder = './data_'+self.classifier_name+'/'

		# 	file = open('cat_to_nom_labels_level_'+str(j+1)+'.dat','w')

		# 	file.write(data_folder + 'cat_label,nom_label\n')
		# 	for cat,nom in zip(cat_labels_unique, label_unique):
		# 		file.write(str(cat) + ',' + str(nom) + '\n')

		# 	file.close()

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

			file = open(material_folder + 'multi_' + type_discr + '_level_' + str(i+1) + work_discr + feat_discr + '.dat', 'w+')
			file.close()

			for j in range(k):

				file = open(material_folder + 'multi_' + type_discr + '_level_' + str(i+1) + work_discr + feat_discr + '.dat', 'a')

				file.write('@fold\n')
				for o, p in zip(oracles_per_fold[j][:,i], predictions_per_fold[j][:,i]):
					file.write(str(o)+' '+str(p)+'\n')

				file.close()

		# Inferring NW metrics per classifier

		for classifier in classifiers_per_fold[0]:

			file = open(material_folder + 'multi_' + type_discr + '_level_' + str(classifier.level+1) + work_discr + feat_discr + '_tag_' + str(classifier.tag) + '.dat', 'w+')
			file.close()
			file = open(material_folder + 'multi_' + type_discr + '_level_' + str(classifier.level+1) + work_discr + feat_discr + '_tag_' + str(classifier.tag) + '_all.dat', 'w+')
			file.close()
			file = open(material_features_folder + 'multi_' + type_discr + '_level_' + str(classifier.level+1) + work_discr + feat_discr + '_tag_' + str(classifier.tag) + '_features.dat', 'w+')
			file.close()
			file = open(material_train_durations_folder + 'multi_' + type_discr + '_level_' + str(classifier.level+1) + work_discr + feat_discr + '_tag_' + str(classifier.tag) + '_test_durations.dat', 'w+')
			file.close()
			
		file = open(material_train_durations_folder + 'multi_' + type_discr + work_discr + feat_discr + '_test_durations.dat', 'w+')
		file.close()

		for fold_n, classifiers in enumerate(classifiers_per_fold):

			for classifier in classifiers:

				file = open(material_folder + 'multi_' + type_discr + '_level_' + str(classifier.level+1) + work_discr + feat_discr + '_tag_' + str(classifier.tag) + '.dat', 'a')

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

				file.close()

				file = open(material_folder + 'multi_' + type_discr + '_level_' + str(classifier.level+1) + work_discr + feat_discr + '_tag_' + str(classifier.tag) + '_all.dat', 'a')

				prediction_all = predictions_per_fold_all[fold_n][classifier.test_index_all, classifier.level]
				oracle_all = oracles_per_fold[fold_n][classifier.test_index_all, classifier.level]

				file.write('@fold\n')
				for o, p in zip(oracle_all, prediction_all):
						file.write(str(o)+' '+str(p)+'\n')

				file.close()

				file = open(material_features_folder + 'multi_' + type_discr + '_level_' + str(classifier.level+1) + work_discr + feat_discr + '_tag_' + str(classifier.tag) + '_features.dat', 'a')

				file.write('@fold\n')
				file.write(self.features_names[classifier.features_index[0]])

				for feature_index in classifier.features_index[1:]:
					file.write(','+self.features_names[feature_index])

				file.write('\n')

				file.close()
				
				file = open(material_train_durations_folder + 'multi_' + type_discr + '_level_' + str(classifier.level+1) + work_discr + feat_discr + '_tag_' + str(classifier.tag) + '_test_durations.dat', 'a')

				file.write('%.6f\n' % (classifier.test_duration))

				file.close()

		# Retrieve train_durations for each classifier
		test_durations_per_fold = []
		
		for classifiers in classifiers_per_fold:
			test_durations_per_fold.append([])
			for classifier in classifiers:
				test_durations_per_fold[-1].append(classifier.test_duration)

		file = open(material_train_durations_folder + 'multi_' + type_discr + work_discr + feat_discr + '_test_durations.dat', 'w+')

		mean_parallel_test_duration = np.mean(np.max(test_durations_per_fold, axis=1))
		std_parallel_test_duration = np.std(np.max(test_durations_per_fold, axis=1))

		mean_sequential_test_duration = np.mean(np.sum(test_durations_per_fold, axis=1))
		std_sequential_test_duration = np.std(np.sum(test_durations_per_fold, axis=1))
	 
		file.write('mean_par,std_par,mean_seq,std_seq\n')
		file.write('%.6f,%.6f,%.6f,%.6f\n' % (mean_parallel_test_duration,std_parallel_test_duration,mean_sequential_test_duration,std_sequential_test_duration))

		file.close()

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

				file = open(material_folder + 'multi_' + type_discr + '_level_' + str(l) + work_discr + feat_discr + '_gamma_'+str(float(i+1)/10.0)+'.dat', 'w+')

				for j in range(k):
					file.write('@fold_cr\n')
					file.write(str(classified_ratio[l, i, j])+'\n')
					for o, p in zip(oracle_gamma[l, i, j], prediction_gamma[l, i, j]):
						file.write(str(o)+' '+str(p)+'\n')

				file.close()

		###

		jvm.stop()

	def features_selection(self,node):
		features_index = []

		if node.features_number != 0 and node.features_number != self.dataset_features_number:

			selector = SelectKBest(mutual_info_classif, k=node.features_number)
			training_set_selected = selector.fit_transform(
				self.training_set[node.train_index, :self.dataset_features_number], np.array(self.ground_truth[node.train_index, node.level], dtype=int))
			training_set_reconstr = selector.inverse_transform(
				training_set_selected)

			i0 = 0
			i1 = 0
			while i0 < node.features_number:
				if np.array_equal(training_set_selected[:, i0], training_set_reconstr[:, i1]):
					features_index.append(i1)
					i0 += 1
				i1 += 1
		else:
			if node.packets_number == 0:
				features_index = [i for i in range(self.dataset_features_number)]
			else:
				features_index = np.r_[0:node.packets_number, self.dataset_features_number/2:self.dataset_features_number/2+node.packets_number]

		return features_index

	def recursive(self,parent):

		for i in range(parent.children_number):
			child = Tree()
			self.classifiers.append(child)

			child.level = parent.level+1
			child.tag = parent.children_tags[i]
			child.parent = parent

			child.train_index = [index for index in parent.train_index if self.ground_truth[index, parent.level] == child.tag]
			child.test_index = [index for index in parent.test_index if self.prediction[index, parent.level] == child.tag]

			child.test_index_all = [index for index in parent.test_index_all if self.oracle[index, parent.level] == child.tag]
			child.children_tags = list(set(self.ground_truth[child.train_index, child.level]))

			child.children_number = len(child.children_tags)

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
				
				#print(child.test_duration)

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
		classifier = SklearnSparkWrapper(self.classifier_name, self.labels_encoders[node.level])
		# Features selection
		node.features_index = self.features_selection(node)
		#classifier.set_encoder(self.ground_truth[:, node.level])
		self.train(node, classifier)
		classifier.set_oracle(self.oracle[node.test_index][:, node.level])
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

	def Weka_NaiveBayes(self, node):

		# Instantation
		classifier = SklearnWekaWrapper(class_name='weka.classifiers.bayes.NaiveBayes', options='-D')

		# Features selection
		node.features_index = self.features_selection(node)

		self.train(node, classifier)
		self.test(node, classifier)

		self.test_all(node, classifier)

	def Weka_BayesNetwork(self, node):

		# Instantation
		classifier = SklearnWekaWrapper(class_name='weka.classifiers.bayes.BayesNet', options='-D -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5')

		# Features selection
		node.features_index = self.features_selection(node)

		self.train(node, classifier)
		self.test(node, classifier)

		self.test_all(node, classifier)

	def Weka_RandomForest(self, node):

		# Instantation
		classifier = SklearnWekaWrapper(class_name='weka.classifiers.trees.RandomForest')

		# Features selection
		node.features_index = self.features_selection(node)

		self.train(node, classifier)
		self.test(node, classifier)

		self.test_all(node, classifier)

	def Weka_J48(self, node):

		# Instantation
		classifier = SklearnWekaWrapper(class_name='weka.classifiers.trees.J48')

		# Features selection
		node.features_index = self.features_selection(node)

		self.train(node, classifier)
		self.test(node, classifier)

		self.test_all(node, classifier)

	def train(self, node, classifier):

		node.test_duration = classifier.fit(self.training_set[node.train_index][:, node.features_index], self.ground_truth[node.train_index][:, node.level])

	def test(self, node, classifier):

		pred = classifier.predict(self.testing_set[node.test_index][:, node.features_index])
		proba = classifier.predict_proba(self.testing_set[node.test_index][:, node.features_index])

		for p, b, i in zip(pred, proba, node.test_index):
			self.prediction[i, node.level] = p
			self.probability[i, node.level] = b

	def test_all(self, node, classifier):

		pred = classifier.predict(self.testing_set[node.test_index_all][:, node.features_index])

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
	}

	input_file = ''
	levels_number = 0
	features_number = 0
	packets_number = 0
	classifier_name = ''
	workers_number = 1

	config_file = ''
	config = ''

	try:
		opts, args = getopt.getopt(
			sys.argv[1:], "hi:n:f:p:c:o:w:", "[input_file=,levels_number=,features_number=,packets_number=,classifier_name=,configuration=,workers_number=]")
	except getopt.GetoptError:
		print('MultilayerClassifier5.0.py -i <input_file> -n <levels_number> (-f <features_number>|-p <packets_number>) -c <classifier_name> -o <configuration_file> -w <workers_number>')
		print('FlatClassifier.py -h (or --help) for a carefuler help')
		sys.exit(2)
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print('MultilayerClassifier5.0.py -i <input_file> -n <levels_number> (-f <features_number>|-p <packets_number>) -c <classifier_name> -o <configuration_file> -w <workers_number>')
			print('Options:\n\t-i: dataset file, must be in arff format\n\t-n: number of levels (number of labels\' columns)')
			print('\t-f or -p: former refers features number, latter refers packets number\n\t-c: classifier name choose from following list:')
			for sc in supported_classifiers:
				print('\t\t-c '+sc+'\t--->\t'+supported_classifiers[sc].split('_')[1]+'\t\timplemented in '+supported_classifiers[sc].split('_')[0])
			print('\t-o: configuration file in JSON format')
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

	if config_file:
		if not config_file.endswith('.json'):
			print('config file must have .json extention')
			sys.exit()

		import json

		with open(config_file) as f:
			config = json.load(f)

	else:

		if packets_number != 0 and features_number != 0 or packets_number == features_number:
			print('-f and -p option should not be used together')
			sys.exit()

		if classifier_name not in supported_classifiers:
			print('Classifier not supported\nList of available classifiers:\n')
			for sc in supported_classifiers:
				print('-c '+sc+'\t--->\t'+supported_classifiers[sc].split('_')[1]+'\t\timplemented in '+supported_classifiers[sc].split('_')[0])
			sys.exit()

	if levels_number == 0:
		print('MultilayerClassifier5.0.py -i <input_file> -n <levels_number> (-f <features_number>|-p <packets_number>) -c <classifier_name>')
		print('Number of level must be positive and non zero')
		sys.exit()

	if not input_file.endswith(".arff"):
		print('input files must be .arff')
		sys.exit()
	 
	executor_cores = 2

	conf = SparkConf().setAll([
		('spark.app.name', '_'.join([sys.argv[0],classifier_name,'w',str(workers_number)])),
		('spark.driver.cores', 30),
		('spark.driver.memory','30g'),
		('spark.executor.memory', '3000m'),
		('spark.master', 'spark://192.168.200.24:7077'),
		('spark.executor.cores', str(executor_cores)),
		('spark.cores.max', str(executor_cores * workers_number))
	])
	sc = SparkContext(conf=conf)
	sc.setLogLevel("ERROR")
	spark = SparkSession(sc)

	#spark = SparkSession.builder.appName('Hiararchical Classifier').getOrCreate()

	hierarchical_classifier = HierarchicalClassifier(input_file=input_file,levels_number=levels_number,features_number=features_number,packets_number=packets_number,classifier_name=classifier_name,workers_number=workers_number)
	
	if config:
		config_name = config_file.split('.')[0]
		hierarchical_classifier.set_config(config_name, config)

	hierarchical_classifier.kfold_validation(k=10)
