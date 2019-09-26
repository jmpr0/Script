#    Implements state-of-the-art deep learning architectures.
#
#    Copyright (C) 2017  Antonio Montieri
#    email: antonio.montieri@unina.it
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as
#    published by the Free Software Foundation, either version 3 of the
#    License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
Created on 13 nov 2017

@author: antonio
'''

import sys, os, time, errno, datetime

import keras, sklearn, numpy, scipy, imblearn.metrics
import tensorflow as tf
from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Conv1D, AveragePooling1D, MaxPooling1D,\
			 Conv2D, MaxPooling2D,\
			 Dense, LSTM, GRU, Flatten, Dropout, Reshape, BatchNormalization,\
			 Input, Bidirectional, concatenate
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical, plot_model
from keras.constraints import maxnorm
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer, Imputer
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from yadlt.models.autoencoders.stacked_denoising_autoencoder import StackedDenoisingAutoencoder
from sklearn.model_selection import train_test_split
import random
import math

import pickle
import logging
from logging.handlers import RotatingFileHandler
from pprint import pprint
import matplotlib.pyplot as plt

import multiprocessing as mp


tf.logging.set_verbosity(tf.logging.INFO)



class DLArchitecturesException(Exception):
	'''
	Exception for DLArchitectures class
	'''
	def __init__(self, message):
		self.message = message
	def __str__(self):
		return str(self.message)



class TimeEpochs(keras.callbacks.Callback):
	'''
	Callback used to calculate per-epoch time
	'''
	
	def on_train_begin(self, logs={}):
		self.times = []
	
	def on_epoch_begin(self, batch, logs={}):
		self.epoch_time_start = time.time()
	
	def on_epoch_end(self, batch, logs={}):
		self.times.append(time.time() - self.epoch_time_start)



class DLArchitectures(object):
	'''
	Implements state-of-the-art deep learning architectures.
	For details, refer to the related papers.
	'''
	
	def __init__(self, pickle_dataset_filenames, dataset, outdir, include_ports = True, is_multimodal = False):
		'''
		Constructor for the class DLArchitectures.
		Input:
		- pickle_dataset_filenames (list): TC datasets in .pickle format; for multimodal approaches len(pickle_dataset_filenames) > 1
		- dataset (int): set the type of dataset to handle the correct set of classes (dataset == 1 is FB/FBM; dataset == 2 is Android; dataset == 3 is iOS)
		- outdir (string): outdir to contain pickle dataset
		- include_ports (boolean): if True, TCP/UDP ports are used as input for lopez2017network approaches
		- is_multimodal (boolean): if True, a multimodal approach is considered and fed with multiple inputs (i.e. len(pickle_dataset_filenames) > 1)
		'''
		
		self.dataset = dataset
		self.outdir = outdir
		
		self.K_binary = [1, 2]
		self.K_multi = [1, 3, 5, 7, 10]
		
		self.gamma_range = numpy.arange(0.0, 1.0, 0.1)
		
		self.soft_comb_values= []
		
		self.label_class_FB_FBM = {
							0: 'FB',
							1: 'FBM'
							}
		
		self.label_class_Android = {
							0: '360Security',
							1: '6Rooms',
							2: '80sMovie',
							3: '9YinZhenJing',
							4: 'Anghami',
							5: 'BaiDu',
							6: 'Crackle',
							7: 'EFood',
							8: 'FrostWire',
							9: 'FsecureFreedomeVPN',
							10: 'Go90',
							11: 'Google+',
							12: 'GoogleAllo',
							13: 'GoogleCast',
							14: 'GoogleMap',
							15: 'GooglePhotos',
							16: 'GooglePlay',
							17: 'GroupMe',
							18: 'Guvera',
							19: 'Hangouts',
							20: 'HidemanVPN',
							21: 'Hidemyass',
							22: 'Hooq',
							23: 'HotSpot',
							24: 'IFengNews',
							25: 'InterVoip',
							26: 'LRR',
							27: 'MeinO2',
							28: 'Minecraft',
							29: 'Mobily',
							30: 'Narutom',
							31: 'NetTalk',
							32: 'NileFM',
							33: 'Palringo',
							34: 'PaltalkScene',
							35: 'PrivateTunnelVPN',
							36: 'PureVPN',
							37: 'QQ',
							38: 'QQReader',
							39: 'QianXunYingShi',
							40: 'RaidCall',
							41: 'Repubblica',
							42: 'RiyadBank',
							43: 'Ryanair',
							44: 'SayHi',
							45: 'Shadowsocks',
							46: 'SmartVoip',
							47: 'Sogou',
							48: 'eBay'
		}
		
	
		self.label_class_CICIDS17 = {
							0: 'BENIGN',
							1: 'Bot',
							2: 'DDoS',
							3: 'DoSGoldenEye',
							4: 'DoSHulk',
							5: 'DoSSlowhttptest',
							6: 'DoSslowloris',
							7: 'FTP-Patator',
							8: 'Heartbleed',
							9: 'Infiltration',
							10: 'PortScan',
							11: 'SSH-Patator',
							12: 'WebAttackBruteForce',
							13: 'WebAttackSqlInjection',
							14: 'WebAttackXSS'
		}

		self.time_callback = TimeEpochs()
		
		if self.dataset == 1:
			self.K = self.K_binary
			self.label_class = self.label_class_FB_FBM
		elif self.dataset == 2:
			self.K = self.K_multi
			self.label_class = self.label_class_Android
		elif self.dataset == 3:
			self.K = self.K_multi
			self.label_class = self.label_class_IOS
		elif self.dataset == 4:
			self.K = self.K_multi
			self.label_class = self.label_class_CICIDS17
		
		self.debug_log = self.setup_logger('debug', '%s/deep_learning_architectures.log' % self.outdir, logging.DEBUG)
		
		for pickle_dataset_filename in pickle_dataset_filenames:
			if os.path.isfile(pickle_dataset_filename):
				self.debug_log.debug('The dataset file %s exists.' % pickle_dataset_filename)
			else:
				err_str = 'The dataset file %s does not exist. Please, provide a valid .pickle file.' % pickle_dataset_filename
				self.debug_log.error(err_str)
				raise DLArchitecturesException(err_str)
		
		self.pickle_dataset_filenames = pickle_dataset_filenames
		self.deserialize_dataset(include_ports, is_multimodal)
	
	
	def setup_logger(self, logger_name, log_file, level=logging.INFO):
		'''
		Return a log object associated to <log_file> with specific formatting and rotate handling.
		'''

		l = logging.getLogger(logger_name)

		if not getattr(l, 'handler_set', None):
			logging.Formatter.converter = time.gmtime
			formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')

			rotatehandler = RotatingFileHandler(log_file, mode='a', maxBytes=10485760, backupCount=30)
			rotatehandler.setFormatter(formatter)

			#TODO: Debug stream handler for development
			#streamHandler = logging.StreamHandler()
			#streamHandler.setFormatter(formatter)
			#l.addHandler(streamHandler)

			l.addHandler(rotatehandler)
			l.setLevel(level)
			l.handler_set = True
		
		elif not os.path.exists(log_file):
			logging.Formatter.converter = time.gmtime
			formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')

			rotatehandler = RotatingFileHandler(log_file, mode='a', maxBytes=10485760, backupCount=30)
			rotatehandler.setFormatter(formatter)

			#TODO: Debug stream handler for development
			#streamHandler = logging.StreamHandler()
			#streamHandler.setFormatter(formatter)
			#l.addHandler(streamHandler)

			l.addHandler(rotatehandler)
			l.setLevel(level)
			l.handler_set = True
		
		return l
		
	
	#RF Classifier	
	def taylor2016appscanner_RF (self, samples_train, categorical_labels_train, samples_test, categorical_labels_test):
		'''
		Implements the RF proposed in taylor2016appscanner used as a baseline for DL-networks evaluation
		'''
		
		self.debug_log.info('Starting execution of taylor2016appscanner_RF approach')

		
		estimators_number = 150  #150
		
		# Define RF classifier with parameters specified in taylor2016appcanner
		clf = RandomForestClassifier(criterion='gini', max_features='sqrt', n_estimators=estimators_number)
		self.debug_log.info('taylor2016appscanner_RF: classifier defined')
		
		# Training phase
		train_time_begin = time.time()
		#print("TAYLOR", samples_train.shape, categorical_labels_train.shape)
		clf.fit(samples_train, categorical_labels_train) 
		train_time_end = time.time()
		self.train_time = train_time_end - train_time_begin
		self.debug_log.info('taylor2016appscanner_RF: Training phase completed')
		print('Training phase RF completed')
		
		# Test with predict_classes: test_model(model, transformed_samples_test, categorical_labels_test)
		predictions, accuracy,macro_recall, macro_precision,fmeasure, gmean, macro_gmean, training_accuracy, training_macro_recall,training_macro_precision, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios = \
		self.test_classifier(clf, samples_train, categorical_labels_train,samples_test, categorical_labels_test)
		self.debug_log.info('taylor2016appcanner_RF: Test phase completed')
		#print("IN TAYLOR PREDICTION ", predictions.shape, categorical_labels_test.shape)
		print('Test phase completed')
		
		self.debug_log.info('Ending execution of taylor2016appscanner_RF approach')
		
		return predictions, accuracy,macro_recall,macro_precision, fmeasure, gmean, macro_gmean, training_accuracy,training_macro_recall, training_macro_precision, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios
	
	
	#NBGaussian Classifier
	def gaussianNB (self, samples_train, categorical_labels_train, samples_test, categorical_labels_test):
			self.debug_log.info('Starting execution of gaussianNB approach')
			
			
			# Define GaussianNB classifier
			clf = GaussianNB()
			self.debug_log.info('gaussianNB: classifier defined')

				
			# Training phase
			train_time_begin = time.time()
			clf.fit(samples_train, categorical_labels_train)
			train_time_end = time.time()
			self.train_time = train_time_end - train_time_begin
			self.debug_log.info('GaussianNB: Training phase completed')
			print('Training phase GNB completed')
			
			
			# Test with predict_classes: test_model(model, transformed_samples_test, categorical_labels_test)
			predictions, accuracy, macro_recall,macro_precision,fmeasure, gmean, macro_gmean, training_accuracy , training_macro_recall,training_macro_precision,training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
			filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
			filtered_norm_cnf_matrices, filtered_classified_ratios = \
			self.test_classifier(clf, samples_train, categorical_labels_train, samples_test, categorical_labels_test)
			print("PREDICTION DI GAUSSIAN", predictions.shape)
			self.debug_log.info('GaussianNB: Test phase completed')
			print('Test phase completed')
			
			self.debug_log.info('Ending execution of GaussianNB approach')
			
			return predictions, accuracy,macro_recall, macro_precision,fmeasure, gmean, macro_gmean, training_accuracy, training_macro_recall,training_macro_precision, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
			filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
			filtered_norm_cnf_matrices, filtered_classified_ratios
	

	#DecisionTree(CART) Classifier
	def decisionTree(self, samples_train, categorical_labels_train, samples_test, categorical_labels_test):
		'''
		Implements the RF proposed in taylor2016appscanner used as a baseline for DL-networks evaluation
		'''
		
		self.debug_log.info('Starting execution of decisionTree approach')

	
		# Define DT classifier 
		clf = DecisionTreeClassifier(criterion='gini', max_features='sqrt')
		self.debug_log.info('decisionTree: classifier defined')
		
		# Training phase
		train_time_begin = time.time()
		clf.fit(samples_train, categorical_labels_train) 
		train_time_end = time.time()
		self.train_time = train_time_end - train_time_begin
		self.debug_log.info('decisionTree: Training phase completed')
		print('Training phase DT completed')
		
		# Test with predict_classes: test_model(model, transformed_samples_test, categorical_labels_test)
		predictions, accuracy,macro_recall, macro_precision,fmeasure, gmean, macro_gmean, training_accuracy, training_macro_recall,training_macro_precision, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios = \
		self.test_classifier(clf,samples_train, categorical_labels_train,samples_test, categorical_labels_test)
		self.debug_log.info('decisionTree: Test phase completed')
		print('Test phase completed')
		
		self.debug_log.info('Ending execution of decisionTree approach')
		
		return predictions, accuracy,macro_recall,macro_precision, fmeasure, gmean, macro_gmean, training_accuracy, training_macro_recall,training_macro_precision, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios
	
	#Hard no-Trainable : MVC
	def MajorityVotingCombiner_10Fold (self,samples_train,categorical_labels_train,samples_test,categorical_labels_test):
		
		self.train_time = 0
		self.test_time = 0
		self.debug_log.info('Starting execution of MultiClassification approach')

		###########################################
		
		
		estimators_number = 150
		
		# Define RF classifier 
		clf_RF = RandomForestClassifier(criterion='gini', max_features='sqrt', n_estimators=estimators_number)
		self.debug_log.info('taylor2016appscanner_RF: classifier defined')
		
		# Training phase
		train_time_begin = time.time()
		clf_RF.fit(samples_train, categorical_labels_train) 
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		self.debug_log.info('taylor2016appscanner_RF: Training phase completed')
		print('Training phase RF completed')
		

		#Test phase	
		test_time_begin = time.time()
		predictions_RF = clf_RF.predict(samples_test)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		self.debug_log.info('taylor2016appcanner_RF: Test phase completed')
		print('Test phase RF completed')
		
		self.debug_log.info('Ending execution of taylor2016appscanner_RF approach')
		
		#################################
		
		self.debug_log.info('Starting execution of decisionTree approach')

	
		# Define DT classifier 
		clf_DT = DecisionTreeClassifier(criterion='gini', max_features='sqrt')
		self.debug_log.info('decisionTree: classifier defined')
		
		# Training phase
		train_time_begin = time.time()
		clf_DT.fit(samples_train, categorical_labels_train) 
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		self.debug_log.info('decisionTree: Training phase completed')
		print('Training phase DT completed')
		
		#Test phase
		test_time_begin = time.time()
		predictions_DT = clf_DT.predict(samples_test)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		self.debug_log.info('decisionTree: Test phase completed')
		print('Test phase DT completed')
		
		self.debug_log.info('Ending execution of decisionTree approach')
		
		########################################
		
		
		
		self.debug_log.info('Starting execution of gaussianNB approach')
			
			
		# Define GaussianNB classifier
		clf_GNB = GaussianNB()
		self.debug_log.info('gaussianNB: classifier defined')

			
		# Training phase
		train_time_begin = time.time()
		clf_GNB.fit(samples_train, categorical_labels_train)
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		self.debug_log.info('GaussianNB: Training phase completed')
		print('Training phase GNB completed')
		
		
		#Test phase
		test_time_begin = time.time()
		predictions_GNB = clf_GNB.predict(samples_test)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		self.debug_log.info('GaussianNB: Test phase completed')
		print('Test phase GNB completed')
		
		self.debug_log.info('Ending execution of GaussianNB approach')
				
		
		#####################################
			
		
		#Define Majority Voting Hard Combiner
		predictions = self.Predict_MajorityVoting_Combiner([predictions_RF, predictions_DT, predictions_GNB])
		
		
		##########################################
		
		#Perfomance Evaluation  -------------------------------------------------------------------
		
		predictions_RF = clf_RF.predict(samples_train)
		predictions_DT = clf_DT.predict(samples_train)
		predictions_GNB = clf_GNB.predict(samples_train)
		training_predictions = self.Predict_MajorityVoting_Combiner([predictions_RF, predictions_DT, predictions_GNB])
		
		# Accuracy, F-measure, and g-mean
		accuracy = sklearn.metrics.accuracy_score(categorical_labels_test, predictions)
		macro_recall = sklearn.metrics.recall_score(categorical_labels_test, predictions,average='macro') 
		macro_precision=numpy.mean(sklearn.metrics.precision_score(categorical_labels_test, predictions,average=None))
		fmeasure = sklearn.metrics.f1_score(categorical_labels_test, predictions, average='macro')
		gmean = self.compute_g_mean(categorical_labels_test, predictions)
		macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_test, predictions, average=None))
		
		# Accuracy, F-measure, and g-mean on training set
		training_accuracy = sklearn.metrics.accuracy_score(categorical_labels_train, training_predictions)
		training_macro_recall = sklearn.metrics.recall_score(categorical_labels_train, training_predictions,average='macro')
		training_macro_precision=numpy.mean(sklearn.metrics.precision_score(categorical_labels_train, training_predictions,average=None))
		training_fmeasure = sklearn.metrics.f1_score(categorical_labels_train, training_predictions, average='macro')
		training_gmean = self.compute_g_mean(categorical_labels_train, training_predictions)
		training_macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_train, training_predictions, average=None))
 
		
		# Confusion matrix
		norm_cnf_matrix = self.compute_norm_confusion_matrix(categorical_labels_test, predictions)
		
		# Top-K accuracy
		topk_accuracies = []

		# Accuracy, F-measure, g-mean, and confusion matrix with reject option (filtered)
		filtered_categorical_labels_test_list = []
		filtered_predictions_list = []
		filtered_classified_ratios = []
		filtered_accuracies = []
		filtered_fmeasures = []
		filtered_gmeans = []
		filtered_macro_gmeans = []
		filtered_norm_cnf_matrices = []
			
		###############################################
		
		return predictions, accuracy,macro_recall,macro_precision, fmeasure, gmean, macro_gmean, training_accuracy, training_macro_recall, training_macro_precision,training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios
		
			
	def Predict_MajorityVoting_Combiner(self,prediction_list) :
		voter = [0] * len(self.label_class) #vettore contenente il  conteggio dei voti per le classi. 
		decision= [] #lista delle decisioni/predizioni
		
		for i in range(0,len(prediction_list[0])): #ciclo sulle predizioni dei classificatori
			for j in range(0,len(prediction_list)): #ciclo sui classificatori 
				voter [prediction_list[j][i]] +=1 # incremento del voto per la classe decisa
				
				
			#Regola Tie-Breaking	
			if voter.count(max(voter))<2: 
				decision_class=voter.index(max(voter))
			else:
				decision_class=random.choice([i for i,x in enumerate(voter) if x==max(voter)])
			voter=[0]*len(self.label_class)
			
			decision.append(decision_class)
		
		return decision
		
		
	def Predict_WMV_Combiner(self, prediction_list,confidence_vector):

		
		voter = [0] * len(self.label_class)
		decision= []
		for i in range(0,len(prediction_list[0])):	#ciclo sulle predizioni dei classificatori								
			for j in range(0,len(prediction_list)): #ciclo sui classificatori 
				voter [prediction_list[j][i]] +=confidence_vector[j] #calcolo pesi per le classi

			#Tie-Breaking
			if voter.count(max(voter))<2:
				decision_class=voter.index(max(voter))
			else:

				decision_class=random.choice([i for i,x in enumerate(voter) if x==max(voter)])
			
			voter=[0]*len(self.label_class)
			decision.append(decision_class)

			
		return decision
	

	def Fit_WMV_Combiner(self, prediction_list,validation_labels):
		
		confidence_vector= [0] * len(prediction_list) #vettore che contiene i pesi dei classificatori

		#Train phase
		for i in range(0,len(prediction_list[0])): #ciclo sulle predizioni dei classificatori
			for j in range(0,len(prediction_list)): #ciclo sui classificatori 
					if validation_labels[i] == prediction_list[j][i]: #incremento punteggio del classificatore che decide correttamente
						confidence_vector[j] +=1
		
	
		confidence_vector=[float(i)/len(prediction_list[0]) for i in confidence_vector] #accuratezza per classificatore
		
		#WMV (Kuncheva)
		for i in range(0,len(prediction_list)):	
			if 	(float(confidence_vector[i])) != 0 and (1-float(confidence_vector[i])) != 0 :
				confidence_vector[i]= math.log(float(confidence_vector[i])/(1-float(confidence_vector[i]))) + math.log(len(self.label_class) - 1)
			else :
				if (float(confidence_vector[i])) == 0:  
					confidence_vector[i]= math.log((sys.float_info.epsilon)/(1-float(confidence_vector[i]))) + math.log(len(self.label_class) - 1)
				else : 
					confidence_vector[i]= math.log((float(confidence_vector[i])/(1- sys.float_info.epsilon))) + math.log(len(self.label_class) - 1)
		
		return confidence_vector
				
	#Hard Trainable : WMC	
	def WeightedMajorityVotingCombiner_10Fold(self,samples_train, categorical_labels_train, samples_test, categorical_labels_test):
		
		# random seed 
		seed = int(time.time())
		
		#Training set splitting : 65% Training, 35% Validation
		
		# Splitting training set in : training e validation set
		training_set, validation_set, training_labels, validation_labels = train_test_split(samples_train,categorical_labels_train,test_size=0.65,train_size=0.35,random_state=seed,stratify=categorical_labels_train)
		
				
	
		self.train_time = 0
		self.test_time = 0
		self.debug_log.info('Starting execution of MultiClassification approach')
		
		###########################################
		
		
		estimators_number = 150
		
		# Define RF classifier
		clf_RF = RandomForestClassifier(criterion='gini', max_features='sqrt', n_estimators=estimators_number)
		self.debug_log.info('taylor2016appscanner_RF: classifier defined')
		
		# Training phase
		train_time_begin = time.time()
		clf_RF.fit(training_set, training_labels) 
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		self.debug_log.info('taylor2016appscanner_RF: Training phase completed')

		

		#Validattion phase	
		test_time_begin = time.time()
		predictions_RF = clf_RF.predict(validation_set)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		self.debug_log.info('taylor2016appcanner_RF: Test phase completed')

		
		self.debug_log.info('Ending execution of taylor2016appscanner_RF approach')
		
		#################################
	
		self.debug_log.info('Starting execution of decisionTree approach')

	
		# Define DT classifier 
		clf_DT = DecisionTreeClassifier(criterion='gini', max_features='sqrt')
		self.debug_log.info('decisionTree: classifier defined')
		
		# Training phase
		train_time_begin = time.time()
		clf_DT.fit(training_set, training_labels) 
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		self.debug_log.info('decisionTree: Training phase completed')
		
		#Test phase
		test_time_begin = time.time()
		predictions_DT = clf_DT.predict(validation_set)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		self.debug_log.info('decisionTree: Test phase completed')
		
		
		self.debug_log.info('Ending execution of decisionTree approach')
		
		########################################
		
		
		self.debug_log.info('Starting execution of gaussianNB approach')
			
			
		# Define GaussianNB classifier
		clf_GNB = GaussianNB()
		#clf_GNB = BernoulliNB()
		self.debug_log.info('gaussianNB: classifier defined')

			
		# Training phase
		train_time_begin = time.time()
		clf_GNB.fit(training_set, training_labels)
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		self.debug_log.info('GaussianNB: Training phase completed')

		
		
		#Test phase
		test_time_begin = time.time()
		predictions_GNB = clf_GNB.predict(validation_set)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		self.debug_log.info('GaussianNB: Test phase completed')
	
		
		self.debug_log.info('Ending execution of GaussianNB approach')
						
		#####################################			
		
		#Train phase Combiner (Weighted Majority Voting)	
		train_time_begin = time.time()
		confidence_vector=self.Fit_WMV_Combiner([predictions_RF, predictions_DT, predictions_GNB], validation_labels)
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		print('Training and Validation phases completed')

		
		#Test phase Combiner
		
		#RF
		test_time_begin = time.time()
		predictions_RF = clf_RF.predict(samples_test)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		#DT
		test_time_begin = time.time()
		predictions_DT = clf_DT.predict(samples_test)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		#GNB
		est_time_begin = time.time()
		predictions_GNB = clf_GNB.predict(samples_test)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		
		#Combiner Testing
		predictions=self.Predict_WMV_Combiner([predictions_RF, predictions_DT, predictions_GNB],confidence_vector)

		
		
		##########################################
		#Perfomance Evaluation  -------------------------------------------------------------------
		
		predictions_RF = clf_RF.predict(validation_set)
		predictions_DT = clf_DT.predict(validation_set)
		predictions_GNB = clf_GNB.predict(validation_set)
		
		#training_predictions = predizioni relative validation_set
		training_predictions = self.Predict_WMV_Combiner([predictions_RF, predictions_DT, predictions_GNB],confidence_vector)
			
		
		# Accuracy, F-measure, and g-mean
		accuracy = sklearn.metrics.accuracy_score(categorical_labels_test, predictions)
		macro_recall = sklearn.metrics.recall_score(categorical_labels_test, predictions,average='macro') 
		macro_precision=numpy.mean(sklearn.metrics.precision_score(categorical_labels_test, predictions,average=None))
		fmeasure = sklearn.metrics.f1_score(categorical_labels_test, predictions, average='macro')
		gmean = self.compute_g_mean(categorical_labels_test, predictions)
		macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_test, predictions, average=None))
		
		# Accuracy, F-measure, and g-mean on training set
		training_accuracy = sklearn.metrics.accuracy_score(validation_labels, training_predictions)
		training_macro_recall = sklearn.metrics.recall_score(validation_labels, training_predictions,average='macro')
		training_macro_precision=numpy.mean(sklearn.metrics.precision_score(validation_labels, training_predictions,average=None))
		training_fmeasure = sklearn.metrics.f1_score(validation_labels, training_predictions, average='macro')
		training_gmean = self.compute_g_mean(validation_labels, training_predictions)
		training_macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(validation_labels, training_predictions, average=None))
 
		# Confusion matrix
		norm_cnf_matrix = self.compute_norm_confusion_matrix(categorical_labels_test, predictions)
		
		# Top-K accuracy
		topk_accuracies = []

		# Accuracy, F-measure, g-mean, and confusion matrix with reject option (filtered)
		filtered_categorical_labels_test_list = []
		filtered_predictions_list = []
		filtered_classified_ratios = []
		filtered_accuracies = []
		filtered_fmeasures = []
		filtered_gmeans = []
		filtered_macro_gmeans = []
		filtered_norm_cnf_matrices = []
		
		
		###############################################
		
		return predictions, accuracy, macro_recall,macro_precision,fmeasure, gmean, macro_gmean, training_accuracy,training_macro_recall, training_macro_precision, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios
	
	#Soft no-Trainable : Avg,Min,Max
	def SoftCombiner_10Fold (self,samples_train,categorical_labels_train,samples_test,categorical_labels_test):
		
		self.train_time = 0
		self.test_time = 0
		self.debug_log.info('Starting execution of Soft MultiClassification approach')

		###########################################
				
		estimators_number = 150
		
		# Define RF classifier 
		clf_RF = RandomForestClassifier(criterion='gini', max_features='sqrt', n_estimators=estimators_number)
		self.debug_log.info('taylor2016appscanner_RF: classifier defined')
		
		# Training phase
		train_time_begin = time.time()
		clf_RF.fit(samples_train, categorical_labels_train) 
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		self.debug_log.info('taylor2016appscanner_RF: Training phase completed')
		print('Training phase RF completed')
		
		
		#Test phase	
		test_time_begin = time.time()
		predictions_RF = clf_RF.predict_proba(samples_test)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		self.debug_log.info('taylor2016appcanner_RF: Test phase completed')
		print('Test phase RF completed')
		
		self.debug_log.info('Ending execution of taylor2016appscanner_RF approach')
		
		#################################
		
		self.debug_log.info('Starting execution of decisionTree approach')

	
		# Define DT classifier
		clf_DT = DecisionTreeClassifier(criterion='gini', max_features='sqrt')
		self.debug_log.info('decisionTree: classifier defined')
		
		# Training phase
		train_time_begin = time.time()
		clf_DT.fit(samples_train, categorical_labels_train) 
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		self.debug_log.info('decisionTree: Training phase completed')
		print('Training phase DT completed')
		
		#Test phase
		test_time_begin = time.time()
		predictions_DT = clf_DT.predict_proba(samples_test)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		self.debug_log.info('decisionTree: Test phase completed')
		print('Test phase DT completed')
		
		self.debug_log.info('Ending execution of decisionTree approach')
		
		########################################
		
		
		
		self.debug_log.info('Starting execution of gaussianNB approach')
			
			
		# Define GaussianNB classifier
		clf_GNB = GaussianNB()
		self.debug_log.info('gaussianNB: classifier defined')

			
		# Training phase
		train_time_begin = time.time()
		clf_GNB.fit(samples_train, categorical_labels_train)
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		self.debug_log.info('GaussianNB: Training phase completed')
		print('Training phase GNB completed')
		
		
		#Test phase
		test_time_begin = time.time()
		predictions_GNB = clf_GNB.predict_proba(samples_test)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		self.debug_log.info('GaussianNB: Test phase completed')
		print('Test phase GNB completed')
		
		self.debug_log.info('Ending execution of GaussianNB approach')
				
		
		#####################################
			
		
		#Define Soft Combiner
		predictions,soft_comb_values = self.Predict_SoftCombiner([predictions_RF, predictions_DT, predictions_GNB],'max')
		
			

		##########################################
		#Perfomance Evaluation  -------------------------------------------------------------------
	
		predictions_RF = clf_RF.predict_proba(samples_train)
		predictions_DT = clf_DT.predict_proba(samples_train)
		predictions_GNB = clf_GNB.predict_proba(samples_train)
		
	
		training_predictions,soft_training_values = self.Predict_SoftCombiner([predictions_RF, predictions_DT, predictions_GNB],'max')
		
		
		# Accuracy, F-measure, and g-mean
		accuracy = sklearn.metrics.accuracy_score(categorical_labels_test, predictions)
		macro_recall = sklearn.metrics.recall_score(categorical_labels_test, predictions,average='macro') 
		macro_precision=numpy.mean(sklearn.metrics.precision_score(categorical_labels_test, predictions,average=None))
		fmeasure = sklearn.metrics.f1_score(categorical_labels_test, predictions, average='macro')
		gmean = self.compute_g_mean(categorical_labels_test, predictions)
		macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_test, predictions, average=None))
		
		# Accuracy, F-measure, and g-mean on training set
		training_accuracy = sklearn.metrics.accuracy_score(categorical_labels_train, training_predictions)
		training_macro_recall = sklearn.metrics.recall_score(categorical_labels_train, training_predictions,average='macro')
		training_macro_precision=numpy.mean(sklearn.metrics.precision_score(categorical_labels_train, training_predictions,average=None))
		training_fmeasure = sklearn.metrics.f1_score(categorical_labels_train, training_predictions, average='macro')
		training_gmean = self.compute_g_mean(categorical_labels_train, training_predictions)
		training_macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_train, training_predictions, average=None))
 	
		
		# Confusion matrix
		norm_cnf_matrix = self.compute_norm_confusion_matrix(categorical_labels_test, predictions)
			
		# Top-K accuracy
		topk_accuracies = []
		for k in self.K:
				try:
					topk_accuracy = self.compute_topk_accuracy(soft_comb_values, categorical_labels_test, k)
					topk_accuracies.append(topk_accuracy)
				except Exception as exc:
					sys.stderr.write('%s\n' % exc)
		
		# Accuracy, F-measure, g-mean, and confusion matrix with reject option (filtered)
		filtered_categorical_labels_test_list = []
		filtered_predictions_list = []
		filtered_classified_ratios = []
		filtered_accuracies = []
		filtered_fmeasures = []
		filtered_gmeans = []
		filtered_macro_gmeans = []
		filtered_norm_cnf_matrices = []
		
	
		for gamma in self.gamma_range:
			categorical_labels_test_filtered, filtered_predictions, filtered_accuracy, filtered_fmeasure, filtered_gmean, filtered_macro_gmean, filtered_norm_cnf_matrix, filtered_classified_ratio = \
			self.compute_classifier_filtered_performance_soft_comb(predictions, soft_comb_values, categorical_labels_test, gamma)
			filtered_categorical_labels_test_list.append(categorical_labels_test_filtered)
			filtered_predictions_list.append(filtered_predictions)
			filtered_accuracies.append(filtered_accuracy)
			filtered_fmeasures.append(filtered_fmeasure)
			filtered_gmeans.append(filtered_gmean)
			filtered_macro_gmeans.append(filtered_macro_gmean)
			filtered_norm_cnf_matrices.append(filtered_norm_cnf_matrix)
			filtered_classified_ratios.append(filtered_classified_ratio)
		
		###############################################
		
		return predictions, accuracy, macro_recall,macro_precision,fmeasure, gmean, macro_gmean, training_accuracy,training_macro_recall,training_macro_precision, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios
				
				
	def Predict_SoftCombiner (self, prediction_list, type_comb = 'max'):	
		
	
		class_prob= [0] * len(prediction_list[0][0])  #vettore probabilità classi (Avg,max,min)
		temp = [0] * len(prediction_list) #vettore temporaneo per mantenere i sotf output dei classificatori relativi ad una classe
		sum_temp =0
		decision = []
		if type_comb != 'avg' and type_comb !='min' and type_comb != 'max':
			print ("input type not correct")
			self.debug_log.error('Soft Combiner INPUT TYPE not correct')
		
		else :
			soft_comb_values=[]
			#ciclo biflusso
			for i in range(0,len(prediction_list[0])): #ciclo sulle predizioni dei classificatori
				#ciclo class
				for j in range(0,len(prediction_list[0][0])): #ciclo sulle classi
					#ciclo classifier					
					for k in range(0,len(prediction_list)):  #ciclo sui classificatori
						if type_comb == 'avg':
							sum_temp = sum_temp + prediction_list[k][i][j]					
						
						elif type_comb == 'max' or type_comb == 'min':
							temp[k]=prediction_list[k][i][j]	
													
					if type_comb == 'max' :
						class_prob[j]=max(temp)
					
					elif type_comb == 'min' :
						class_prob[j]=min(temp)
				
					elif type_comb == 'avg' :
						try:
							class_prob[j]=sum_temp / len(prediction_list)
							sum_temp = 0
						except ZeroDivisionError as Err:
							print("Soft Combinare avg division by zero")
					
				
				#normalizzazione
				soft_comb_values.append([float(i)/sum(class_prob) for i in class_prob])
				
				#Tie-Breaking
				if class_prob.count(max(class_prob))<2:
					decision_class=class_prob.index(max(class_prob))
				else:
					decision_class=random.choice([i for i,x in enumerate(class_prob) if x==max(class_prob)])
				
				decision.append(decision_class)
				class_prob = [0] * len(prediction_list[0][0])
				temp = [0] * len(prediction_list)
				
		soft_comb_values=numpy.asarray(soft_comb_values)
		return decision, soft_comb_values
		
		
	def Fit_Recall (self, prediction_list, validation_labels):		
	
		confidence_class_matrix = []
		missclassification = [] # per ogni classe conta quante volte si è sbagliato per quella classe ( false negative )
		offset= [0] * len(self.label_class)
		
		#Creazione struttura matrice True Positive  e matrice False Negative
		for i in range(0,len(prediction_list)): #ciclo sui classificatori
			confidence_class= [0] * len(self.label_class)
			confidence_class_matrix.append(confidence_class)
			mis_class= [0] * len(self.label_class)
			missclassification.append(mis_class)
		
		for i in range(0,len(prediction_list[0])): #ciclo sulle predizioni dei classificatori
			for j in range(0,len(prediction_list)): #ciclo su ogni classificatore
					if validation_labels[i] == prediction_list[j][i]: 
						confidence_class_matrix[j][prediction_list[j][i]] +=1 #incremento dei True Positive per classificatore
					else :
						missclassification[j][validation_labels[i]] +=1 #incremento dei False Negative per classificatore

		#Recall per Classe : TP / (TP + FN)	
		for i in range(0,len(prediction_list)):	  #ciclo su ogni classificatore
			for j in range(0,len(confidence_class_matrix[0])):
				confidence_class_matrix[i][j]=float(confidence_class_matrix[i][j])/(float(confidence_class_matrix[i][j]) + float(missclassification[i][j]))
		
		
		# Calcolo Offset per classe
		for i in range(0, len(self.label_class)):
			for j in range(0, len(prediction_list)):
				if (1-float(confidence_class_matrix[j][i])) != 0:
					offset[i]= offset[i] + math.log((1 - float(confidence_class_matrix[j][i])))
				else :
					offset[i]= offset[i] + math.log(sys.float_info.epsilon)
		

		
		#Recall (Kuncheva)
		for i in range(0,len(prediction_list)):	
			for j in range(0,len(confidence_class_matrix[0])):
				if  (1- float(confidence_class_matrix[i][j])) != 0 and  (float(confidence_class_matrix[i][j])) != 0 :
					confidence_class_matrix[i][j]= math.log(float(confidence_class_matrix[i][j])) - math.log((1 - float(confidence_class_matrix[i][j]))) + math.log(len(self.label_class) -1)
				else :
					if(1- float(confidence_class_matrix[i][j])) == 0 :
						confidence_class_matrix[i][j]= math.log(float(confidence_class_matrix[i][j])) - math.log(sys.float_info.epsilon) + math.log(len(self.label_class) -1)
					else : 
						confidence_class_matrix[i][j]= math.log(sys.float_info.epsilon) - math.log((1 - float(confidence_class_matrix[i][j])))  + math.log(len(self.label_class) -1)
	
		return confidence_class_matrix,offset
	
	
	def Predict_Recall (self, prediction_list, confidence_class_matrix,offset):
		
		voter = [0] * len(self.label_class) 
		decision= []
		
		# Offset per classe
		for i in range(0,len(self.label_class)):
			voter[i] = offset[i]
		
		for i in range(0,len(prediction_list[0])):									
			for j in range(0,len(prediction_list)):
				voter [prediction_list[j][i]] +=confidence_class_matrix[j][prediction_list[j][i]]
			#Tie-Breaking
			if voter.count(max(voter))<2: 
				decision_class=voter.index(max(voter))
			else:		
				decision_class=random.choice([i for i,x in enumerate(voter) if x==max(voter)])
			
			voter=[0]*len(self.label_class)
			decision.append(decision_class)

			
		return decision

		
	#Hard Trainable : Recall Combiner
	def Recall_Combiner_10Fold (self,samples_train, categorical_labels_train, samples_test, categorical_labels_test):
		
		# random seed 
		seed = int(time.time())
		
		#Training set splitting : 65% Training, 35% Validation
		
		# Training_set
		training_set, validation_set, training_labels, validation_labels = train_test_split(samples_train,categorical_labels_train,test_size=0.7,train_size=0.3,random_state=seed,stratify=categorical_labels_train)
		
				
	
		self.train_time = 0
		self.test_time = 0
		self.debug_log.info('Starting execution of MultiClassification approach')
		
		###########################################
		
		
		estimators_number =150
		
		# Define RF classifier
		clf_RF = RandomForestClassifier(criterion='gini', max_features='sqrt', n_estimators=estimators_number)
		self.debug_log.info('taylor2016appscanner_RF: classifier defined')
		
		# Training phase
		train_time_begin = time.time()
		clf_RF.fit(training_set, training_labels) 
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		self.debug_log.info('taylor2016appscanner_RF: Training phase completed')

		

		#Validattion phase	
		test_time_begin = time.time()
		predictions_RF = clf_RF.predict(validation_set)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		self.debug_log.info('taylor2016appcanner_RF: Test phase completed')

		
		self.debug_log.info('Ending execution of taylor2016appscanner_RF approach')
		
		#################################
	
		self.debug_log.info('Starting execution of decisionTree approach')

	
		# Define DT classifier 
		clf_DT = DecisionTreeClassifier(criterion='gini', max_features='sqrt')
		self.debug_log.info('decisionTree: classifier defined')
		
		# Training phase
		train_time_begin = time.time()
		clf_DT.fit(training_set, training_labels) 
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		self.debug_log.info('decisionTree: Training phase completed')
		
		
		#Test phase
		test_time_begin = time.time()
		predictions_DT = clf_DT.predict(validation_set)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		self.debug_log.info('decisionTree: Test phase completed')
		
		
		self.debug_log.info('Ending execution of decisionTree approach')
		
		########################################
		
		
		self.debug_log.info('Starting execution of gaussianNB approach')
			
			
		# Define GaussianNB classifier
		clf_GNB = GaussianNB()
		#clf_GNB = BernoulliNB()
		self.debug_log.info('gaussianNB: classifier defined')

			
		# Training phase
		train_time_begin = time.time()
		clf_GNB.fit(training_set, training_labels)
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		self.debug_log.info('GaussianNB: Training phase completed')
		
		
		
		#Test phase
		test_time_begin = time.time()
		predictions_GNB = clf_GNB.predict(validation_set)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		self.debug_log.info('GaussianNB: Test phase completed')
		
		
		self.debug_log.info('Ending execution of GaussianNB approach')
						
		#####################################			
		
		#Train phase Combiner (Recall)	

		train_time_begin = time.time()
		confidence_class_matrix,offset =self.Fit_Recall([predictions_RF, predictions_DT, predictions_GNB], validation_labels)
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		print('Training and Validation phases completed')
		
		
		#Test phase Combiner
		
		#RF
		test_time_begin = time.time()
		predictions_RF = clf_RF.predict(samples_test)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		#DT
		test_time_begin = time.time()
		predictions_DT = clf_DT.predict(samples_test)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		#GNB
		est_time_begin = time.time()
		predictions_GNB = clf_GNB.predict(samples_test)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		
		#Combiner Testing
		predictions=self.Predict_Recall([predictions_RF, predictions_DT, predictions_GNB],confidence_class_matrix,offset)

		
		
		##########################################
		#Perfomance Evaluation  -------------------------------------------------------------------
		
				
		predictions_RF = clf_RF.predict(validation_set)
		predictions_DT = clf_DT.predict(validation_set)
		predictions_GNB = clf_GNB.predict(validation_set)
		
		training_predictions = self.Predict_Recall([predictions_RF, predictions_DT, predictions_GNB],confidence_class_matrix,offset)
		
		# Accuracy, F-measure, and g-mean
		accuracy = sklearn.metrics.accuracy_score(categorical_labels_test, predictions)
		macro_recall = sklearn.metrics.recall_score(categorical_labels_test, predictions,average='macro') 
		macro_precision=numpy.mean(sklearn.metrics.precision_score(categorical_labels_test, predictions,average=None))
		fmeasure = sklearn.metrics.f1_score(categorical_labels_test, predictions, average='macro')
		gmean = self.compute_g_mean(categorical_labels_test, predictions)
		macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_test, predictions, average=None))
		
		# Accuracy, F-measure, and g-mean on training set
		training_accuracy = sklearn.metrics.accuracy_score(validation_labels, training_predictions)
		training_macro_recall = sklearn.metrics.recall_score(validation_labels, training_predictions,average='macro')
		training_macro_precision=numpy.mean(sklearn.metrics.precision_score(validation_labels, training_predictions,average=None))
		training_fmeasure = sklearn.metrics.f1_score(validation_labels, training_predictions, average='macro')
		training_gmean = self.compute_g_mean(validation_labels, training_predictions)
		training_macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(validation_labels, training_predictions, average=None))
 
		# Confusion matrix
		norm_cnf_matrix = self.compute_norm_confusion_matrix(categorical_labels_test, predictions)
		
		# Top-K accuracy
		topk_accuracies = []

		# Accuracy, F-measure, g-mean, and confusion matrix with reject option (filtered)
		filtered_categorical_labels_test_list = []
		filtered_predictions_list = []
		filtered_classified_ratios = []
		filtered_accuracies = []
		filtered_fmeasures = []
		filtered_gmeans = []
		filtered_macro_gmeans = []
		filtered_norm_cnf_matrices = []
		
		
		###############################################
		
		return predictions, accuracy, macro_recall,macro_precision,fmeasure, gmean, macro_gmean, training_accuracy,training_macro_recall, training_macro_precision, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios
	
	
	def Fit_NB_Combiner (self, prediction_list, validation_labels):
	
		conf_matrix_list = []
		
		for i in range(0, len(prediction_list)):		
			norm_cnf_matrix = self.compute_norm_confusion_matrix(validation_labels, prediction_list[i])				
			conf_matrix_list.append(norm_cnf_matrix)
		
		return conf_matrix_list
	
	
	def Predict_NB_Combiner (self, prediction_list, conf_matrix_list):	
		
		naive_vec=[1]*len(self.label_class)
		decision= []
		
		# NB Maximum-Likelihood
		for i in range(0, len(prediction_list[0])): #ciclo sulle predizioni dei classificatori
			naive_vec =[1] * len(self.label_class) 
			for j in range(0,len(prediction_list)):  #ciclo su ogni classificatore
				for k in range(0, len(self.label_class)): #ciclo sulle classi
					naive_vec[k]=conf_matrix_list[j][k][prediction_list[j][i]] * naive_vec[k]
			#Tie-Breaking		
			if naive_vec.count(max(naive_vec))<2:
				decision_class=naive_vec.index(max(naive_vec))
			else:		
				decision_class=random.choice([i for i,x in enumerate(naive_vec) if x==max(naive_vec)])
			
			naive_vec=[1]*len(self.label_class)
			decision.append(decision_class)

			
		return decision
			
	#Hard Trainable : Naive Bayes	
	def NB_Trainable_Combiner_10Fold (self,samples_train, categorical_labels_train, samples_test, categorical_labels_test):
		# random seed 
		seed = int(time.time())
		
		#Training set splitting : 65% Training, 35% Validation
		
		# Training_set
		training_set, validation_set, training_labels, validation_labels = train_test_split(samples_train,categorical_labels_train,test_size=0.7,train_size=0.3,random_state=seed,stratify=categorical_labels_train)
		
				
	
		self.train_time = 0
		self.test_time = 0
		self.debug_log.info('Starting execution of MultiClassification approach')
		
		###########################################
		
		
		estimators_number = 150
		
		# Define RF classifier
		clf_RF = RandomForestClassifier(criterion='gini', max_features='sqrt', n_estimators=estimators_number)
		self.debug_log.info('taylor2016appscanner_RF: classifier defined')
		
		# Training phase
		train_time_begin = time.time()
		clf_RF.fit(training_set, training_labels) 
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		self.debug_log.info('taylor2016appscanner_RF: Training phase completed')

		

		#Validattion phase	
		test_time_begin = time.time()
		predictions_RF = clf_RF.predict(validation_set)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		self.debug_log.info('taylor2016appcanner_RF: Test phase completed')
		
		
		self.debug_log.info('Ending execution of taylor2016appscanner_RF approach')
		
		#################################
	
		self.debug_log.info('Starting execution of decisionTree approach')

	
		# Define DT classifier 
		clf_DT = DecisionTreeClassifier(criterion='gini', max_features='sqrt')
		self.debug_log.info('decisionTree: classifier defined')
		
		# Training phase
		train_time_begin = time.time()
		clf_DT.fit(training_set, training_labels) 
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		self.debug_log.info('decisionTree: Training phase completed')
		
		
		#Test phase
		test_time_begin = time.time()
		predictions_DT = clf_DT.predict(validation_set)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		self.debug_log.info('decisionTree: Test phase completed')

		
		self.debug_log.info('Ending execution of decisionTree approach')
		
		########################################
		
		
		self.debug_log.info('Starting execution of gaussianNB approach')
			
			
		# Define GaussianNB classifier
		clf_GNB = GaussianNB()
		#clf_GNB = BernoulliNB()
		self.debug_log.info('gaussianNB: classifier defined')

			
		# Training phase
		train_time_begin = time.time()
		clf_GNB.fit(training_set, training_labels)
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		self.debug_log.info('GaussianNB: Training phase completed')

		
		
		#Test phase
		test_time_begin = time.time()
		predictions_GNB = clf_GNB.predict(validation_set)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		self.debug_log.info('GaussianNB: Test phase completed')
		
		
		self.debug_log.info('Ending execution of GaussianNB approach')
		
		
		############################################
		train_time_begin = time.time()
		conf_matrix_list=self.Fit_NB_Combiner([predictions_RF, predictions_DT, predictions_GNB], validation_labels)
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		print('Training and Validation phases completed')
		
		#Test phase Combiner
		
		#RF
		test_time_begin = time.time()
		predictions_RF = clf_RF.predict(samples_test)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		#DT
		test_time_begin = time.time()
		predictions_DT = clf_DT.predict(samples_test)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		#GNB
		est_time_begin = time.time()
		predictions_GNB = clf_GNB.predict(samples_test)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		
		#Combiner Testing
		predictions=self.Predict_NB_Combiner([predictions_RF, predictions_DT, predictions_GNB],conf_matrix_list)

		
		
		##########################################
		#Perfomance Evaluation  -------------------------------------------------------------------
		
				
		predictions_RF = clf_RF.predict(validation_set)
		predictions_DT = clf_DT.predict(validation_set)
		predictions_GNB = clf_GNB.predict(validation_set)
		
		training_predictions = self.Predict_NB_Combiner([predictions_RF, predictions_DT, predictions_GNB],conf_matrix_list)
		
		# Accuracy, F-measure, and g-mean
		accuracy = sklearn.metrics.accuracy_score(categorical_labels_test, predictions)
		macro_recall = sklearn.metrics.recall_score(categorical_labels_test, predictions,average='macro') 
		macro_precision=numpy.mean(sklearn.metrics.precision_score(categorical_labels_test, predictions,average=None))
		fmeasure = sklearn.metrics.f1_score(categorical_labels_test, predictions, average='macro')
		gmean = self.compute_g_mean(categorical_labels_test, predictions)
		macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_test, predictions, average=None))
		
		
		
		# Accuracy, F-measure, and g-mean on training set
		training_accuracy = sklearn.metrics.accuracy_score(validation_labels, training_predictions)
		training_macro_recall = sklearn.metrics.recall_score(validation_labels, training_predictions,average='macro')
		training_macro_precision=numpy.mean(sklearn.metrics.precision_score(validation_labels, training_predictions,average=None))
		training_fmeasure = sklearn.metrics.f1_score(validation_labels, training_predictions, average='macro')
		training_gmean = self.compute_g_mean(validation_labels, training_predictions)
		training_macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(validation_labels, training_predictions, average=None))
 
		
		
		# Confusion matrix
		norm_cnf_matrix = self.compute_norm_confusion_matrix(categorical_labels_test, predictions)
		
		# Top-K accuracy
		topk_accuracies = []

		# Accuracy, F-measure, g-mean, and confusion matrix with reject option (filtered)
		filtered_categorical_labels_test_list = []
		filtered_predictions_list = []
		filtered_classified_ratios = []
		filtered_accuracies = []
		filtered_fmeasures = []
		filtered_gmeans = []
		filtered_macro_gmeans = []
		filtered_norm_cnf_matrices = []
		
		
		
		
		return predictions, accuracy, macro_recall,macro_precision,fmeasure, gmean, macro_gmean, training_accuracy,training_macro_recall, training_macro_precision, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios
	
	
	def parallel_count(self,prediction_val_start_index,prediction_val_end_index,prediction_list_val, validation_labels, data_vec,output):
		training_vec = [0] * len(prediction_list_val)
		#N_dwl = [0] * len(self.label_class) #N. istanze di validation corrispondenti all' L-esima classe appartenente al d-test. d-test =sottoinsieme di prediction_list_val=prediction_list_test
		#N_wl = [0] * len(self.label_class) #N. istanze di validation corrispondenti all' L-esima classe
		N_dwl = numpy.zeros(len(self.label_class))
		N_wl= numpy.zeros(len(self.label_class))
	
	
		for k in range(prediction_val_start_index,prediction_val_end_index):
			for l in range(0, len(prediction_list_val)):
				training_vec[l]= prediction_list_val[l][k]
			if training_vec==data_vec:
				N_dwl[validation_labels[k]] +=1
			N_wl[validation_labels[k]] +=1
		
		output.put( (N_dwl,N_wl))

		
	def Fit_Predict_BKS_Combiner (self, prediction_list_val, validation_labels, prediction_list_test):
	
		output = mp.Queue()
		num_proc=4
		
		BSK_ML_vec= [0] * len(self.label_class)
		decision = []
		data_vec = [0] * len(prediction_list_test)
		#training_vec = [0] * len(prediction_list_val)
		#N_dwl_tot = [0] * len(self.label_class) #N. istanze di validation corrispondenti all' L-esima classe appartenente al d-test. d-test =sottoinsieme di prediction_list_val=prediction_list_test
		#N_wl_tot = [0] * len(self.label_class) #N. istanze di validation corrispondenti all' L-esima classe
		N_dwl_tot = numpy.zeros(len(self.label_class))
		N_wl_tot = numpy.zeros(len(self.label_class))
		
		print("start_BKS")
		for i in range(0, len(prediction_list_test[0])):
			#time_begin = time.time()
			for j in range(0, len(prediction_list_test)):
				data_vec[j]= prediction_list_test[j][i]
			
			# Conteggio parallelo
			start_index= [0] * num_proc
			end_index =  [0] * num_proc
			
			
			
			#set index
			chunk= math.floor(len(prediction_list_val[0])/num_proc)
			for n in range(0,num_proc):
				start_index[n]= n * chunk
				end_index[n]= start_index[n] + chunk
			end_index[num_proc-1]=len(prediction_list_val[0])
			
			processes = [mp.Process(target=self.parallel_count, args=(start_index[x],end_index[x],prediction_list_val, validation_labels, data_vec,output)) for x in range(num_proc)]
			
			# Run processes
			for p in processes:
				p.start()
				
			# Exit the completed processes
			for p in processes:
				p.join()	
				
				
			# Get process results from the output queue
			results = [output.get() for p in processes]
			
			for r in results:
				N_dwl_tot = numpy.add(N_dwl_tot,r[0])
				N_wl_tot = numpy.add(N_dwl_tot,r[1])
			

			#print(N_dwl_tot)
			#print(N_wl_tot)
			
			for r in range(0, len(self.label_class)):
				BSK_ML_vec[r]= (float(N_dwl_tot[r])+1) / (float(N_wl_tot[r])+ (len(self.label_class))**(len(prediction_list_test)))
			
			#Tie-Breaking			
			if BSK_ML_vec.count(max(BSK_ML_vec))<2:
				decision_class=BSK_ML_vec.index(max(BSK_ML_vec))
			else:
				decision_class=random.choice([i for i,x in enumerate(BSK_ML_vec) if x==max(BSK_ML_vec)])
			
			BSK_ML_vec=[0]*len(self.label_class)
			N_dwl_tot = numpy.zeros(len(self.label_class))
			N_wl_tot = numpy.zeros(len(self.label_class))
			
			decision.append(decision_class)
			#time_end = time.time()
			#print("time",time_end - time_begin)
			
		return decision

		
	def BKS_Combiner_10Fold (self,samples_train, categorical_labels_train, samples_test, categorical_labels_test):	
		
		# random seed 
		seed = int(time.time())
		
		#Training set splitting : 65% Training, 35% Validation
		
		# Training_set
		training_set, validation_set, training_labels, validation_labels = train_test_split(samples_train,categorical_labels_train,test_size=0.7,train_size=0.3,random_state=seed,stratify=categorical_labels_train)
		
				

		self.train_time = 0
		self.test_time = 0
		self.debug_log.info('Starting execution of MultiClassification approach')
		
		###########################################
		
		
		estimators_number = 150
		
		# Define RF classifier
		clf_RF = RandomForestClassifier(criterion='gini', max_features='sqrt', n_estimators=estimators_number)
		self.debug_log.info('taylor2016appscanner_RF: classifier defined')
		
		# Training phase
		train_time_begin = time.time()
		clf_RF.fit(training_set, training_labels) 
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		self.debug_log.info('taylor2016appscanner_RF: Training phase completed')
		
		

		#Validation phase	
		test_time_begin = time.time()
		predictions_RF_val = clf_RF.predict(validation_set)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		self.debug_log.info('taylor2016appcanner_RF: Test phase completed')

		
		self.debug_log.info('Ending execution of taylor2016appscanner_RF approach')
		
		#################################

		self.debug_log.info('Starting execution of decisionTree approach')


		# Define DT classifier 
		clf_DT = DecisionTreeClassifier(criterion='gini', max_features='sqrt')
		self.debug_log.info('decisionTree: classifier defined')
		
		# Training phase
		train_time_begin = time.time()
		clf_DT.fit(training_set, training_labels) 
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		self.debug_log.info('decisionTree: Training phase completed')
		
		#Test phase
		test_time_begin = time.time()
		predictions_DT_val = clf_DT.predict(validation_set)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		self.debug_log.info('decisionTree: Test phase completed')

		
		self.debug_log.info('Ending execution of decisionTree approach')
		
		########################################
		
		
		self.debug_log.info('Starting execution of gaussianNB approach')
			
			
		# Define GaussianNB classifier
		clf_GNB = GaussianNB()
		#clf_GNB = BernoulliNB()
		self.debug_log.info('gaussianNB: classifier defined')

			
		# Training phase
		train_time_begin = time.time()
		clf_GNB.fit(training_set, training_labels)
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		self.debug_log.info('GaussianNB: Training phase completed')

		
		
		#Test phase
		test_time_begin = time.time()
		predictions_GNB_val = clf_GNB.predict(validation_set)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		self.debug_log.info('GaussianNB: Test phase completed')
		
		
		self.debug_log.info('Ending execution of GaussianNB approach')
		
		
		############################################
		
		test_time_begin = time.time()
		predictions_RF = clf_RF.predict(samples_test)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		
		test_time_begin = time.time()
		predictions_DT = clf_DT.predict(samples_test)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		test_time_begin = time.time()
		predictions_GNB = clf_GNB.predict(samples_test)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		train_time_begin = time.time()
		predictions=self.Fit_Predict_BKS_Combiner([predictions_RF_val, predictions_DT_val, predictions_GNB_val], validation_labels, [predictions_RF, predictions_DT, predictions_GNB])
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		print('Training and Validation phases completed')
		
		
		
		
		
		##########################################
		#Perfomance Evaluation  -------------------------------------------------------------------
		
		# N.B. La valutazione delle prestazioni relative alla fase di training vengono evitate perché è richiesto molto tempo per eseguirla		
		
		# training_predictions = predictions=self.Fit_Predict_BKS_Combiner([predictions_RF_val, predictions_DT_val, predictions_GNB_val], validation_labels, [predictions_RF_val, predictions_DT_val, predictions_GNB_val])
		
		# Accuracy, F-measure, and g-mean
		accuracy = sklearn.metrics.accuracy_score(categorical_labels_test, predictions)
		macro_recall = sklearn.metrics.recall_score(categorical_labels_test, predictions,average='macro')
		macro_precision=numpy.mean(sklearn.metrics.precision_score(categorical_labels_test, predictions,average=None))
		fmeasure = sklearn.metrics.f1_score(categorical_labels_test, predictions, average='macro')
		gmean = self.compute_g_mean(categorical_labels_test, predictions)
		macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_test, predictions, average=None))
		
		
		# # Accuracy, F-measure, and g-mean on training set
		# training_accuracy = sklearn.metrics.accuracy_score(validation_labels, training_predictions)
		# training_macro_recall = sklearn.metrics.recall_score(validation_labels, training_predictions,average='macro')
		# training_macro_precision=numpy.mean(sklearn.metrics.precision_score(validation_labels, training_predictions,average=None))
		# training_fmeasure = sklearn.metrics.f1_score(validation_labels, training_predictions, average='macro')
		# training_gmean = self.compute_g_mean(validation_labels, training_predictions)
		# training_macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(validation_labels, training_predictions, average=None))

		training_accuracy = 0
		training_macro_recall = 0
		training_macro_precision=0
		training_fmeasure = 0
		training_gmean = 0
		training_macro_gmean =0
		
		# Confusion matrix
		norm_cnf_matrix = self.compute_norm_confusion_matrix(categorical_labels_test, predictions)
		
		# Top-K accuracy
		topk_accuracies = []

		# Accuracy, F-measure, g-mean, and confusion matrix with reject option (filtered)
		filtered_categorical_labels_test_list = []
		filtered_predictions_list = []
		filtered_classified_ratios = []
		filtered_accuracies = []
		filtered_fmeasures = []
		filtered_gmeans = []
		filtered_macro_gmeans = []
		filtered_norm_cnf_matrices = []
		
		
		###############################################
		
		return predictions, accuracy, macro_recall,macro_precision,fmeasure, gmean, macro_gmean, training_accuracy, training_macro_recall, training_macro_precision,training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios

		
	def Fit_Decision_Template (self, decision_profile_list, validation_labels):
	
		vect_cont = [0] * len(self.label_class)
		decision_template = numpy.zeros((len(self.label_class),len(decision_profile_list) ,len(self.label_class))) #Decision Template [LxKxL]
		
		dp=[] #lista dei profili di decisione corrispondenti ad ogni biflusso
		dp_temp=[] #lista temporanea usata per costruire un decision profile a partire dai sotf output dei classificatori
		
		#costruzione lista decision profile
		for i in range(0,len(decision_profile_list[0])): #ciclo sulle predizioni dei classificatori
			for j in range(0, len(decision_profile_list)): #ciclo sui classificatori
				dp_temp.append(decision_profile_list[j][i])
			dp.append(dp_temp)
			dp_temp=[]
			
		#Calcolo Decision Template(Decision Profile medi)
		for i in range(0,len(validation_labels)): #ciclo sulla ground truth del validation set
			decision_template[validation_labels[i]]= numpy.add(decision_template[validation_labels[i]], dp[i])
			vect_cont[validation_labels[i]] +=1
		
		
		for i in range(0, decision_template.shape[0]): #Ciclo sul numero di classi
			decision_template[i]= decision_template[i]/vect_cont[i] 
			
		return decision_template
	
	
	def Predict_Decision_Template (self, decision_profile_list, decision_template):
		
		mu_DT_SE=[0]* len(self.label_class) #vettore delle "distanze" tra  il Decision Profile e il Decision Template
		decision = []
		soft_comb_values=[]
		
		dp_predict=[]
		dp_temp_predict=[]
		
		for i in range(0,len(decision_profile_list[0])):#ciclo sulle predizioni dei classificatori
			for j in range(0, len(decision_profile_list)): #ciclo sui classificatori
				dp_temp_predict.append(decision_profile_list[j][i])
			dp_predict.append(dp_temp_predict)
			dp_temp_predict=[]
		
		dp_predict=numpy.asarray(dp_predict)
		
		#Distanza Euclidea quadratica
		for i in range(0,len(dp_predict)):
			for j in range(0,len(self.label_class)): #ciclo sulle classi
				mu_DT_SE[j]= 1 - (1/((len(self.label_class))*(len(dp_predict[0])))) * (numpy.linalg.norm(decision_template[j] - dp_predict[i])**2)
				
			
			soft_comb_values.append(mu_DT_SE)
			
			#tie-Braking
			if mu_DT_SE.count(max(mu_DT_SE))<2:
				decision_class=mu_DT_SE.index(max(mu_DT_SE))
			else:
				decision_class=random.choice([i for i,x in enumerate(mu_DT_SE) if x==max(mu_DT_SE)])
			
			mu_DT_SE=[0]*len(self.label_class)
			decision.append(decision_class)
		
		
		#soft value per le filtered performance
		soft_comb_values=numpy.asarray(soft_comb_values)
		
		x_min=soft_comb_values.min()
		x_max=soft_comb_values.max()
					
		for i in range(0,soft_comb_values.shape[0]):
			for j in range(0,soft_comb_values.shape[1]):
				soft_comb_values[i][j]=(soft_comb_values[i][j] - x_min) /(x_max - x_min)
		
		
		
		return decision,soft_comb_values
	
	#Sfot Trainable : DT
	def Decision_Template_Combiner_10Fold (self,samples_train, categorical_labels_train, samples_test, categorical_labels_test ):
		# random seed 
		seed = int(time.time())
		
		#Training set splitting : 65% Training, 35% Validation
		
		# Training_set
		training_set, validation_set, training_labels, validation_labels = train_test_split(samples_train,categorical_labels_train,test_size=0.7,train_size=0.3,random_state=seed,stratify=categorical_labels_train)
		
				
	
		self.train_time = 0
		self.test_time = 0
		self.debug_log.info('Starting execution of MultiClassification approach')
		
		###########################################
		
		
		estimators_number = 150
		
		# Define RF classifier
		clf_RF = RandomForestClassifier(criterion='gini', max_features='sqrt', n_estimators=estimators_number)
		self.debug_log.info('taylor2016appscanner_RF: classifier defined')
		
		# Training phase
		train_time_begin = time.time()
		clf_RF.fit(training_set, training_labels) 
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		self.debug_log.info('taylor2016appscanner_RF: Training phase completed')
		
		

		#Validattion phase	
		test_time_begin = time.time()
		DP_RF = clf_RF.predict_proba(validation_set)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		self.debug_log.info('taylor2016appcanner_RF: Test phase completed')
	
		
		self.debug_log.info('Ending execution of taylor2016appscanner_RF approach')
		
		#################################
	
		self.debug_log.info('Starting execution of decisionTree approach')

	
		# Define DT classifier 
		clf_DT = DecisionTreeClassifier(criterion='gini', max_features='sqrt')
		self.debug_log.info('decisionTree: classifier defined')
		
		# Training phase
		train_time_begin = time.time()
		clf_DT.fit(training_set, training_labels) 
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		self.debug_log.info('decisionTree: Training phase completed')
	
		
		#Test phase
		test_time_begin = time.time()
		DP_DT = clf_DT.predict_proba(validation_set)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		self.debug_log.info('decisionTree: Test phase completed')
		
		
		self.debug_log.info('Ending execution of decisionTree approach')
		
		########################################
		
		
		self.debug_log.info('Starting execution of gaussianNB approach')
			
			
		# Define GaussianNB classifier
		clf_GNB = GaussianNB()
		#clf_GNB = BernoulliNB()
		self.debug_log.info('gaussianNB: classifier defined')

			
		# Training phase
		train_time_begin = time.time()
		clf_GNB.fit(training_set, training_labels)
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		self.debug_log.info('GaussianNB: Training phase completed')
		
		
		
		#Test phase
		test_time_begin = time.time()
		DP_GNB = clf_GNB.predict_proba(validation_set)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		self.debug_log.info('GaussianNB: Test phase completed')
		
		
		self.debug_log.info('Ending execution of GaussianNB approach')
		
		
		############################################
		train_time_begin = time.time()
		decision_template=self.Fit_Decision_Template([DP_RF, DP_DT, DP_GNB], validation_labels)
		train_time_end = time.time()
		self.train_time = self.train_time + (train_time_end - train_time_begin)
		print('Training and Validation phases completed')
		
		#Test phase Combiner
		
		#RF
		test_time_begin = time.time()
		DP_RF = clf_RF.predict_proba(samples_test)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		#DT
		test_time_begin = time.time()
		DP_DT = clf_DT.predict_proba(samples_test)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		#GNB
		est_time_begin = time.time()
		DP_GNB = clf_GNB.predict_proba(samples_test)
		test_time_end = time.time()
		self.test_time = self.test_time + (test_time_end - test_time_begin)
		
		
		#Combiner Testing
		predictions,soft_comb_values=self.Predict_Decision_Template([DP_RF, DP_DT, DP_GNB],decision_template)
	
		##########################################
		#Perfomance Evaluation  -------------------------------------------------------------------
		
				
		DP_RF = clf_RF.predict_proba(validation_set)
		DP_DT = clf_DT.predict_proba(validation_set)
		DP_GNB = clf_GNB.predict_proba(validation_set)
		
		training_predictions,training_soft = self.Predict_Decision_Template([DP_RF, DP_DT, DP_GNB],decision_template)
		
		# Accuracy, F-measure, and g-mean
		accuracy = sklearn.metrics.accuracy_score(categorical_labels_test, predictions)
		macro_recall = sklearn.metrics.recall_score(categorical_labels_test, predictions,average='macro') 
		macro_precision=numpy.mean(sklearn.metrics.precision_score(categorical_labels_test, predictions,average=None))
		fmeasure = sklearn.metrics.f1_score(categorical_labels_test, predictions, average='macro')
		gmean = self.compute_g_mean(categorical_labels_test, predictions)
		macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_test, predictions, average=None))
		
		
		# Accuracy, F-measure, and g-mean on training set
		training_accuracy = sklearn.metrics.accuracy_score(validation_labels, training_predictions)
		training_macro_recall = sklearn.metrics.recall_score(validation_labels, training_predictions,average='macro')
		training_macro_precision=numpy.mean(sklearn.metrics.precision_score(validation_labels, training_predictions,average=None))
		training_fmeasure = sklearn.metrics.f1_score(validation_labels, training_predictions, average='macro')
		training_gmean = self.compute_g_mean(validation_labels, training_predictions)
		training_macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(validation_labels, training_predictions, average=None))
 
		
		# Confusion matrix
		norm_cnf_matrix = self.compute_norm_confusion_matrix(categorical_labels_test, predictions)
		
		# Top-K accuracy
		topk_accuracies = []
		for k in self.K:
				try:
					topk_accuracy = self.compute_topk_accuracy(soft_comb_values, categorical_labels_test, k)
					topk_accuracies.append(topk_accuracy)
				except Exception as exc:
					sys.stderr.write('%s\n' % exc)
		
		# Accuracy, F-measure, g-mean, and confusion matrix with reject option (filtered)
		filtered_categorical_labels_test_list = []
		filtered_predictions_list = []
		filtered_classified_ratios = []
		filtered_accuracies = []
		filtered_fmeasures = []
		filtered_gmeans = []
		filtered_macro_gmeans = []
		filtered_norm_cnf_matrices = []
		
		
	
		for gamma in self.gamma_range:
			categorical_labels_test_filtered, filtered_predictions, filtered_accuracy, filtered_fmeasure, filtered_gmean, filtered_macro_gmean, filtered_norm_cnf_matrix, filtered_classified_ratio = \
			self.compute_classifier_filtered_performance_soft_comb(predictions, soft_comb_values, categorical_labels_test, gamma)
			filtered_categorical_labels_test_list.append(categorical_labels_test_filtered)
			filtered_predictions_list.append(filtered_predictions)
			filtered_accuracies.append(filtered_accuracy)
			filtered_fmeasures.append(filtered_fmeasure)
			filtered_gmeans.append(filtered_gmean)
			filtered_macro_gmeans.append(filtered_macro_gmean)
			filtered_norm_cnf_matrices.append(filtered_norm_cnf_matrix)
			filtered_classified_ratios.append(filtered_classified_ratio)
		
		
		return predictions, accuracy, macro_recall,macro_precision,fmeasure, gmean, macro_gmean, training_accuracy,training_macro_recall, training_macro_precision, training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
		filtered_norm_cnf_matrices, filtered_classified_ratios
	
	
	def Predict_Oracle(self, prediction_list,categorical_labels_test):
		
		predictions=[]
		
		for i,label in enumerate(categorical_labels_test):
			classifier_predictions=[]
			for j in range(0,len(prediction_list)):
				classifier_predictions.append(prediction_list[j][i])
			if label in classifier_predictions :
				predictions.append(label)				
			else : 
				predictions.append(prediction_list[0][i]) #si setta come predizione sbagliata sempre quella del primo classificatore
				
		predictions= numpy.asarray(predictions)
					
		return predictions

	#Oracle	
	def OracleCombiner (self,samples_train,categorical_labels_train,samples_test,categorical_labels_test):
		
		
		self.debug_log.info('Starting execution of Oracle Combiner')

		###########################################
		
		
		estimators_number = 150
		
		# Define RF classifier 
		clf_RF = RandomForestClassifier(criterion='gini', max_features='sqrt', n_estimators=estimators_number)
		self.debug_log.info('taylor2016appscanner_RF: classifier defined')
		
		# Training phase
		clf_RF.fit(samples_train, categorical_labels_train) 
		self.debug_log.info('taylor2016appscanner_RF: Training phase completed')
		print('Training phase RF completed')
		

		#Test phase	
		
		predictions_RF = clf_RF.predict(samples_test)
		
		self.debug_log.info('taylor2016appcanner_RF: Test phase completed')
		print('Test phase RF completed')
		
		self.debug_log.info('Ending execution of taylor2016appscanner_RF approach')
		
		#################################
		
		self.debug_log.info('Starting execution of decisionTree approach')

	
		# Define DT classifier 
		clf_DT = DecisionTreeClassifier(criterion='gini', max_features='sqrt')
		self.debug_log.info('decisionTree: classifier defined')
		
		# Training phase
		
		clf_DT.fit(samples_train, categorical_labels_train) 
		
		self.debug_log.info('decisionTree: Training phase completed')
		print('Training phase DT completed')
		
		#Test phase
		
		predictions_DT = clf_DT.predict(samples_test)
		
		self.debug_log.info('decisionTree: Test phase completed')
		print('Test phase DT completed')
		
		self.debug_log.info('Ending execution of decisionTree approach')
		
		########################################
		
		
		
		self.debug_log.info('Starting execution of gaussianNB approach')
			
			
		# Define GaussianNB classifier
		clf_GNB = GaussianNB()
		self.debug_log.info('gaussianNB: classifier defined')

			
		# Training phase
		
		clf_GNB.fit(samples_train, categorical_labels_train)
		
		self.debug_log.info('GaussianNB: Training phase completed')
		print('Training phase GNB completed')
		
		
		#Test phase
		predictions_GNB = clf_GNB.predict(samples_test)
		
		self.debug_log.info('GaussianNB: Test phase completed')
		print('Test phase GNB completed')
		
		self.debug_log.info('Ending execution of GaussianNB approach')
				
		
		#####################################
			
		
		# ORACLE predictions	
		predictions = self.Predict_Oracle([predictions_RF, predictions_DT, predictions_GNB],categorical_labels_test)
				
		##########################################
		
		#Perfomance Evaluation  -------------------------------------------------------------------
		
		# Accuracy, Recall
		accuracy = sklearn.metrics.accuracy_score(categorical_labels_test, predictions)
		
		macro_recall = sklearn.metrics.recall_score(categorical_labels_test, predictions,average='macro') 												############################
		
			
				
			
		###############################################
		
		return accuracy,macro_recall
			
	
	def stratified_cross_validation(self, dl_model, num_fold, seed, do_expand_dims=True, do_normalization=False, is_multimodal=False):
		'''
		Perform a stratified cross validation of the state-of-the-art DL (multimodal) model given as input
		'''
		print("10 fold")
		self.debug_log.info('Starting execution of stratified cross validation')
		
		if dl_model.__name__ != "OracleCombiner":
		
			with open('%s/performance/%s_performance.dat' % (self.outdir, dl_model.__name__), 'w') as performance_file:
				performance_file.write('%s\tAccuracy\tMacr_Recall\tMacr_Precision\tMacro_F-measure\tG-mean\tMacro_G-mean\n' % dl_model.__name__)
			
			with open('%s/performance/%s_training_performance.dat' % (self.outdir, dl_model.__name__), 'w') as training_performance_file:
				training_performance_file.write('%s\tTraining_Accuracy\tTraining_Macro_Recall\tTraining_Macro_Precision\tTraining_Macro_F-measure\tTraining_G-mean\tTraining_Macro_G-mean\n' % dl_model.__name__)
			
			with open('%s/performance/%s_top-k_accuracy.dat' % (self.outdir, dl_model.__name__), 'w') as topk_accuracy_file:
				topk_accuracy_file.write('%s\t' % dl_model.__name__)
				for k in self.K:
					topk_accuracy_file.write('K=%d\t' % k)
				topk_accuracy_file.write('\n')
			
			with open('%s/performance/%s_filtered_performance.dat' % (self.outdir, dl_model.__name__), 'w') as filtered_perfomance_file:
				filtered_perfomance_file.write('%s\t' % dl_model.__name__)
				for gamma in self.gamma_range:
					filtered_perfomance_file.write('c=%f\t' % gamma)
				filtered_perfomance_file.write('\n')
		else:
		
			with open('%s/performance/%s_performance.dat' % (self.outdir, dl_model.__name__), 'w') as performance_file:
				performance_file.write('%s\tAccuracy\tMacro_Recall\n' % dl_model.__name__)
			
		
		
		if dl_model.__name__ == "gaussianNB":
			time_gaussianNB_filename = '%s/performance/%s_gaussianNB_training_times.dat' % (self.outdir, dl_model.__name__)
			with open(time_gaussianNB_filename, 'w') as time_gaussianNB_file:
				time_gaussianNB_file.write('%s\tTraining_time\n' % dl_model.__name__)
		if dl_model.__name__ == "decisionTree":
			time_decisionTree_filename = '%s/performance/%s_decisionTree_training_times.dat' % (self.outdir, dl_model.__name__)
			with open(time_decisionTree_filename, 'w') as time_decisionTree_file:
				time_decisionTree_file.write('%s\tTraining_time\n' % dl_model.__name__)
		
		if dl_model.__name__ == "taylor2016appscanner_RF":
			time_RF_filename = '%s/performance/%s_RF_training_times.dat' % (self.outdir, dl_model.__name__)
			with open(time_RF_filename, 'w') as time_RF_file:
				time_RF_file.write('%s\tTraining_time\n' % dl_model.__name__)
				
		if dl_model.__name__ == "MajorityVotingCombiner_10Fold":
			time_MajorityVotingCombiner_10Fold_filename = '%s/performance/%s_MajorityVotingCombiner_10Fold_training_times.dat' % (self.outdir, dl_model.__name__)
			with open(time_MajorityVotingCombiner_10Fold_filename, 'w') as time_MajorityVotingCombiner_10Fold_file:
				time_MajorityVotingCombiner_10Fold_file.write('%s\tTraining_time\n' % dl_model.__name__)		
				
		if dl_model.__name__ == "SoftCombiner_10Fold":
			time_SoftCombiner_10Fold_filename = '%s/performance/%s_SoftCombiner_10Fold_training_times.dat' % (self.outdir, dl_model.__name__)
			with open(time_SoftCombiner_10Fold_filename, 'w') as time_SoftCombiner_10Fold_file:
				time_SoftCombiner_10Fold_file.write('%s\tTraining_time\n' % dl_model.__name__)	

		if dl_model.__name__ == "WeightedMajorityVotingCombiner_10Fold":
			time_WeightedMajorityVotingCombiner_10Fold_filename = '%s/performance/%s_WeightedMajorityVotingCombiner_10Fold_training_times.dat' % (self.outdir, dl_model.__name__)
			with open(time_WeightedMajorityVotingCombiner_10Fold_filename, 'w') as time_WeightedMajorityVotingCombiner_10Fold_file:
				time_WeightedMajorityVotingCombiner_10Fold_file.write('%s\tTraining_time\n' % dl_model.__name__)	
		
		if dl_model.__name__ == "Recall_Combiner_10Fold":
			time_Recall_Combiner_10Fold_filename = '%s/performance/%s_Recall_Combiner_10Fold_training_times.dat' % (self.outdir, dl_model.__name__)
			with open(time_Recall_Combiner_10Fold_filename, 'w') as time_Recall_Combiner_10Fold_file:
				time_Recall_Combiner_10Fold_file.write('%s\tTraining_time\n' % dl_model.__name__)	
		if dl_model.__name__ == "NB_Trainable_Combiner_10Fold":
			time_NB_Trainable_Combiner_10Fold_filename = '%s/performance/%s_NB_Trainable_Combiner_10Fold_training_times.dat' % (self.outdir, dl_model.__name__)
			with open(time_NB_Trainable_Combiner_10Fold_filename, 'w') as time_NB_Trainable_Combiner_10Fold_file:
				time_NB_Trainable_Combiner_10Fold_file.write('%s\tTraining_time\n' % dl_model.__name__)	
		if dl_model.__name__ == "BKS_Combiner_10Fold":
			time_BKS_Combiner_10Fold_filename = '%s/performance/%s_BKS_Combiner_10Fold_training_times.dat' % (self.outdir, dl_model.__name__)
			with open(time_BKS_Combiner_10Fold_filename, 'w') as time_BKS_Combiner_10Fold_file:
				time_BKS_Combiner_10Fold_file.write('%s\tTraining_time\n' % dl_model.__name__)	
		if dl_model.__name__ == "Decision_Template_Combiner_10Fold":
			time_Decision_Template_Combiner_10Fold_filename = '%s/performance/%s_Decision_Template_Combiner_10Fold_training_times.dat' % (self.outdir, dl_model.__name__)
			with open(time_Decision_Template_Combiner_10Fold_filename, 'w') as time_Decision_Template_Combiner_10Fold_file:
				time_Decision_Template_Combiner_10Fold_file.write('%s\tTraining_time\n' % dl_model.__name__)		
		if dl_model.__name__ != "OracleCombiner":
			test_time_filename = '%s/performance/%s_test_times.dat' % (self.outdir, dl_model.__name__)
			with open(test_time_filename, 'w') as test_time_file:
					test_time_file.write('%s\tTest_time\n' % dl_model.__name__)
				
		
		cvscores_pred = []
		macro_recall_pred = []
		macro_precision_pred = []
		fscores_pred = []
		gmeans_pred = []
		macro_gmeans_pred = []
		training_cvscores_pred = []
		training_recall_pred=[]
		training_precision_pred=[]
		training_fscores_pred = []
		training_gmeans_pred = []
		training_macro_gmeans_pred = []
		norm_cms_pred = []
		topk_accuracies_pred = []
		filtered_cvscores_pred = []
		filtered_fscores_pred = []
		filtered_gmeans_pred = []
		filtered_macro_gmeans_pred = []
		filtered_norm_cms_pred = []
		filtered_classified_ratios_pred = []
		
		current_fold = 1
		kfold = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=seed)


	
		
		for train, test in kfold.split(self.samples, self.categorical_labels):
			self.debug_log.info('stratified_cross_validation: Starting fold %d execution' % current_fold)


			
			if is_multimodal:
				# In the multimodal approaches, preprocessing operations strongly depend on the specific DL architectures and inputs considered.
				samples_train_list = []
				samples_test_list = []
				
				for dataset_samples in self.samples_list:
					samples_train_list.append(dataset_samples[train])
					samples_test_list.append(dataset_samples[test])
				
				
				
				
				predictions, accuracy,macro_recall,macro_precision ,fmeasure, gmean, macro_gmean, training_accuracy, training_macro_recall,training_macro_precision,training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
				filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
				filtered_norm_cnf_matrices, filtered_classified_ratios = \
				dl_model(samples_train_list, self.categorical_labels[train], samples_test_list, self.categorical_labels[test])
			
			else:

				samples_train = self.samples[train]  
				samples_test = self.samples[test]
				if do_expand_dims:
					
					samples_train = numpy.expand_dims(samples_train, axis=2)
					samples_test = numpy.expand_dims(samples_test, axis=2)


				if dl_model.__name__ != "OracleCombiner":
					predictions, accuracy,macro_recall, macro_precision,fmeasure, gmean, macro_gmean, training_accuracy, training_macro_recall,training_macro_precision,training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
					filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, \
					filtered_norm_cnf_matrices, filtered_classified_ratios = \
					dl_model(samples_train, self.categorical_labels[train], samples_test, self.categorical_labels[test])
				else:
					accuracy,macro_recall= dl_model(samples_train, self.categorical_labels[train], samples_test, self.categorical_labels[test])
				
			if dl_model.__name__ == "OracleCombiner":
				accuracy = accuracy * 100
				macro_recall = macro_recall * 100
				

				cvscores_pred.append(accuracy)
				macro_recall_pred.append(macro_recall)
				
				# Evaluation of performance metrics
				print("Fold %d accuracy: %.4f%%" % (current_fold, accuracy))
				print("Fold %d macro recall: %.4f%%" % (current_fold, macro_recall))
				
				with open('%s/performance/%s_performance.dat' % (self.outdir, dl_model.__name__), 'a') as performance_file:
					performance_file.write('Fold_%d\t%.4f%%\t%.4f%%\n' % (current_fold, accuracy, macro_recall))
								
			
			else:
				accuracy = accuracy * 100
				macro_recall = macro_recall * 100
				macro_precision=macro_precision * 100
				fmeasure = fmeasure * 100
				gmean = gmean * 100
				macro_gmean = macro_gmean * 100
				training_accuracy = training_accuracy * 100
				training_macro_recall=training_macro_precision * 100
				training_fmeasure = training_fmeasure * 100
				training_gmean = training_gmean * 100
				training_macro_gmean = training_macro_gmean * 100
				norm_cnf_matrix = norm_cnf_matrix * 100
				topk_accuracies[:] = [topk_acc * 100 for topk_acc in topk_accuracies]
				filtered_accuracies[:] = [filtered_accuracy * 100 for filtered_accuracy in filtered_accuracies]
				filtered_fmeasures[:] = [filtered_fmeasure * 100 for filtered_fmeasure in filtered_fmeasures]
				filtered_gmeans[:] = [filtered_gmean * 100 for filtered_gmean in filtered_gmeans]
				filtered_macro_gmeans[:] = [filtered_macro_gmean * 100 for filtered_macro_gmean in filtered_macro_gmeans]
				filtered_norm_cnf_matrices[:] = [filtered_norm_cnf_matrix * 100 for filtered_norm_cnf_matrix in filtered_norm_cnf_matrices]
				filtered_classified_ratios[:] = [filtered_classified_ratio * 100 for filtered_classified_ratio in filtered_classified_ratios]
				
				sorted_labels = sorted(set(self.categorical_labels))
				
				cvscores_pred.append(accuracy)
				macro_recall_pred.append(macro_recall)
				macro_precision_pred.append(macro_precision)
				fscores_pred.append(fmeasure)
				gmeans_pred.append(gmean)
				macro_gmeans_pred.append(macro_gmean)
				training_cvscores_pred.append(training_accuracy)
				training_recall_pred.append(training_macro_recall)
				training_precision_pred.append(training_macro_precision)
				training_fscores_pred.append(training_fmeasure)
				training_gmeans_pred.append(training_gmean)
				training_macro_gmeans_pred.append(training_macro_gmean)
				norm_cms_pred.append(norm_cnf_matrix)
				topk_accuracies_pred.append(topk_accuracies)
				filtered_cvscores_pred.append(filtered_accuracies)
				filtered_fscores_pred.append(filtered_fmeasures)
				filtered_gmeans_pred.append(filtered_gmeans)
				filtered_macro_gmeans_pred.append(filtered_macro_gmeans)
				filtered_norm_cms_pred.append(filtered_norm_cnf_matrices)
				filtered_classified_ratios_pred.append(filtered_classified_ratios)
				
				# Evaluation of performance metrics
				print("Fold %d accuracy: %.4f%%" % (current_fold, accuracy))
				print("Fold %d macro recall: %.4f%%" % (current_fold, macro_recall))
				print("Fold %d macro precision: %.4f%%" % (current_fold, macro_precision))
				print("Fold %d macro F-measure: %.4f%%" % (current_fold, fmeasure))
				print("Fold %d g-mean: %.4f%%" % (current_fold, gmean))
				print("Fold %d macro g-mean: %.4f%%" % (current_fold, macro_gmean))
				
				with open('%s/performance/%s_performance.dat' % (self.outdir, dl_model.__name__), 'a') as performance_file:
					performance_file.write('Fold_%d\t%.4f%%\t%.4f%%\t%.4f%%\t%.4f%%\t%.4f%%\t%.4f%%\n' % (current_fold, accuracy,macro_recall,macro_precision, fmeasure, gmean, macro_gmean))
				
				# Evaluation of performance metrics on training set
				print("Fold %d training accuracy: %.4f%%" % (current_fold, training_accuracy))
				print("Fold %d training macro recall: %.4f%%" % (current_fold, training_macro_recall))
				print("Fold %d training macro precisio: %.4f%%" % (current_fold, training_macro_precision))
				print("Fold %d training macro F-measure: %.4f%%" % (current_fold, training_fmeasure))
				print("Fold %d training g-mean: %.4f%%" % (current_fold, training_gmean))
				print("Fold %d training macro g-mean: %.4f%%" % (current_fold, training_macro_gmean))
				
				with open('%s/performance/%s_training_performance.dat' % (self.outdir, dl_model.__name__), 'a') as training_performance_file:
					training_performance_file.write('Fold_%d\t%.4f%%\t%.4f%%\t%.4f%%\t%.4f%%\t%.4f%%\t%.4f%%\n' % (current_fold, training_accuracy,training_macro_recall,training_macro_precision, training_fmeasure, training_gmean, training_macro_gmean))
				
				# Confusion matrix
				print("Fold %d confusion matrix:" % current_fold)
				for i, row in enumerate(norm_cnf_matrix):
					for occurences in row:
						sys.stdout.write('%s,' % occurences)
					sys.stdout.write('%s\n' % sorted_labels[i])
				
				# Top-K accuracy
				with open('%s/performance/%s_top-k_accuracy.dat' % (self.outdir, dl_model.__name__), 'a') as topk_accuracy_file:
					topk_accuracy_file.write('Fold_%d' % current_fold)
					for i, topk_accuracy in enumerate(topk_accuracies):
						print("Fold %d top-%d accuracy: %.4f%%" % (current_fold, i, topk_accuracy))
						topk_accuracy_file.write('\t%.4f' % topk_accuracy)
					topk_accuracy_file.write('\n')
				
				# Accuracy, F-measure, and g-mean with reject option
				with open('%s/performance/%s_filtered_performance.dat' % (self.outdir, dl_model.__name__), 'a') as filtered_perfomance_file:
					sys.stdout.write('Fold %d filtered accuracy:' % current_fold)
					filtered_perfomance_file.write('Fold_%d_filtered_accuracy' % current_fold)
					for filtered_accuracy in filtered_accuracies:
						sys.stdout.write('\t%.4f%%' % filtered_accuracy)
						filtered_perfomance_file.write('\t%.4f' % filtered_accuracy)
					sys.stdout.write('\n')
					filtered_perfomance_file.write('\n')
					
					sys.stdout.write('Fold %d filtered F-measure:' % current_fold)
					filtered_perfomance_file.write('Fold_%d_filtered_F-measure' % current_fold)
					for filtered_fmeasure in filtered_fmeasures:
						sys.stdout.write('\t%.4f%%' % filtered_fmeasure)
						filtered_perfomance_file.write('\t%.4f' % filtered_fmeasure)
					sys.stdout.write('\n')
					filtered_perfomance_file.write('\n')
					
					sys.stdout.write('Fold %d filtered g-mean:' % current_fold)
					filtered_perfomance_file.write('Fold_%d_filtered_g-mean' % current_fold)
					for filtered_gmean in filtered_gmeans:
						sys.stdout.write('\t%.4f%%' % filtered_gmean)
						filtered_perfomance_file.write('\t%.4f' % filtered_gmean)
					sys.stdout.write('\n')
					filtered_perfomance_file.write('\n')
					
					sys.stdout.write('Fold %d filtered macro g-mean:' % current_fold)
					filtered_perfomance_file.write('Fold_%d_filtered_macro_g-mean' % current_fold)
					for filtered_macro_gmean in filtered_macro_gmeans:
						sys.stdout.write('\t%.4f%%' % filtered_macro_gmean)
						filtered_perfomance_file.write('\t%.4f' % filtered_macro_gmean)
					sys.stdout.write('\n')
					filtered_perfomance_file.write('\n')
					
					sys.stdout.write('Fold %d filtered classified ratio:' % current_fold)
					filtered_perfomance_file.write('Fold_%d_filtered_classified_ratio' % current_fold)
					for filtered_classified_ratio in filtered_classified_ratios:
						sys.stdout.write('\t%.4f%%' % filtered_classified_ratio)
						filtered_perfomance_file.write('\t%.4f' % filtered_classified_ratio)
					sys.stdout.write('\n')
					filtered_perfomance_file.write('\n')
					
					try:
						self.extract_per_epoch_times(current_fold, time_epochs_filename)
					except:
						try:
							self.write_SAE_training_times(current_fold, time_sdae_filename)
						except:
							if dl_model.__name__ == "taylor2016appscanner_RF":
								self.write_classifier_training_times(current_fold, time_RF_filename)
							elif  dl_model.__name__ == "gaussianNB":
								self.write_classifier_training_times(current_fold, time_gaussianNB_filename)
							elif dl_model.__name__ == "decisionTree":
								self.write_classifier_training_times(current_fold, time_decisionTree_filename)
							elif  dl_model.__name__ == "MajorityVotingCombiner_10Fold":
								self.write_classifier_training_times(current_fold, time_MajorityVotingCombiner_10Fold_filename)
							elif  dl_model.__name__ == "SoftCombiner_10Fold":
								self.write_classifier_training_times(current_fold, time_SoftCombiner_10Fold_filename)
							elif  dl_model.__name__ == "WeightedMajorityVotingCombiner_10Fold":
								self.write_classifier_training_times(current_fold, time_WeightedMajorityVotingCombiner_10Fold_filename)	
							elif  dl_model.__name__ == "Recall_Combiner_10Fold":
								self.write_classifier_training_times(current_fold, time_Recall_Combiner_10Fold_filename)	
							elif  dl_model.__name__ == "NB_Trainable_Combiner_10Fold":
								self.write_classifier_training_times(current_fold, time_NB_Trainable_Combiner_10Fold_filename)	
							elif  dl_model.__name__ == "BKS_Combiner_10Fold":
								self.write_classifier_training_times(current_fold, time_BKS_Combiner_10Fold_filename)	
							elif  dl_model.__name__ == "Decision_Template_Combiner_10Fold":
								self.write_classifier_training_times(current_fold, time_Decision_Template_Combiner_10Fold_filename)	
							
							
							
					
					
					self.write_test_times(current_fold, test_time_filename)
					
					self.write_predictions(self.categorical_labels[test], predictions, dl_model, current_fold)
					self.write_filtered_predictions(filtered_categorical_labels_test_list, filtered_predictions_list, dl_model, current_fold)
			
			self.debug_log.info('stratified_cross_validation: Ending fold %d execution' % current_fold)
			current_fold += 1
		
		
		if dl_model.__name__ == "OracleCombiner":
			# Evaluation of macro performance metrics
			mean_accuracy = numpy.mean(cvscores_pred)
			std_accuracy = numpy.std(cvscores_pred)
			
			# Evaluation of macro performance metrics
			mean_macro_recall = numpy.mean(macro_recall_pred)
			std_macro_recall = numpy.std(macro_recall_pred)
		
			print("Macro accuracy: %.4f%% (+/- %.4f%%)" % (mean_accuracy, std_accuracy))
			print("Macro recall: %.4f%% (+/- %.4f%%)" % (mean_macro_recall, std_macro_recall))
			
			with open('%s/performance/%s_performance.dat' % (self.outdir, dl_model.__name__), 'a') as performance_file:
				performance_file.write('Macro_total\t%.4f%% (+/- %.4f%%)\t%.4f%% (+/- %.4f%%)\n' % \
				(mean_accuracy, std_accuracy, mean_macro_recall, std_macro_recall))
			
		
		else:
			# Evaluation of macro performance metrics
			mean_accuracy = numpy.mean(cvscores_pred)
			std_accuracy = numpy.std(cvscores_pred)
			
			mean_macro_recall = numpy.mean(macro_recall_pred)
			std_macro_recall = numpy.std(macro_recall_pred)
			
			mean_macro_precision = numpy.mean(macro_precision_pred)
			std_precision = numpy.std(macro_precision_pred)
			
			mean_fmeasure = numpy.mean(fscores_pred)
			std_fmeasure = numpy.std(fscores_pred)
			
			mean_gmean = numpy.mean(gmeans_pred)
			std_gmean = numpy.std(gmeans_pred)
			
			mean_macro_gmean = numpy.mean(macro_gmeans_pred)
			std_macro_gmean = numpy.std(macro_gmeans_pred)
			
			print("Macro accuracy: %.4f%% (+/- %.4f%%)" % (mean_accuracy, std_accuracy))
			print("Macro recall: %.4f%% (+/- %.4f%%)" % (mean_macro_recall, std_macro_recall))
			print("Macro precision: %.4f%% (+/- %.4f%%)" % (mean_macro_precision, std_precision))
			print("Macro F-measure: %.4f%% (+/- %.4f%%)" % (mean_fmeasure, std_fmeasure))
			print("Average g-mean: %.4f%% (+/- %.4f%%)" % (mean_gmean, std_gmean))
			print("Average macro g-mean: %.4f%% (+/- %.4f%%)" % (mean_macro_gmean, std_macro_gmean))
			
			with open('%s/performance/%s_performance.dat' % (self.outdir, dl_model.__name__), 'a') as performance_file:
				performance_file.write('Macro_total\t%.4f%% (+/- %.4f%%)\t%.4f%% (+/- %.4f%%)\t%.4f%% (+/- %.4f%%)\t%.4f%% (+/- %.4f%%)\t%.4f%% (+/- %.4f%%)\t%.4f%% (+/- %.4f%%)\n' % \
				(mean_accuracy, std_accuracy,mean_macro_recall,std_macro_recall,mean_macro_precision,std_precision, mean_fmeasure, std_fmeasure, mean_gmean, std_gmean, mean_macro_gmean, std_macro_gmean))
			
			# Evaluation of macro performance metrics on training set
			training_mean_accuracy = numpy.mean(training_cvscores_pred)
			training_std_accuracy = numpy.std(training_cvscores_pred)
			
			training_mean_macro_recall = numpy.mean(training_recall_pred)
			training_std_macro_recall = numpy.std(training_recall_pred)
			
			training_mean_macro_precision = numpy.mean(training_precision_pred)
			training_std_precision = numpy.std(training_precision_pred)
			
			training_mean_fmeasure = numpy.mean(training_fscores_pred)
			training_std_fmeasure = numpy.std(training_fscores_pred)
			
			training_mean_gmean = numpy.mean(training_gmeans_pred)
			training_std_gmean = numpy.std(training_gmeans_pred)
			
			training_mean_macro_gmean = numpy.mean(training_macro_gmeans_pred)
			training_std_macro_gmean = numpy.std(training_macro_gmeans_pred)
			
			print("Training macro accuracy: %.4f%% (+/- %.4f%%)" % (training_mean_accuracy, training_std_accuracy))
			print("Training macro recall: %.4f%% (+/- %.4f%%)" % (training_mean_macro_recall, training_std_macro_recall))
			print("Training macro precision: %.4f%% (+/- %.4f%%)" % (training_mean_macro_precision, training_std_precision))
			print("Training macro F-measure: %.4f%% (+/- %.4f%%)" % (training_mean_fmeasure, training_std_fmeasure))
			print("Training average g-mean: %.4f%% (+/- %.4f%%)" % (training_mean_gmean, training_std_gmean))
			print("Training average macro g-mean: %.4f%% (+/- %.4f%%)" % (training_mean_macro_gmean, training_std_macro_gmean))
			
			with open('%s/performance/%s_training_performance.dat' % (self.outdir, dl_model.__name__), 'a') as training_performance_file:
				training_performance_file.write('Macro_total\t%.4f%% (+/- %.4f%%)\t%.4f%% (+/- %.4f%%)\t%.4f%% (+/- %.4f%%)\t%.4f%% (+/- %.4f%%)\t%.4f%% (+/- %.4f%%)\t%.4f%% (+/- %.4f%%)\n' % \
				(training_mean_accuracy, training_std_accuracy, training_mean_macro_recall,training_std_macro_recall, training_mean_macro_precision, training_std_precision,training_mean_fmeasure, training_std_fmeasure, training_mean_gmean, training_std_gmean, training_mean_macro_gmean, training_std_macro_gmean))
			
			# Average confusion matrix
			average_cnf_matrix = self.compute_average_confusion_matrix(norm_cms_pred, num_fold)
			average_cnf_matrix_filename = '%s/confusion_matrix/%s_cnf_matrix.dat' % (self.outdir, dl_model.__name__)
			sys.stdout.write('Macro_total_confusion_matrix:\n')
			self.write_average_cnf_matrix(average_cnf_matrix, average_cnf_matrix_filename)
			
			
			
			
			# Macro top-K accuracy
			mean_topk_accuracies = []
			std_topk_accuracies = []
			
			topk_accuracies_pred_trans = list(map(list, zip(*topk_accuracies_pred)))
			for cvtopk_accuracies in topk_accuracies_pred_trans:
				mean_topk_accuracies.append(numpy.mean(cvtopk_accuracies))
				std_topk_accuracies.append(numpy.std(cvtopk_accuracies))
			
			with open('%s/performance/%s_top-k_accuracy.dat' % (self.outdir, dl_model.__name__), 'a') as topk_accuracy_file:
				topk_accuracy_file.write('Macro_total')
				for i, mean_topk_accuracy in enumerate(mean_topk_accuracies):
					print("Macro top-%d accuracy: %.4f%% (+/- %.4f%%)" % (i, mean_topk_accuracy, std_topk_accuracies[i]))
					topk_accuracy_file.write('\t%.4f\t%.4f' % (mean_topk_accuracy, std_topk_accuracies[i]))
				topk_accuracy_file.write('\n')
			
			# Macro accuracy, F-measure, and g-mean with reject option
			mean_filtered_accuracies = []
			std_filtered_accuracies = []
			
			mean_filtered_fmeasures = []
			std_filtered_fmeasures = []
			
			mean_filtered_gmeans = []
			std_filtered_gmeans = []
			
			mean_filtered_macro_gmeans = []
			std_filtered_macro_gmeans = []
			
			mean_filtered_classified_ratios = []
			std_filtered_classified_ratios = []
			
			filtered_cvscores_pred_trans = list(map(list, zip(*filtered_cvscores_pred)))
			filtered_fscores_pred_trans = list(map(list, zip(*filtered_fscores_pred)))
			filtered_gmeans_pred_trans = list(map(list, zip(*filtered_gmeans_pred)))
			filtered_macro_gmeans_pred_trans = list(map(list, zip(*filtered_macro_gmeans_pred)))
			filtered_classified_ratios_pred_trans = list(map(list, zip(*filtered_classified_ratios_pred)))
			for i, filtered_cvscores in enumerate(filtered_cvscores_pred_trans):
				mean_filtered_accuracies.append(numpy.mean(filtered_cvscores))
				std_filtered_accuracies.append(numpy.std(filtered_cvscores))
				mean_filtered_fmeasures.append(numpy.mean(filtered_fscores_pred_trans[i]))
				std_filtered_fmeasures.append(numpy.std(filtered_fscores_pred_trans[i]))
				mean_filtered_gmeans.append(numpy.mean(filtered_gmeans_pred_trans[i]))
				std_filtered_gmeans.append(numpy.std(filtered_gmeans_pred_trans[i]))
				mean_filtered_macro_gmeans.append(numpy.mean(filtered_macro_gmeans_pred_trans[i]))
				std_filtered_macro_gmeans.append(numpy.std(filtered_macro_gmeans_pred_trans[i]))
				mean_filtered_classified_ratios.append(numpy.mean(filtered_classified_ratios_pred_trans[i]))
				std_filtered_classified_ratios.append(numpy.std(filtered_classified_ratios_pred_trans[i]))
			
			with open('%s/performance/%s_filtered_performance.dat' % (self.outdir, dl_model.__name__), 'a') as filtered_perfomance_file:
				sys.stdout.write('Macro_total_filtered_accuracy:')
				filtered_perfomance_file.write('Macro_total_filtered_accuracy')
				for i, mean_filtered_accuracy in enumerate(mean_filtered_accuracies):
					sys.stdout.write('\t%.4f%%\t%.4f%%' % (mean_filtered_accuracy, std_filtered_accuracies[i]))
					filtered_perfomance_file.write('\t%.4f\t%.4f' % (mean_filtered_accuracy, std_filtered_accuracies[i]))
				sys.stdout.write('\n')
				filtered_perfomance_file.write('\n')
			
			with open('%s/performance/%s_filtered_performance.dat' % (self.outdir, dl_model.__name__), 'a') as filtered_perfomance_file:
				sys.stdout.write('Macro_total_filtered_F-measure:')
				filtered_perfomance_file.write('Macro_total_filtered_F-measure')
				for i, mean_filtered_fmeasure in enumerate(mean_filtered_fmeasures):
					sys.stdout.write('\t%.4f%%\t%.4f%%' % (mean_filtered_fmeasure, std_filtered_fmeasures[i]))
					filtered_perfomance_file.write('\t%.4f\t%.4f' % (mean_filtered_fmeasure, std_filtered_fmeasures[i]))
				sys.stdout.write('\n')
				filtered_perfomance_file.write('\n')
			
			with open('%s/performance/%s_filtered_performance.dat' % (self.outdir, dl_model.__name__), 'a') as filtered_perfomance_file:
				sys.stdout.write('Average_total_filtered_g-mean:')
				filtered_perfomance_file.write('Average_total_filtered_g-mean')
				for i, mean_filtered_gmean in enumerate(mean_filtered_gmeans):
					sys.stdout.write('\t%.4f%%\t%.4f%%' % (mean_filtered_gmean, std_filtered_gmeans[i]))
					filtered_perfomance_file.write('\t%.4f\t%.4f' % (mean_filtered_gmean, std_filtered_gmeans[i]))
				sys.stdout.write('\n')
				filtered_perfomance_file.write('\n')
			
			with open('%s/performance/%s_filtered_performance.dat' % (self.outdir, dl_model.__name__), 'a') as filtered_perfomance_file:
				sys.stdout.write('Average_total_filtered_macro_g-mean:')
				filtered_perfomance_file.write('Average_total_filtered_macro_g-mean')
				for i, mean_filtered_macro_gmean in enumerate(mean_filtered_macro_gmeans):
					sys.stdout.write('\t%.4f%%\t%.4f%%' % (mean_filtered_macro_gmean, std_filtered_macro_gmeans[i]))
					filtered_perfomance_file.write('\t%.4f\t%.4f' % (mean_filtered_macro_gmean, std_filtered_macro_gmeans[i]))
				sys.stdout.write('\n')
				filtered_perfomance_file.write('\n')
			
			with open('%s/performance/%s_filtered_performance.dat' % (self.outdir, dl_model.__name__), 'a') as filtered_perfomance_file:
				sys.stdout.write('Macro_total_filtered_classified_ratio:')
				filtered_perfomance_file.write('Macro_total_filtered_classified_ratio')
				for i, mean_filtered_classified_ratio in enumerate(mean_filtered_classified_ratios):
					sys.stdout.write('\t%.4f%%\t%.4f%%' % (mean_filtered_classified_ratio, std_filtered_classified_ratios[i]))
					filtered_perfomance_file.write('\t%.4f\t%.4f' % (mean_filtered_classified_ratio, std_filtered_classified_ratios[i]))
				sys.stdout.write('\n')
				filtered_perfomance_file.write('\n')
				
			# Average filtered confusion matrix
			filtered_norm_cms_pred_trans = list(map(list, zip(*filtered_norm_cms_pred)))
			for i, filtered_norm_cms in enumerate(filtered_norm_cms_pred_trans):
				try:
					filtered_average_cnf_matrix = self.compute_average_confusion_matrix(filtered_norm_cms, num_fold)
					filtered_average_cnf_matrix_filename = '%s/confusion_matrix/%s_filtered_cnf_matrix_%s.dat' % (self.outdir, dl_model.__name__, str(round(self.gamma_range[i], 1)).replace('.', '_'))
					sys.stdout.write('Macro_total_filtered_confusion_matrix_%s:\n' % str(round(self.gamma_range[i], 1)).replace('.', '_'))
					self.write_average_cnf_matrix(filtered_average_cnf_matrix, filtered_average_cnf_matrix_filename)
				except (ValueError, IndexError) as exc:
					sys.stderr.write('%s\n' % exc)
					sys.stderr.write('Error in computing macro total filtered confusion_matrix for gamma %s.\n' % str(round(self.gamma_range[i], 1)).replace('.', '_'))
		
		self.debug_log.info('Ending execution of stratified cross validation')
	
	
	def test_classifier(self, clf, samples_train, categorical_labels_train, samples_test, categorical_labels_test):
		'''
		Test the sklearn classifier given as input with predict_classes()
		'''
		
		predictions = clf.predict(samples_test)
		
		# Calculate soft predictions and test time
		
		test_time_begin = time.time()
		soft_values = clf.predict_proba(samples_test)

		test_time_end = time.time()
		self.test_time = test_time_end - test_time_begin
	
		self.test_time =0
		training_predictions = clf.predict(samples_train)
		
		# print(len(categorical_labels_test))

		
		
		# Accuracy, F-measure, Precision and g-mean
		accuracy = sklearn.metrics.accuracy_score(categorical_labels_test, predictions)
		macro_recall = sklearn.metrics.recall_score(categorical_labels_test, predictions,average='macro') 
		macro_precision=numpy.mean(sklearn.metrics.precision_score(categorical_labels_test, predictions,average=None))
		fmeasure = sklearn.metrics.f1_score(categorical_labels_test, predictions, average='macro')
		gmean = self.compute_g_mean(categorical_labels_test, predictions)
		macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_test, predictions, average=None))
		
		# Accuracy, F-measure, and g-mean on training set
		training_accuracy = sklearn.metrics.accuracy_score(categorical_labels_train, training_predictions)
		training_macro_recall = sklearn.metrics.recall_score(categorical_labels_train, training_predictions,average='macro')
		training_fmeasure = sklearn.metrics.f1_score(categorical_labels_train, training_predictions, average='macro')
		training_macro_precision=numpy.mean(sklearn.metrics.precision_score(categorical_labels_train, training_predictions,average=None))
		training_gmean = self.compute_g_mean(categorical_labels_train, training_predictions)
		training_macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_train, training_predictions, average=None))
		
		# Confusion matrix
		norm_cnf_matrix = self.compute_norm_confusion_matrix(categorical_labels_test, predictions)
		
		# Top-K accuracy
		topk_accuracies = []
		
			
		for k in self.K:
			try:
				topk_accuracy = self.compute_topk_accuracy(soft_values, categorical_labels_test, k)
				topk_accuracies.append(topk_accuracy)
			except Exception as exc:
				sys.stderr.write('%s\n' % exc)
	
		# Accuracy, F-measure, g-mean, and confusion matrix with reject option (filtered)
		filtered_categorical_labels_test_list = []
		filtered_predictions_list = []
		filtered_classified_ratios = []
		filtered_accuracies = []
		filtered_fmeasures = []
		filtered_gmeans = []
		filtered_macro_gmeans = []
		filtered_norm_cnf_matrices = []
		
		
		for gamma in self.gamma_range:
			categorical_labels_test_filtered, filtered_predictions, filtered_accuracy, filtered_fmeasure, filtered_gmean, filtered_macro_gmean, filtered_norm_cnf_matrix, filtered_classified_ratio = \
			self.compute_classifier_filtered_performance(clf, samples_test, soft_values, categorical_labels_test, gamma)
			filtered_categorical_labels_test_list.append(categorical_labels_test_filtered)
			filtered_predictions_list.append(filtered_predictions)
			filtered_accuracies.append(filtered_accuracy)
			filtered_fmeasures.append(filtered_fmeasure)
			filtered_gmeans.append(filtered_gmean)
			filtered_macro_gmeans.append(filtered_macro_gmean)
			filtered_norm_cnf_matrices.append(filtered_norm_cnf_matrix)
			filtered_classified_ratios.append(filtered_classified_ratio)
		
		return predictions, accuracy, macro_recall,macro_precision,fmeasure, gmean, macro_gmean, training_accuracy, training_macro_recall,training_macro_precision,training_fmeasure, training_gmean, training_macro_gmean, norm_cnf_matrix, topk_accuracies, \
		filtered_categorical_labels_test_list, filtered_predictions_list, filtered_accuracies, filtered_fmeasures, filtered_gmeans, filtered_macro_gmeans, filtered_norm_cnf_matrices, filtered_classified_ratios
	
	
	def compute_norm_confusion_matrix(self, y_true, y_pred):
		'''
		Compute normalized confusion matrix
		'''
		
		sorted_labels = sorted(set(self.categorical_labels))
		
		cnf_matrix = confusion_matrix(y_true, y_pred, labels=sorted_labels)
		norm_cnf_matrix = preprocessing.normalize(cnf_matrix, axis=1, norm='l1')
		
		return norm_cnf_matrix
	
	
	def compute_average_confusion_matrix_old(self, norm_cms_pred, num_fold):
		'''
		Compute average confusion matrix over num_fold
		'''
		
		# Average confusion matrix
		average_cnf_matrix = numpy.zeros((self.num_classes, self.num_classes), dtype=float)
		
		for norm_cnf_matrix in norm_cms_pred:
			average_cnf_matrix += norm_cnf_matrix
		average_cnf_matrix = average_cnf_matrix / num_fold
		
		return average_cnf_matrix
		
		
	def compute_average_confusion_matrix(self, norm_cms_pred, num_fold):
		'''
		Compute average confusion matrix over num_fold
		'''
		
		average_cnf_matrix = []
		
		for i in range(self.num_classes):
			average_cnf_matrix_row = numpy.zeros(self.num_classes, dtype=float)
			non_zero_rows = 0
			
			for norm_cnf_matrix in norm_cms_pred:
				if numpy.any(norm_cnf_matrix[i]):
					average_cnf_matrix_row = numpy.add(average_cnf_matrix_row, norm_cnf_matrix[i])
					non_zero_rows += 1
			
			average_cnf_matrix_row = average_cnf_matrix_row / non_zero_rows
			average_cnf_matrix.append(average_cnf_matrix_row)
		
		return average_cnf_matrix
	
	
	def compute_g_mean(self, y_true, y_pred):
		'''
		Compute g-mean as the geometric mean of the recalls of all classes
		'''
		
		recalls = sklearn.metrics.recall_score(y_true, y_pred, average=None)
		nonzero_recalls = recalls[recalls != 0]
		
		is_zero_recall = False
		unique_y_true = list(set(y_true))
		
		for i, recall in enumerate(recalls):
			if recall == 0 and i in unique_y_true:
				is_zero_recall = True
				self.debug_log.error('compute_g_mean: zero-recall obtained, class %s has no sample correcly classified.' % self.label_class[i])
		
		if is_zero_recall:
			gmean = scipy.stats.mstats.gmean(recalls)
		else:
			gmean = scipy.stats.mstats.gmean(nonzero_recalls)
		
		return gmean
	
	
	def compute_topk_accuracy(self, soft_values, categorical_labels_test, k):
		'''
		Compute top-k accuracy for a given value of k
		'''
		
		predictions_indices = numpy.argsort(-soft_values, 1)
		predictions_topk_indices = predictions_indices[:,0:k]
		
		accuracies = numpy.zeros(categorical_labels_test.shape)
		for i in range(k):
			accuracies = accuracies + numpy.equal(predictions_topk_indices[:,i], categorical_labels_test)
		
		topk_accuracy = sum(accuracies) / categorical_labels_test.size
		
		return topk_accuracy
	
	
	def compute_filtered_performance(self, model, samples_test, soft_values, categorical_labels_test, gamma):
		'''
		Compute filtered performance for a given value of gamma
		'''
		
		pred_indices = numpy.greater(soft_values.max(1), gamma)
		num_rej_samples = categorical_labels_test.shape - numpy.sum(pred_indices)		#  of unclassified samples
		
		samples_test_filtered = samples_test[numpy.nonzero(pred_indices)]
		categorical_labels_test_filtered = categorical_labels_test[numpy.nonzero(pred_indices)]
		
		try:
			filtered_predictions = model.predict_classes(samples_test_filtered, verbose=2)
		except AttributeError:
			filtered_logit_predictions = model.predict(samples_test_filtered)		# Predictions are logit values
			try:
				filtered_predictions = filtered_logit_predictions.argmax(1)
			except AttributeError:
				self.debug_log.error('Empty list returned by model.predict()')
				sys.stderr.write('Empty list returned by model.predict()\n')
				return numpy.nan, [], numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.array([[numpy.nan] * self.num_classes, [numpy.nan] * self.num_classes]), numpy.nan
		
		filtered_accuracy = sklearn.metrics.accuracy_score(categorical_labels_test_filtered, filtered_predictions)
		filtered_fmeasure = sklearn.metrics.f1_score(categorical_labels_test_filtered, filtered_predictions, average='macro')
		filtered_gmean = self.compute_g_mean(categorical_labels_test_filtered, filtered_predictions)
		filtered_macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_test_filtered, filtered_predictions, average=None))
		filtered_norm_cnf_matrix = self.compute_norm_confusion_matrix(categorical_labels_test_filtered, filtered_predictions)
		filtered_classified_ratio = numpy.sum(pred_indices) / categorical_labels_test.shape
		
		return categorical_labels_test_filtered, filtered_predictions, filtered_accuracy, filtered_fmeasure, filtered_gmean, filtered_macro_gmean, filtered_norm_cnf_matrix, filtered_classified_ratio
	
	
	def compute_classifier_filtered_performance_soft_comb(self,prediction, soft_values,categorical_labels_test, gamma):
		'''
		Compute filtered performance of a sklearn classifier for a given value of gamma
		'''
		
		soft_values_temp=numpy.asarray(soft_values)
		prediction_temp=numpy.asarray(prediction)

		pred_indices = numpy.greater(soft_values_temp.max(1), gamma)
		

		num_rej_samples = categorical_labels_test.shape - numpy.sum(pred_indices)		# Number of unclassified samples
		categorical_labels_test_filtered = categorical_labels_test[numpy.nonzero(pred_indices)]
		filtered_predictions=prediction_temp[numpy.nonzero(pred_indices)]
	
		#print(" gamma , categorical_labels_test_filtered , filtered_predictions" , gamma , categorical_labels_test_filtered.shape, filtered_predictions.shape)
		filtered_accuracy = sklearn.metrics.accuracy_score(categorical_labels_test_filtered, filtered_predictions)
		filtered_fmeasure = sklearn.metrics.f1_score(categorical_labels_test_filtered, filtered_predictions, average='macro')
		filtered_gmean = self.compute_g_mean(categorical_labels_test_filtered, filtered_predictions)
		filtered_macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_test_filtered, filtered_predictions, average=None))
		filtered_norm_cnf_matrix = self.compute_norm_confusion_matrix(categorical_labels_test_filtered, filtered_predictions)
		filtered_classified_ratio = numpy.sum(pred_indices) / categorical_labels_test.shape
		
		return categorical_labels_test_filtered, filtered_predictions, filtered_accuracy, filtered_fmeasure, filtered_gmean, filtered_macro_gmean, filtered_norm_cnf_matrix, filtered_classified_ratio
	
	
	def compute_classifier_filtered_performance(self, clf, samples_test, soft_values, categorical_labels_test, gamma):
		'''
		Compute filtered performance of a sklearn classifier for a given value of gamma
		'''
		
		pred_indices = numpy.greater(soft_values.max(1), gamma)
		num_rej_samples = categorical_labels_test.shape - numpy.sum(pred_indices)		# Number of unclassified samples
		
		samples_test_filtered = samples_test[numpy.nonzero(pred_indices)]
		categorical_labels_test_filtered = categorical_labels_test[numpy.nonzero(pred_indices)]
		
		filtered_predictions = clf.predict(samples_test_filtered)
		
		filtered_accuracy = sklearn.metrics.accuracy_score(categorical_labels_test_filtered, filtered_predictions)
		filtered_fmeasure = sklearn.metrics.f1_score(categorical_labels_test_filtered, filtered_predictions, average='macro')
		filtered_gmean = self.compute_g_mean(categorical_labels_test_filtered, filtered_predictions)
		filtered_macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_test_filtered, filtered_predictions, average=None))
		filtered_norm_cnf_matrix = self.compute_norm_confusion_matrix(categorical_labels_test_filtered, filtered_predictions)
		filtered_classified_ratio = numpy.sum(pred_indices) / categorical_labels_test.shape
		
		return categorical_labels_test_filtered, filtered_predictions, filtered_accuracy, filtered_fmeasure, filtered_gmean, filtered_macro_gmean, filtered_norm_cnf_matrix, filtered_classified_ratio
	
	
	def compute_multimodal_filtered_performance(self, multimodal_model, samples_test_list, soft_values, categorical_labels_test, gamma):
		'''
		Compute multimodal filtered performance for a given value of gamma
		'''
		
		pred_indices = numpy.greater(soft_values.max(1), gamma)
		num_rej_samples = categorical_labels_test.shape - numpy.sum(pred_indices)		# Number of unclassified samples
		
		samples_test_filtered_list = []
		for samples_test in samples_test_list:
			samples_test_filtered_list.append(samples_test[numpy.nonzero(pred_indices)])	
		categorical_labels_test_filtered = categorical_labels_test[numpy.nonzero(pred_indices)]
		
		filtered_soft_values = multimodal_model.predict(samples_test_filtered_list)
		filtered_predictions = filtered_soft_values.argmax(axis=-1)
		
		filtered_accuracy = sklearn.metrics.accuracy_score(categorical_labels_test_filtered, filtered_predictions)
		filtered_fmeasure = sklearn.metrics.f1_score(categorical_labels_test_filtered, filtered_predictions, average='macro')
		filtered_gmean = self.compute_g_mean(categorical_labels_test_filtered, filtered_predictions)
		filtered_macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_test_filtered, filtered_predictions, average=None))
		filtered_norm_cnf_matrix = self.compute_norm_confusion_matrix(categorical_labels_test_filtered, filtered_predictions)
		filtered_classified_ratio = numpy.sum(pred_indices) / categorical_labels_test.shape
		
		return categorical_labels_test_filtered, filtered_predictions, filtered_accuracy, filtered_fmeasure, filtered_gmean, filtered_macro_gmean, filtered_norm_cnf_matrix, filtered_classified_ratio
	
	
	def extract_per_epoch_times(self, current_fold, time_epochs_filename):
		'''
		Extract and print per-epoch times
		'''
		
		epoch_times = self.time_callback.times
		with open(time_epochs_filename, 'a') as time_epochs_file:
			time_epochs_file.write('Fold_%d\t' % current_fold)
			for epoch_time in epoch_times:
				time_epochs_file.write('\t%.4f' % epoch_time)
			time_epochs_file.write('\n')
		
	
	def write_average_cnf_matrix(self, average_cnf_matrix, average_cnf_matrix_filename):
		'''
		Write average confusion matrix (potentially filtered)
		'''
		
		with open(average_cnf_matrix_filename, 'w') as average_cnf_matrix_file:
			sorted_labels = sorted(set(self.categorical_labels))
			for label in sorted_labels:
				average_cnf_matrix_file.write('%s,' % label)
				sys.stdout.write('%s,' % label)
			average_cnf_matrix_file.write(',\n')
			sys.stdout.write(',\n')
			
			for i, row in enumerate(average_cnf_matrix):
				for occurences in row:
					average_cnf_matrix_file.write('%s,' % occurences)
					sys.stdout.write('%s,' % occurences)								########
					#sys.stdout.write('%.2f,' % float(occurences))
				average_cnf_matrix_file.write('%s,%s\n' % (sorted_labels[i], self.label_class[int(sorted_labels[i])]))
				sys.stdout.write('%s,%s\n' % (sorted_labels[i], self.label_class[int(sorted_labels[i])]))
	
	
	def write_classifier_training_times(self, current_fold, time_classifier_filename):
		'''
		Write sklearn classifier training times
		'''
		
		print('Training time: %.4f' % self.train_time)
		
		with open(time_classifier_filename, 'a') as time_classifier_file:
			time_classifier_file.write('Fold_%d\t%.4f\n' % (current_fold, self.train_time))
	
	
	def write_test_times(self, current_fold, test_time_filename):
		'''
		Write test times
		'''
		
		print('Test time: %.4f' % self.test_time)
		
		with open(test_time_filename, 'a') as test_time_file:
			test_time_file.write('Fold_%d\t%.4f\n' % (current_fold, self.test_time))
	
	
	def write_predictions(self, categorical_labels_test, predictions, dl_model, current_fold):
		'''
		Write (multimodal) model / classifier predictions
		'''
		
		try:
			os.makedirs('%s/predictions/fold_%d' % (self.outdir, current_fold))
		except OSError as exception:
			if exception.errno != errno.EEXIST:
				traceback.print_exc(file=sys.stderr)
		
		predictions_filename = '%s/predictions/fold_%d/%s_fold_%d_predictions.dat' % (self.outdir, current_fold, dl_model.__name__, current_fold)
		with open(predictions_filename, 'w') as predictions_file:
			predictions_file.write('Actual\t%s\n' % dl_model.__name__)
			for i, prediction in enumerate(predictions):
				predictions_file.write('%s\t%s\n' % (categorical_labels_test[i], prediction))
	
	
	def write_filtered_predictions(self, filtered_categorical_labels_test_list, filtered_predictions_list, dl_model, current_fold):
		'''
		Write (multimodal) model / classifier filtered predictions
		'''
		
		try:
			os.makedirs('%s/predictions/fold_%d' % (self.outdir, current_fold))
		except OSError as exception:
			if exception.errno != errno.EEXIST:
				traceback.print_exc(file=sys.stderr)
		
		for i, filtered_predictions in enumerate(filtered_predictions_list):
			filtered_predictions_filename = '%s/predictions/fold_%d/%s_fold_%d_filtered_predictions_%s.dat' % \
			(self.outdir, current_fold, dl_model.__name__, current_fold, str(round(self.gamma_range[i], 1)).replace('.', '_'))
			with open(filtered_predictions_filename, 'w') as filtered_predictions_file:
				filtered_predictions_file.write('Actual\t%s\n' % dl_model.__name__)
				for j, filtered_prediction in enumerate(filtered_predictions):
					filtered_predictions_file.write('%s\t%s\n' % (filtered_categorical_labels_test_list[i][j], filtered_prediction))
	
	
	def deserialize_dataset(self, include_ports, is_multimodal):
		'''
		Deserialize TC datasets to extract samples and (categorical) labels to perform classification using one of the state-of-the-art DL (multimodal) approaches
		'''
		
		self.debug_log.info('Starting datasets deserialization')
		
		samples_list = []
		categorical_labels_list = []
		for pickle_dataset_filename in self.pickle_dataset_filenames:
			with open(pickle_dataset_filename, 'rb') as pickle_dataset_file:
				samples_list.append(pickle.load(pickle_dataset_file))
				categorical_labels_list.append(pickle.load(pickle_dataset_file))
			self.debug_log.debug('Dataset %s deserialized' % pickle_dataset_filename)
		
		if not is_multimodal:
			self.samples = samples_list[0]
			self.categorical_labels = categorical_labels_list[0]
			
			self.num_samples = numpy.shape(self.samples)[0]
			self.input_dim = numpy.shape(self.samples)[1]
			self.num_classes = numpy.max(self.categorical_labels) + 1
			
			# if dl_network == 4 or dl_network == 5 or dl_network == 6 or dl_network == 11:
				# if include_ports:
					# self.samples = numpy.asarray(self.samples)
				# else:
					# self.samples = numpy.asarray(self.samples)[:,:,2:6]
				# self.input_dim_2 = numpy.shape(self.samples)[2]
			
			# print(self.samples)
			# print(self.categorical_labels)
			# print(self.num_samples)
			# print(self.input_dim)
			# if dl_network == 4 or dl_network == 5 or dl_network == 6 or dl_network == 11: print(self.input_dim_2)
			# print(self.num_classes)
		
		else:
			num_samples_list = [numpy.shape(samples)[0] for samples in samples_list]
			num_classes_list = [numpy.max(categorical_labels) for categorical_labels in categorical_labels_list]
			
			if len(set(num_samples_list)) <= 1 and len(set(num_classes_list)) <= 1:
				self.samples = samples_list[0]					# used to calculate indices for 10-fold validation
				self.categorical_labels = categorical_labels_list[0]
				self.num_samples = num_samples_list[0]
				self.num_classes = num_classes_list[0] + 1
			
			elif len(set(num_samples_list)) > 1:
				raise DLArchitecturesException('The datasets provided have a different number of samples.')
			
			else:
				raise DLArchitecturesException('The datasets provided have a different number of classes.')
			
			self.samples_list = samples_list
			self.categorical_labels_list = categorical_labels_list
			
			if not numpy.array_equal(self.categorical_labels, self.categorical_labels_list[1]):
				raise DLArchitecturesException('The datasets provided have different labels.')
			
			self.input_dims_list = []
			
			for i, pickle_dataset_filename in enumerate(self.pickle_dataset_filenames):
				if 'lopez' in pickle_dataset_filename.lower():
					if include_ports:
						self.samples_list[i] = numpy.asarray(self.samples_list[i])
					else:
						self.samples_list[i] = numpy.asarray(self.samples_list[i])[:,:,2:6]
					self.input_dims_list.append(numpy.shape(self.samples_list[i])[1:])
				else:
					self.input_dims_list.append(numpy.shape(self.samples_list[i])[1])
			
			# print(self.samples_list)
			# print(self.categorical_labels_list)
			# print(self.num_samples)
			# print(self.input_dims_list)
			# print(self.num_classes)
		
		self.debug_log.info('Ending datasets deserialization')


if __name__ == "__main__":
	
	if len(sys.argv) < 6:
		print('usage:', sys.argv[0], '<PICKLE_DATASET_NUMBER>', '<PICKLE_DATASET_1>', '[PICKLE_DATASET_2]', '...', '[PICKLE_DATASET_N]', '<DL_NETWORK_INDEX>', '<DATASET_INDEX>', '<OUTDIR>')
		sys.exit(1)
	elif len(sys.argv) < (int(sys.argv[1]) + 5):
		print('usage:', sys.argv[0], sys.argv[1], '<PICKLE_DATASET_1>', '...', '<PICKLE_DATASET_%s>' % sys.argv[1], '<DL_NETWORK_INDEX>', '<DATASET_INDEX>', '<OUTDIR>')
		sys.exit(1)
	
	try:	
		pickle_dataset_number = int(sys.argv[1])
	except ValueError:
		raise DLArchitecturesException('pickle_dataset_number must be an integer.')
	
	pickle_dataset_filenames = []
	for i in range(pickle_dataset_number):
		pickle_dataset_filenames.append(sys.argv[i + 2])
	
	is_multimodal = False
	if len(pickle_dataset_filenames) > 1:
		is_multimodal = True
	
	try:
		dl_network = int(sys.argv[pickle_dataset_number + 2])
	except ValueError:
		dl_network = -1
	
	# dataset == 1 is FB/FBM; dataset == 2 is Android; dataset == 3 is iOS; dataset == 4 CICIDS17.
	try:
		dataset = int(sys.argv[pickle_dataset_number + 3])
	except ValueError:
		dataset = -1
	
	outdir = sys.argv[pickle_dataset_number + 4]
	
	try:
		os.makedirs('%s/confusion_matrix' % outdir)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			traceback.print_exc(file=sys.stderr)
	
	try:
		os.makedirs('%s/performance' % outdir)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			traceback.print_exc(file=sys.stderr)
	
	try:
		os.makedirs('%s/predictions' % outdir)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			traceback.print_exc(file=sys.stderr)
	
	# Set the number of folds for stratified cross-validation
	num_fold = 10
	
	# Set random seed for reproducibility
	seed = int(time.time())
	# print(seed)
	numpy.random.seed(seed)
	
	include_ports = False				# It should be set to False to avoid biased (port-dependent) inputs
	dl_architectures = DLArchitectures(pickle_dataset_filenames, dataset, outdir, include_ports, is_multimodal)
	
	if dl_network == 1:
		dl_architectures.stratified_cross_validation(dl_architectures.taylor2016appscanner_RF, num_fold, seed, do_expand_dims = False, do_normalization = False)
	elif dl_network == 2:
		dl_architectures.stratified_cross_validation(dl_architectures.gaussianNB, num_fold, seed, do_expand_dims = False, do_normalization = False)
	elif dl_network == 3:
		dl_architectures.stratified_cross_validation(dl_architectures.decisionTree, num_fold, seed, do_expand_dims = False, do_normalization = False)
	elif dl_network == 4:
		dl_architectures.stratified_cross_validation(dl_architectures.MajorityVotingCombiner_10Fold, num_fold, seed, do_expand_dims = False, do_normalization = False)
	elif dl_network == 5:
		dl_architectures.stratified_cross_validation(dl_architectures.SoftCombiner_10Fold, num_fold, seed, do_expand_dims = False, do_normalization = False)
	elif dl_network == 6:
		dl_architectures.stratified_cross_validation(dl_architectures.WeightedMajorityVotingCombiner_10Fold, num_fold, seed, do_expand_dims = False, do_normalization = False)
	elif dl_network == 7:
		dl_architectures.stratified_cross_validation(dl_architectures.Recall_Combiner_10Fold, num_fold, seed, do_expand_dims = False, do_normalization = False)	
	elif dl_network == 8:
		dl_architectures.stratified_cross_validation(dl_architectures.NB_Trainable_Combiner_10Fold, num_fold, seed, do_expand_dims = False, do_normalization = False)		
	elif dl_network == 9:
		dl_architectures.stratified_cross_validation(dl_architectures.Decision_Template_Combiner_10Fold, num_fold, seed, do_expand_dims = False, do_normalization = False)			
	elif dl_network == 10:
		dl_architectures.stratified_cross_validation(dl_architectures.BKS_Combiner_10Fold, num_fold, seed, do_expand_dims = False, do_normalization = False)			
	elif dl_network == 11:
		dl_architectures.stratified_cross_validation(dl_architectures.OracleCombiner, num_fold, seed, do_expand_dims = False, do_normalization = False)	
	
	else:
		sys.stderr.write("Please, provide a valid DL network index in the range 1-11.\n")
