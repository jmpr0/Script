#    Implements hard and soft fusion rules of classifier results.
#
#    Copyright (C) 2018  Antonio Montieri
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
Created on 6 dec 2018

@author: antonio
'''

import sys, os, time, errno, datetime

import glob

from hard_combiners import non_trainable
from soft_combiners import CC_non_trainable

import sklearn, numpy, scipy, imblearn.metrics
#import keras
#import tensorflow as tf
#from keras import optimizers
#from keras.models import Model, Sequential
#from keras.layers import Conv1D, AveragePooling1D, MaxPooling1D,\
#			 Conv2D, MaxPooling2D,\
#			 Dense, LSTM, GRU, Flatten, Dropout, Reshape, BatchNormalization,\
#			 Input, Bidirectional, concatenate
#from keras.regularizers import l2
#from keras.callbacks import EarlyStopping
#from keras.utils import to_categorical, plot_model
#from keras.constraints import maxnorm
from sklearn import preprocessing
#from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer, Imputer
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
#from yadlt.models.autoencoders.stacked_denoising_autoencoder import StackedDenoisingAutoencoder

#import pickle
import logging
from logging.handlers import RotatingFileHandler
from pprint import pprint



class FusionRulesException(Exception):
	'''
	Exception for FusionRules class
	'''
	def __init__(self, message):
		self.message = message
	def __str__(self):
		return str(self.message)



class FusionRules(object):
	'''
	Implements hard and soft classifier fusion rules.
	'''
	
	def __init__(self, classifier_results_dirs, dataset, outdir):
		'''
		Constructor for the class FusionRules.
		Input:
		- classifier_results_dirs (list): list of directories containing the results of classifiers to combine
		- dataset (int): set the type of dataset to handle the correct set of classes (dataset == 1 is FB/FBM; dataset == 2 is Android; dataset == 3 is iOS)
		- outdir (string): outdir to contain combiner results
		'''
		
		self.dataset = dataset
		self.outdir = outdir
		
		self.K_binary = [1, 2]
		self.K_multi = [1, 3, 5, 7, 10]
		
		self.gamma_range = numpy.arange(0.0, 1.0, 0.1)
		
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
		
		self.label_class_IOS = {
							0: '360Security',
							1: '6Rooms',
							2: '80sMovie',
							3: 'Anghami',
							4: 'AppleiCloud',
							5: 'BaiDu',
							6: 'Brightcove',
							7: 'Crackle',
							8: 'EFood',
							9: 'FsecureFreedomeVPN',
							10: 'Go90',
							11: 'Google+',
							12: 'GoogleAllo',
							13: 'GoogleCast',
							14: 'GoogleMap',
							15: 'GooglePhotos',
							16: 'GroupMe',
							17: 'Guvera',
							18: 'Hangouts',
							19: 'HiTalk',
							20: 'HidemanVPN',
							21: 'Hidemyass',
							22: 'Hooq',
							23: 'HotSpot',
							24: 'IFengNews',
							25: 'LRR',
							26: 'MeinO2',
							27: 'Minecraft',
							28: 'Mobily',
							29: 'Narutom',
							30: 'NetTalk',
							31: 'NileFM',
							32: 'Palringo',
							33: 'PaltalkScene',
							34: 'PrivateTunnelVPN',
							35: 'PureVPN',
							36: 'QQReader',
							37: 'QianXunYingShi',
							38: 'Repubblica',
							39: 'Ryanair',
							40: 'SayHi',
							41: 'Shadowsocks',
							42: 'Sogou',
							43: 'eBay',
							44: 'iMessage'
		}
		
		if self.dataset == 1:
			self.K = self.K_binary
			self.label_class = self.label_class_FB_FBM
		elif self.dataset == 2:
			self.K = self.K_multi
			self.label_class = self.label_class_Android
		elif self.dataset == 3:
			self.K = self.K_multi
			self.label_class = self.label_class_IOS
		
		self.debug_log = self.setup_logger('debug', '%s/fusion_rules.log' % self.outdir, logging.DEBUG)
		
		for classifier_results_dir in classifier_results_dirs:
			if os.path.isdir(classifier_results_dir):
				self.debug_log.debug('The results directory %s exists.' % classifier_results_dir)
			else:
				err_str = 'The results directory %s does not exist. Please, provide a valid directory.' % classifier_results_dir
				self.debug_log.error(err_str)
				raise FusionRulesException(err_str)
		
		self.classifier_results_dirs = classifier_results_dirs
		
		self.extract_predictions()
		self.extract_soft_values()
	
	
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
	
	
	def per_fold_fusion_rule(self, fusion_rule, is_soft_combiner = False):
		'''
		Runs classifier fusion rule given as input on different folds.
		:param predictions_list: i-th list contains the predictions of i-th classifier
		'''
		
		self.debug_log.info('%s: Starting execution of fusion rule' % fusion_rule.__name__)
		
		with open('%s/performance/%s_performance.dat' % (self.outdir, fusion_rule.__name__), 'w') as performance_file:
			performance_file.write('%s\tAccuracy\tMacro_F-measure\tG-mean\tMacro_G-mean\n' % fusion_rule.__name__)
		
		with open('%s/performance/%s_top-k_accuracy.dat' % (self.outdir, fusion_rule.__name__), 'w') as topk_accuracy_file:
			topk_accuracy_file.write('%s\t' % fusion_rule.__name__)
			for k in self.K:
				topk_accuracy_file.write('K=%d\t' % k)
			topk_accuracy_file.write('\n')
		
		if is_soft_combiner:
			per_fold_classifiers_results_list = self.per_fold_classifiers_soft_values_list
		else:
			per_fold_classifiers_results_list = self.per_fold_classifiers_predictions_list
		
		num_fold = len(self.folds)
		
		cvscores_pred = []
		fscores_pred = []
		gmeans_pred = []
		macro_gmeans_pred = []
#		training_cvscores_pred = []
#		training_fscores_pred = []
#		training_gmeans_pred = []
#		training_macro_gmeans_pred = []
		norm_cms_pred = []
		topk_accuracies_pred = []
#		filtered_cvscores_pred = []
#		filtered_fscores_pred = []
#		filtered_gmeans_pred = []
#		filtered_macro_gmeans_pred = []
#		filtered_norm_cms_pred = []
#		filtered_classified_ratios_pred = []
		
		for fold in self.folds:
			self.debug_log.info('per_fold_fusion_rule: Starting %s execution' % fold)
			
			per_fold_results = [per_fold_classifier_results[fold] for per_fold_classifier_results in per_fold_classifiers_results_list]
			per_fold_actual_classes = numpy.array(self.per_fold_classifiers_actual_classes_list[0][fold])
			
			self.per_fold_categorical_labels = per_fold_actual_classes
			self.num_classes = numpy.max(per_fold_actual_classes) + 1
			
			per_fold_combiner_results = numpy.array(fusion_rule(per_fold_results))
			accuracy, fmeasure, gmean, macro_gmean, norm_cnf_matrix, topk_accuracies = self.test_combiner(per_fold_combiner_results, per_fold_actual_classes, is_soft_combiner)
			
			accuracy = accuracy * 100
			fmeasure = fmeasure * 100
			gmean = gmean * 100
			macro_gmean = macro_gmean * 100
#			training_accuracy = training_accuracy * 100
#			training_fmeasure = training_fmeasure * 100
#			training_gmean = training_gmean * 100
#			training_macro_gmean = training_macro_gmean * 100
			norm_cnf_matrix = norm_cnf_matrix * 100
			topk_accuracies[:] = [topk_acc * 100 for topk_acc in topk_accuracies]
#			filtered_accuracies[:] = [filtered_accuracy * 100 for filtered_accuracy in filtered_accuracies]
#			filtered_fmeasures[:] = [filtered_fmeasure * 100 for filtered_fmeasure in filtered_fmeasures]
#			filtered_gmeans[:] = [filtered_gmean * 100 for filtered_gmean in filtered_gmeans]
#			filtered_macro_gmeans[:] = [filtered_macro_gmean * 100 for filtered_macro_gmean in filtered_macro_gmeans]
#			filtered_norm_cnf_matrices[:] = [filtered_norm_cnf_matrix * 100 for filtered_norm_cnf_matrix in filtered_norm_cnf_matrices]
#			filtered_classified_ratios[:] = [filtered_classified_ratio * 100 for filtered_classified_ratio in filtered_classified_ratios]
			
			per_fold_sorted_labels = sorted(set(self.per_fold_categorical_labels))
			
			cvscores_pred.append(accuracy)
			fscores_pred.append(fmeasure)
			gmeans_pred.append(gmean)
			macro_gmeans_pred.append(macro_gmean)
#			training_cvscores_pred.append(training_accuracy)
#			training_fscores_pred.append(training_fmeasure)
#			training_gmeans_pred.append(training_gmean)
#			training_macro_gmeans_pred.append(training_macro_gmean)
			norm_cms_pred.append(norm_cnf_matrix)
			topk_accuracies_pred.append(topk_accuracies)
#			filtered_cvscores_pred.append(filtered_accuracies)
#			filtered_fscores_pred.append(filtered_fmeasures)
#			filtered_gmeans_pred.append(filtered_gmeans)
#			filtered_macro_gmeans_pred.append(filtered_macro_gmeans)
#			filtered_norm_cms_pred.append(filtered_norm_cnf_matrices)
#			filtered_classified_ratios_pred.append(filtered_classified_ratios)
			
			# Evaluation of performance metrics
			print("%s accuracy: %.4f%%" % (fold, accuracy))
			print("%s macro F-measure: %.4f%%" % (fold, fmeasure))
			print("%s g-mean: %.4f%%" % (fold, gmean))
			print("%s macro g-mean: %.4f%%" % (fold, macro_gmean))
			
			with open('%s/performance/%s_performance.dat' % (self.outdir, fusion_rule.__name__), 'a') as performance_file:
				performance_file.write('%s\t%.4f%%\t%.4f%%\t%.4f%%\t%.4f%%\n' % (fold, accuracy, fmeasure, gmean, macro_gmean))
			
			# Confusion matrix
			print("%s confusion matrix:" % fold)
			for i, row in enumerate(norm_cnf_matrix):
				for occurences in row:
					sys.stdout.write('%s,' % occurences)
				sys.stdout.write('%s\n' % per_fold_sorted_labels[i])
			
			# Top-K accuracy
			with open('%s/performance/%s_top-k_accuracy.dat' % (self.outdir, fusion_rule.__name__), 'a') as topk_accuracy_file:
				topk_accuracy_file.write(fold)
				for i, topk_accuracy in enumerate(topk_accuracies):
					print("%s top-%d accuracy: %.4f%%" % (fold, i, topk_accuracy))
					topk_accuracy_file.write('\t%.4f' % topk_accuracy)
				topk_accuracy_file.write('\n')
			
			self.debug_log.info('per_fold_fusion_rule: Ending %s execution' % fold)
		
		# Evaluation of macro performance metrics
		mean_accuracy = numpy.mean(cvscores_pred)
		std_accuracy = numpy.std(cvscores_pred)
		
		mean_fmeasure = numpy.mean(fscores_pred)
		std_fmeasure = numpy.std(fscores_pred)
		
		mean_gmean = numpy.mean(gmeans_pred)
		std_gmean = numpy.std(gmeans_pred)
		
		mean_macro_gmean = numpy.mean(macro_gmeans_pred)
		std_macro_gmean = numpy.std(macro_gmeans_pred)
		
		print("Macro accuracy: %.4f%% (+/- %.4f%%)" % (mean_accuracy, std_accuracy))
		print("Macro F-measure: %.4f%% (+/- %.4f%%)" % (mean_fmeasure, std_fmeasure))
		print("Average g-mean: %.4f%% (+/- %.4f%%)" % (mean_gmean, std_gmean))
		print("Average macro g-mean: %.4f%% (+/- %.4f%%)" % (mean_macro_gmean, std_macro_gmean))
		
		with open('%s/performance/%s_performance.dat' % (self.outdir, fusion_rule.__name__), 'a') as performance_file:
			performance_file.write('Macro_total\t%.4f%% (+/- %.4f%%)\t%.4f%% (+/- %.4f%%)\t%.4f%% (+/- %.4f%%)\t%.4f%% (+/- %.4f%%)\n' % \
			(mean_accuracy, std_accuracy, mean_fmeasure, std_fmeasure, mean_gmean, std_gmean, mean_macro_gmean, std_macro_gmean))
		
		# Average confusion matrix
		average_cnf_matrix = self.compute_average_confusion_matrix(norm_cms_pred, num_fold)
		average_cnf_matrix_filename = '%s/confusion_matrix/%s_cnf_matrix.dat' % (self.outdir, fusion_rule.__name__)
		sys.stdout.write('Macro_total_confusion_matrix:\n')
		self.write_average_cnf_matrix(average_cnf_matrix, average_cnf_matrix_filename)
		
		# Macro top-K accuracy
		mean_topk_accuracies = []
		std_topk_accuracies = []
		
		topk_accuracies_pred_trans = list(map(list, zip(*topk_accuracies_pred)))
		for cvtopk_accuracies in topk_accuracies_pred_trans:
			mean_topk_accuracies.append(numpy.mean(cvtopk_accuracies))
			std_topk_accuracies.append(numpy.std(cvtopk_accuracies))
		
		with open('%s/performance/%s_top-k_accuracy.dat' % (self.outdir, fusion_rule.__name__), 'a') as topk_accuracy_file:
			topk_accuracy_file.write('Macro_total')
			for i, mean_topk_accuracy in enumerate(mean_topk_accuracies):
				print("Macro top-%d accuracy: %.4f%% (+/- %.4f%%)" % (i, mean_topk_accuracy, std_topk_accuracies[i]))
				topk_accuracy_file.write('\t%.4f\t%.4f' % (mean_topk_accuracy, std_topk_accuracies[i]))
			topk_accuracy_file.write('\n')
		
		self.debug_log.info('%s: Starting execution of fusion rule' % fusion_rule.__name__)
	
	
	def test_combiner(self, per_fold_combiner_results, categorical_labels_test, is_soft_combiner):
		'''
		Test the hard/soft combiner
		'''
		
		if is_soft_combiner:
			predictions = per_fold_combiner_results.argmax(axis=-1)
		else:
			predictions = per_fold_combiner_results
		
		# print(len(categorical_labels_test))
		# print(categorical_labels_test)
		
		# print(len(predictions))
		# print(predictions)
		
		accuracy = sklearn.metrics.accuracy_score(categorical_labels_test, predictions)
		fmeasure = sklearn.metrics.f1_score(categorical_labels_test, predictions, average='macro')
		gmean = self.compute_g_mean(categorical_labels_test, predictions)
		macro_gmean = numpy.mean(imblearn.metrics.geometric_mean_score(categorical_labels_test, predictions, average=None))
		
		# Confusion matrix
		norm_cnf_matrix = self.compute_norm_confusion_matrix(categorical_labels_test, predictions)
		
		# Top-K accuracy
		topk_accuracies = []
		
		if is_soft_combiner:
			for k in self.K:
				topk_accuracy = self.compute_topk_accuracy(per_fold_combiner_results, categorical_labels_test, k)
				topk_accuracies.append(topk_accuracy)
		
		return accuracy, fmeasure, gmean, macro_gmean, norm_cnf_matrix, topk_accuracies
	
	
	def compute_norm_confusion_matrix(self, y_true, y_pred):
		'''
		Compute normalized confusion matrix
		'''
		
		per_fold_sorted_labels = sorted(set(self.per_fold_categorical_labels))
		
		cnf_matrix = confusion_matrix(y_true, y_pred, labels=per_fold_sorted_labels)
		norm_cnf_matrix = preprocessing.normalize(cnf_matrix, axis=1, norm='l1')
		
		return norm_cnf_matrix
	
	
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
	
	
	def write_average_cnf_matrix(self, average_cnf_matrix, average_cnf_matrix_filename):
		'''
		Write average confusion matrix (potentially filtered)
		'''
		
		with open(average_cnf_matrix_filename, 'w') as average_cnf_matrix_file:
			per_fold_sorted_labels = sorted(set(self.per_fold_categorical_labels))
			for label in per_fold_sorted_labels:
				average_cnf_matrix_file.write('%s,' % label)
				sys.stdout.write('%s,' % label)
			average_cnf_matrix_file.write(',\n')
			sys.stdout.write(',\n')
			
			for i, row in enumerate(average_cnf_matrix):
				for occurences in row:
					average_cnf_matrix_file.write('%s,' % occurences)
					sys.stdout.write('%s,' % occurences)
				average_cnf_matrix_file.write('%s,%s\n' % (per_fold_sorted_labels[i], self.label_class[int(per_fold_sorted_labels[i])]))
				sys.stdout.write('%s,%s\n' % (per_fold_sorted_labels[i], self.label_class[int(per_fold_sorted_labels[i])]))
	
	
	def extract_predictions(self):
		'''
		Extracts per-fold predictions from each classifier results directory.
		'''
		
		self.debug_log.info('Starting predictions extraction')
		
		self.per_fold_classifiers_actual_classes_list = []
		self.per_fold_classifiers_predictions_list = []
		
		for classifier_results_dir in classifier_results_dirs:
			predictions_dir = '%s/predictions/*' % classifier_results_dir
			predictions_fold_dirs = sorted(glob.glob(predictions_dir))
			
			self.folds = []
			
			per_fold_classifier_predictions = {}
			per_fold_classifier_actual_classes = {}
			for fold_dir in predictions_fold_dirs:
				
				with open(glob.glob('%s/*dat' % fold_dir)[0]) as per_fold_file:
					per_fold_lines = per_fold_file.readlines()[1:]
					
					per_fold_actual_classes = []
					per_fold_predictions = []
					for per_fold_line in per_fold_lines:
						per_fold_actual_classes.append(int(per_fold_line.strip().split('\t')[0]))
						per_fold_predictions.append(int(per_fold_line.strip().split('\t')[1]))
				
				fold = fold_dir.split('/')[-1]
				self.folds.append(fold)
				
				per_fold_classifier_actual_classes[fold] = per_fold_actual_classes
				per_fold_classifier_predictions[fold] = per_fold_predictions
			
			self.per_fold_classifiers_actual_classes_list.append(per_fold_classifier_actual_classes)
			self.per_fold_classifiers_predictions_list.append(per_fold_classifier_predictions)
		
		if not self.per_fold_classifiers_actual_classes_list[1:] == self.per_fold_classifiers_actual_classes_list[:-1]:
			raise FusionRulesException('ground truth is not reliable: please check results directories of each classifier that should be combined.')
		
		self.debug_log.info('Ending predictions extraction')
	
	
	def extract_soft_values(self):
		'''
		Extracts per-fold soft values from each classifier results directory.
		'''
		
		self.debug_log.info('Starting soft values extraction')
		
		self.per_fold_classifiers_actual_classes_support_list = []
		self.per_fold_classifiers_soft_values_list = []
		
		for classifier_results_dir in classifier_results_dirs:
			soft_values_dir = '%s/soft_values/*' % classifier_results_dir
			soft_values_fold_dirs = sorted(glob.glob(soft_values_dir))
			
			per_fold_classifier_soft_values = {}
			per_fold_classifier_actual_classes = {}
			for fold_dir in soft_values_fold_dirs:
			
				with open(glob.glob('%s/*dat' % fold_dir)[0]) as per_fold_file:
					per_fold_lines = per_fold_file.readlines()[1:]
					
					per_fold_actual_classes = []
					per_fold_soft_values = []
					for per_fold_line in per_fold_lines:
						per_fold_actual_classes.append(int(per_fold_line.strip().split('\t')[0]))
						per_fold_soft_values.append(list(map(float, per_fold_line.strip().split('\t')[1].split(','))))
				
				fold = fold_dir.split('/')[-1]
				
				per_fold_classifier_actual_classes[fold] = per_fold_actual_classes
				per_fold_classifier_soft_values[fold] = per_fold_soft_values
			
			self.per_fold_classifiers_actual_classes_support_list.append(per_fold_classifier_actual_classes)
			self.per_fold_classifiers_soft_values_list.append(per_fold_classifier_soft_values)
		
		if not self.per_fold_classifiers_actual_classes_list == self.per_fold_classifiers_actual_classes_support_list:
			raise FusionRulesException('ground truth is not reliable: please check classifier results directories.')
		
		self.debug_log.info('Ending soft values extraction')



if __name__ == "__main__":
	
	if len(sys.argv) < 6:
		print('usage:', sys.argv[0], '<NUMBER_OF_CLASSIFIERS>', '<CLASSIFIER_1>', '[CLASSIFIER_2]', '...', '[CLASSIFIER_N]', '<COMBINER_INDEX>', '<DATASET_INDEX>', '<OUTDIR>')
		sys.exit(1)
	elif len(sys.argv) < (int(sys.argv[1]) + 5):
		print('usage:', sys.argv[0], sys.argv[1], '<CLASSIFIER_1>', '...', '<CLASSIFIER_%s>' % sys.argv[1], '<COMBINER_INDEX>', '<DATASET_INDEX>', '<OUTDIR>')
		sys.exit(1)
	
	try:
		number_classifiers = int(sys.argv[1])
	except ValueError:
		raise FusionRulesException('number_classifiers must be an integer.')
	
	classifier_results_dirs = []
	for i in range(number_classifiers):
		classifier_results_dirs.append(sys.argv[i + 2])
	
	try:
		combiner = int(sys.argv[number_classifiers + 2])
	except ValueError:
		combiner = -1
	
	# dataset == 1 is FB/FBM; dataset == 2 is Android; dataset == 3 is iOS.
	try:
		dataset = int(sys.argv[number_classifiers + 3])
	except ValueError:
		dataset = -1
	
	outdir = sys.argv[number_classifiers + 4]
	
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
	
	# Defines the parameters used to perform stratified cross-validation
	
	# Set the number of folds for stratified cross-validation
	# num_fold = 10
	
	# Set random seed for reproducibility
	# seed = 412
	
	# Initialize numpy random stateself.debug_log = self.setup_logger('debug', '%s/deep_learning_architectures.log' % self.outdir, logging.DEBUG)
	# numpy.random.seed(seed)
	
	# Calculates the indices used for 10-fold validation
	# kfold = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=seed)
	
	# include_ports = False					# It should be set to False to avoid biased (port-dependent) inputs
	
	fusion_rules = FusionRules(classifier_results_dirs, dataset, outdir)
	
	if combiner == 1:
		print('Selected fusion rule: %s' % non_trainable.MV.__name__)
		fusion_rules.per_fold_fusion_rule(non_trainable.MV, is_soft_combiner = False)
	if combiner == 2:
		print('Selected fusion rule: %s' % CC_non_trainable.CC_Mean.__name__)
		fusion_rules.per_fold_fusion_rule(CC_non_trainable.CC_Mean, is_soft_combiner = True)



