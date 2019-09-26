#!/usr/bin/python3

import numpy as np,copy
import sys,getopt,os
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
import seaborn as sns
import itertools
from numpy import interp
import hashlib

pd.options.mode.chained_assignment = None  # default='warn'
epsilon = 1e-3
threshold = 1e-2

class ROCGenerator(object):

	def __init__(self, input_file, file_label, multiple, hash_tag, style, true_class, scale, fold_n, print_std):
		self.input_file = input_file
		self.file_label = file_label
		self.classifier_name = ' '.join(self.input_file.split('/')[0].split('_')[1:])
		self.multiple = multiple
		self.hash_tag = hash_tag
		self.color = style[0]
		self.linestyle = style[1]
		self.marker = style[2]
		self.true_class = true_class
		self.true_class_index = None
		self.scale = scale
		self.fold_n = fold_n
		self.print_std = print_std

	def ROC_compute(self):

		print('\n'+self.input_file+' elaboration...\n')
		#load .dat file
		file = open(self.input_file, 'r')
		lines = file.readlines()
		file.close()

		file_name = os.path.basename(self.input_file)

		level = int([file_name.split('_')[i+1] for i in range(0,len(file_name.split('_'))) if file_name.split('_')[i]=='level'][0])

		oracles = list()
		probabilities = list()

		classified_ratios = list()

		save_cr = False

		fold_cnt = -1	# first iteration make fold_cnt -eq to 0
		cr_cnt = -1

		for i,line in enumerate(lines):
			if '@fold_cr' not in line:
				if not save_cr and '@fold' not in line:
					if line.split()[1:][0] != 'nan':
						oracles[-1][-1].append(line.split(' ')[0])
						probabilities[-1][-1].append([float(l) for l in line.split()[1:]])
				elif '@fold' in line:
					try:
						if '@fold' in lines[i+1]:
							continue
						if len(line.split()) > 3 and self.true_class_index is None:
							self.true_class_index = int([ i for i,v in enumerate(line.split()[2:]) if v == self.true_class ][0])
						if lines[i+1].split()[1:][0] != 'nan':
							if (fold_cnt+1) % self.fold_n == 0:
								oracles.append(list())
								probabilities.append(list())
							oracles[-1].append(list())
							probabilities[-1].append(list())
							fold_cnt += 1
					except:
						pass
				else:
					classified_ratios[cr_cnt] = float(line)
					save_cr = False
					if fold_cnt+1 % self.fold_n == 0:
						oracles.append(list())
						probabilities.append(list())
					oracles[-1].append(list())
					probabilities[-1].append(list())
					fold_cnt += 1
			else:
				classified_ratios.append(0)
				save_cr = True
				cr_cnt += 1

		k = self.fold_n

		if self.multiple:
			image_folder = './image_multiple/roc_curve/'
			if not os.path.exists('./image_multiple'):
				os.makedirs('./image_multiple')
				os.makedirs(image_folder)
			elif not os.path.exists(image_folder):
				os.makedirs(image_folder)
		else:
			classifier_name = self.classifier_name.split(' ')[0]
			image_folder = './image_'+classifier_name+'/roc_curve/'
			if not os.path.exists('./image_'+classifier_name):
				os.makedirs('./image_'+classifier_name)
				os.makedirs(image_folder)
			elif not os.path.exists(image_folder):
				os.makedirs(image_folder)

		if self.scale == 'linear':
			mean_fpr = np.linspace(0,1,1000)
		if self.scale == 'log':
			mean_fpr = np.logspace(-3,0,1000)
		tprs = []
		thrs = []
		roc_aucs = []
		roc_aucs_bound = []
		mean_tpr = []
		std_tpr = []
		mean_thr = []
		std_thr = []
		fig = plt.figure(1)
		average_n = len(oracles)
		for a in range(average_n):
			tprs.append([])
			thrs.append([])
			roc_aucs.append([])
			roc_aucs_bound.append([])
			mean_tpr.append([])
			std_tpr.append([])
			for i in range(k):
				if self.true_class in oracles[a][i]:
					oracle = oracles[a][i]
					try:
						probability = [ p[self.true_class_index] for p in probabilities[a][i] ]
					except:
						probability = probabilities[a][i]
					fpr, tpr, thresholds = roc_curve(oracle,probability,pos_label=self.true_class)
					roc_auc = roc_auc_score(oracle,probability)
					roc_auc_bound = roc_auc_score(oracle,probability,max_fpr=threshold)
					tprs[-1].append(interp(mean_fpr, fpr, tpr))
					thrs[-1].append(interp(mean_fpr, fpr, thresholds))
					tprs[-1][-1][0] = 0.0
					roc_aucs[-1].append(roc_auc)
					roc_aucs_bound[-1].append(roc_auc_bound)
			mean_tpr[-1] = np.mean(tprs[-1], axis=0)
			std_tpr[-1] = np.std(tprs[-1], axis=0)
			mean_tpr[-1][-1] = 1.0

		roc_aucs = np.mean(roc_aucs, axis=0)
		roc_aucs_bound = np.mean(roc_aucs_bound, axis=0)
		mean_tpr = np.mean(mean_tpr, axis=0)
		std_tpr = np.mean(std_tpr, axis=0)
		mean_thr = np.mean(thrs, axis=0)
		std_thr = np.mean(thrs, axis=0)
		mean_roc_auc = np.mean(roc_aucs)
		std_roc_auc = np.std(roc_aucs)
		mean_roc_auc_bound = np.mean(roc_aucs_bound)
		std_roc_auc_bound = np.std(roc_aucs_bound)

		print(mean_roc_auc, self.compute_pAUC(mean_fpr,mean_tpr))
		print(mean_roc_auc_bound, self.compute_pAUC(mean_fpr,mean_tpr,.03))

		if self.classifier_name.startswith('kdae 3 elu a'):
			self.classifier_name = 'LSG-DAE'
		if self.classifier_name.startswith('kdae 3 elu n'):
			self.classifier_name = 'DAE'
		if self.classifier_name == 'sif':
			self.classifier_name = 'IF'
		elif self.classifier_name == 'slo':
			self.classifier_name = 'LOF'
		elif self.classifier_name == 'ssv':
			self.classifier_name = 'OC-SVM'
		label = r'%s' % (' '.join([ self.classifier_name, self.file_label ]))

		if multiple:
			color = self.color
			linestyle = self.linestyle
			marker = self.marker
		else:
			try:
				color = classifiers[self.classifier_name][0]
				linestyle = classifiers[self.classifier_name][1]
				marker = self.marker
			except:
				color = 'darkred'
				linestyle = '--'
				marker = self.marker

		if self.scale == 'linear':
			dot_x = list(np.linspace(epsilon, 1, 10))
			index_y = [ int(i) for i in np.linspace(0, 999, 10) ]
		if self.scale == 'log':
			dot_x = list(np.logspace(-3, 0, 10))
			index_y = [ 111*i for i in range(10) ]
			index_y[0] = 1

		plt.plot(mean_fpr, mean_tpr, color=color, linestyle=linestyle,
			lw=1, alpha=0.8, markersize=7)
		plt.scatter(dot_x, list(mean_tpr[index_y]), edgecolor=color, facecolor='none',
			marker=marker, alpha=0.8)
		plt.plot([-1], [-1],
			color=color,
			linestyle=linestyle,
			label=label,
			markeredgecolor=color,
			markersize=7,
			fillstyle='none',
			marker=marker,
			lw=1,
			alpha=0.8
			)

		leg = plt.legend(loc='best')
		leg.get_frame().set_linewidth(0.0)

		plt.plot(np.linspace(epsilon,1,1000), np.linspace(0,1,1000), color='black', lw=1, linestyle='--')
		# plt.plot([ threshold ]*1000, np.linspace(0,1,1000), color='darkred', lw=1, linestyle='-')

		if self.print_std:
			tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
			tprs_lower = np.maximum(mean_tpr - std_tpr, epsilon)
			plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color[4:], alpha=.2)

		plt.grid(linestyle=':')
		plt.xticks(np.arange(0, 1, step=0.1))
		plt.yticks(np.arange(0, 1, step=0.1))

		plt.xlim([epsilon,1.0])
		# plt.xlim([0.0,1.0])
		plt.ylim([0.0,1.0])
		if len(self.scale.split('_')) > 1:
			plt.xscale(self.scale.split('_')[0])
			plt.yscale(self.scale.split('_')[1])
		else:
			plt.xscale(self.scale)

		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')

		plt.savefig(image_folder+'roc_curve_'+self.hash_tag+'.pdf', format='pdf', dpi=300, bbox_inches='tight')
		with open(image_folder+'AUC_'+self.hash_tag+'.dat', 'a') as f:
			f.write('%s AUC_1 %0.5f %0.5f\n' % (label, mean_roc_auc, std_roc_auc))
			f.write('%s AUC_%.2f %.5f %.5f\n' % (label, threshold, mean_roc_auc_bound, std_roc_auc_bound))
		with open(image_folder+'pAUC%.2f_'%threshold+self.hash_tag+'_latex.dat', 'a') as f:
			f.write('%s\n$%0.2f\\pm%0.2f$\n' % (label, mean_roc_auc_bound*100, std_roc_auc_bound*100))
		with open(image_folder+'AUC_'+self.hash_tag+'_latex.dat', 'a') as f:
			f.write('%s\n$%0.2f\\pm%0.2f$\n' % (label, mean_roc_auc*100, std_roc_auc*100))

		fprs = [.01, .03, .05, .07, .1]
		save=False
		j=0
		with open(image_folder+'thresholds.dat', 'a') as f:
			for i,fpr in enumerate(mean_fpr):
				if not save and fpr < fprs[j]:
					continue
				elif not save:
					save=True
				if save:
					j+=1
					f.write('%s %s %s'%(label.strip(), fpr, mean_tpr[i]))
					for k in range(10):
						f.write(' %s'%mean_thr[k][i])
					f.write('\n')
					save=False
					if j==len(fprs):
						break

	def multi_to_byn(self, vector, true_label):
		vector = [ 1 if v == true_label else 0 for v in vector ]
		return vector

	def compute_pAUC(self,fpr,tpr,p=1.):
		'''
		Compute the pAUC given fpr and tpr vectors
		'''
		tprs = [ v for i,v in enumerate(tpr) if fpr[i] <= p ]
		pAUC = np.mean(tprs)
		return pAUC

if __name__ == "__main__":
	
	np.random.seed(0)
	input_file = ''
	classifier_name = 'wrf'
	true_class = ''
	classifiers = {
		'slo' : ['darkred','--'],
		'sif' : ['darkblue','-.'],
		'ssv' : ['darkgreen',':'],
		'kae' : ['darkred','--'],
		'kdae' : ['darkblue','-.'],
		'k1dcnnae' : ['darkgreen',':']}
	linestyles = [
		['darkred','-','v'],
		['darkblue','-','o'],
		['darkgreen','-','s'],
		['darkorange','-','*'],
		['darkviolet','-','D'],
		['darkcyan','-','h'],
		['darkmagenta','-','+'],
		['darkred','-','x'],
		['darkblue','-','^'],
		['darkgreen','-','p'],
	]
	multiple = False

	if len(sys.argv) < 3 :
		print('RocCurveGenerator.py <N_FILES> <INPUT_FILE_1>§(<LABEL_FILE_1>) ... <INPUT_FILE_N>§(<LABEL_FILE_N>) <TRUE_CLASS> (<SCALE>) (<FOLD_NUM>) (<STD_DEV>) (<RESULT_LABEL>)')
		exit(1)
	try:
		n_files = int(sys.argv[1])
	except:
		print('RocCurveGenerator.py <N_FILES> <INPUT_FILE_1>§(<LABEL_FILE_1>) ... <INPUT_FILE_N>§(<LABEL_FILE_N>) <TRUE_CLASS> (<SCALE>) (<FOLD_NUM>) (<STD_DEV>) (<RESULT_LABEL>)')
		exit(1)
	if len(sys.argv) < 2 + n_files:
		print('RocCurveGenerator.py <N_FILES> <INPUT_FILE_1>§(<LABEL_FILE_1>) ... <INPUT_FILE_N>§(<LABEL_FILE_N>) <TRUE_CLASS> (<SCALE>) (<FOLD_NUM>) (<STD_DEV>) (<RESULT_LABEL>)')
		exit(1)

	input_files = []
	file_labels = []

	if n_files > 1:
		multiple = True

	for i in range(n_files):
		input_file = sys.argv[2+i]
		if len(input_file.split('§')) == 2:
			input_files.append(input_file.split('§')[0])
			file_labels.append(input_file.split('§')[1])
		elif len(input_file.split('§')) == 1:
			input_files.append(input_file)
			file_labels.append('')
		else:
			raise ValueError('Input files had to be passed as <INPUT_FILE_i>§<LABEL_FILE_i> to specify a specific label, or simply <INPUT_FILE_i>.\
				File name cannot contain character "§".')

	true_classes = sys.argv[3+i].split(',')
	while len(true_classes) < len(input_files):
		true_classes.append(true_classes[-1])
	scale='linear'
	fold_n=10
	print_std=False
	result_label=''
	try:
		scale = sys.argv[4+i]
		fold_n = int(sys.argv[5+i])
		print_std = bool(int(sys.argv[6+i]))
		result_label = sys.argv[7+i]
	except:
		pass
	sorted_index = np.argsort(input_files)
	input_files = [ input_files[i] for i in sorted_index ]
	file_labels = [ file_labels[i] for i in sorted_index ]
	true_classes = [ true_classes[i] for i in sorted_index ]

	m = hashlib.md5()
	for input_file in input_files:
		m.update(input_file.encode('utf-8'))
	hash_tag = result_label + '_' + str(m.hexdigest()[:6]) + '_' + scale

	for i,input_file in enumerate(input_files):
		generator = ROCGenerator(input_file, file_labels[i], multiple, hash_tag, linestyles[i], true_classes[i], scale, fold_n, print_std)
		generator.ROC_compute()