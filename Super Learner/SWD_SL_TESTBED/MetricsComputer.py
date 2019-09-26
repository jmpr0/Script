#!/usr/bin/python3

import numpy as np,copy
import sys,getopt,os
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,recall_score
from scipy.stats.mstats import gmean
from imblearn.metrics import geometric_mean_score
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

class Measure(object):
	def __init__(self):
		self.mean = None
		self.std = None

class Metrics(object):
	def __init__(self):
		self.accuracy = None
		self.f_measure = None
		self.g_mean_multiclass = None
		self.g_mean_macro = None
		self.classified_ratio = None

def main(argv):

	def compute_g_mean_multiclass(y_true, y_pred):
		'''
		Compute g-mean as the geometric mean of the recalls of all classes
		'''
		recalls = recall_score(y_true, y_pred, average=None)
		nonzero_recalls = recalls[recalls != 0]
		is_zero_recall = False
		unique_y_true = list(set(y_true))
		for i, recall in enumerate(recalls):
			if recall == 0 and i in unique_y_true:
				is_zero_recall = True
		if is_zero_recall:
			gmean_ret = gmean(recalls)
		else:
			gmean_ret = gmean(nonzero_recalls)
		return gmean_ret

	np.random.seed(0)

	input_file = ''

	classifier_name = 'rf'

	try:
		opts, args = getopt.getopt(argv,"hi:c:","[input_file=]")
	except getopt.GetoptError:
		print('MetricsComputer.py -i <input_file>')
		sys.exit(2)
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print('MetricsComputer.py -i <input_file>')
			sys.exit()
		if opt in ("-i", "--input_file"):
			input_file = arg
		if opt in ("-c"):
			classifier_name = arg

	print('\n'+input_file+' elaboration...\n')
	#load .dat file
	file = open(input_file, 'r')
	lines = file.readlines()
	file.close()

	file_name = os.path.basename(input_file)

	oracles = list()
	predictions = list()

	classified_ratios = list()

	save_cr = False

	fold_cnt = -1	# first iteration make fold_cnt -eq to 0
	cr_cnt = -1
	for line in lines:
		if '@fold_cr' not in line:
			if not save_cr and '@fold' not in line:
				oracles[fold_cnt].append(line.split()[0])
				predictions[fold_cnt].append(line.split()[1])
			elif '@fold' in line:
				oracles.append(list())
				predictions.append(list())
				fold_cnt += 1
			else:
				classified_ratios[cr_cnt] = float(line)
				save_cr = False
				oracles.append(list())
				predictions.append(list())
				fold_cnt += 1
		else:
			classified_ratios.append(0)
			save_cr = True
			cr_cnt += 1

	k = fold_cnt+1

	metrics = list()

	if save_cr:
		metrics_cr = list()

	for i in range(0,k):
		temp_metrics = Metrics()

		temp_metrics.accuracy = accuracy_score(oracles[i],predictions[i]) if not np.isnan(accuracy_score(oracles[i], predictions[i])) else 0.0
		temp_metrics.f_measure = f1_score(oracles[i],predictions[i],average='macro') if not np.isnan(f1_score(oracles[i],predictions[i],average='macro')) else 0.0
		temp_metrics.g_mean_multiclass = compute_g_mean_multiclass(oracles[i],predictions[i]) if not np.isnan(compute_g_mean_multiclass(oracles[i],predictions[i])) else 0.0
		temp_metrics.g_mean_macro = geometric_mean_score(oracles[i],predictions[i],average='macro') if not np.isnan(geometric_mean_score(oracles[i],predictions[i],average='macro')) else 0.0

		metrics.append(temp_metrics)

		if save_cr:
			temp_metrics = copy.copy(temp_metrics)

			temp_metrics.accuracy *= classified_ratios[i]
			temp_metrics.f_measure *= classified_ratios[i]
			temp_metrics.g_mean_multiclass *= classified_ratios[i]
			temp_metrics.g_mean_macro *= classified_ratios[i]

			metrics_cr.append(temp_metrics)

	data_folder = './data_'+classifier_name+'/metrics_per_fold/'

	if not os.path.exists('./data_'+classifier_name):
		os.makedirs('./data_'+classifier_name)
		os.makedirs(data_folder)
	elif not os.path.exists(data_folder):
		os.makedirs(data_folder)

	f0 = open(data_folder+file_name[:-4]+'_metrics_per_fold.dat','w+')
	# f1 = open(data_folder+file_name[:-4]+'_metrics_cr_per_fold.dat','w+')

	f0.write('acc f1 gmulti gmacro')
	if save_cr:
		f0.write(' cr\n')
	else:
		f0.write('\n')
	# f1.write('acc,f1,gmulti,gmacro\n')
	
	f0.close()
	# f1.close()

	file.close()

	file = open(data_folder+file_name[:-4]+'_metrics_per_fold.dat','a')
	for i in range(0,k):
		file.write(
			str(metrics[i].accuracy)+' '+
			str(metrics[i].f_measure)+' '+
			str(metrics[i].g_mean_multiclass)+' '+
			str(metrics[i].g_mean_macro))
		if save_cr:
			file.write(','+str(classified_ratios[i])+'\n')
		else:
			file.write('\n')

	file.close()

	# if save_cr:
	# 	file = open(data_folder+file_name[:-4]+'_metrics_cr_per_fold.dat','a')
	# 	for i in range(0,k):
	# 		file.write(
	# 			str(metrics_cr[i].accuracy)+' '+
	# 			str(metrics_cr[i].f_measure)+' '+
	# 			str(metrics_cr[i].g_mean_multiclass)+' '+
	# 			str(metrics_cr[i].g_mean_macro)+'\n')

	# 	file.close()

	data_folder = './data_'+classifier_name+'/metrics/'

	if not os.path.exists('./data_'+classifier_name):
		os.makedirs('./data_'+classifier_name)
		os.makedirs(data_folder)
	elif not os.path.exists(data_folder):
		os.makedirs(data_folder)

	f0 = open(data_folder+file_name[:-4]+'_metrics.dat','w+')
	# f1 = open(data_folder+file_name[:-4]+'_metrics_cr.dat','w+')

	f0.write('acc_mean acc_std f1_mean f1_std gmulti_mean gmulti_std gmacro_mean gmacro_std')
	if save_cr:
		f0.write(' cr_mean cr_std\n')
	else:
		f0.write('\n')
	# f1.write('acc_mean,acc_std,f1_mean,f1_std,gmulti_mean,gmulti_std,gmacro_mean,gmacro_std\n')

	f0.close()
	# f1.close()

	file = open(data_folder+file_name[:-4]+'_metrics.dat','a')
	file.write(
		str(np.mean([temp.accuracy for temp in metrics]))+' '+str(np.std([temp.accuracy for temp in metrics]))+' '+
		str(np.mean([temp.f_measure for temp in metrics]))+' '+str(np.std([temp.f_measure for temp in metrics]))+' '+
		str(np.mean([temp.g_mean_multiclass for temp in metrics]))+' '+str(np.std([temp.g_mean_multiclass for temp in metrics]))+' '+
		str(np.mean([temp.g_mean_macro for temp in metrics]))+' '+str(np.std([temp.g_mean_macro for temp in metrics])))
	if save_cr:
		file.write(','+str(np.mean([cr for cr in classified_ratios]))+' '+str(np.std([cr for cr in classified_ratios]))+'\n')
	else:
		file.write('\n')

	file.close()

	# if save_cr:
	# 	file = open(data_folder+file_name[:-4]+'_metrics_cr.dat','a')
	# 	file.write(
	# 		str(np.mean([temp.accuracy for temp in metrics_cr]))+' '+str(np.std([temp.accuracy for temp in metrics_cr]))+' '+
	# 		str(np.mean([temp.f_measure for temp in metrics_cr]))+' '+str(np.std([temp.f_measure for temp in metrics_cr]))+' '+
	# 		str(np.mean([temp.g_mean_multiclass for temp in metrics_cr]))+' '+str(np.std([temp.g_mean_multiclass for temp in metrics_cr]))+' '+
	# 		str(np.mean([temp.g_mean_macro for temp in metrics_cr]))+' '+str(np.std([temp.g_mean_macro for temp in metrics_cr]))+'\n')

	# 	file.close()

if __name__ == "__main__":
	os.environ["DISPLAY"] = ":0" # used to show xming display
	main(sys.argv[1:])