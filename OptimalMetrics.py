#!/usr/bin/python3

import numpy as np,sys,os,getopt

def main(argv):

	classifier_name = 'rf'

	try:
		opts, args = getopt.getopt(argv,"hc:")
	except getopt.GetoptError:
		print('OptimalMetrics.py -c <classifier_tag>')
		sys.exit(2)
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print('OptimalMetrics.py -c <classifier_tag>')
			sys.exit()
		if opt in ("-c"):
			classifier_name = arg

	files = os.listdir('./data_'+classifier_name+'/metrics/join_fp/')
	data_fold = './data_'+classifier_name+'/metrics/join_fp/'

	fout = open('./optimal_' + classifier_name + '.dat','w+')
	fout.close()

	for file in sorted(files, reverse=True):
		if 'gamma' not in file and 'optimal' not in file:
			lines = open(data_fold+file,'r').readlines()
			lines_splitted = [line.split() for line in lines]
			xs = [int(line_splitted[0]) for line_splitted in lines_splitted]
			accuracies = [float(line_splitted[1]) for line_splitted in lines_splitted]
			f_measures = [float(line_splitted[3]) for line_splitted in lines_splitted]
			racall_1 = [float(line_splitted[5]) for line_splitted in lines_splitted]
			recall_0 = [float(line_splitted[7]) for line_splitted in lines_splitted]
			g_mean_multiclass = [float(line_splitted[9]) for line_splitted in lines_splitted]
			g_mean_macro = [float(line_splitted[11]) for line_splitted in lines_splitted]
			accuracies_std = [float(line_splitted[2]) for line_splitted in lines_splitted]
			f_measures_std = [float(line_splitted[4]) for line_splitted in lines_splitted]
			racall_1_std = [float(line_splitted[6]) for line_splitted in lines_splitted]
			recall_0_std = [float(line_splitted[8]) for line_splitted in lines_splitted]
			g_mean_multiclass_std = [float(line_splitted[10]) for line_splitted in lines_splitted]
			g_mean_macro_std = [float(line_splitted[12]) for line_splitted in lines_splitted]
			fout = open('./optimal_' + classifier_name + '.dat','a')

			if 'inferred' not in file:
				try:
					max_fmeasure_index = f_measures.index(max(f_measures))
				except:
					print('error',file)

			fout.write(
				file+'\t\t'+
				str(accuracies[max_fmeasure_index])+'\t'+
				str(f_measures[max_fmeasure_index])+'\t'+
				str(racall_1[max_fmeasure_index])+'\t'+
				str(recall_0[max_fmeasure_index])+'\t'+
				str(g_mean_multiclass[max_fmeasure_index])+'\t'+
				str(g_mean_macro[max_fmeasure_index])+'\t'+
				str(accuracies_std[max_fmeasure_index])+'\t'+
				str(f_measures_std[max_fmeasure_index])+'\t'+
				str(racall_1_std[max_fmeasure_index])+'\t'+
				str(recall_0_std[max_fmeasure_index])+'\t'+
				str(g_mean_multiclass_std[max_fmeasure_index])+'\t'+
				str(g_mean_macro_std[max_fmeasure_index])+'\t'+
				str(xs[max_fmeasure_index])+'\n')


if __name__ == "__main__":
	os.environ["DISPLAY"] = ":0" # used to show xming display
	main(sys.argv[1:])