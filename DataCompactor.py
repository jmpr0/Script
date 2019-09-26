#!/usr/bin/python3

import numpy as np,sys,os,getopt,shutil

def main(argv):

	classifier_name = 'rf'

	try:
		opts, args = getopt.getopt(argv,"hc:")
	except getopt.GetoptError:
		print('DataCompactor.py -c <classifier_tag>')
		sys.exit(2)
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print('DataCompactor.py -c <classifier_tag>')
			sys.exit()
		if opt in ("-c"):
			classifier_name = arg

	base_path = './data_'+classifier_name+'/metrics/'

	if not os.path.exists(base_path):
		print(classifier_name + ' path does not exist.')
		exit(1)

	files = os.listdir(base_path)
	files_splitted = list()

	lines = {}
	gamma_lines = {}

	files = [file for file in files if 'hierarchical' not in file]

	for file in files:
		files_splitted.append(file.split('_'))

	for file,file_splitted in zip(files,files_splitted):
		if file.endswith('.dat'):
			base_index = file_splitted.index('level')
			root = '_'.join(np.r_[file_splitted[:base_index+2],file_splitted[base_index+4:]])

			if file_splitted[base_index+2] == 'w':
				x = file_splitted[base_index+3].split('.')[0]
			else:
				x = file_splitted[base_index+3]

			fin = open(base_path+file,'r')

			if root not in lines:
				lines[root] = list()
			lines[root].append(x+' '+fin.readlines()[-1])
			
			fin.close()

			if 'gamma' in file:
				root = '_'.join(np.r_[file_splitted[:6],file_splitted[8:]])

				gamma = file_splitted[7]

				fin = open(base_path+file,'r')

				if root not in gamma_lines:
					gamma_lines[root] = list()
				gamma_lines[root].append(gamma+' '+fin.readlines()[-1])

				fin.close()

	data_folder = base_path + 'join_fp/'

	if os.path.exists(data_folder):
		shutil.rmtree(data_folder)
	os.makedirs(data_folder)

	for key in lines:
		fout = open(data_folder+key,'w+')
		fout.close()
		fout = open(data_folder+key,'a')

		to_sort = [float(v.split()[0]) for v in lines[key]]
		sorting = np.argsort(to_sort)
		lines[key] = [lines[key][i] for i in sorting]

		fout.writelines(lines[key])

		fout.close()

	data_folder = base_path + 'join_gamma/'

	if os.path.exists(data_folder):
		shutil.rmtree(data_folder)
	os.makedirs(data_folder)

	for key in gamma_lines:
		fout = open(data_folder+key,'w+')
		fout.close()
		fout = open(data_folder+key,'a')

		to_sort = [float(v.split()[0]) for v in gamma_lines[key]]
		sorting = np.argsort(to_sort)
		gamma_lines[key] = [gamma_lines[key][i] for i in sorting]

		fout.writelines(gamma_lines[key])

		fout.close()

if __name__ == "__main__":
	os.environ["DISPLAY"] = ":0" # used to show xming display
	main(sys.argv[1:])