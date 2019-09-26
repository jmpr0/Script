#!/usr/bin/python3
import sys, os, numpy as np

if sys.argv[1] == '-h':
	print('Usage: %s <normal_label> <labels_file> <probas_file>\n' % sys.argv[0])
	exit()

labels_per_fold=[]
probas_per_fold=[]
normal_label=sys.argv[1]

with open(sys.argv[2],'r') as labels_per_fold_file:
	lines=labels_per_fold_file.readlines()

for i,line in enumerate(lines):
	try:
		if '@fold' in lines[i+1]:
			continue
	except:
		break
	if '@fold' in line:
		labels_per_fold.append([])
	else:
		labels_per_fold[-1].append(line.strip())

with open(sys.argv[3],'r') as probas_per_fold_file:
	lines=probas_per_fold_file.readlines()

for i,line in enumerate(lines):
	try:
		if '@fold' in lines[i+1]:
			continue
	except:
		break
	if '@fold' in line:
		probas_per_fold.append([])
	else:
		probas_per_fold[-1].append(line.split()[-1].strip())

labels_u = [l for l in np.unique(labels_per_fold)[0] if l != normal_label]
fout_names={}
for label in labels_u:
	fout_names[label] = '%s_%s.dat'%(sys.argv[3].split('.dat')[0],label)
	with open(fout_names[label],'w') as fout:
		pass

# for i,(labels,probas) in enumerate(zip(labels_per_fold,probas_per_fold)):
# 	print('Elaborating fold n.%s...'%(i+1))
# 	for label in fout_names:
# 		with open(fout_names[label],'a') as fout:
# 			fout.write('@fold\n')
# 	for label,proba in zip(labels,probas):
# 		if label == normal_label:
# 			for l in fout_names:
# 				with open(fout_names[l],'a') as fout:
# 					fout.write('0 %s\n'%(proba))
# 		else:
# 			with open(fout_names[label],'a') as fout:
# 				fout.write('1 %s\n'%(proba))

for i,(labels,probas) in enumerate(zip(labels_per_fold,probas_per_fold)):
	print('Elaborating fold n.%s...'%(i+1))
	fname='%s_noDoS_noDDoS.dat'%(sys.argv[3].split('.dat')[0])
	with open(fname,'a') as fout:
		fout.write('@fold\n')
		for label,proba in zip(labels,probas):
			if label not in ['DoS', 'DDoS']:
				if label == normal_label:
					fout.write('0 %s\n'%(proba))
				else:
					fout.write('1 %s\n'%(proba))