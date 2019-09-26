#!/usr/bin/python3

import os, sys, numpy as np

file_proba=sys.argv[1]
anomaly_class=sys.argv[2]
file_out_base=sys.argv[3]
fin=open(file_proba)
lines=fin.readlines()
from sklearn.metrics import f1_score, precision_score, recall_score

gt=[]
proba=[]
attributes=''

for i,line in enumerate(lines):
	try:
		if '@fold' in line:
			while '@fold' in lines[i+1]:
				i+=1
				line=lines[i]
			line_s = line.split(' ')
			if attributes == '':
				attributes = [ a.strip() for a in line_s[2:] ]
			gt.append([])
			proba.append([])
		else:
			line_s = line.split(' ')
			gt[-1].append(line_s[0])
			proba[-1].append([ float(p.strip()) for p in line_s[1:] ])
	except:
		pass

k=len(gt)

x = np.arange(0,1,.1)
y_f1 = []
y_recall = []
y_precision = []

for i in x:
	f1=[]
	recall=[]
	precision=[]
	for j in range(k):
		pred = []
		for t,p in zip(gt[j],proba[j]):
			p_max = np.max(p)
			if p_max > i:
				pred.append(attributes[p.index(p_max)])
			else:
				pred.append(anomaly_class)
		f1.append(f1_score(gt[j],pred,average='macro'))
		recall.append(recall_score([ 0 if t!=anomaly_class else 1 for t in gt[j] ],[ 0 if p!=anomaly_class else 1 for t in pred ]))
		precision.append(precision_score([ 0 if t!=anomaly_class else 1 for t in gt[j] ],[ 0 if p!=anomaly_class else 1 for t in pred ]))
	with open('%s_%.2f.dat' % (file_out_base,i),'w') as fout:
		fout.write('Mean_F1,Std_F1,Mean_Rec,Std_Rec,Mean_Prec,Std_Prec\n')
		fout.write('%s,%s,%s,%s,%s,%s\n' % (np.mean(f1),np.std(f1),np.mean(recall),np.std(recall),np.mean(precision),np.std(precision)))

	y_f1.append(np.mean(f1))
	y_recall.append(np.mean(recall))
	y_precision.append(np.mean(precision))

os.environ["DISPLAY"] = ":0"
import matplotlib.pyplot as plt
plt.plot(x,y_f1,'r')
plt.plot(x,y_recall,'b')
plt.plot(x,y_precision,'g')
plt.savefig('%s_%.2f.pdf' % (file_out_base,i))
