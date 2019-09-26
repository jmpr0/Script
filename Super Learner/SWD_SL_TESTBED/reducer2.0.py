import sys,os,numpy as np
from sklearn.utils import resample
from time import time

inputs=sys.argv[1:]

if len(inputs) < 6:
	print('usage',sys.argv[0],'<DATASET_FILENAME> <LABEL_INDEX> <LABEL_NUMBER> <LABEL_TAG0,LABEL_TAG1,...> <PERC_TAG0,PERC_TAG1,...> <SEED>')
	exit(1)

dataset_filename = inputs[0]
label_index = int(inputs[1])
label_number = int(inputs[2])

percs = {}

try:

	label_tags = inputs[3].split(',')

	for i,label_tag in enumerate(label_tags):
		percs[label_tag] = float(inputs[4].split(',')[i])

	seed = int(inputs[5])

except:

	print('have you specified all the label_tags as more as the label_number states?')
	exit(1)

with open(dataset_filename,'r') as f:
	if dataset_filename.endswith('.csv'):
		attributes = f.readline()
		dataset = f.readlines()

	elif dataset_filename.endswith('.arff'):
		all_dataset = f.readlines()
		data_start = all_dataset.index('@data\n')
		attributes = all_dataset[:data_start+1]
		dataset = all_dataset[data_start+1:]
		del all_dataset

try:
	labels = [ row.split(',')[label_index].strip() for row in dataset ]
except:
	print('maybe label_index inserted is out of possible indexes')
	exit(1)

labels_count = np.unique(labels, return_counts=True)

labels_newcount_dict = {}

for label,count in zip(labels_count[0], labels_count[1]):
	if label in label_tags:
		count *= percs[label] / 100
	labels_newcount_dict[label] = int(count)

new_dataset = []

for label in labels_currcount_dict:
	indexes = [ i for i,l in enumerate(labels) if l == label ]

	indexes_red = resample(indexes, replace=False, n_samples=labels_newcount_dict[label], random_state=seed)

	new_dataset += [ dataset[i] for i in indexes_red ]

with open(dataset_filename.split('.')[0]+'_red.'+dataset_filename.split('.')[1],'w') as f:
	f.writelines(attributes)
	f.writelines(new_dataset)

with open(dataset_filename+'.log','a') as f:
	f.write(str(time())+' '+dataset_filename.split('.')[0]+'_red'+dataset_filename.split('.')[1])
	f.write('\nprevious_counts:\n'+str(labels_count)+'\n')
	f.write('\nactual_counts: '+str(labels_newcount_dict)+'\n')
