#!/usr/bin/python3
'''
Implementation of code to generate Confidence Histograms and Reliability Diagrams as showed in:
Guo, Chuan, et al. "On calibration of modern neural networks." Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017.

Giampaolo Bovenzi - 19/06/2019
'''
import sys, os, numpy as np, matplotlib.pyplot as plt, matplotlib.patches as mpatches

os.environ["DISPLAY"] = ":0"

if len(sys.argv) < 3:
	print('Usage:\npython3 %s <CONFIDENCE_FILE> <NUM_BINS>' % sys.argv[0])
	exit()

f_name=sys.argv[1]
M=int(sys.argv[2])

samples_per_fold = []

'''
Files contain confidence value are formatted as follows:

@fold Actual <Label_1> <Label_2> ... <Label_n>
<Actual_11> <Confidence_111> <Confidence_121> ... <Confidence_1n1>
...
<Actual_1i> <Confidence_11i> <Confidence_12i> ... <Confidence_1ni>
...
<Actual_1m> <Confidence_11m> <Confidence_12m> ... <Confidence_1nm>
@fold Actual <Label_1> <Label_2> ... <Label_n>
<Actual_21> <Confidence_211> <Confidence_221> ... <Confidence_2n1>
...
@fold Actual <Label_1> <Label_2> ... <Label_n>
<Actual_f1> <Confidence_f11> <Confidence_f21> ... <Confidence_fn1>
...
<Actual_fi> <Confidence_f1i> <Confidence_f2i> ... <Confidence_fni>
...
<Actual_fm> <Confidence_f1m> <Confidence_f2m> ... <Confidence_fnm>

So @fold starts list of confidence values for each sample.

'''

# For each fold we save the list of samples, of which we save Actual tag, predicted tag, and max confidence (max proba)
with open(f_name) as fin:
	lines=fin.readlines()
	for i,line in enumerate(lines):
		try:
			if '@fold' in line:
				# Sometimes we could found more @fold at start of file, so we need to jump them
				if '@fold' in lines[i+1]:
					continue
				# As previously commented, values on @fold row after "Actual" are the labels in the order we found confidence values
				tags = line.strip().split(' ')[2:]
				samples_per_fold.append([])
			else:
				line_s = line.strip().split(' ')
				# Tag is Actual
				tag = line_s[0]
				probas = [ float(p) for p in line_s[1:] ]
				max_proba = np.max(probas)
				max_proba_index = probas.index(max_proba)
				# Pred tag is Predicted class, corresponding to max confidence value
				pred_tag = tags[max_proba_index]
				samples_per_fold[-1].append([tag,pred_tag,max_proba])
		except:
			pass

L = len(tags)

# Divide samples in respective bin
bins_per_fold=[]
for samples in samples_per_fold:
	bins_per_fold.append([])
	for i in range(1,M+1):
		bins_per_fold[-1].append([])
		for sample in samples:
			if (i-1)/M < sample[2] <= i/M:
				bins_per_fold[-1][-1].append(sample)

# Compute accuracy for each bin
accuracy_per_fold=[]
confidence_per_fold=[]
for bins in bins_per_fold:
	accuracy_per_fold.append([])
	confidence_per_fold.append([])
	for bin in bins:
		if len(bin)>0:
			accuracy_per_fold[-1].append(np.sum([ 1 if s[0]==s[1] else 0 for s in bin ])/len(bin))
			confidence_per_fold[-1].append(np.sum(s[2] for s in bin)/len(bin))
		else:
			accuracy_per_fold[-1].append(0)
			confidence_per_fold[-1].append(0)

# Compute ECE and MCE for each bin
sample_num = len(samples_per_fold[0])
expected_confidence_error_per_fold=[]
maximum_calibration_error_per_fold=[]
for bins,accuracy,confidence in zip(bins_per_fold,accuracy_per_fold,confidence_per_fold):
	expected_confidence_error_per_fold.append(np.sum([len(bin)*np.abs(bin_acc-bin_conf) for bin,bin_acc,bin_conf in zip(bins,accuracy,confidence)])/sample_num)
	maximum_calibration_error_per_fold.append(np.max([np.abs(bin_acc-bin_conf) for bin_acc,bin_conf in zip(accuracy,confidence)]))

# Each measure for each bin is provided over k-fold validation, now we provide the mean and std values over k fold
mean_binlen=np.mean(np.array([[ len(bin) for bin in bins ] for bins in bins_per_fold]),axis=0)
std_binlen=np.std(np.array([[ len(bin) for bin in bins ] for bins in bins_per_fold]),axis=0)
mean_accuracy=np.mean(np.array(accuracy_per_fold),axis=0)
std_accuracy=np.std(np.array(accuracy_per_fold),axis=0)
mean_confidence=np.mean(np.array(confidence_per_fold),axis=0)
std_confidence=np.std(np.array(confidence_per_fold),axis=0)
mean_ece=np.mean(np.array(expected_confidence_error_per_fold),axis=0)
std_ece=np.std(np.array(expected_confidence_error_per_fold),axis=0)
mean_mce=np.mean(np.array(maximum_calibration_error_per_fold),axis=0)
std_mce=np.std(np.array(maximum_calibration_error_per_fold),axis=0)

# Average accuracy and average confidence are computed in a weighted way
average_accuracy=np.average(mean_accuracy,weights=mean_binlen)
average_confidence=np.average(mean_confidence,weights=mean_binlen)

fig=plt.figure(1)
# Puts grid behind the graph
plt.rc('axes', axisbelow=True)
plt.rc('font', size=14)
# Removes margins
plt.margins(x=0,y=0)
# Plots histogram of bin length
plt.bar(np.arange(M)/M,mean_binlen/np.sum(mean_binlen),align='edge',width=1/M,color='royalblue',edgecolor='k')
# Plots average accuracy line
avg_acc=plt.plot([average_accuracy]*2, [0,1], color='crimson', ls='--', lw=3, label='Accuracy')
# Plots average confidence line
avg_conf=plt.plot([average_confidence]*2, [0,1], color='green', ls='--', lw=3, label='Average confidence')
plt.legend(loc='upper left',fontsize=14)
# Inserts the label for previous two lines, according the respective position
# if average_accuracy > average_confidence:
# 	plt.text(average_accuracy+.02,.9,'Accuracy',rotation=90,color='k',fontsize=14)
# 	plt.text(average_confidence-.07,.9,'Average confidence',rotation=90,color='k',fontsize=14)
# else:
# 	plt.text(average_accuracy-.07,.9,'Accuracy',rotation=90,color='k',fontsize=14)
# 	plt.text(average_confidence+.02,.9,'Average confidence',rotation=90,color='k',fontsize=14)
# Plots grid
plt.grid(ls='--')
# Sets axix limits
plt.xlim(0,1)
plt.ylim(0,1)
# Force equality in axis representation
plt.axis('square')
plt.savefig('%s_confidence_histo.pdf'%f_name)

fig=plt.figure(2)
plt.rc('axes', axisbelow=True)
plt.rc('font', size=14)
plt.margins(x=0,y=0)
# Compute lower gap, i.e. the difference among confidence and accuracy when accuracy is lt confidence
lower_gap = np.arange(M)/M+1/(2*M)
# Initialize upper gap as zeros vector
upper_gap = np.zeros(len(lower_gap))
lower_acc = np.array(mean_accuracy)
upper_acc = np.array(mean_accuracy)
# To test upper_gap visualization comment subsequent line
# accuracy_to_plot[5] = .8
# If accuracy for a bin is higher than confidence, we want to draw a gap bar that include accuracy
for i,(b,a) in enumerate(zip(lower_gap,lower_acc)):
	if a > b:
		lower_gap[i] = 0
		upper_gap[i] = a
		lower_acc[i] = b
# Plotting red bar without edges with alpha=.5
plt.bar(np.arange(M)/M,lower_gap,align='edge',width=1/M,color='crimson',edgecolor='none',alpha=.5)
# Adding black edges and hatch to previous bar
plt.bar(np.arange(M)/M,lower_gap,align='edge',width=1/M,color='none',edgecolor='k',hatch="//")
# Plotting red bar without edges with alpha=.5
plt.bar(np.arange(M)/M,upper_gap,align='edge',width=1/M,color='crimson',edgecolor='none',alpha=.5)
# Plotting blue bar without edges with alpha=.5, to simulate overlap of accuracy and gap
plt.bar(np.arange(M)/M,upper_gap,align='edge',width=1/M,color='royalblue',edgecolor='none',alpha=.5)
# Adding black edges and hatch to previous bar
plt.bar(np.arange(M)/M,upper_gap,align='edge',width=1/M,color='none',edgecolor='k',hatch="//")
# Plotting solid blue bar without edges, accuracy greater than diagonal is not colored
plt.bar(np.arange(M)/M,lower_acc,align='edge',width=1/M,color='royalblue',edgecolor='none')
# Adding black edges and hatch to previous bar, including overlapping portion
plt.bar(np.arange(M)/M,upper_acc,align='edge',width=1/M,color='none',edgecolor='k')
# Creating the Patches for label
gap=(mpatches.Patch(facecolor='crimson',alpha=.5),mpatches.Patch(facecolor='none',hatch='//',edgecolor='k'))
out=mpatches.Patch(facecolor='royalblue',alpha=1,edgecolor='k')
plt.legend([gap,out],['Gap','Outputs'],loc='upper left',fontsize=14,numpoints=1)
plt.plot([0,1], [0,1], color='gray', ls='--', lw=3)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
plt.text(.7,.05,'ECE=%.2f'%mean_ece,bbox=props,fontsize=14)
plt.grid(ls='--')
plt.xlim(0,1)
plt.ylim(0,1)
plt.axis('square')
# plt.show()
plt.savefig('%s_reliability_diag.pdf'%f_name)