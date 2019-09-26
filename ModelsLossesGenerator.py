#!/usr/bin/python3

import os, sys

os.environ["DISPLAY"] = ":0"

files = []
for file in sys.argv[1:]:
	files.append(file)

models_losses = []
models_names = []
models_jrs = []
models_gls = []

for file in sorted(files):
 with open(file,'r') as f:
  lines=f.readlines()
  models_losses.append([ float(l.split(' ')[0]) for l in lines ])
with open(file,'r') as f:
 lines=f.readlines()
 models_names = [ l.split(' ')[2].strip() for l in lines ]
 models_jrs = [ int(name.split('_')[1]) for name in models_names ]
 models_gls = [ int(name.split('_')[2]) for name in models_names ]

import matplotlib.pyplot as plt
import numpy as np

labels = [ ('(%s;%s)' % (jr,gl)) if gl != 4 else ('(%s;-1)' % (jr)) for jr,gl in zip(models_jrs,models_gls) ]

models_losses_mean = np.mean(models_losses, axis = 0)
models_losses_std = np.std(models_losses, axis = 0)

sorted_index = np.argsort([ np.average([ int(i) for i in l[1:-1].split(';')],weights=[2,1]) for l in labels ])
models_losses_mean, models_losses_std, labels = [models_losses_mean[i] for i in sorted_index], [models_losses_std[i] for i in sorted_index], [labels[i] for i in sorted_index]

fig,ax=plt.subplots()

colors = [ 'crimson' if mlm == np.min(models_losses_mean) else 'green' if l == '(3;0)' else 'royalblue' for mlm,l in zip(models_losses_mean,labels) ]

plt.bar(range(len(models_losses_mean)),
	models_losses_mean,
	tick_label=labels,
	yerr = models_losses_std,
	color = colors
	)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)
plt.xlabel('Configuration',fontsize=20)
plt.ylabel('Loss',fontsize=20)

plt.savefig('losses_bar.pdf')

with open('losses_bar.dat','w') as f:
 for model_name,model_losses_mean,model_losses_std in zip(models_names,models_losses_mean,models_losses_std):
  f.write('%s %s %s\n' % (model_name,model_losses_mean,model_losses_std))
