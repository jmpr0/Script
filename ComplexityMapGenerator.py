#!/usr/bin/python3

import os, sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import itertools

os.environ["DISPLAY"] = ":0"

def draw_square(ax, top_left_coordinates, side, ls = 'solid', lw = 1, ec = 'black', a=1):
 '''
 Draw a square given as input top-left coordinates [tuple] and side length [float] with a certain ls (linestyle) [string].
 '''
 ax.add_patch(patches.Rectangle(top_left_coordinates, side, side, fill = False, linestyle = ls, linewidth = lw, edgecolor = ec, alpha = a))

try:
 summary_num = int(sys.argv[1])
 summary_files = []
 for file in sys.argv[2:summary_num+2]:
  summary_files.append(file)
 best_num = int(sys.argv[summary_num+2])
 best_files=[]
 for file in sys.argv[summary_num+3:]:
  best_files.append(file)
except Exception as e:
 print(e)
 print('Usage: %s <summary_num> <summary_1> ... <summary_n> <best_num> <best_1> ... <best_m>'%sys.argv[0])
 exit(1)

models_params = []
models_jrs = []
models_gls = []

for file in sorted(summary_files):
 with open(file,'r') as f:
  lines=f.readlines()
  models_params.append(int([ l for l in lines if 'Trainable params:' in l ][0].split(' ')[-1].replace(',','').strip()))
 models_jrs.append(int(file.split('_')[-2].split('.')[0][2:]))
 models_gls.append(int(file.split('_')[-1].split('.')[0][2:]))

models_gls = [mgls if mgls != np.unique(models_gls).shape[0]+1 else -1 for mgls in models_gls ]

models_jrs = sorted(models_jrs)
models_gls = sorted(models_gls)

models_names = [ 'dae %s %s' % (jr,gl) for jr,gl in itertools.product(np.unique(models_jrs),np.unique(models_gls)) ]

models_counters = [0]*best_num
for file in sorted(best_files):
 best_model = ' '.join(file.split('.')[-2].split('_')[-3:])[1:]
 if best_model.endswith('%s'%(np.unique(models_gls).shape[0]+1)):
  best_model = best_model[:-len('%s'%(np.unique(models_gls).shape[0]+1))]+'-1'
 models_counters[models_names.index(best_model)] += 1

ys=np.reshape(models_params,(np.unique(models_gls).shape[0],np.unique(models_jrs).shape[0]))

xs=np.reshape(np.array([ '%s,%s' % (jr,gl) for jr,gl in zip(models_jrs,models_gls) ]),(np.unique(models_gls).shape[0],np.unique(models_jrs).shape[0]))
print(xs)
# input()
print(np.rot90(xs))
# input()
print(np.rot90(xs)[1:,:])
# input()
print(np.rot90(xs)[0,:])
# input()
print(np.r_['0,2',np.rot90(xs)[1:,:],np.rot90(xs)[0,:]])
# input()
# print(np.rot90(xs,2)[-1,:])
# input()
# print(np.rot90(xs,2)[1::-1,:])
# input()
# print(np.r_['0,2', np.rot90(xs,2)[-1,:], np.rot90(xs,2)[1::-1,:]])
# input()
# print(np.rot90(np.r_['0,2', np.rot90(xs,2)[-1,:], np.rot90(xs,2)[1::-1,:]],-1))
# input()

plt.clf()
# plt.savefig('complexity_map.eps')

fig,ax=plt.subplots()

#plt.imshow(np.c_[ys[:,0],np.roll(ys[:,1:],1,axis=1)],cmap='Reds')
plt.imshow(np.r_['0,2',np.rot90(ys)[1:,:],np.rot90(ys)[0,:]],cmap='RdYlGn_r')
ax.set_xticks(range(np.unique(models_jrs).shape[0]))
ax.set_yticks(range(np.unique(models_gls).shape[0]))
ax.set_xticklabels(sorted(np.unique(models_jrs)), fontsize=18)
ax.set_yticklabels(reversed(sorted(np.unique(models_gls))), fontsize=18)
plt.xlabel(r'$S_r$',fontsize=20)
plt.ylabel(r'$G_d$',fontsize=20)
clb=plt.colorbar(format='%.2e')
clb.ax.set_ylabel('Number of Trainable Params',fontsize=20)
clb.ax.set_yticklabels(clb.ax.get_yticklabels(),fontsize=18)

base = -.45
step = 1

xs = [ -.45 + i for i in range(len(np.unique(models_jrs))) ]
ys = [ y for y in reversed([ -.45 + i for i in range(len(np.unique(models_gls))) ]) ]

base_model_jr=3
base_model_gl=0

draw_square(ax, (xs[np.where(np.unique(models_jrs)==base_model_jr)[0][0]],ys[np.where(np.unique(models_gls)==base_model_gl)[0][0]]),
	.9, lw=3, ls='dashed', ec='green')
for model_name,model_counter in zip(models_names,models_counters):
 model_jr = int(model_name.split()[1])
 model_gl = int(model_name.split()[2])
 draw_square(ax, (xs[np.where(np.unique(models_jrs)==model_jr)[0][0]],ys[np.where(np.unique(models_gls)==model_gl)[0][0]]),
 	.9, lw=3, ec='crimson', a=model_counter/len(models_counters))
 plt.text(xs[np.where(np.unique(models_jrs)==model_jr)[0][0]]+step/2-1/20,ys[np.where(np.unique(models_gls)==model_gl)[0][0]]+step/2-1/20, model_counter,
 	ha='center', va='center', fontsize=20)

plt.savefig('complexity_map.pdf')

with open('complexity_map.dat','w') as f:
 for model_name,model_params,model_counter in zip(models_names,models_params,models_counters):
  f.write('%s %s %s\n' % (model_name,model_params,model_counter))
