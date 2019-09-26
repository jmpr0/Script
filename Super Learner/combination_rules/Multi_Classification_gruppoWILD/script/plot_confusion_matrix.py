import itertools
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys



cnf_matrix= np.loadtxt(sys.argv[1], delimiter=",",skiprows=1, usecols = range(0, 15))

class_names = np.loadtxt(sys.argv[1], delimiter=",",skiprows=1, usecols = (16), dtype='U21')

name=sys.argv[1]
name=name.split('/')[3].split('_')[0]


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    
    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' #if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black") #colore del testo in ogni cella

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



# Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Confusion matrix ' + name )

plt.show()


