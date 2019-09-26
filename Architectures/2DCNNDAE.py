import keras
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Deconvolution2D
from keras.layers import Input
from keras.models import Model
from keras.regularizers import *


def get_model():
	aliases = {}
	Input_1 = Input(shape=(3, 32, 32), name='Input_1')
	Convolution2D_1 = Convolution2D(name='Convolution2D_1',activation= 'relu' ,nb_col= 3,nb_row= 3,nb_filter= 16)(Input_1)
	MaxPooling2D_1 = MaxPooling2D(name='MaxPooling2D_1',pool_size= (3, 3))(Convolution2D_1)
	Convolution2D_2 = Convolution2D(name='Convolution2D_2',activation= 'relu' ,nb_col= 3,nb_row= 3,nb_filter= 2)(MaxPooling2D_1)
	MaxPooling2D_2 = MaxPooling2D(name='MaxPooling2D_2')(Convolution2D_2)
	UpSampling2D_10 = UpSampling2D(name='UpSampling2D_10')(MaxPooling2D_2)
	Deconvolution2D_2 = Deconvolution2D(name='Deconvolution2D_2',nb_col= 3,nb_row= 3,subsample= (1,1),activation= 'relu' ,nb_filter= 16,output_shape= (None,16,10,10))(UpSampling2D_10)
	UpSampling2D_1 = UpSampling2D(name='UpSampling2D_1',size= (3,3))(Deconvolution2D_2)
	Deconvolution2D_3 = Deconvolution2D(name='Deconvolution2D_3',nb_col= 3,nb_row= 3,subsample= (1,1),activation= 'sigmoid' ,nb_filter= 3,output_shape= (None,3,32,32))(UpSampling2D_1)

	model = Model([Input_1],[Deconvolution2D_3])
	return aliases, model


from keras.optimizers import *

def get_optimizer():
	return Adadelta()

def is_custom_loss_function():
	return False

def get_loss_function():
	return 'mean_squared_error'

def get_batch_size():
	return 32

def get_num_epoch():
	return 10

def get_data_config():
	return '{"datasetLoadOption": "batch", "samples": {"split": 1, "test": 0, "training": 48000, "validation": 12000}, "kfold": 1, "mapping": {"img": {"shape": "", "port": "InputPort0,OutputPort0", "type": "Image", "options": {"height_shift_range": 0, "width_shift_range": 0, "vertical_flip": false, "Width": 28, "Normalization": true, "Height": 28, "horizontal_flip": false, "Resize": false, "rotation_range": 0, "Scaling": 1, "shear_range": 0, "pretrained": "None", "Augmentation": false}}, "label": {"shape": "", "port": "", "type": "Categorical", "options": {}}}, "numPorts": 1, "dataset": {"samples": 60000, "name": "cifar-10", "type": "public"}, "shuffle": false}'