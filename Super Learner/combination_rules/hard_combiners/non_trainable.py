#    Implements state-of-the-art deep learning architectures.
#
#    Copyright (C) 2018  Antonio Montieri
#    email: antonio.montieri@unina.it
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as
#    published by the Free Software Foundation, either version 3 of the
#    License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
Created on 6 dec 2018

@author: antonio
'''

import random
	

def mode(array):
	'''
	Returns the (multi-)mode of an array
	'''
	
	most = max(list(map(array.count, array)))
	return list(set(filter(lambda x: array.count(x) == most, array)))		
		

def MV(predictions_list):
	'''
	Implements majority voting combination rule.
	:param predictions_list: i-th list contains the predictions of i-th classifier
	'''
	
	sample_predictions_list = list(zip(*predictions_list))
	
	MV_predictions = []
	for sample_predictions in sample_predictions_list:
		modes = mode(sample_predictions)
		MV_predictions.append(random.choice(modes))
	
	return MV_predictions
