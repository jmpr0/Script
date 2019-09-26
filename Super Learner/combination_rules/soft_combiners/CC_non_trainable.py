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

import numpy
	


def CC_Mean(soft_values_list):
	'''
	Implements Class Conscious (CC) Mean soft combination rule.
	:param soft_values_list: i-th 2D-list contains the soft values of i-th classifier
	'''
	
	sample_soft_values_list = list(zip(*soft_values_list))
	
	CC_Mean_soft_values = []
	for sample_soft_values in sample_soft_values_list:
		sample_soft_values_mean = numpy.array(sample_soft_values).mean(0).tolist()
		CC_Mean_soft_values.append(sample_soft_values_mean)
	
	return CC_Mean_soft_values
