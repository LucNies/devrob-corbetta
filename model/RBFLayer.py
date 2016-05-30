# -*- coding: utf-8 -*-
"""
Created on Sun May 29 14:17:28 2016

@author: luc
"""

import theano.tensor as T
import lasagne

class RBFLayer(lasagne.layers.Layer):
    
    
    def __init__(self, incoming, prototypes, beta = 0.5, **kwargs):
        super(RBFLayer, self).__init__(incoming, **kwargs)
        self.num_units = prototypes.shape[0]
        self.prototypes = self.add_param(prototypes, prototypes.shape ,name='prototypes')
        self.beta = beta #now arbitarliy set to 0.5, look into other values
     
        
    
    def get_output_for(self, input, **kwargs):
        return T.exp(-self.beta*T.sqr(input-self.prototypes))
    
    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)
       
        