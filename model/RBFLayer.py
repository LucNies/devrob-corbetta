# -*- coding: utf-8 -*-

import theano.tensor as T
import theano
import lasagne
from IPython import embed


class RBFLayer(lasagne.layers.Layer):
    """
    Creates an RBF layer using lasagne
    """
    
    def __init__(self, incoming, prototypes, beta = 0.5, **kwargs):
        super(RBFLayer, self).__init__(incoming, **kwargs)
        self.num_units = prototypes.shape[0]
        self.prototypes = self.add_param(prototypes, prototypes.shape ,name='prototypes')
        self.beta = beta #now arbitarliy set to 0.5, look into other values
     
        
    
    def get_output_for(self, input, **kwargs):


        result, updates = theano.scan(fn=lambda row, prototypes: prototypes - row,
                                      sequences=[input],
                                      non_sequences=self.prototypes)
        a = T.sum(T.sqr(result), axis=2) #+ shape1*1
        b = -self.beta*a
        c = T.exp(b)
        return c#T.exp(-self.beta*T.sum(T.sqr(result), axis=1))
    
    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)
       
        