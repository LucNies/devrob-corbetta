# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:53:21 2016

@author: luc
"""

import pickle
import numpy as np
from IPython import embed
import arm
import eye

def iterate_data(data_file = 'train_data.p', batch_size = 1000):
    
    try:
        with open(data_file, 'rb') as f_in:
            data_points = np.array(pickle.load(f_in), dtype = "float32")
        
    except IOError:
        print "Could not load file: {}".format(data_file)
        return

    n_data_points = len(data_points)        
    n_batches = n_data_points/batch_size
    
    #print "Got {} datapoints, good for {} batches".format(n_data_points, n_batches)
     
    
    current_batch = 0
    while current_batch < n_batches:
        batch = data_points[current_batch*batch_size:current_batch*batch_size+batch_size]
        batch = np.split(batch, 2, axis = 1)
        yield batch[1].reshape((batch_size, 2)), batch[0].reshape((batch_size, 2)) #input, output
        current_batch+=1
         
        

def evaluate(x, y):
    
    diff = x-y
    squared = np.square(diff)
    summed = squared.sum(axis = 1)
    rooted = np.sqrt(summed)
    return rooted, rooted.mean(), rooted.std()
    
        
def combine_prototypes(proto_a, proto_b):
    
    #[100][2] --> [100*100][4]
    
    result = np.zeros(shape = (proto_a.shape[0]*proto_b.shape[0], proto_a.shape[1]+proto_b.shape[1]))
    rep_size = proto_a.shape[0]
    
    for i, proto in enumerate(proto_b):
        proto = np.tile(proto, rep_size).reshape((rep_size, proto_b.shape[1]))
        result[i*rep_size:i*rep_size+rep_size] = np.hstack((proto_a, proto))

    
    return result

if __name__ == "__main__":
    
    a = arm.Arm(origin = 12, visualize = False)
    eyes = eye.Eyes(origin = 12, visualize = False)
    proto_arm = a.create_prototypes()
    proto_eyes = eyes.create_prototypes()
    combi = combine_prototypes(proto_arm, proto_eyes)
    embed()
    
    