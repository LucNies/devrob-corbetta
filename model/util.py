# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:53:21 2016

@author: luc
"""

import pickle
import numpy as np
from IPython import embed

def iterate_data(data_file = 'train_data.p', batch_size = 1000):
    
    try:
        with open(data_file, 'rb') as f_in:
            data_points = np.array(pickle.load(f_in), dtype = "float32")
        
    except IOError:
        print "Could not load file: {}".format(data_file)
        return

    n_data_points = len(data_points)        
    n_batches = n_data_points/batch_size
    
    print "Got {} datapoints, good for {} batches".format(n_data_points, n_batches)
     
    
    current_batch = 0
    while current_batch < n_batches:
        batch = data_points[current_batch*batch_size:current_batch*batch_size+batch_size]
        batch = np.split(batch, 2, axis = 1)
        yield batch[1].reshape((batch_size, 2)), batch[0].reshape((batch_size, 2)) #input, output
        current_batch+=1
         
        

def evaluate(x, y):
    
    diff = x-y
    summed = diff.sum(axis = 1)
    squared = np.square(summed)
    return squared, squared.mean(), squared.std()
    
    
        


if __name__ == "__main__":
    
    for inputs, targets in iterate_data():
        dist, mean, std = evaluate(inputs, inputs+0.5)
        print mean, std