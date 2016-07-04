# -*- coding: utf-8 -*-
"""
Misc functions
"""

import pickle
import numpy as np
from IPython import embed
import arm
import eye
import math


def iterate_data(data_file = 'train_data.p', batch_size = 1000):
    """
    Iterates over dataset and returns batches
    :param data_file:
    :param batch_size:
    :return: batch of siye batch_size
    """
    try:
        with open(data_file, 'rb') as f_in:
            data_points = np.array(pickle.load(f_in), dtype = "float32")
        
    except:
        print "Could not load file: {}".format(data_file)
        return

    n_data_points = len(data_points)        
    n_batches = n_data_points/batch_size
    
    current_batch = 0
    while current_batch < n_batches:
        batch = data_points[current_batch*batch_size:current_batch*batch_size+batch_size]
        batch = np.split(batch, 2, axis = 1)
        yield batch[1].reshape((batch_size, 2)), batch[0].reshape((batch_size, 2)) #input, output
        current_batch+=1


def evaluate(x, y):
    """
    Computes the euclidean distance
    :param x:
    :param y:
    :return: (rooted, mean, std)
    """
    diff = x-y
    squared = np.square(diff)
    summed = squared.sum(axis = 1)
    rooted = np.sqrt(summed)
    return rooted, rooted.mean(), rooted.std()
    
        
def combine_prototypes(proto_a, proto_b):
    """
    Concatenates prototypes

    :param proto_a:
    :param proto_b:
    :return: concatenated arraz of a and b
    """

    #[100][2] --> [100*100][4]
    result = np.zeros(shape = (proto_a.shape[0]*proto_b.shape[0], proto_a.shape[1]+proto_b.shape[1]))
    rep_size = proto_a.shape[0]
    
    for i, proto in enumerate(proto_b):
        proto = np.tile(proto, rep_size).reshape((rep_size, proto_b.shape[1]))
        result[i*rep_size:i*rep_size+rep_size] = np.hstack((proto_a, proto))

    
    return result.astype('float32')


def sign(x):
    """
    Evaluates the sign of a value x
    :param x:
    :return: 1 if pos, -1 if neg
    """
    if x >= 0:
        return 1 # its positive!
    else:
        return -1 # its negative!


def calc_intersect(left, right, inter_eye_distance=6):
    """
    Calculates the intersection point of two lines
    :param left:
    :param right:
    :return: intersection point
    """
    
    left = -sign(left) * (90 - abs(left))
    right = -sign(right) * (90 - abs(right))

    # steepnesses
    m1 = math.tan(math.radians(left))
    m2 = math.tan(math.radians(right))

    # y-intersections
    n1 = (inter_eye_distance/2) * m1
    n2 = (-inter_eye_distance/2) * m2

    #line intersects
    x = (n2-n1)/(m1-m2)
    y = (m1*x+n1)
    
    return x, y