from lasagne.layers import InputLayer, DenseLayer
from lasagne import layers
from lasagne.nonlinearities import sigmoid
from RBFLayer import RBFLayer
import theano.tensor as T
import lasagne
import theano
import numpy as np
import pickle
import random
from IPython import embed
from arm import Arm
from tqdm import tqdm
from util import iterate_data, evaluate

batch_size = 10000

def create_network(prototypes, n_output = 2):
    #print "prototpyes"+ str(prototypes.shape)
    l_in  = InputLayer((None, 2))
    
    
    l_rbf = RBFLayer(l_in, prototypes)
    
    l_out = DenseLayer(l_rbf, num_units = n_output) #might need leaky rectify, espacially during test time 

    return l_out

def train_network(network):
    
    input_var = T.fmatrix()
    target_var = T.fmatrix()
    pred = lasagne.layers.get_output(network, inputs = input_var)
    loss = lasagne.objectives.squared_error(pred, target_var)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.1, momentum=0.9)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)#LEAKY!

    val_prediction = layers.get_output(network, inputs = input_var, deterministic = True)
    val_loss = lasagne.objectives.squared_error(val_prediction, target_var)
    val_loss = val_loss.mean()
    val_fn = theano.function([input_var, target_var], [val_prediction, val_loss])

    print "Train network"
    
    for input_batch, output_batch in tqdm(iterate_data()):
        loss = train_fn(input_batch, output_batch)
        print "loss: {}".format(loss)
        
        print "Testing validation set... "
        total_mean = 0
        total_std = 0
        n = 0
        for inp_val, out_val in tqdm(iterate_data(data_file = 'validation_data.p')):
            predictions, loss = val_fn(inp_val, out_val)
            dist, mean, std = evaluate(predictions, out_val)
            n+=1 
            delta_mean = mean - total_mean
            total_mean = delta_mean/n
            delta_std = std - total_std
            total_std = delta_std/n
        print "Mean distance between predictions {} with an std of {}".format(total_mean, total_std)
    
        


def load_data():
    print "Load data" 
    with open('output.dat', 'rb') as f_out:
        data = np.array(pickle.load(f_out), dtype = "float32")
    return np.split(data, 2, axis = 1)


def main():
    with open('output.dat', 'rb') as f_in:
        lst = pickle.load(f_in)
        random.shuffle(lst)

    y_train, X_train = zip(*lst[:int(len(lst) * 0.75)])
    y_test, X_test   = zip(*lst[int(len(lst) * 0.75 + 1):])


    


if __name__ == '__main__':
    arm = Arm(visualize = False)
    network = create_network(arm.create_prototypes(redraw = False))
    train_network(network)
    """
    arm = Arm(visualize=False)
    prototypes = arm.create_prototypes()
    test(prototypes)
    """