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

def create_network(prototypes):
    l_in  = InputLayer((None,2))
    l_rbf = RBFLayer(l_in, prototypes)
    l_out = DenseLayer(l_rbf, layers.get_output_shape(l_rbf), nonlinearity=sigmoid)

    return l_out

def train_network(network):
    
    input_var = T.irow('inpt')
    target_var = T.vector('target')
    pred = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(pred, target_var)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.1, momentum=0.9
    )
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    embed()

def main():
    with open('output.dat', 'rb') as f_in:
        lst = pickle.load(f_in)
        random.shuffle(lst)

    y_train, X_train = zip(*lst[:int(len(lst) * 0.75)])
    y_test, X_test   = zip(*lst[int(len(lst) * 0.75 + 1):])

def test(prototypes):
    l_in = InputLayer(shape = (None, 2))
    print layers.get_output_shape(l_in)
      
    #RBF layer
    l_rbf = RBFLayer(l_in, prototypes)
    print layers.get_output_shape(l_rbf)
    
    print layers.get_output(l_rbf, inputs = np.array([[0,0]]))
    
    


if __name__ == '__main__':
    arm = Arm(visualize=False)
    prototypes = arm.create_prototypes()
    test(prototypes)