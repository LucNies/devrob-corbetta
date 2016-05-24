from lasagne.layers import InputLayer, DenseLayer
from lasagne.nonlinearities import sigmoid
import theano.tensor as T
import lasagne
import theano

import pickle
import random
from IPython import embed

def init_network(input_var):
    l_in  = InputLayer((None,9), input_var=input_var)
    l_h   = DenseLayer(l_in, num_units=7, nonlinearity=sigmoid)
    l_out = DenseLayer(l_h, num_units=6, nonlinearity=sigmoid)

    return l_out

def train_network():
    print 'Getting network..'
    input_var = T.irow('inpt')
    target_var = T.vector('target')
    network = init_network(input_var)
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

if __name__ == '__main__':
    main()