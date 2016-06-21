from lasagne.layers import InputLayer, DenseLayer, batch_norm
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
from util import iterate_data, evaluate, calc_intersect
import matplotlib.pyplot as plt
from eye import Eyes


batch_size = 10000

def create_network(prototypes, n_output = 2):
    #print "prototpyes"+ str(prototypes.shape)
    l_in  = InputLayer((None, 2))
    
    
    l_rbf = RBFLayer(l_in, prototypes)
    
    l_out = DenseLayer(l_rbf, num_units = n_output, nonlinearity=lasagne.nonlinearities.LeakyRectify()) #might need leaky rectify, espacially during test time

    return l_out
    
def train_network_double(network, network1, train_data1 = 'train_data.p', val_data = 'validation_data.p'):
    
    input_var = T.fmatrix()
    target_var = T.fmatrix()
    pred = lasagne.layers.get_output(network, inputs = input_var)
    loss = lasagne.objectives.squared_error(pred, target_var)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    val_prediction = layers.get_output(network, inputs = input_var, deterministic = True)
    val_loss = lasagne.objectives.squared_error(val_prediction, target_var)
    val_loss = val_loss.mean()
    val_fn = theano.function([input_var, target_var], [val_prediction, val_loss])

    epochs = 150
    means = np.zeros(epochs)
    stds = np.zeros(epochs)
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)


    print "Train network"

    for e in tqdm(range(epochs)):
        for input_batch, output_batch in iterate_data(data_file = train_data):
            train_loss = train_fn(input_batch, output_batch)
        #print " train loss: \t\t{}".format(loss)

        #print "Testing validation set... "
        total_mean = 0
        total_std = 0
        n = 0
        for inp_val, out_val in iterate_data(data_file = val_data):
            predictions, loss = val_fn(inp_val, out_val)
            dist, mean, std = evaluate(predictions, out_val)
            n+=1
            total_mean+=mean
            total_std+= std

        #print "Validation loss: \t{}".format(loss)
        #print "Mean distance between predictions {} with an std of {}".format(total_mean/n, total_std/n)
        means[e] = total_mean/n
        stds[e] = total_std/n
        train_losses[e] = train_loss
        val_losses[e] = loss

    plt.figure()
    meanplot, = plt.plot(means, label = 'mean')
    stdplot, = plt.plot(stds, label = 'std')
    plt.legend(handles = [meanplot, stdplot])
    plt.show()

    plt.figure()
    trainplot, = plt.plot(train_losses, label = 'train loss')
    valplot, = plt.plot(val_losses, label = 'val loss')
    plt.legend(handles = [trainplot, valplot])
    plt.show()
    
    np.save('network_eyes.np', network)    
    
    test(predictions, out_val, inp_val)
    
    return network, predictions

def train_network(network, train_data = 'train_data.p', val_data = 'validation_data.p'):
    
    input_var = T.fmatrix()
    target_var = T.fmatrix()
    pred = lasagne.layers.get_output(network, inputs = input_var)
    loss = lasagne.objectives.squared_error(pred, target_var)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    val_prediction = layers.get_output(network, inputs = input_var, deterministic = True)
    val_loss = lasagne.objectives.squared_error(val_prediction, target_var)
    val_loss = val_loss.mean()
    val_fn = theano.function([input_var, target_var], [val_prediction, val_loss])

    epochs = 150
    means = np.zeros(epochs)
    stds = np.zeros(epochs)
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)


    print "Train network"

    for e in tqdm(range(epochs)):
        for input_batch, output_batch in iterate_data(data_file = train_data):
            train_loss = train_fn(input_batch, output_batch)
        #print " train loss: \t\t{}".format(loss)

        #print "Testing validation set... "
        total_mean = 0
        total_std = 0
        n = 0
        for inp_val, out_val in iterate_data(data_file = val_data):
            predictions, loss = val_fn(inp_val, out_val)
            dist, mean, std = evaluate(predictions, out_val)
            n+=1
            total_mean+=mean
            total_std+= std

        #print "Validation loss: \t{}".format(loss)
        #print "Mean distance between predictions {} with an std of {}".format(total_mean/n, total_std/n)
        means[e] = total_mean/n
        stds[e] = total_std/n
        train_losses[e] = train_loss
        val_losses[e] = loss
        np.save('network_epoch' + str(e), layers.get_all_param_values(network))   

    plt.figure()
    meanplot, = plt.plot(means, label = 'mean')
    stdplot, = plt.plot(stds, label = 'std')
    plt.legend(handles = [meanplot, stdplot])
    plt.show()

    plt.figure()
    trainplot, = plt.plot(train_losses, label = 'train loss')
    valplot, = plt.plot(val_losses, label = 'val loss')
    plt.legend(handles = [trainplot, valplot])
    plt.show()
    
    print "saving network"
    np.save('network_eyes', layers.get_all_param_values(network))    
    print "done saving"
        
    
    #test(predictions, out_val, inp_val)
    embed()    
    
    return network, predictions

def load_data():
    print "Load data" 
    with open('output.dat', 'rb') as f_out:
        data = np.array(pickle.load(f_out), dtype = "float32")
    return np.split(data, 2, axis = 1)


def load_network(prototypes, network_path = 'network_eyes.npy'):
    network = create_network(prototypes)
    
    layers.set_all_param_values(network, np.load(network_path))
    
    return network

def test(prototypes, network_path = 'network_eyes.np'):
    
    network = load_network(prototypes)    
    
    input_var = T.fmatrix()
    target_var = T.fmatrix()
    val_prediction = layers.get_output(network, inputs = input_var, deterministic = True)
    val_loss = lasagne.objectives.squared_error(val_prediction, target_var)
    val_loss = val_loss.mean()
    val_fn = theano.function([input_var, target_var], [val_prediction, val_loss])    
    
    total_mean = 0
    total_std = 0
    n = 0
    print "validation data in test"
    for inp_val, out_val in tqdm(iterate_data(data_file = 'validation_data_eyes.p')):
        predictions, loss = val_fn(inp_val, out_val)
        dist, mean, std = evaluate(predictions, out_val)
        n+=1
        total_mean+=mean
        total_std+= std
    
    
    
    
    #eyes = Eyes(origin = 0, visualize = True)
    #eyes.set_dominance(0)
    for i, [left, right] in enumerate(predictions):
        #print left, right
        x, y = calc_intersect(left, right)
        print "predicted \t x: {} y: {}".format(x, y)
        #eyes.redraw()
        #point_target = eyes.move_eyes(out_val[i][0], out_val[i][1])
        
        x1, y1 = calc_intersect(out_val[i][0], out_val[i][1])
        print "target \t\t x: {} y: {}".format(x1,y1)
        print "should be \t x: {} y: {}".format(inp_val[i][0], inp_val[i][1])
        #eyes.redraw()
        embed()
        #eyes.attended_points = np.array()


def main():
    with open('output.dat', 'rb') as f_in:
        lst = pickle.load(f_in)
        random.shuffle(lst)

    y_train, X_train = zip(*lst[:int(len(lst) * 0.75)])
    y_test, X_test   = zip(*lst[int(len(lst) * 0.75 + 1):])


    


if __name__ == '__main__':
    #arm = Arm(visualize = False)
    eyes = Eyes(visualize = False, origin = 0)
    proto = eyes.create_prototypes(shape = (20,20))
    #eyes.create_dataset(n_datapoints=1000000)
    #test(proto)
    network = create_network(proto)
    #network = create_network(arm.create_prototypes(shape=(20,20),redraw = False))
    #train_network(network)
    train_network(network, train_data = 'train_data_eyes_new.p', val_data = 'validation_data_eyes_new.p' )
    """
    arm = Arm(visualize=False)
    prototypes = arm.create_prototypes()
    test(prototypes)
    """