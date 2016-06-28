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

def create_network(prototypes, n_outputs = 2, n_inputs = 2):
    #print "prototpyes"+ str(prototypes.shape)
    l_in  = InputLayer((None, n_inputs))
    
    
    l_rbf = RBFLayer(l_in, prototypes)

    l_hidden = DenseLayer(l_rbf, num_units=prototypes.shape[0], nonlinearity=lasagne.nonlinearities.LeakyRectify())
    
    l_out = DenseLayer(l_hidden, num_units = n_outputs, nonlinearity=lasagne.nonlinearities.LeakyRectify()) #might need leaky rectify, espacially during test time

    input_var = T.fmatrix()
    target_var = T.fmatrix()
    pred = lasagne.layers.get_output(l_out, inputs = input_var)
    loss = lasagne.objectives.squared_error(pred, target_var)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(l_out, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
    train_fn = theano.function([input_var, target_var], [pred, loss], updates=updates)

    val_prediction = layers.get_output(l_out, inputs = input_var, deterministic = True)
    val_loss = lasagne.objectives.squared_error(val_prediction, target_var)
    val_loss = val_loss.mean()
    val_fn = theano.function([input_var, target_var], [val_prediction, val_loss])


    return l_out, train_fn, val_fn
    
def train_network_double(prototypes1, prototypes2, origin, train_data = 'train_data.p', val_data = 'validation_data.p'):
    
    network1, train_fn1, val_fn1 = create_network(prototypes1)
    network2, train_fn2, val_fn2 = create_network(prototypes2, n_inputs = 4)    
    
    eyes = Eyes(origin = origin)

    epochs = 150
    means_arm = np.zeros(epochs)
    stds_arm = np.zeros(epochs)
    train_losses_arm = np.zeros(epochs)
    val_losses_arm = np.zeros(epochs)
    means_eye = np.zeros(epochs)
    stds_eye = np.zeros(epochs)
    train_losses_eye = np.zeros(epochs)
    val_losses_eye = np.zeros(epochs)
    print "Train network"

    for e in tqdm(range(epochs)):
        for input_batch, output_batch in iterate_data(data_file = train_data):
            pred1, train_loss1 = train_fn1(input_batch, output_batch)
            eye_angles = [eyes.attend_to(x,y) for [x,y] in output_batch]
            eye_input = np.hstack((eye_angles, pred1)).astype('float32')
            pred2, train_loss2 = train_fn2(eye_input, output_batch)
            
        #print " train loss: \t\t{}".format(loss)

        #print "Testing validation set... "
        total_mean_arm = 0
        total_std_arm = 0
        total_mean_eye = 0
        total_std_eye = 0
        n = 0
        for inp_val, out_val in iterate_data(data_file = val_data):
            predictions_arm, loss_arm = val_fn1(inp_val, out_val)
            dist_arm, mean_arm, std_arm = evaluate(predictions_arm, out_val)#dist is for debugging
            
            eye_angles = [eyes.attend_to(x,y) for [x,y] in out_val]
            eye_input = np.hstack((eye_angles, out_val))
            prediction_eye, loss_eye = val_fn2(eye_angles)
            dist_eye, mean_eye, std_eye = evaluate(prediction_eye, out_val)
            
            n += 1
            total_mean_arm += mean_arm
            total_std_arm += std_arm
            total_mean_eye += mean_eye
            total_std_eye += std_eye
        #print "Validation loss: \t{}".format(loss)
        #print "Mean distance between predictions {} with an std of {}".format(total_mean/n, total_std/n)
        means_arm[e] = total_mean_arm/n
        stds_arm[e] = total_std_arm/n
        train_losses_arm[e] = train_loss1
        val_losses_arm[e] = loss_arm
        means_eye[e] = total_mean_eye/n
        stds_eye[e] = total_std_eye/n
        train_losses_eye[e] = train_loss2
        val_losses_eye[e] = loss_eye

    plt.figure()
    meanplot_arm, = plt.plot(means_arm, label = 'mean arm')
    stdplot_arm, = plt.plot(stds_arm, label = 'std arm')
    meanplot_eye = plt.plot(means_arm, label = 'mean eye')    
    stdplot_eye = plt.plot(stds_eye,  label = 'std eye')
        
    plt.legend(handles = [meanplot_arm, stdplot_arm, meanplot_eye, stdplot_eye])
    plt.show()

    plt.figure()
    trainplot_arm, = plt.plot(train_losses_arm, label = 'train loss arm')
    valplot_arm, = plt.plot(val_losses_arm, label = 'val loss arm')
    trainplot_eye = plt.plot(train_losses_eye, label = 'train loss eye')
    valplot_eye = plt.plot(val_losses_eye, label = 'val loss eye')
    
    plt.legend(handles = [trainplot_arm, valplot_arm, trainplot_eye, valplot_eye])
    plt.show()
    
    np.save('network_arm', layers.get_all_param_values(network1))    
    np.save('network_eye', layers.get_all_param_values(network2))   
    
    #test(predictions, out_val, inp_val)
    
    return #network, predictions

def train_network(network, train_data = 'train_data.p', val_data = 'validation_data.p'):
    
    input_var = T.fmatrix()
    target_var = T.fmatrix()
    pred = lasagne.layers.get_output(network, inputs = input_var)
    train_loss = lasagne.objectives.squared_error(pred, target_var)
    train_loss = lasagne.objectives.aggregate(train_loss, mode='mean')
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(train_loss, params, learning_rate=0.01, momentum=0.9)
    train_fn = theano.function([input_var, target_var], train_loss, updates=updates)

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
    np.save('network_arm', layers.get_all_param_values(network))
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
    origin      = 0
    arm = Arm(visualize = False, origin = origin)
    proto_arm = arm.create_prototypes(shape=(5,5))
    arm = arm.create_dataset(n_datapoints = 10000)
    eyes = Eyes(visualize = False, origin = origin)
    proto_eye = eyes.create_prototypes(shape = (5,5))
    #eyes.create_dataset(n_datapoints=1000)
    #test(proto)
    #network = create_network(proto)
    #network = create_network(arm.create_prototypes(shape=(20,20),redraw = False))
    #train_network(network)
    train_network_double(proto_arm, proto_eye, origin = origin,  train_data = 'train_data.p', val_data = 'validation_data.p' )
    """
    arm = Arm(visualize=False)
    prototypes = arm.create_prototypes()
    test(prototypes)
    """
