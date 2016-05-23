#from lasagne.layers import InputLayer, DenseLayer
#from lasagne.nonlinearities import sigmoid
#import theano.tensor as T
#import lasagne
#import theano
import matplotlib.pyplot as plt
import math
import numpy as np
import time
import random

from IPython import embed

class Arm(object):
    def __init__(self, origin=0):
        self.L = 5
        self.origin = origin
        self.shoulder_joint = (self.origin, 0)
        self.elbow_joint = (self.origin, self.L)
        self.wrist_joint = (self.origin, self.L*2)

        self._init_graphics()

    def _init_graphics(self):
        # Init figure
        plt.ion()
        self.canvas, self.canvas_ax = plt.subplots()
        self.canvas_ax = plt.axes(xlim=(-25, self.L*5), ylim=(-25, self.L*5))#to make it better visiable as it rotates in 360 degrees
        self.canvas_ax.set_autoscale_on(False)

        # Init drawable elements
        self.upper_arm_line = plt.Line2D([], [], linewidth=2, color='k')
        self.lower_arm_line = plt.Line2D([], [], linewidth=2, color='k')
        self.canvas_ax.add_line(self.upper_arm_line)
        self.canvas_ax.add_line(self.lower_arm_line)

        self.shoulder_joint_circle = plt.Circle((0,0), radius=0.2, color='k', fill=True)
        self.elbow_joint_circle = plt.Circle((0,0), radius=0.2, color='k', fill=True)
        self.canvas_ax.add_patch(self.shoulder_joint_circle)
        self.canvas_ax.add_patch(self.elbow_joint_circle)

        self.reachable_space_circle = plt.Circle((self.origin, 0), radius=self.L*2, color='b', fill=False)
        self.canvas_ax.add_patch(self.reachable_space_circle)

    def rotate_shoulder(self, theta):

        #rotates elbow joint around shoulder joint
       self.elbow_joint = (((self.elbow_joint[0]-self.shoulder_joint[0])*math.cos(theta))  - ((self.elbow_joint[1]-self.shoulder_joint[1])*math.sin(theta))+self.shoulder_joint[0],
                           ((self.elbow_joint[0]-self.shoulder_joint[0])*math.sin(theta) )+ ((self.elbow_joint[1]-self.shoulder_joint[1])*math.cos(theta)) +self.shoulder_joint[1] )
        #rotates wrist joint around shoulder joint (if elbow moves, wrist automatically moves)
       self.wrist_joint = (((self.wrist_joint[0]-self.shoulder_joint[0])*math.cos(theta))  - ((self.wrist_joint[1]-self.shoulder_joint[1])*math.sin(theta))+self.shoulder_joint[0],
                           ((self.wrist_joint[0]-self.shoulder_joint[0])*math.sin(theta) )+ ((self.wrist_joint[1]-self.shoulder_joint[1])*math.cos(theta)) +self.shoulder_joint[1] )

    def rotate_elbow(self, theta):
          #rotates wrist joint around elbow joint
        self.wrist_joint = (((self.wrist_joint[0]-self.elbow_joint[0])*math.cos(theta))  - ((self.wrist_joint[1]-self.elbow_joint[1])*math.sin(theta))+self.elbow_joint[0],
                           ((self.wrist_joint[0]-self.elbow_joint[0])*math.sin(theta) )+ ((self.wrist_joint[1]-self.elbow_joint[1])*math.cos(theta)) +self.elbow_joint[1] )



    def redraw(self):
        # Adjust arm segments
        upper_arm = [self.shoulder_joint, self.elbow_joint]
        lower_arm = [self.elbow_joint, self.wrist_joint]

        (upper_arm_xs, upper_arm_ys) = zip(*upper_arm)
        (lower_arm_xs, lower_arm_ys) = zip(*lower_arm)

        self.upper_arm_line.set_data(upper_arm_xs, upper_arm_ys)
        self.lower_arm_line.set_data(lower_arm_xs, lower_arm_ys)
        
        # Adjust arm joints
        self.shoulder_joint_circle.center = self.shoulder_joint
        self.elbow_joint_circle.center = self.elbow_joint

        # Redraw
        plt.plot()
        plt.pause(0.05)

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
    #train_network()
    arm = Arm(origin=12)
    while True:
        arm.rotate_shoulder(theta=random.randint(0, 360)) # Looks retarded as f*ck
        arm.rotate_elbow(theta=random.randint(0, 360)) # Looks retarded as f*ck
        arm.redraw()

if __name__ == '__main__':
    main()