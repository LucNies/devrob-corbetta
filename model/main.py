#from lasagne.layers import InputLayer, DenseLayer
#from lasagne.nonlinearities import sigmoid
#import theano.tensor as T
#import lasagne
#import theano
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import math

from IPython import embed

def init_network(input_var):
    l_in  = InputLayer((None,9), input_var=input_var)
    l_h   = DenseLayer(l_in, num_units=7, nonlinearity=sigmoid)
    l_out = DenseLayer(l_h, num_units=6, nonlinearity=sigmoid)

    return l_out

class Arm(object):
    def __init__(self, origin=0):
        self.L = 5
        self.origin = origin
        self.shoulder_joint = (self.origin, 0)
        self.elbow_joint = (self.origin, self.L)

    def rotate_shoulder(self, angle):
        self.elbow_joint = (math.cos(angle) * self.elbow_joint[0] - math.sin(angle) * self.elbow_joint[1],
                            math.sin(angle) * self.elbow_joint[0] + math.cos(angle) * self.elbow_joint[1])

    def draw(self):
        fig, ax = plt.subplots()
        ax = plt.axes(xlim=(0, self.L*10), ylim=(0, self.L*10))
        upper_arm = [(self.origin, 0), (self.origin, self.L)]  # [(x, y), (x, y)]
        lower_arm = [(upper_arm[0][0], self.L), (self.origin, self.L * 2)]

        (upper_arm_xs, upper_arm_ys) = zip(*upper_arm)
        (lower_arm_xs, lower_arm_ys) = zip(*lower_arm)

        ax.add_line(lines.Line2D(upper_arm_xs, upper_arm_ys, linewidth=2, color='black'))
        ax.add_line(lines.Line2D(lower_arm_xs, lower_arm_ys, linewidth=2, color='black'))
        plt.plot()
        plt.show()

def main():
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

if __name__ == '__main__':
    #main()
    arm = Arm(origin=4)
    arm.draw()
    arm.rotate_shoulder(.10)
    arm.draw()