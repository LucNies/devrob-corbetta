"""
Simulates a 2D arm.
"""

import matplotlib.pyplot as plt
import math
import numpy as np
import random
import pickle
from fractions import gcd
import scipy.optimize
from IPython import embed


class Arm(object):
    def __init__(self, origin=0, visualize=True):
        self.L = 5
        self.origin = origin
        self.max_elbow_angle = 160
        self.max_shoulder_angle = 100
        self.shoulder_joint = (self.origin, 0)
        self.elbow_joint = (self.origin, self.L)
        self.wrist_joint = (self.origin, self.L*2)

        self.shoulder_angle = 90
        self.elbow_angle = 0

        self.reached_points = []

        self.visualize = visualize
        if self.visualize:
            self._init_graphics()

    def _init_graphics(self):
        # Init figure
        plt.ion()
        self.canvas, self.canvas_ax = plt.subplots()
        self.canvas_ax = plt.axes(xlim=(0, self.L*5), ylim=(0, self.L*5))#to make it better visiable as it rotates in 360 degrees
        self.canvas_ax.set_autoscale_on(False)

        # Init drawable elements
        ## ... arm
        self.upper_arm_line = plt.Line2D([], [], linewidth=2, color='k')
        self.lower_arm_line = plt.Line2D([], [], linewidth=2, color='k')
        self.canvas_ax.add_line(self.upper_arm_line)
        self.canvas_ax.add_line(self.lower_arm_line)

        ## ... joints
        self.shoulder_joint_circle = plt.Circle((0,0), radius=0.2, color='k', fill=True)
        self.elbow_joint_circle    = plt.Circle((0,0), radius=0.2, color='k', fill=True)
        self.canvas_ax.add_patch(self.shoulder_joint_circle)
        self.canvas_ax.add_patch(self.elbow_joint_circle)

        ## ... reachable space
        self.reachable_space_circle = plt.Circle((self.origin, 0), radius=self.L*2, color='b', fill=False)
        self.canvas_ax.add_patch(self.reachable_space_circle)
        self.reachable_scatter = plt.scatter([], [])

    def random_arm_pos(self):
        """
        Picks a random shoulder and elbow angle and remembers destination point
        """
        shoulder_angle  = random.randint(0, self.max_shoulder_angle) * (math.pi / 180)
        changed = False
        if self.shoulder_joint[1] + self.L * math.sin(shoulder_angle) > 0:
            self.elbow_joint = (self.shoulder_joint[0] + self.L * math.cos(shoulder_angle),
                                self.shoulder_joint[1] + self.L * math.sin(shoulder_angle))

            while changed == False:
                elbow_angle = shoulder_angle + random.randint(0, self.max_elbow_angle) * (math.pi / 180) #160 degrees is in my opinion the most the arm can bend and can only bend one way
                if self.elbow_joint[1] + self.L * math.sin(elbow_angle) > 0:
                    self.wrist_joint = (self.elbow_joint[0] + self.L * math.cos(elbow_angle),
                                        self.elbow_joint[1] + self.L * math.sin(elbow_angle))
                    datapoint = (self.wrist_joint, (shoulder_angle, elbow_angle))
                    self.reached_points.append(datapoint)
                    changed = True

    def calculate_angles(self, x, y):
        """
        Calculates inverse kinematics for given position
        :param x:
        :param y:
        :return: joint angles
        """

        def distance_to_default(q, *args):
            q0 = np.array([math.radians(self.shoulder_angle),math.radians(self.elbow_angle)])
            weight = [1,1]
            return np.sqrt(np.sum([(qi - q0i)**2 * wi for qi, q0i, wi in zip(q, q0, weight)]))

        def x_constraint(q, xy):
            new_x = (self.L*np.cos(math.radians(q[0])) + self.L*np.cos(math.radians(q[0]) + math.radians(q[1]))) - xy[0]
            return new_x

        def y_constraint(q, xy):
            new_y = (self.L*np.sin(math.radians(q[0])) + self.L*np.sin(math.radians(q[0]) + math.radians(q[1]))) - xy[1]
            return new_y

        angles = scipy.optimize.fmin_slsqp(func=distance_to_default,
                                          x0=[math.radians(self.shoulder_angle), math.radians(self.elbow_angle)], eqcons=[x_constraint, y_constraint],
                                          args=((x,y),))
        return [math.degrees(angle) for angle in angles]

    def move_arm(self, shoulder_angle, elbow_angle, redraw=True):
        """
        Moves arm according to shoulder and elbow angles. Assumes the given anlges are allowed.

        returns (x,y)
        """
        shoulder_angle *= (math.pi / 180)
        elbow_angle = shoulder_angle + elbow_angle*(math.pi / 180)
        self.elbow_joint = (self.shoulder_joint[0] + self.L * math.cos(shoulder_angle),
                            self.shoulder_joint[1] + self.L * math.sin(shoulder_angle))
        self.wrist_joint = (self.elbow_joint[0] + self.L * math.cos(elbow_angle),
                            self.elbow_joint[1] + self.L * math.sin(elbow_angle))

        datapoint = (self.wrist_joint, (shoulder_angle, elbow_angle))
        lower_arm = [self.elbow_joint, self.wrist_joint]
        x, y = lower_arm[1]
        self.reached_points.append(datapoint)
        if redraw:
            self.redraw()

        return x, y
    
    
    def create_dataset(self, n_datapoints=100000, train_file='train_data.p', val_file='validation_data.p', test_file='test_data.p', validation_size=0.1, test_size=0.1):
        """
        Creates a dataset with n_datapoints and saves it. Not always exactly n_datapoints due to rounding errors

        returns [n_datapoints][[shoulder_angle, elbow_angle], [x,y]]
        """

        print "Create datapoints"
        step_size = math.sqrt((self.max_elbow_angle*self.max_shoulder_angle)/(n_datapoints*1.0))
        
        shoulder_angles = np.arange(0, self.max_shoulder_angle, step_size)
        elbow_angles = np.arange(0, self.max_elbow_angle, step_size)
        data_points = np.zeros(shape = (len(shoulder_angles)*len(elbow_angles), 2, 2))
        
        i = 0

        for shoulder_angle in shoulder_angles:
            for elbow_angle in elbow_angles:
                x, y = self.move_arm(shoulder_angle, elbow_angle, redraw = False)
                data_points[i] = [[shoulder_angle, elbow_angle],[x,y]]
                i+=1

        np.random.shuffle(data_points)
        print "Datapoints created, saving to file..."
        train_size = len(data_points)*(1-validation_size-test_size)
        validation_size = len(data_points)*validation_size
        
        train_data = data_points[:train_size]
        val_data = data_points[train_size:train_size+validation_size]
        test_data = data_points[train_size+validation_size: len(data_points)-1]
        
        with open(train_file, 'wb') as f_out:
            pickle.dump(train_data, f_out)
            
        with open(val_file, 'wb') as f_out:
            pickle.dump(val_data, f_out)

        with open(test_file, 'wb') as f_out:
            pickle.dump(test_data, f_out)
                            
        print "Done saving" 
        return data_points
        
    def create_prototypes(self, shape=(10,10), redraw=False):
        """
        Creates a uniformly distributed set of points in the reachable space as prototypes for the RBFs
        :param shape:
        :param redraw:
        :return: coordinates for prototypes
        """
        prototypes = np.zeros(shape = (shape[0]*shape[1], 2)) 
        
        gcd_shoulder = gcd(shape[0],self.max_shoulder_angle)
        gcd_elbow = gcd(shape[1], self.max_elbow_angle)

        i = 0
        for shoulder_angle in np.arange(0, self.max_shoulder_angle, self.max_shoulder_angle/gcd_shoulder+1):
            for elbow_angle in np.arange(0, self.max_elbow_angle , self.max_elbow_angle/gcd_elbow+1):
                x, y = self.move_arm(shoulder_angle, elbow_angle, redraw = redraw)
                prototypes[i] = [x, y]
                i += 1
        
        return prototypes

    def redraw(self):
        """
        Redraws the arm visualization
        """
        if self.visualize:
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

            # Update scatter plot of reached points
            points = zip(*self.reached_points)[0]
            self.reachable_scatter.set_offsets(np.array(points))

            # Redraw
            plt.plot()
            plt.pause(0.00001)

def save_data(arm):
    """
    Save arm positions to file
    :param arm:
    """
    with open('output.dat', 'wb') as f_out:
        pickle.dump(arm.reached_points, f_out)
    
    print "Saved" 

def main():
    arm = Arm(origin=12, visualize=True)
    #arm.create_dataset(n_datapoints=50000)
    embed()
    #arm.create_prototypes(redraw= True)
    wait = raw_input("Press enter when done...")
    """ 
    for i in tqdm(range(1000000)):
        
        arm.random_arm_pos()
        arm.redraw()

    save_data(arm)
    """

if __name__ == '__main__':
    main()