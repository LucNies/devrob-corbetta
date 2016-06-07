import matplotlib.pyplot as plt
import math
import numpy as np
import random
import pickle
from fractions import gcd

class Arm(object):
    def __init__(self, origin=0, visualize=True):
        self.L = 5
        self.origin = origin
        self.max_elbow_angle = 160
        self.max_shoulder_angle = 100
        self.shoulder_joint = (self.origin, 0)
        self.elbow_joint = (self.origin, self.L)
        self.wrist_joint = (self.origin, self.L*2)

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
        

    """
    Assumes the given anlges are allowed
    """    
    def move_arm(self, shoulder_angle, elbow_angle, redraw = True):
        shoulder_angle *= (math.pi / 180)
        elbow_angle = shoulder_angle + elbow_angle*(math.pi / 180)
        self.elbow_joint = (self.shoulder_joint[0] + self.L * math.cos(shoulder_angle),
                            self.shoulder_joint[1] + self.L * math.sin(shoulder_angle))
        self.wrist_joint = (self.elbow_joint[0] + self.L * math.cos(elbow_angle),
                            self.elbow_joint[1] + self.L * math.sin(elbow_angle))

#        self.elbow_joint = (self.L * math.cos(shoulder_angle), self.L * math.sin(shoulder_angle))
 #       self.wrist_joint = (self.L * math.cos(elbow_angle), self.L * math.sin(elbow_angle))
        datapoint = (self.wrist_joint, (shoulder_angle, elbow_angle))
        lower_arm = [self.elbow_joint, self.wrist_joint]
        x, y = lower_arm[1]
        self.reached_points.append(datapoint)
        if redraw:
            self.redraw()
            
        return x, y
    
    
    """
    Not always exactly n_datapoints due to rounding errors
    
    returns [n_datapoints][[shoulder_angle, elbow_angle], [x,y]]
    """
    def create_dataset(self, n_datapoints = 100000, train_file = 'train_data.p', val_file = 'validation_data.p', test_file = 'test_data.p', validation_size = 0.1, test_size = 0.1):
                
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
        test_size = len(data_points)*validation_size
        
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
        
    def create_prototypes(self, shape = (10,10), redraw = False):
        commands = np.zeros(shape = (shape[0]*shape[1], 2)) 
        
        gcd_shoulder = gcd(shape[0],self.max_shoulder_angle)
        gcd_elbow = gcd(shape[1], self.max_elbow_angle)

        i = 0
        for shoulder_angle in np.arange(0, self.max_shoulder_angle, self.max_shoulder_angle/gcd_shoulder+1):
            for elbow_angle in np.arange(0, self.max_elbow_angle , self.max_elbow_angle/gcd_elbow+1):
                commands[i] = [shoulder_angle, elbow_angle]
                self.move_arm(shoulder_angle, elbow_angle, redraw = redraw)
                i+=1
                #print shoulder_angle, elbow_angle
        
        return commands

    def redraw(self):
        if self.visualize:
            # Adjust arm segments
            upper_arm = [self.shoulder_joint, self.elbow_joint]
            lower_arm = [self.elbow_joint, self.wrist_joint]

            (upper_arm_xs, upper_arm_ys) = zip(*upper_arm)
            (lower_arm_xs, lower_arm_ys) = zip(*lower_arm)
            #embed()
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
    with open('output.dat', 'wb') as f_out:
        pickle.dump(arm.reached_points, f_out)
    
    print "Saved" 

def main():
    
    arm = Arm(origin=12, visualize=False)
    arm.create_dataset()

    #arm.create_prototypes(redraw= True)
    #wait = raw_input("Press enter when done...")
    """ 
    for i in tqdm(range(1000000)):
        
        arm.random_arm_pos()
        arm.redraw()

    save_data(arm)
    """

if __name__ == '__main__':
    main()