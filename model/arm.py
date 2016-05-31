import matplotlib.pyplot as plt
import math
import numpy as np
import random
import pickle
import time
from sympy.geometry import Line as SymLine
from sympy.geometry import Point as SymPoint
from shapely.geometry import Point, LineString
from shapely import affinity
from fractions import gcd
import time
from tqdm import tqdm

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

        self.reached_points = []

        self.visualize = visualize
        if self.visualize:
            self._init_graphics()

    def _init_graphics(self):
        # Init figure
        plt.ion()
        self.canvas, self.canvas_ax = plt.subplots()
        self.canvas_ax = plt.axes(xlim=(-25, self.L*5), ylim=(-25, self.L*5))#to make it better visiable as it rotates in 360 degrees
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
        self.reached_points.append(datapoint)
        if redraw:
            self.redraw()
                
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

class Eyes(object):
    def __init__(self, origin, left_dominant = True):
        self.center_origin = origin
        self.inter_eye_distance = 6
        self.attended_points = []
        self.max_distance = 15
        self.visual_space = Point(self.center_origin, 0).buffer(self.max_distance)
        self.max_angle = 60
        self.left_dominant = left_dominant

        self.calculate_lines()
        self._init_graphics()

    def _init_graphics(self):
        plt.ion()
        self.canvas, self.canvas_ax = plt.subplots()
        self.canvas_ax = plt.axes(xlim=(-25, 25), ylim=(-25, 25))

        # Init elements
        self.left_eye_circle = plt.Circle((self.center_origin - self.inter_eye_distance / 2, 0), radius=0.5, fill=False)
        self.right_eye_circle = plt.Circle((self.center_origin + self.inter_eye_distance / 2, 0), radius=0.5, fill=False)
        self.visual_space_circle = plt.Circle((self.center_origin, 0), radius=self.max_distance, fill=False)
        self.canvas_ax.add_patch(self.left_eye_circle)
        self.canvas_ax.add_patch(self.right_eye_circle)
        self.canvas_ax.add_patch(self.visual_space_circle)

        self.canvas_ax.add_line(self.to_Line2D(self.l_center_line))
        self.canvas_ax.add_line(self.to_Line2D(self.r_center_line))
        self.canvas_ax.add_line(self.to_Line2D(self.l_outer_bound))
        self.canvas_ax.add_line(self.to_Line2D(self.r_outer_bound))
        self.canvas_ax.add_line(self.to_Line2D(self.l_inner_bound))
        self.canvas_ax.add_line(self.to_Line2D(self.r_inner_bound))
        #self.canvas_ax.add_line(self.to_Line2D(self.min_angle_line))

        # Focus lines
        self.focus_line_leye = plt.Line2D([], [], linewidth=2, color='r')
        self.focus_line_reye = plt.Line2D([], [], linewidth=2, color='r')
        self.right_eye_line = plt.Line2D([], [], linewidth=2, color='g')
        self.canvas_ax.add_line(self.focus_line_leye)
        self.canvas_ax.add_line(self.focus_line_reye)
        self.canvas_ax.add_line(self.right_eye_line)

        self.reachable_scatter = plt.scatter([], [])

    def redraw(self):
        # Draw attended points
        self.reachable_scatter.set_offsets(self.attended_points)

        # Adjust focus lines
        self.focus_line_leye.set_data([self.l_focus_line.coords[0][0], self.l_focus_line.coords[1][0]], [self.l_focus_line.coords[0][1], self.l_focus_line.coords[1][1]])
        self.focus_line_reye.set_data([self.r_focus_line.coords[0][0], self.r_focus_line.coords[1][0]], [self.r_focus_line.coords[0][1], self.r_focus_line.coords[1][1]])

        plt.plot()
        plt.pause(0.000000000001)

    def calc_angle_submissive_eye(self, angle):
        if self.left_dominant:
            left_focus_line = self.rotate_line(self.l_center_line, angle)
            intersection_point = self.get_pos_intersection(left_focus_line, self.visual_space.boundary)
            self.right_angle_line = LineString([(self.r_center_line.coords[0][0], 0), intersection_point.coords[0]])
            #embed()
            angle_between = self.get_angle_between(
                self.right_angle_line,
                self.r_center_line
            )
            angle = math.degrees(angle_between)
            if math.atan2(self.right_angle_line.coords[1][1], self.right_angle_line.coords[1][0] - self.inter_eye_distance/2) - math.pi/2 < 0:
                angle = -angle

        else:
            right_focus_line = self.rotate_line(self.r_center_line, angle)
            intersection_point = self.get_pos_intersection(right_focus_line, self.visual_space.boundary)
            self.left_angle_line = LineString([(self.l_center_line.coords[0][0], 0), intersection_point.coords[0]])
            #embed()
            angle_between = self.get_angle_between(
                self.left_angle_line,
                self.l_center_line
            )
            angle = math.degrees(angle_between)
            if math.atan2(self.left_angle_line.coords[1][1], self.left_angle_line.coords[1][0] + self.inter_eye_distance/2) - math.pi/2 < 0:
                angle = -angle

        return angle

    def calculate_lines(self):
        # Center lines for left and right eye
        self.l_center_line = LineString([Point(self.center_origin -self.inter_eye_distance / 2, 0),
                                  Point(self.center_origin - self.inter_eye_distance / 2, math.sqrt(self.max_distance**2 - (self.inter_eye_distance / 2)**2))])
        self.r_center_line = LineString([Point(self.center_origin + self.inter_eye_distance / 2, 0),
                                  Point(self.center_origin + self.inter_eye_distance / 2,
                                        math.sqrt(self.max_distance**2 - (self.inter_eye_distance / 2) ** 2))])

        # Inner and outer bounds for left and right eye
        self.l_outer_bound = self.rotate_line(self.l_center_line, + self.max_angle)
        self.r_outer_bound = self.rotate_line(self.r_center_line, - self.max_angle)
        self.l_inner_bound = self.rotate_line(self.l_center_line, - self.max_angle)
        self.r_inner_bound = self.rotate_line(self.r_center_line, + self.max_angle)

        self.min_angle_line = LineString([self.l_center_line.coords[0], self.get_pos_intersection(self.r_inner_bound, self.visual_space.boundary).coords[0]])
        self.min_angle = math.degrees(self.get_angle_between(
            self.min_angle_line,
            self.l_center_line
        ))

    def to_Line2D(self, line):
        ''' Converts a Shapely line to a matplotlib Line2D '''
        return plt.Line2D((line.coords[0][0], line.coords[1][0]), (line.coords[0][1], line.coords[1][1]))

    def rotate_line(self, line, angle):
        ''' Wrapper function for rotating the line to an angle and then clipping the rotated line to the visual space. '''
        new_line = affinity.rotate(line, angle, origin=line.coords[0])
        intersection_point = self.get_pos_intersection(new_line, self.visual_space.boundary)        
        return LineString([Point(line.coords[0][0], line.coords[0][1]), intersection_point])

    def get_pos_intersection(self, e1, e2):
        ''' Calculates the intersection between two entities. '''
        # Hack: extend the line to make sure there is an intersection point
        extended_point = Point(e1.coords[1][0] + (e1.coords[1][0] - e1.coords[0][0]) / e1.length * e1.length**2,
                               e1.coords[1][1] + (e1.coords[1][1] - e1.coords[0][1]) / e1.length * e1.length**2)
        e1 = LineString([Point(e1.coords[0][0], e1.coords[0][1]), extended_point])
        intersection_point = e1.intersection(e2)

        return intersection_point

    def get_angle_between(self, line1, line2):
        ''' Uses SymPy to calculate the angle between two lines. Assumes input formatted as Shapely lines. '''
        return SymLine.angle_between(
            SymLine(SymPoint(*line1.coords[0]), SymPoint(*line1.coords[1])),
            SymLine(SymPoint(*line2.coords[0]), SymPoint(*line2.coords[1]))
        )

    def random_eye_pos(self):
        angle_leye = random.uniform(-self.max_angle, self.min_angle)
        angle_reye = random.uniform(self.calc_angle_submissive_eye(angle_leye), self.max_angle)
        self.l_focus_line = self.rotate_line(self.l_center_line, angle_leye)
        self.r_focus_line = self.rotate_line(self.r_center_line, angle_reye)
        focus_point = self.get_pos_intersection(self.l_focus_line, self.r_focus_line)
        if focus_point:
            # Hacky ... sometimes there is no intersection point? 
            self.attended_points.append((focus_point.x, focus_point.y))

def save_data(arm):
    with open('output.dat', 'wb') as f_out:
        pickle.dump(arm.reached_points, f_out)
    
    print "Saved" 

def main():
    

    arm = Arm(origin=12, visualize=False)
    #arm.create_prototypes()
    #wait = raw_input("Press enter when done...")
    for i in tqdm(range(1000000)):
        
        arm.random_arm_pos()
        arm.redraw()

    save_data(arm)
    
    '''
=======
   # save_data(arm)

>>>>>>> e2baa33a38e9bf7016cbc2de96f5876b6a1a6e59
    eyes = Eyes(origin=0)
    while True:
        eyes.random_eye_pos()
        eyes.redraw()
if __name__ == '__main__':
    main()