import matplotlib.pyplot as plt
import math
import numpy as np
import random
import pickle
import time
from sympy import *
from sympy.geometry import *

from IPython import embed

class Arm(object):
    def __init__(self, origin=0, visualize=True):
        self.L = 5
        self.origin = origin
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
        shoulder_angle  = random.randint(0, 360) * (math.pi / 180)
        changed = False
        if self.shoulder_joint[1] + self.L * math.sin(shoulder_angle) > 0:
            self.elbow_joint = (self.shoulder_joint[0] + self.L * math.cos(shoulder_angle),
                                self.shoulder_joint[1] + self.L * math.sin(shoulder_angle))

            while changed == False:
                elbow_angle = shoulder_angle + random.randint(0, 160) * (math.pi / 180) #160 degrees is in my opinion the most the arm can bend and can only bend one way
                if self.elbow_joint[1] + self.L * math.sin(elbow_angle) > 0:
                    self.wrist_joint = (self.elbow_joint[0] + self.L * math.cos(elbow_angle),
                                        self.elbow_joint[1] + self.L * math.sin(elbow_angle))
                    datapoint = (self.wrist_joint, (shoulder_angle, elbow_angle))
                    self.reached_points.append(datapoint)
                    changed = True

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
    def __init__(self, origin):
        self.center_origin = origin
        self.inter_eye_distance = 6
        self.left_eye_angle = 0
        self.right_eye_angle = 0
        self.attended_points = []
        self.max_distance = 15
        self.visual_space = Circle(Point(self.center_origin, 0), self.max_distance)
        self.max_angle = math.radians(60)

        self.calculate_lines()
        self._init_graphics()

    def _init_graphics(self):
        stime = time.time()
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

        self.canvas_ax.add_line(self.convert_to_Line2D(self.l_center_line))
        self.canvas_ax.add_line(self.convert_to_Line2D(self.r_center_line))
        self.canvas_ax.add_line(self.convert_to_Line2D(self.l_outer_bound))
        self.canvas_ax.add_line(self.convert_to_Line2D(self.r_outer_bound))
        self.canvas_ax.add_line(self.convert_to_Line2D(self.l_inner_bound))
        self.canvas_ax.add_line(self.convert_to_Line2D(self.r_inner_bound))

        self.reachable_scatter = plt.scatter([], [])
        print '_init_graphics: %s' % (time.time() - stime)

    def redraw(self):
        self.reachable_scatter.set_offsets(self.attended_points)

        plt.plot()
        plt.pause(0.000001)

    def random_angle(self, min, max):
        return random.uniform(min, max)

    def calc_angle_submissive_eye(self, angle):
        focus_line = self.rotate_line(self.l_center_line, angle)
        stime = time.time()
        intersection_point = intersection(focus_line, self.visual_space)[0]

        line = Line(Point(self.r_center_line.p1.x, 0), intersection_point)

        angle_between = Line.angle_between(line, self.r_center_line)
        print 'calc_angle_submissive_eye: %s' % (time.time() - stime)
        return angle_between

    def calculate_lines(self):
        stime = time.time()
        self.l_center_line = Line(Point(self.center_origin -self.inter_eye_distance / 2, 0),
                                  Point(self.center_origin - self.inter_eye_distance / 2, math.sqrt(self.max_distance**2 - (self.inter_eye_distance / 2)**2)))
        self.r_center_line = Line(Point(self.center_origin + self.inter_eye_distance / 2, 0),
                                  Point(self.center_origin + self.inter_eye_distance / 2,
                                        math.sqrt(self.max_distance ** 2 - (self.inter_eye_distance / 2) ** 2)))

        self.l_outer_bound = self.rotate_line(self.l_center_line, + self.max_angle)
        self.r_outer_bound = self.rotate_line(self.r_center_line, - self.max_angle)
        self.l_inner_bound = self.rotate_line(self.l_center_line, - self.max_angle)
        self.r_inner_bound = self.rotate_line(self.r_center_line, + self.max_angle)

        self.min_angle = Line.angle_between(
            Line(self.l_center_line.p1, self.get_pos_intersection(self.r_inner_bound, self.visual_space)),
            self.l_center_line)
        print 'calculate_lines: %s' % (time.time() - stime)

    def convert_to_Line2D(self, line):
        return plt.Line2D((line.p1.x, line.p2.x),
                   (line.p1.y, line.p2.y))

    def rotate_line(self, line, angle):
        stime = time.time()
        line_x = line.p1.x

        rotated_x = math.cos(angle) * 0 - math.sin(angle) * line.p2.y
        rotated_y = math.sin(angle) * 0 + math.cos(angle) * line.p2.y

        temp_line = Line(Point(line_x, line.p1.y), Point(rotated_x + line_x, rotated_y))

        intersection_point = self.get_pos_intersection(temp_line, self.visual_space)

        new_line = Line(temp_line.p1, intersection_point)
        print 'rotate_line: %s' % (time.time() - stime)
        return new_line

    def get_pos_intersection(self, e1, e2):
        intersection_point = None
        for point in intersection(e1, e2):
            if point.y >= 0:
                intersection_point = point

        return intersection_point

    def random_eye_pos(self):
        stime = time.time()
        # do stuff
        rand_angle_leye = self.random_angle(-self.max_angle, self.min_angle)
        angle_reye = self.random_angle(self.calc_angle_submissive_eye(rand_angle_leye), self.max_angle)
        l_focus_line = self.rotate_line(self.l_center_line, rand_angle_leye)
        r_focus_line = self.rotate_line(self.r_center_line, angle_reye)
        focus_point = self.get_pos_intersection(l_focus_line, r_focus_line)
        self.attended_points.append((focus_point.x, focus_point.y))
        print 'random_eye_pos: %s' % (time.time() - stime)

def save_data(arm):
    with open('output.dat', 'wb') as f_out:
        pickle.dump(arm.reached_points, f_out)

def main():
    ''''
    arm = Arm(origin=12, visualize=True)
    while len(arm.reached_points) < 1000000:
        arm.random_arm_pos()
        arm.redraw()

        if len(arm.reached_points) % 100000 == 0:
            print len(arm.reached_points)

    save_data(arm)
    '''

    eyes = Eyes(origin=0)
    while True:
        eyes.random_eye_pos()
        eyes.redraw()

if __name__ == '__main__':
    main()