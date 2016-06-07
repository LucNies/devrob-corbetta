import matplotlib.pyplot as plt
import math
import numpy as np
import random
import pickle
from sympy.geometry import Line as SymLine
from sympy.geometry import Point as SymPoint
from shapely.geometry import Point, LineString, GeometryCollection
from shapely import affinity
from fractions import gcd

from tqdm import tqdm
import time
from IPython import embed



class Eyes(object):
    def __init__(self, origin, visualize=True):
        self.center_origin = origin
        self.inter_eye_distance = 6
        self.attended_points = []
        self.max_distance = 15
        self.visual_space = Point(self.center_origin, 0).buffer(self.max_distance)
        self.max_angle = 60
        self.left_dominant = True
        self.visualize = visualize

        self.calculate_lines()
        if self.visualize:
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

        self.line_dom_center = plt.Line2D([], [], linewidth=2)
        self.line_sub_center = plt.Line2D([], [], linewidth=2)
        self.line_dom_inner_bound = plt.Line2D([], [], linewidth=2)
        self.line_sub_inner_bound = plt.Line2D([], [], linewidth=2)
        self.canvas_ax.add_line(self.line_dom_center)
        self.canvas_ax.add_line(self.line_sub_center)
        #self.canvas_ax.add_line(self.to_Line2D(self.dom_outer_bound))
        #self.canvas_ax.add_line(self.to_Line2D(self.sub_outer_bound))
        self.canvas_ax.add_line(self.line_dom_inner_bound)
        self.canvas_ax.add_line(self.line_sub_inner_bound)
        #self.canvas_ax.add_line(self.to_Line2D(self.min_angle_line))

        # Focus lines
        self.focus_line_dom_eye = plt.Line2D([], [], linewidth=2)
        self.focus_line_sub_eye = plt.Line2D([], [], linewidth=2)
        self.canvas_ax.add_line(self.focus_line_dom_eye)
        self.canvas_ax.add_line(self.focus_line_sub_eye)

        # Attended points
        self.reachable_scatter = plt.scatter([], [])

    def redraw(self):
        if self.visualize:
            # Draw attended points
            self.reachable_scatter.set_offsets(self.attended_points)

            self.line_dom_center.set_data([self.dom_center_line.coords[0][0], self.dom_center_line.coords[1][0]], [self.dom_center_line.coords[0][1], self.dom_center_line.coords[1][1]])
            self.line_sub_center.set_data([self.sub_center_line.coords[0][0], self.sub_center_line.coords[1][0]], [self.sub_center_line.coords[0][1], self.sub_center_line.coords[1][1]])
            self.line_dom_inner_bound.set_data([self.dom_inner_bound.coords[0][0], self.dom_inner_bound.coords[1][0]], [self.dom_inner_bound.coords[0][1], self.dom_inner_bound.coords[1][1]])
            self.line_sub_inner_bound.set_data([self.sub_inner_bound.coords[0][0], self.sub_inner_bound.coords[1][0]], [self.sub_inner_bound.coords[0][1], self.sub_inner_bound.coords[1][1]])

            # Adjust focus lines
            self.focus_line_dom_eye.set_data([self.dom_focus_line.coords[0][0], self.dom_focus_line.coords[1][0]], [self.dom_focus_line.coords[0][1], self.dom_focus_line.coords[1][1]])
            self.focus_line_dom_eye.set_color('r')
            self.focus_line_sub_eye.set_data([self.sub_focus_line.coords[0][0], self.sub_focus_line.coords[1][0]], [self.sub_focus_line.coords[0][1], self.sub_focus_line.coords[1][1]])
            self.focus_line_sub_eye.set_color('g')

            plt.plot()
            plt.pause(0.000001)

    def calc_angle_submissive_eye(self, angle):
        tstart= time.time()
        dom_focus_line = self.rotate_line(self.dom_center_line, angle)
        intersection_point = self.get_pos_intersection(dom_focus_line, self.visual_space.boundary)
        self.sub_angle_line = LineString([(self.sub_center_line.coords[0][0], 0), intersection_point.coords[0]])

        angle_between = self.get_angle_between(
            self.sub_angle_line,
            self.sub_center_line
        )
        angle = math.degrees(angle_between)
        if self.left_dominant:
            if math.atan2(self.sub_angle_line.coords[1][1], self.sub_angle_line.coords[1][0] - self.inter_eye_distance/2) - math.pi/2 < 0:
                angle = -angle
        else:
            if math.atan2(self.sub_angle_line.coords[1][1], self.sub_angle_line.coords[1][0] + self.inter_eye_distance/2) - math.pi/2 < 0:
                angle = -angle

        #print 'calc_angle: %s' % (time.time() - tstart)

        return angle

    def calculate_lines(self):
        # In case of Left dominance
        self.Ldom_center_line = LineString([Point(self.center_origin - self.inter_eye_distance / 2, 0),
                              Point(self.center_origin - self.inter_eye_distance / 2, math.sqrt(self.max_distance**2 - (self.inter_eye_distance / 2)**2))])
        self.Lsub_center_line = LineString([Point(self.center_origin + self.inter_eye_distance / 2, 0),
                            Point(self.center_origin + self.inter_eye_distance / 2,
                            math.sqrt(self.max_distance**2 - (self.inter_eye_distance / 2) ** 2))])

        #self.Ldom_outer_bound = self.rotate_line(self.dom_center_line, + self.max_angle)
        #self.Lsub_outer_bound = self.rotate_line(self.sub_center_line, - self.max_angle)
        self.Ldom_inner_bound = self.rotate_line(self.Ldom_center_line, - self.max_angle)
        self.Lsub_inner_bound = self.rotate_line(self.Lsub_center_line, + self.max_angle)

        # In case of Right dominance
        self.Rdom_center_line = LineString([Point(self.center_origin + self.inter_eye_distance / 2, 0),
                  Point(self.center_origin + self.inter_eye_distance / 2, math.sqrt(self.max_distance**2 - (self.inter_eye_distance / 2)**2))])
        self.Rsub_center_line = LineString([Point(self.center_origin -self.inter_eye_distance / 2, 0),
                  Point(self.center_origin - self.inter_eye_distance / 2, math.sqrt(self.max_distance**2 - (self.inter_eye_distance / 2)**2))])

        #self.Rdom_outer_bound = self.rotate_line(self.dom_center_line, - self.max_angle)
        #self.Rsub_outer_bound = self.rotate_line(self.sub_center_line, + self.max_angle)
        self.Rdom_inner_bound = self.rotate_line(self.Rdom_center_line, + self.max_angle)
        self.Rsub_inner_bound = self.rotate_line(self.Rsub_center_line, - self.max_angle)

    def set_dominance(self, dominance):
        tstart = time.time()
        self.left_dominant = bool(dominance)

        self.dom_center_line = self.Ldom_center_line if self.left_dominant else self.Rdom_center_line
        self.sub_center_line = self.Lsub_center_line if self.left_dominant else self.Rsub_center_line
        self.dom_inner_bound = self.Ldom_inner_bound if self.left_dominant else self.Rdom_inner_bound
        self.sub_inner_bound = self.Lsub_inner_bound if self.left_dominant else self.Rsub_inner_bound
        #self.dom_outer_bound = self.Ldom_outer_bound if self.left_dominant else self.Rdom_outer_bound
        #self.sub_outer_bound = self.Lsub_outer_bound if self.left_dominant else self.Rsub_outer_bound

        self.min_angle_line = LineString([self.dom_center_line.coords[0], self.get_pos_intersection(self.sub_inner_bound, self.visual_space.boundary).coords[0]])
        self.min_angle = math.degrees(self.get_angle_between(
            self.min_angle_line,
            self.dom_center_line
        ))
        #print 'set_dominance: %s' % (time.time()-tstart)

    def to_Line2D(self, line):
        ''' Converts a Shapely line to a matplotlib Line2D '''
        return plt.Line2D((line.coords[0][0], line.coords[1][0]), (line.coords[0][1], line.coords[1][1]))

    def rotate_line(self, line, angle):
        ''' Wrapper function for rotating the line to an angle and then clipping the rotated line to the visual space. '''
        tstart = time.time()
        new_line = affinity.rotate(line, angle, origin=line.coords[0])
        intersection_point = self.get_pos_intersection(new_line, self.visual_space.boundary)
        #print 'rotate_line: %s'  % (time.time() - tstart)
        return LineString([Point(line.coords[0][0], line.coords[0][1]), intersection_point])

    def get_pos_intersection(self, e1, e2):
        ''' Calculates the intersection between two entities. '''
        # Hack: extend the line to make sure there is an intersection point
        tstart = time.time()
        extended_point = Point(e1.coords[1][0] + (e1.coords[1][0] - e1.coords[0][0]) / e1.length * e1.length**2,
                               e1.coords[1][1] + (e1.coords[1][1] - e1.coords[0][1]) / e1.length * e1.length**2)
        e1 = LineString([Point(e1.coords[0][0], e1.coords[0][1]), extended_point])
        intersection_point = e1.intersection(e2)
        #print 'get_pos_intersection: %s' % (time.time() - tstart)
        return intersection_point

    def get_angle_between(self, line1, line2):
        ''' Uses SymPy to calculate the angle between two lines. Assumes input formatted as Shapely lines. '''
        tstart = time.time()
        return SymLine.angle_between(
            SymLine(SymPoint(*line1.coords[0]), SymPoint(*line1.coords[1])),
            SymLine(SymPoint(*line2.coords[0]), SymPoint(*line2.coords[1]))
        )
        #print 'get_angle_between: %s' % (time.time() - tstart)

    def move_eyes(self, angle_dom_eye, angle_sub_eye):
        tstart = time.time()
        self.dom_focus_line = self.rotate_line(self.dom_center_line, angle_dom_eye)
        self.sub_focus_line = self.rotate_line(self.sub_center_line, angle_sub_eye)
        focus_point = self.get_pos_intersection(self.dom_focus_line, self.sub_focus_line)

        #print 'move_eyes: %s'  % (time.time() - tstart)

        if type(focus_point) != GeometryCollection:
            return focus_point

        return None

    def create_prototypes(self, shape = (10,10)):
            prototypes = np.zeros(shape=(shape[0] * shape[1], 2))

            gcd_shoulder = gcd(shape[0])
            gcd_elbow = gcd(shape[1], self.max_elbow_angle)

            i = 0
            for shoulder_angle in np.arange(0, self.max_shoulder_angle, self.max_shoulder_angle / gcd_shoulder + 1):
                for elbow_angle in np.arange(0, self.max_elbow_angle, self.max_elbow_angle / gcd_elbow + 1):
                    x, y = self.move_arm(shoulder_angle, elbow_angle)
                    prototypes[i] = [x, y]
                    i += 1
                    # print shoulder_angle, elbow_angle

            return prototypes

    def create_dataset(self, n_datapoints=100000, train_file='train_data.p', val_file='validation_data.p',
                           test_file='test_data.p', validation_size=0.1, test_size=0.1):
        print "Create datapoints"
        data_points = np.zeros((n_datapoints, 2, 2), dtype=np.float32)
        dominance_left_set = False
        dominance_right_set = False
        for i in tqdm(range(n_datapoints)):
            if not dominance_left_set:
                self.set_dominance(1)
                dominance_left_set = True
            if i > n_datapoints / 2 and not dominance_right_set:
                self.set_dominance(0)
                dominance_right_set = True

            if self.left_dominant:
                angle_dom_eye = random.uniform(-self.max_angle, self.min_angle)
                angle_sub_eye = random.uniform(self.calc_angle_submissive_eye(angle_dom_eye), self.max_angle)
                point = self.move_eyes(angle_dom_eye, angle_sub_eye)
                data_points[i] = [[angle_dom_eye, angle_sub_eye], [point.x, point.y]]
            else:
                angle_dom_eye = random.uniform(-self.min_angle, self.max_angle)
                angle_sub_eye = random.uniform(-self.max_angle, self.calc_angle_submissive_eye(angle_dom_eye))
                point = self.move_eyes(angle_dom_eye, angle_sub_eye)
                data_points[i] = [[angle_sub_eye, angle_dom_eye], [point.x, point.y]]
            if i % 1000 == 0:
                print i

        embed()

        print "Datapoints created, saving to file..."
        train_size = len(data_points) * (1 - validation_size - test_size)
        validation_size = len(data_points) * validation_size
        test_size = len(data_points) * validation_size

        train_data = data_points[:train_size]
        val_data = data_points[train_size:train_size + validation_size]
        test_data = data_points[train_size + validation_size: len(data_points) - 1]

        with open(train_file, 'wb') as f_out:
            pickle.dump(train_data, f_out)

        with open(val_file, 'wb') as f_out:
            pickle.dump(val_data, f_out)

        with open(test_file, 'wb') as f_out:
            pickle.dump(test_data, f_out)

        print "Done saving"
        return data_points