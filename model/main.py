from arm import Arm
from eye import Eyes
from IPython import embed
import util

def main():
    eyes = Eyes(origin=0, visualize=True)
    data = eyes.create_dataset(n_datapoints=100)
    for point in data:
        x, y = util.calc_intersect(point[0][0], point[0][1])
        print "actual\t x: {}, \t y: {}".format(point[1][0], point[1][1])
        print "calc \t x: {}, \t y: {}".format(x,y)

if __name__ == '__main__':
    main()