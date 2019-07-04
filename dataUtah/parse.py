import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt
import scipy.io
import math
import os


def read_utah_data():
    '''Read utah data
    Return:
        trace ()
        location ()
    '''
    savedSig = scipy.io.loadmat('dataUtah/savedSig/savedSig.mat')
    trace = savedSig['savedSig']
    num = trace.shape[0]
    mean = np.zeros((num, num))
    stds = np.zeros((num, num))
    for tx in range(num):
        for rx in range(num):
            if tx == rx:
                continue
            meas_num = len(trace[tx][rx])
            powers = []
            for i in range(meas_num):
                iq_samples = trace[tx][rx][i]
                amplitude  = np.absolute(iq_samples)
                powers.extend(list(20*np.log10(amplitude)))
            powers = np.array(powers)
            argsort = np.flip(np.argsort(powers))
            peaks = powers[argsort[0:15]]          # select highest 15 signals
            mean[tx][rx] = peaks.mean()
            stds[tx][rx] = peaks.std()
    
    locations = scipy.io.loadmat('dataUtah/savedSig/deviceLocs.mat')
    locations = locations['deviceLocs']
    locations = np.array(locations)
    return mean, np.mean(stds, 0), locations


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return '{},{}'.format(self.x, self.y)


class Segment:
    def __init__(self, point1, point2):
        self.p1 = point1
        self.p2 = point2
    
    def __str__(self):
        return '{} -- {}'.format(str(self.p1), str(self.p2))


class Wall:
    def __init__(self):
        self.wall = []

    def __str__(self):
        return '\n'.join(map(str, self.wall))

    def add(self, segment):
        self.wall.append(segment)

    def count_intersect(self, p3, p4):
        '''Count the number of intersections between segment p3-p4 and all the walls
        Args:
            p3, p4 (Point)
        Return:
            (bool)
        '''
        counter = 0
        for seg in self.wall:
            p1, p2 = seg.p1, seg.p2
            if Wall.doIntersect(p1, p2, p3, p4):
                counter += 1
                # print(seg)
        return counter

    @staticmethod
    def onSegment(p1, p2, p3):
        '''Given 3 colinear points p1, p2, p3, checks if p2 lies on segment p1-p3
        Args:
            p1, p2, p3 (Point)
        Return:
            (bool)
        '''
        if (p2.x <= max(p1.x, p3.x) and p2.x >= min(p1.x, p3.x) and p2.y <= max(p1.y, p3.y) and p2.y >= min(p1.y, p3.y)):
            return True
        return False
    
    @staticmethod
    def orientation(p1, p2, p3):
        '''Find the orientation of ordered triplet (p1, p2, p3)
        Args:
            p1, p2, p3 (Point)
        Return:
            (int)
            0 --> p1, p2, p3 are colinear
            1 --> p1, p2, p3 clockwise
            2 --> p1, p2, p3 counter clockwise
        '''
        val = (p2.y - p1.y)*(p3.x - p2.x) - (p3.y - p2.y)*(p2.x - p1.x)

        if val == 0:
            return 0
        if val > 0:
            return 1
        if val < 0:
            return 2
    
    @staticmethod
    def doIntersect(p1, p2, p3, p4):
        '''Check if segment p1-p2 and p3-p4 intersect
        Args:
            p1, p2, p3, p4 (Point)
        Return:
            (bool)
        '''
        o1 = Wall.orientation(p1, p2, p3)
        o2 = Wall.orientation(p1, p2, p4)
        o3 = Wall.orientation(p3, p4, p1)
        o4 = Wall.orientation(p3, p4, p2)

        if o1 != o2 and o3 != o4:
            return True
        
        if o1 == 0 and Wall.onSegment(p1, p3, p2):
            return True
        if o2 == 0 and Wall.onSegment(p1, p4, p2):
            return True
        if o3 == 0 and Wall.onSegment(p3, p1, p4):
            return True
        if o4 == 0 and Wall.onSegment(p3, p2, p4):
            return True

        return False


def read_utah_wall(filename):
    with open(filename, 'r') as f:
        wall = Wall()
        for line in f:
            line = line.replace('\n', '')
            point1, point2 = line.split(' ')
            x, y = point1.split(',')
            p1 = Point(float(x), float(y))
            x, y = point2.split(',')
            p2 = Point(float(x), float(y))
            seg = Segment(p1, p2)
            wall.add(seg)
    return wall

def preprocess_rawdata(input_filename, output_filename):
    '''Convert the raw pixels into coordinates, write to a file
    Args:
        wall (Wall)
    '''
    anchor1_pix = (92, 69)              # point 1  in utah floor map, minimal x
    anchor2_pix = (432, 490)            # point 30 in utah floor map, minimal y
    anchor1     = (-4.1148, 12.1158)
    anchor2     = ( 4.5720, -0.2286)
    min_x_pix   = 92
    min_y_pix   = 490
    min_x       = -4.1148
    min_y       = -0.2286
    step_x      = (anchor2[0] - anchor1[0]) / (anchor2_pix[0] - anchor1_pix[0])
    step_y      = (anchor2[1] - anchor1[1]) / (anchor2_pix[1] - anchor1_pix[1])
    
    outfile = open(output_filename, 'w')
    with open(input_filename, 'r') as f:
        for line in f:
            line = line.replace('\n', '')
            point1, point2 = line.split(' ')
            
            x_pix, y_pix = point1.split(',')
            p1 = Point(float(x_pix), float(y_pix))
            p1.x = min_x + (p1.x - min_x_pix)*step_x
            p1.y = min_y + (p1.y - min_y_pix)*step_y

            x_pix, y_pix = point2.split(',')
            p2 = Point(float(x_pix), float(y_pix))
            p2.x = min_x + (p2.x - min_x_pix)*step_x
            p2.y = min_y + (p2.y - min_y_pix)*step_y

            print('{:.2f},{:.2f} {:.2f},{:.2f}'.format(p1.x, p1.y, p2.x, p2.y), file=outfile)
    outfile.close()


if __name__ == '__main__':
    wall = read_utah_wall('dataUtah/wall.txt')
    # preprocess_rawdata('dataUtah/wall_raw.txt', 'dataUtah/wall.txt')
    # p1 = Point(100, 124)
    # p2 = Point(596, 124)
    # num = wall.count_intersect(p1, p2)
    # print(num)
    mean, stds, locations = read_utah_data()
    # print(mean)
    # print(stds)
    # print(locations)
    try:
        while True:
            p1 = input('Enter point 1: ')
            p2 = input('Enter point 2: ')
            p1, p2 = int(p1), int(p2)
            x, y = locations[p1-1]
            p1 = Point(x, y)
            x, y = locations[p2-1]
            p2 = Point(x, y)
            print('Num of walls: ', wall.count_intersect(p1, p2), '\n')
    except Exception as e:
        print(e)
