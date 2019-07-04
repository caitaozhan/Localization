'''The wall attenuation model
'''

import numpy as np
from utility import Point, Segment, Wall, distance
from collections import defaultdict

class WAF:
    '''P(d)[dBm] = P(d0)[dBm] - 10*n*log(d/d0) - nW*WAF 
    '''
    def __init__(self, means, locations, lt, wall):
        '''
        Args:
            means (np.ndarray, n=2)
            locations (np.ndarray, n=2)
            wall (WAll)
            lt (LocationTransform)
        '''
        self.means = means           # means of training samples
        self.locations = locations   # locations of training samples
        self.wall = wall             # Wall
        self.lt   = lt               # Location
        self.p_d0 = 0                # model paramaters
        self.d0   = 0                # model paramaters
        self.n    = 0                # model paramaters
        self.waf  = 0                # model paramaters
        self.C    = 0                # model paramaters
        self.train_model()
    
    def __str__(self):
        return 'P(d0) = {}, d0 = {}, n = {}, waf = {}, C = {}'.format(self.p_d0, self.d0, self.n, self.waf, self.C)


    def train_model(self):
        self.C = 4
        # step 1: find d0 and p_d0
        self.d0, self.p_d0 = self.find_d0_pd0()
        print(self)


    def find_d0_pd0(self):
        '''find the smallest distance pair of locations that do not have wall in between
        '''
        dic = defaultdict(list)
        for i in range(len(self.locations)):
            for j in range(i+1, len(self.locations)):
                dist = distance(self.locations[i], self.locations[j])
                dic[dist].append((i, j))
        for key, val in sorted(dic.items()):  # distance --> [ (loc1, loc2) ...]
            for pair in val:
                i, j = pair
                tx = Point(self.locations[i][0], self.locations[i][1])
                rx = Point(self.locations[j][0], self.locations[j][1])
                if self.wall.count_intersect(tx, rx) == 0:
                    return key, self.means[i][j]
    

    def predict(self, tx, rx):
        '''Predice the receiver's RSS at rx, when the transmitter is at tx
        Args:
            tx (tuple(float, float))
            rx (tuple(float, float))
        '''
        tx = Point(tx[0], tx[1])
        rx = Point(rx[0], rx[1])
        nW = self.wall.count_intersect(tx, rx)
        nW = 4 if nW > 4 else nW
        return nW

