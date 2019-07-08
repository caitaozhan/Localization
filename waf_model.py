'''The wall attenuation model
'''

import numpy as np
import math
from sklearn.linear_model import LinearRegression
from collections import defaultdict
import matplotlib.pyplot as plt
from utility import Point, Segment, Wall, distance

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
        return 'P(d0) = {:5.2f}, d0 = {:5.2f}, n = {:5.2f}, waf = {:5.2f}, C = {:5.2f}, itcpt = {:5.2f}'.format(self.p_d0, self.d0, self.n, self.waf, self.C, self.intercept)


    def train_model(self):
        self.C = 3.5
        # step 1: find d0 and p_d0
        self.d0, self.p_d0 = self.find_d0_pd0(num=4)
        # step 2: regression the n and WAF
        self.n, self.waf, self.intercept = self.regresssion_n_waf()
        self.n /= -10
        self.waf *= -1
        print(self)


    def find_d0_pd0(self, num=4):
        '''find some largest RSS, average them as the P(d0) and d0
        Args:
            num (int): 
        '''
        means = []
        dic = defaultdict(list)
        for i in range(len(self.locations)):
            for j in range(len(self.locations)):
                if i == j:
                    continue
                dist = distance(self.locations[i], self.locations[j])
                dic[dist].append((i, j, self.means[i, j]))
                means.append(self.means[i][j])
        # X, Y = [], []
        threshold = sorted(np.array(means))[-num]
        d0, pd0 = [], []
        for key, vals in sorted(dic.items()):  # distance --> [ (loc1, loc2) ...]
            for val in vals:
                i, j, mean = val
                if mean >= threshold:
                    d0.append(key)
                    pd0.append(mean)
                    if len(d0) == num:
                        return np.array(d0).mean(), np.array(pd0).mean()
                
                # X.append(key)
                # Y.append(mean)
        # plt.figure(figsize=(10, 10))
        # plt.scatter(X, Y)
        # plt.savefig('visualize/RSS-dist.png')

    def regresssion_n_waf(self):
        '''Do a regression to get the n and waf
        '''
        X, y = [], []
        for i in range(len(self.locations)):
            for j in range(len(self.locations)):
                if i == j:
                    continue
                dist = distance(self.locations[i], self.locations[j])
                tx = Point(self.locations[i][0], self.locations[i][1])
                rx = Point(self.locations[j][0], self.locations[j][1])
                nW = self.wall.count_intersect(tx, rx)
                nW = nW if nW <= self.C else self.C
                y.append(self.means[i][j] - self.p_d0)
                # y.append(self.means[i][j])
                X.append([math.log10(dist/self.d0), nW])
                # X.append([math.log10(dist), nW])

        reg = LinearRegression(fit_intercept=True).fit(X, y)
        print('Regression score:', reg.score(X, y))
        return reg.coef_[0], reg.coef_[1], reg.intercept_


    def predict(self, tx, rx):
        '''Predice the receiver's RSS at rx, when the transmitter is at tx
        Args:
            tx (tuple(float, float))
            rx (tuple(float, float))
        '''
        dist = distance(tx, rx)
        tx = Point(tx[0], tx[1])
        rx = Point(rx[0], rx[1])
        nW = self.wall.count_intersect(tx, rx)
        nW = nW if nW <= self.C else self.C
        # offset = 0
        # if dist < 2 and nW == 0:
        #     offset = 6.37
        return self.p_d0 - 10*self.n*math.log10(dist/self.d0) - nW*self.waf + self.intercept
    

    def correct(self):
        X, Y = [], []
        for i in range(len(self.locations)):
            for j in range(len(self.locations)):
                if i == j:
                    continue
                dist = distance(self.locations[i], self.locations[j])
                tx = Point(self.locations[i][0], self.locations[i][1])
                rx = Point(self.locations[j][0], self.locations[j][1])
                nW = self.wall.count_intersect(tx, rx)
                nW = nW if nW <= self.C else self.C
                X.append(dist)
                Y.append(self.means[i][j] + nW*self.waf)
                # Y.append(self.means[i][j])
        
        plt.rcParams['font.size'] = 20
        plt.figure(figsize=(10, 10))
        plt.scatter(np.log10(X), Y)
        # plt.scatter(X, Y)
        plt.title('Wall Correction')
        plt.xlabel('Log Distance (m)')
        plt.ylabel('RSS (dBm)')
        plt.ylim([-85, -30])
        plt.savefig('visualize/RSS-logdist-wallcorrect.png')
