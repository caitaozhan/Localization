'''
Build a duplex transformation between {real world float locations} and {grid cell}
'''
import scipy.io
import numpy as np
import pandas as pd
from collections import defaultdict
from utility import read_utah_data, distance, plot_cdf


class LocationTransform:
    def __init__(self, real_location, cell_len=1.):
        '''
        Args:
            real_location (2D-array-like): location of hypothesis
            grid_len (int)   : the length of the grid, it is a square
            cell_len (float) : the length of a grid cell, also a square
        '''
        self.cell_len = cell_len
        self.real_location = real_location
        self.x_min, self.y_min = np.min(self.real_location, 0)             # x_min and y_min will be the ZERO of the grid
        self.grid_location = np.array([self.real_2_gridcell(real_loc) for real_loc in self.real_location])
        self.grid_x_len, self.grid_y_len = np.max(self.grid_location, 0) + 1
        self.grid_len = max(self.grid_x_len, self.grid_y_len)
        self.check()


    def __str__(self):
        return '\n'.join(map(lambda x: '({:5.2f}, {:5.2f}) --> ({:2d}, {:2d})'.format(x[0][0], x[0][1], x[1][0], x[1][1]), zip(self.real_location, self.grid_location))) + \
               '\ncell len   = {}\n'.format(self.cell_len) + \
               'real x min = {:.2f}, real y min = {:.2f}\n'.format(self.x_min, self.y_min) + \
               'grid x len = {:5d}, grid y len = {:5d}\n'.format(self.grid_x_len, self.grid_y_len)
                

    def check(self):
        '''check: see how many different float locations fall into a same grid cell
        '''
        d = defaultdict(lambda: 0)
        for cell in self.grid_location:
            d[tuple(cell)] += 1
            if d[tuple(cell)] >= 2:
                print(tuple(cell), d[tuple(cell)])


    def real_2_gridcell(self, real_loc):
        '''put a real number location into a grid cell
        Args:
            real_loc ([float, float])
        Return:
            grid_loc ([int, int])
        '''
        x = int((real_loc[0] - self.x_min) / self.cell_len)
        y = int((real_loc[1] - self.y_min) / self.cell_len)
        return [x, y]


    def gridcell_2_real(self, gridcell):
        '''get the real location of the CENTER of the grid cell
        Args:
            gridcell ([int, int])
        Return:
            real_loc ([float, float])
        '''
        x = self.x_min + (gridcell[0] + 0.5)*self.cell_len  # 0.5 represents the center of grid cell
        y = self.y_min + (gridcell[1] + 0.5)*self.cell_len
        return [round(x, 4), round(y, 4)]


class GPSLocationTransform:
    def __init__(self, gps_locations, cell_len):
        self.gps_locations = gps_locations
        self.cell_len = cell_len



def main1():
    mean, stds, locations, wall = read_utah_data()
    # print(mean)
    # print(stds)
    # print(locations)
    lt = LocationTransform(locations, cell_len=1)
    # print(lt)

    num = len(mean)
    with open('dataUtah/means.txt', 'w') as f:
        for i in range(num):
            for j in range(num):
                f.write('{:6.2f} '.format(mean[i][j]))
            f.write('\n')
    with open('dataUtah/stds.txt', 'w') as f:
        for std in stds:
            f.write('{:.3f}\n'.format(std))
    with open('dataUtah/locations.txt', 'w') as f:
        for i in range(num):
            cell = lt.grid_location[i]
            real_loc = lt.real_location[i]
            f.write('{:3d} - '.format(cell[0]*lt.grid_len + cell[1]))
            f.write('({:2d}, {:2d}) - '.format(cell[0], cell[1]))
            f.write('({:7.4f}, {:7.4f})\n'.format(real_loc[0], real_loc[1]))

    errors = []
    for real_loc in lt.real_location:
        gridcell = lt.real_2_gridcell(real_loc)
        real_loc2 = lt.gridcell_2_real(gridcell)
        error = distance(real_loc, real_loc2)
        errors.append(error)
        print('({:5.2f}, {:5.2f}) -> ({:2d}, {:2d}) -> ({:5.2f}, {:5.2f}); error = {:3.2f}'.format(real_loc[0], real_loc[1], gridcell[0], gridcell[1], real_loc2[0], real_loc2[1], error))
    plot_cdf(errors)


def main2():
    '''Output files in forms class Localize can init
    '''
    means, stds, locations, wall = read_utah_data()
    lt = LocationTransform(locations, cell_len=1)
    num = len(means)
    with open('dataUtah/hypothesis', 'w') as f:
        for t in range(num):
            t_x, t_y = lt.grid_location[t]     # transmitter
            for s in range(num):
                s_x, s_y = lt.grid_location[s] # sensor
                mean = means[t][s] if t!=s else -100
                std  = stds[s]
                f.write('{} {} {} {} {} {}\n'.format(t_x, t_y, s_x, s_y, mean, std))
    with open('dataUtah/sensors', 'w') as f:
        for s in range(num):
            s_x, s_y = lt.grid_location[s]
            std  = stds[s]
            cost = 1
            f.write('{} {} {} {}\n'.format(s_x, s_y, std, cost))
    with open('dataUtah/cov', 'w') as f:
        for i in range(num):
            for j in range(num):
                if i == j:
                    f.write('{} '.format(stds[i]**2))
                else:
                    f.write('{} '.format(0.0))
            f.write('\n')


if __name__ == '__main__':
    # main1()
    main2()    


