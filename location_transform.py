'''
Build a duplex transformation between {real world float locations} and {grid cell integer locations}
'''
import scipy.io
import numpy as np
import pandas as pd
import math
from utility import read_utah_data


class LocationTransform:
    def __init__(self, real_location, grid_len=16, cell_len=1.):
        '''
        Args:
            real_location (2D-array-like): location of hypothesis
            grid_len (int)   : the length of the grid, it is a square
            cell_len (float) : the length of a grid cell, also a square
        '''
        self.real_location  = real_location
        self.grid_len = grid_len
        self.cell_len = cell_len
        self.check()

    def check(self):
        '''check: make sure that different float locations fall into different grid cells
        '''
        pass

    def real_2_gridcell(self, real):
        pass
    
    def gridcell_2_gridcenter(self, gridcell):
        pass



if __name__ == '__main__':
    mean, stds, locations = read_utah_data()
    print(mean)
    print(stds)
    print(locations)

