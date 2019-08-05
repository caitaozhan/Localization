import numpy as np
import sys
import random

from utility import read_utah_data, distance, plot_cdf, generate_intruders_utah, set_intruders_utah
from location_transform import LocationTransform
from localize import Localization
from plots import visualize_localization
from interpolate import Interpolate


def main1():
    random.seed(0)
    np.random.seed(0)
    ll = Localization(grid_len=10)
    # ll.init_data()
    inter = Interpolate(ll.sensors, ll.means, ll.stds) #''' args... '''
    sensors_inter, means_inter, stds_inter = inter.idw_inter(factor=4)
    ll_inter = Localization(grid_len=40)
    ll_inter.init_data_direct(ll.covariance, sensors_inter, means_inter, stds_inter)



if __name__ == '__main__':
    main1()
