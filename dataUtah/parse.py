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


if __name__ == '__main__':
    mean, stds, locations = read_utah_data()
    print(mean)
    print(stds)
    print(locations)
