'''
Module for class Transmitter. A transmitter is essentially a hypothesis
'''

import numpy as np

class Transmitter:
    '''Encapsulate a transmitter
    Attributes:
        x (int): location -  first dimension
        y (int): location -  second dimension
        mean_vec (np.ndarray):     mean vector, length is the number of sensors
        mean_vec_sub (np.ndarray): mean vector for subset of sensors
        multivariant_gaussian(scipy.stats.multivariate_normal): each hypothesis corresponds to a multivariant guassian distribution
        powers (list): a list of powers, contains a 0 in the center: the power read from the hypothesis file
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.hypothesis = 0
        self.mean_vec = np.zeros(0)
        self.mean_vec_sub = np.zeros(0)
        self.multivariant_gaussian = None
        self.powers = [0]


    def set_mean_vec_sub(self, subset_index):
        '''Given a subset_index list, set the mean_vec_sub
        Attributes:
            subset_index (list): a list of index
        '''
        self.mean_vec_sub = self.mean_vec[subset_index]


    def write_mean_vec(self, filename):
        '''append the mean vector to filename
        '''
        with open(filename, 'a') as f:      # when the file already exists, it will not overwirte
            f.write(str(self.mean_vec) + '\n')


    def __str__(self):
        str1 = "(%d, %d) ".ljust(10) % (self.x, self.y)
        return str1 + str(self.error)


if __name__ == '__main__':
    transmitter = Transmitter(3, 5)
    transmitter2 = Transmitter(7, 9)
    print(transmitter)
    print(transmitter2)
