'''
Select sensor and detect transmitter
'''

import random
import math
import copy
import time
import os
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import multivariate_normal, norm
from joblib import Parallel, delayed, dump, load
from sensor import Sensor
from transmitter import Transmitter
from utility import read_config, ordered_insert, power_2_db, power_2_db_, db_2_power, db_2_power_, find_elbow#, print_results
try:
    from numba import cuda
    from cuda_kernals import o_t_approx_kernal, o_t_kernal, o_t_approx_dist_kernal, \
                         o_t_approx_kernal2, o_t_approx_dist_kernal2, update_dot_of_selected_kernal, sum_reduce
except Exception as e:
    pass
from itertools import combinations
import line_profiler
from sklearn.cluster import KMeans
from scipy.optimize import nnls
from plots import visualize_sensor_output, visualize_cluster, visualize_localization, visualize_q_prime, visualize_q, visualize_splot
from utility import generate_intruders, generate_intruders_2
from skimage.feature import peak_local_max
import itertools

class SelectSensor:
    '''Near-optimal low-cost sensor selection

    Attributes:
        config (json):               configurations - settings and parameters
        sen_num (int):               the number of sensors
        grid_len (int):              the length of the grid
        grid_priori (np.ndarray):    the element is priori probability of hypothesis - transmitter
        grid_posterior (np.ndarray): the element is posterior probability of hypothesis - transmitter
        transmitters (list):         a list of Transmitter
        sensors (dict):              a dictionary of Sensor. less than 10% the # of transmitter
        data (ndarray):              a 2D array of observation data
        covariance (np.ndarray):     a 2D array of covariance. each data share a same covariance matrix
        mean_stds (dict):            assume sigal between a transmitter-sensor pair is normal distributed
        subset (dict):               a subset of all sensors
        subset_index (list):         the linear index of sensor in self.sensors
        meanvec_array (np.ndarray):  contains the mean vector of every transmitter, for CUDA
        TPB (int):                   thread per block
        legal_transmitter (list):    a list of legal transmitters
        lookup_table (np.array):     trade space for time on the q function
    '''
    def __init__(self, grid_len):
        self.grid_len = grid_len
        self.sen_num  = 0
        self.grid_priori = np.zeros(0)
        self.grid_posterior = np.zeros(0)
        self.transmitters = []                 # transmitters are the hypothesises
        self.intruders = []
        self.sensors = []
        self.data = np.zeros(0)
        self.covariance = np.zeros(0)
        self.init_transmitters()
        self.set_priori()
        self.means = np.zeros(0)               # negative mean of intruder
        self.means_primary = np.zeros(0)       # negative mean of intruder plus primary
        self.means_all = np.zeros(0)           # negative mean of intruder plus primary plus secondary (all)
        self.means_rescale = np.zeros(0)       # positive mean of either self.means or self.means_rescale
        self.stds = np.zeros(0)                # for tx, sensor pair
        self.subset = {}
        self.subset_index = []
        self.meanvec_array = np.zeros(0)
        self.TPB = 32
        self.primary_trans = []                # introduce the legal transmitters as secondary user in the Mobicom version
        self.secondary_trans = []              # they include primary and secondary
        self.lookup_table_q = np.array([1. - 0.5*(1. + math.erf(i/1.4142135623730951)) for i in np.arange(0, 8.3, 0.0001)])
        self.lookup_table_norm = norm(0, 1).pdf(np.arange(0, 39, 0.0001))  # norm(0, 1).pdf(39) = 0


    #@profile
    def init_data(self, cov_file, sensor_file, hypothesis_file):
        '''Init everything from collected real data
           1. init covariance matrix
           2. init sensors
           3. init mean and std between every pair of transmitters and sensors
        '''
        cov = pd.read_csv(cov_file, header=None, delimiter=' ')
        del cov[len(cov)]
        #self.covariance = cov.values
        self.covariance = np.zeros(cov.values.shape)
        np.fill_diagonal(self.covariance, 1.)  # std=1 for every sensor NOTE: need to modify three places

        self.sensors = []
        with open(sensor_file, 'r') as f:
            max_gain = 0.5*len(self.transmitters)
            index = 0
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                x, y, std, cost = int(line[0]), int(line[1]), float(line[2]), float(line[3])
                self.sensors.append(Sensor(x, y, 1, cost, gain_up_bound=max_gain, index=index))  # uniform sensors
                index += 1
        self.sen_num = len(self.sensors)

        self.means = np.zeros((self.grid_len * self.grid_len, len(self.sensors)))
        self.stds = np.zeros((self.grid_len * self.grid_len, len(self.sensors)))
        with open(hypothesis_file, 'r') as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                line = line.split(' ')
                tran_x, tran_y = int(line[0]), int(line[1])
                #sen_x, sen_y = int(line[2]), int(line[3])
                mean, std = float(line[4]), float(line[5])
                self.means[tran_x*self.grid_len + tran_y, count] = mean  # count equals to the index of the sensors
                self.stds[tran_x*self.grid_len + tran_y, count] = 1      # std = 1 for every sensor
                count = (count + 1) % len(self.sensors)

        #temp_mean = np.zeros(self.grid_len * self.grid_len, )
        for transmitter in self.transmitters:
            tran_x, tran_y = transmitter.x, transmitter.y
            mean_vec = [0] * len(self.sensors)
            for sensor in self.sensors:
                mean = self.means[self.grid_len*tran_x + tran_y, sensor.index]
                mean_vec[sensor.index] = mean
            transmitter.mean_vec = np.array(mean_vec)


    def vary_power(self, powers):
        '''Varing power
        Args:
            powers (list): an element is a number that denote the difference from the default power read from the hypothesis file
        '''
        for tran in self.transmitters:
            tran.powers = powers


    def interpolate_gradient(self, x, y):
        '''The gradient for location (x, y) in the origin grid, for each sensor
        Args:
            x (int)
            y (int)
        Return:
            grad_x, grad_y (np.array, np.array): gradients in the x and y direction
        '''
        grad_x = np.zeros(len(self.sensors))
        grad_y = np.zeros(len(self.sensors))
        for s_index in range(len(self.sensors)):
            if x + 1 < self.grid_len:
                origin1 = self.means[x*self.grid_len + y][s_index]
                origin2 = self.means[(x+1)*self.grid_len + y][s_index]
                grad_x[s_index] = origin2 - origin1
            else:
                origin1 = self.means[x*self.grid_len + y][s_index]
                origin2 = self.means[(x-1)*self.grid_len + y][s_index]
                grad_x[s_index] = origin1 - origin2
            if y + 1 < self.grid_len:
                origin1 = self.means[x*self.grid_len + y][s_index]
                origin2 = self.means[x*self.grid_len + y+1][s_index]
                grad_y[s_index] = origin2 - origin1
            else:
                origin1 = self.means[x*self.grid_len + y][s_index]
                origin2 = self.means[x*self.grid_len + y-1][s_index]
                grad_y[s_index] = origin1 - origin2
        return grad_x, grad_y


    def interpolate_loc(self, scale, hypo_file, sensor_file):
        '''From M hypothesis to scale^2 * M hypothesis. For localization. Don't change the origin class members, instead create new copies.
           1. self.means_loc
           2. self.transmitters_loc
           3. self.sensors_loc
           4. self.grid_len_loc

        Args:
            scale (int): scaling factor
            hypo_file (str):   need to expand the hypothesis file
            sensor_file (str): need to change the coordinate of each sensor
        '''
        self.grid_len_loc = scale*self.grid_len
        self.means_loc = np.zeros((self.grid_len_loc * self.grid_len_loc, len(self.sensors)))
        self.transmitters_loc = [0] * self.grid_len_loc * self.grid_len_loc
        self.sensors_loc = copy.deepcopy(self.sensors)

        for t_index in range(len(self.transmitters_loc)):
            i = t_index // self.grid_len_loc
            j = t_index %  self.grid_len_loc
            self.transmitters_loc[t_index] = Transmitter(i, j)
        for s_index in range(len(self.sensors_loc)):
            self.sensors_loc[s_index].x = scale*self.sensors[s_index].x
            self.sensors_loc[s_index].y = scale*self.sensors[s_index].y

        for t_index in range(len(self.transmitters)):                          # M
            x = self.transmitters[t_index].x
            y = self.transmitters[t_index].y
            grad_x, grad_y = self.interpolate_gradient(x, y)
            x_loc = scale*x
            y_loc = scale*y
            for i in range(scale):
                for j in range(scale):                                         # scale^2
                    new_t_index = (x_loc+i)*self.grid_len_loc + (y_loc+j)
                    for s_index in range(len(self.sensors)):                   # S
                        origin_rss = self.means[t_index][s_index]              # = O(M*scale^2*S)
                        interpolate = origin_rss + float(i)/scale*grad_x[s_index] + float(j)/scale*grad_y[s_index]
                        self.means_loc[new_t_index][s_index] = interpolate
        
        with open(hypo_file, 'w') as f:
            for t_index in range(len(self.transmitters_loc)):
                trans_x = t_index // self.grid_len_loc
                trans_y = t_index % self.grid_len_loc
                for s_index in range(len(self.sensors_loc)):
                    sen_x = self.sensors_loc[s_index].x
                    sen_y = self.sensors_loc[s_index].y
                    mean = self.means_loc[t_index][s_index]
                    std   = self.sensors_loc[s_index].std
                    f.write('{} {} {} {} {} {}\n'.format(trans_x, trans_y, sen_x, sen_y, mean, std))
       
        with open(sensor_file, 'w') as f:
            for sensor in self.sensors:
                f.write('{} {} {} {}\n'.format(scale*sensor.x, scale*sensor.y, sensor.std, sensor.cost))



    def init_data_from_model(self, num_sensors):
        self.covariance = np.zeros((num_sensors, num_sensors))
        np.fill_diagonal(self.covariance, 1)
        max_gain = 0.5 * len(self.transmitters)

        #diff = (self.grid_len * self.grid_len) // num_sensors
        diff = 3
        count = 0
        for i in range(0, self.grid_len * self.grid_len, diff):
            x = i // self.grid_len
            y = i % self.grid_len
            self.sensors.append(Sensor(x, y, 1, 1, gain_up_bound=max_gain, index=count))
            count += 1

        self.means = np.zeros((self.grid_len * self.grid_len, len(self.sensors)))
        self.stds = np.ones((self.grid_len * self.grid_len, len(self.sensors)))

        for i in range(0, self.grid_len):
            for j in range(0, self.grid_len):
                for sensor in self.sensors:
                    distance = np.sqrt((i - sensor.x) ** 2 + (j - sensor.y) ** 2)
                    #print('distance = ', distance)
                    self.means[i * self.grid_len + j, sensor.index] = 10 - distance * 5
                    self.stds[i * self.grid_len + j, sensor.index] = 1
        print(self.means)
        for transmitter in self.transmitters:
            tran_x, tran_y = transmitter.x, transmitter.y
            mean_vec = [0] * len(self.sensors)
            for sensor in self.sensors:
                mean = self.means[self.grid_len * tran_x + tran_y, sensor.index]
                mean_vec[sensor.index] = mean
            transmitter.mean_vec = np.array(mean_vec)

        # del self.means
        # del self.stds
        print('\ninit done!')

    def set_priori(self):
        '''Set priori distribution - uniform distribution
        '''
        uniform = 1./(self.grid_len * self.grid_len)
        self.grid_priori = np.full((self.grid_len, self.grid_len), uniform)
        self.grid_posterior = np.full((self.grid_len, self.grid_len), uniform)
    

    def init_transmitters(self):
        '''Initiate a transmitter at all locations
        '''
        self.transmitters = [0] * self.grid_len * self.grid_len
        for i in range(self.grid_len):
            for j in range(self.grid_len):
                transmitter = Transmitter(i, j)
                setattr(transmitter, 'hypothesis', i*self.grid_len + j)
                self.transmitters[i*self.grid_len + j] = transmitter


    def setup_primary_transmitters(self, primary_transmitter, primary_hypo_file):
        '''Setup the primary transmitters, then "train" the distribution of them by linearly adding up the milliwatt power
        Args:
            primary_transmitter (list): index of legal primary transmitters
            primary_hypo_file (str):    filename, primary RSS to each sensor
        '''
        print('Setting up primary transmitters...', end=' ')
        self.primary_trans = []
        for trans in primary_transmitter:
            x = self.transmitters[trans].x
            y = self.transmitters[trans].y
            self.primary_trans.append(Transmitter(x, y))

        dic_mean = {}   # (sensor.x, sensor.y) --> [legal_mean1, legal_mean2, ...]
        dic_std  = {}   # (sensor.x, sensor.y) --> []
        for sensor in self.sensors:
            dic_mean[(sensor.x, sensor.y)] = []
            dic_std[(sensor.x, sensor.y)] = sensor.std
            for primary in self.primary_trans:
                pri_index = self.grid_len * primary.x + primary.y
                dic_mean[(sensor.x, sensor.y)].append(self.means[pri_index, sensor.index])

        # y = 20*log10(x)
        # x = 10^(y/20)
        # where y is power in dB and x is the absolute value of iq samples, i.e. amplitude
        # do a addition in the absolute value of iq samples
        with open(primary_hypo_file, 'w') as f:
            for key, value in dic_mean.items():
                amplitudes = np.power(10, np.array(value)/20)
                addition = sum(amplitudes)
                power_db = 20*np.log10(addition)
                f.write('{} {} {} {}\n'.format(key[0], key[1], power_db, dic_std[key]))


    def add_primary(self, primary_hypo_file):
        '''Add primary's RSS to itruder's RSS and save the sum to self.means_primary, write to file optional
        Params:
            primary_hypo_file (str):   filename, primary RSS to each each sensor
            intru_pri_hypo_file (str): filename, primary RSS plus intruder RSS to each each sensor
            write (bool): if True write the sums to intru_pri_hypo_file, if False don't write
        '''
        print('Adding primary...')
        hypothesis_legal = {}
        with open(primary_hypo_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                sen_x = int(line[0])
                sen_y = int(line[1])
                mean  = float(line[2])
                hypothesis_legal[(sen_x, sen_y)] = db_2_power(mean)

        self.means_primary = np.zeros((len(self.transmitters), len(self.sensors)))
        means_amplitute = db_2_power(self.means)
        for trans_index in range(len(self.transmitters)):
            new_means = np.zeros(len(self.sensors))
            for sen_index in range(len(self.sensors)):
                intru_pri_amplitute = means_amplitute[trans_index, sen_index]
                sen_x = self.sensors[sen_index].x
                sen_y = self.sensors[sen_index].y
                lagel_amplitude = hypothesis_legal.get((sen_x, sen_y))
                add_amplitude   = intru_pri_amplitute + lagel_amplitude
                new_means[sen_index] = add_amplitude
            self.means_primary[trans_index, :] = new_means
        self.means_primary = power_2_db(self.means_primary)


    def setup_secondary_transmitters(self, secondary_transmitter, secondary_hypo_file):
        '''Setup the secondary transmitters, then "train" the distribution of them by linearly adding up the milliwatt power
        Args:
            secondary_transmitter (list): index of legal secondary transmitters
            secondary_hypo_file (str):    filename, secondary RSS to each sensor
        '''
        print('Setting up secondary transmitters...', end=' ')
        self.secondary_trans = []                 # a mistake here, forgot to empty it
        for trans in secondary_transmitter:
            x = self.transmitters[trans].x
            y = self.transmitters[trans].y
            self.secondary_trans.append(Transmitter(x, y))

        dic_mean = {}   # (sensor.x, sensor.y) --> [legal_mean1, legal_mean2, ...]
        dic_std  = {}   # (sensor.x, sensor.y) --> []
        for sensor in self.sensors:
            dic_mean[(sensor.x, sensor.y)] = []
            dic_std[(sensor.x, sensor.y)] = sensor.std
            for secondary in self.secondary_trans:
                sec_index = self.grid_len * secondary.x + secondary.y
                dic_mean[(sensor.x, sensor.y)].append(self.means[sec_index, sensor.index])

        # y = 20*log10(x)
        # x = 10^(y/20)
        # where y is power in dB and x is the absolute value of iq samples, i.e. amplitude
        # do a addition in the absolute value of iq samples
        with open(secondary_hypo_file, 'w') as f:
            for key, value in dic_mean.items():
                amplitudes = np.power(10, np.array(value)/20)
                addition = sum(amplitudes)
                power_db = 20*np.log10(addition)
                f.write('{} {} {} {}\n'.format(key[0], key[1], power_db, dic_std[key]))


    #@profile
    def add_secondary(self, secondary_file):
        '''Add secondary's RSS to (itruder's plus primary's) RSS and save the sum to self.means_all, write to file optional
        Params:
            secondary_file (str): filename, secondary RSS to each each sensor
            all_file (str):       filename, intruder RSS + primary RSS + secondary RSS to each each sensor
            write (bool):         if True write the sums to all_file, if False don't write
        '''
        print('Adding secondary...')
        hypothesis_legal = {}
        with open(secondary_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                sen_x = int(line[0])
                sen_y = int(line[1])
                mean  = float(line[2])
                hypothesis_legal[(sen_x, sen_y)] = db_2_power(mean)

        self.means_all = np.zeros((len(self.transmitters), len(self.sensors)))
        means_primary_amplitute = db_2_power(self.means_primary)
        for trans_index in range(len(self.transmitters)):
            new_means = np.zeros(len(self.sensors))
            for sen_index in range(len(self.sensors)):
                intru_pri_amplitute = means_primary_amplitute[trans_index, sen_index]
                sen_x = self.sensors[sen_index].x
                sen_y = self.sensors[sen_index].y
                lagel_amplitude = hypothesis_legal.get((sen_x, sen_y))
                add_amplitude   = intru_pri_amplitute + lagel_amplitude
                new_means[sen_index] = add_amplitude
            self.means_all[trans_index, :] = new_means
        self.means_all = power_2_db(self.means_all)


    def rescale_intruder_hypothesis(self):
        '''Rescale hypothesis, and save it in a new np.array
        '''
        threshold = -80
        num_trans = len(self.transmitters)
        num_sen   = len(self.sensors)
        self.means_rescale = np.zeros((num_trans, num_sen))
        for i in range(num_trans):
            for j in range(num_sen):
                mean = self.means[i, j]
                mean -= threshold
                mean = mean if mean>=0 else 0
                self.means_rescale[i, j] = mean
                self.transmitters[i].mean_vec[j] = mean


    def rescale_all_hypothesis(self):
        '''Rescale hypothesis, and save it in a new np.array # TODO
        '''
        threshold = -80
        num_trans = len(self.transmitters)
        num_sen   = len(self.sensors)
        self.means_rescale = np.zeros((num_trans, num_sen))
        for i in range(num_trans):
            for j in range(num_sen):
                mean = self.means_all[i, j]
                mean -= threshold
                mean = mean if mean>=0 else 0
                self.means_rescale[i, j] = mean
                self.transmitters[i].mean_vec[j] = mean


    def update_subset(self, subset_index):
        '''Given a list of sensor indexes, which represents a subset of sensors, update self.subset
        Args:
            subset_index (list): a list of sensor indexes. guarantee sorted
        '''
        self.subset = []
        self.subset_index = subset_index
        for index in self.subset_index:
            self.subset.append(self.sensors[index])


    def update_transmitters(self):
        '''Given a subset of sensors' index,
           update each transmitter's mean vector sub and multivariate gaussian function
        '''
        for transmitter in self.transmitters:
            transmitter.set_mean_vec_sub(self.subset_index)
            new_cov = self.covariance[np.ix_(self.subset_index, self.subset_index)]
            transmitter.multivariant_gaussian = multivariate_normal(mean=transmitter.mean_vec_sub, cov=new_cov)


    def update_transmitters_loc(self):
        '''Given a subset of sensors' index,
           update each transmitter's mean vector sub and multivariate gaussian function
        '''
        for transmitter in self.transmitters_loc:
            transmitter.set_mean_vec_sub(self.subset_index)
            new_cov = self.covariance[np.ix_(self.subset_index, self.subset_index)]
            transmitter.multivariant_gaussian = multivariate_normal(mean=transmitter.mean_vec_sub, cov=new_cov)


    def update_mean_vec_sub(self, subset_index):
        '''Given a subset of sensors' index,
           update each transmitter's mean vector sub
        Args:
            subset_index (list)
        '''
        for transmitter in self.transmitters:
            transmitter.set_mean_vec_sub(subset_index)

    def update_mean_vec_sub_loc(self, subset_index):
        for transmitter in self.transmitters_loc:
            transmitter.set_mean_vec_sub(subset_index)

    def covariance_sub(self, subset_index):
        '''Given a list of index of sensors, return the sub covariance matrix
        Args:
            subset_index (list): list of index of sensors. should be sorted.
        Return:
            (np.ndarray): a 2D sub covariance matrix
        '''
        sub_cov = self.covariance[np.ix_(subset_index, subset_index)]
        return sub_cov


    #@profile
    def o_t_cpu(self, subset_index):
        '''Given a subset of sensors T, compute the O_T
        Args:
            subset_index (list): a subset of sensors T, guarantee sorted
        Return O_T
        '''
        if not subset_index:  # empty sequence are false
            return 0
        sub_cov = self.covariance_sub(subset_index)
        sub_cov_inv = np.linalg.inv(sub_cov)        # inverse
        o_t = 0
        for i in range(len(self.transmitters)):
            transmitter_i = self.transmitters[i]
            i_x, i_y = transmitter_i.x, transmitter_i.y
            transmitter_i.set_mean_vec_sub(subset_index)
            prob_i = 1
            for j in range(len(self.transmitters)):
                transmitter_j = self.transmitters[j]
                j_x, j_y = transmitter_j.x, transmitter_j.y
                if i_x == j_x and i_y == j_y:
                    continue
                transmitter_j.set_mean_vec_sub(subset_index)
                pj_pi = transmitter_j.mean_vec_sub - transmitter_i.mean_vec_sub
                prob_i *= (1 - norm.sf(0.5 * math.sqrt(np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi))))
            o_t += prob_i * self.grid_priori[i_x][i_y]
        return o_t


    def o_t_approximate(self, subset_index):
        '''Not the accurate O_T, but apprioximating O_T. So that we have a good propertiy of submodular
        Args:
            subset_index (list): a subset of sensors T, needs guarantee sorted
        Return:
            (float): the approximation of O_acc
        '''
        if not subset_index:  # empty sequence are false
            return -99999999999.
        sub_cov = self.covariance_sub(subset_index)
        sub_cov_inv = np.linalg.inv(sub_cov)         # inverse
        prob_error = 0

        for i in range(len(self.transmitters)):
            transmitter_i = self.transmitters[i]     # when the ground truth is at location i
            i_x, i_y = transmitter_i.x, transmitter_i.y
            transmitter_i.set_mean_vec_sub(subset_index)
            prob_i = 0
            for j in range(len(self.transmitters)):
                transmitter_j = self.transmitters[j] # when the classification is at location j
                j_x, j_y = transmitter_j.x, transmitter_j.y
                if i_x == j_x and i_y == j_y:        # correct classification, so no error
                    continue
                transmitter_j.set_mean_vec_sub(subset_index)
                pj_pi = transmitter_j.mean_vec_sub - transmitter_i.mean_vec_sub
                prob_i += norm.sf(0.5 * math.sqrt(np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi)))
            prob_error += prob_i * self.grid_priori[i_x][i_y]
        return 1 - prob_error


    def o_t_approximate2(self, dot_of_selected, candidate):
        '''Not the accurate O_T, but apprioximating O_T. So that we have a good propertiy of submodular
        Args:
            dot_of_selected (np.ndarray): stores the np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi)) of sensors already selected
                                          in previous iterations. shape=(m, m) where m is the number of hypothesis (grid_len^2)
            candidate (int):              a new candidate
        Return:
            (float): the approximation of O_acc
        '''
        prob_error = 0
        for i in range(len(self.transmitters)):
            transmitter_i = self.transmitters[i]     # when the ground truth is at location i
            i_x, i_y = transmitter_i.x, transmitter_i.y
            prob_i = 0
            for j in range(len(self.transmitters)):
                transmitter_j = self.transmitters[j] # when the classification is at location j
                j_x, j_y = transmitter_j.x, transmitter_j.y
                if i_x == j_x and i_y == j_y:        # correct classification, so no error
                    continue

                dot_of_candidate = ((transmitter_j.mean_vec[candidate] - transmitter_i.mean_vec[candidate]) ** 2) / self.covariance[candidate][candidate]
                dot_of_new_subset = dot_of_selected[i][j] + dot_of_candidate
                prob_i += norm.sf(0.5 * math.sqrt(dot_of_new_subset))

            prob_error += prob_i * self.grid_priori[i_x][i_y]
        return 1 - prob_error


    def update_dot_of_selected(self, dot_of_selected, best_candidate):
        '''Update dot_of_selected after a new sensor is seleted
        Args:
            dot_of_selected (np.ndarray): stores the np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi)) of sensors already selected
                                          in previous iterations. shape=(m, m) where m is the number of hypothesis (grid_len^2)
            best_candidate (int):         the best candidate just selected
        '''
        for i in range(len(self.transmitters)):
            transmitter_i = self.transmitters[i]     # when the ground truth is at location i
            i_x, i_y = transmitter_i.x, transmitter_i.y
            for j in range(len(self.transmitters)):
                transmitter_j = self.transmitters[j] # when the classification is at location j
                j_x, j_y = transmitter_j.x, transmitter_j.y
                if i_x == j_x and i_y == j_y:        # correct classification, so no error
                    continue

                dot_of_best_candidate = ((transmitter_j.mean_vec[best_candidate] - transmitter_i.mean_vec[best_candidate]) ** 2) / self.covariance[best_candidate][best_candidate]
                dot_of_selected[i][j] += dot_of_best_candidate


    def select_offline_greedy(self, budget):
        '''Select a subset of sensors greedily. offline + homo version
           The O(BS M^2 B^2) version
        Args:
            budget (int): budget constraint
        Return:
            (list): an element is [str, int, float],
                    where str is the list of subset_index, int is # of sensors, float is O_T
        '''
        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_index = [i for i in range(self.sen_num)] # S\T in the paper
        plot_data = []

        while cost < budget and complement_index:
            maximum = -9999999999                           # L in the paper
            best_candidate = complement_index[0]            # init the best candidate as the first one
            start = time.time()
            for candidate in complement_index:
                ordered_insert(subset_index, candidate)     # guarantee subset_index always be sorted here
                temp = self.o_t_approximate(subset_index)
                #print(subset_index, temp)
                if temp > maximum:
                    maximum = temp
                    best_candidate = candidate
                subset_index.remove(candidate)
            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            print('cost = {}, time = {}, o_t = {}'.format(cost+1, time.time()-start, maximum))
            complement_index.remove(best_candidate)
            plot_data.append([str(subset_index), len(subset_index), maximum])
            cost += 1

        return plot_data


    def select_offline_greedy2(self, budget):
        '''Select a subset of sensors greedily. offline + homo version
           The O(BS M^2) version
        Args:
            budget (int): budget constraint
        Return:
            (list): an element is [str, int, float],
                    where str is the list of subset_index, int is # of sensors, float is O_T
        '''
        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_index = [i for i in range(self.sen_num)] # S\T in the paper
        plot_data = []
        dot_of_selected = np.zeros((len(self.transmitters), len(self.transmitters)))

        while cost < budget and complement_index:
            maximum = -9999999999
            best_candidate = complement_index[0]            # init the best candidate as the first one
            start = time.time()
            for candidate in complement_index:
                temp = self.o_t_approximate2(dot_of_selected, candidate)
                #print(subset_index, temp)
                if temp > maximum:
                    maximum = temp
                    best_candidate = candidate
            self.update_dot_of_selected(dot_of_selected, best_candidate)
            print('cost = {}, time = {}, o_t = {}'.format(cost+1, time.time()-start, maximum))
            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            complement_index.remove(best_candidate)
            plot_data.append([str(subset_index), len(subset_index), maximum])
            cost += 1

        return plot_data


    def select_offline_greedy_p(self, budget, cores):
        '''(Parallel version) Select a subset of sensors greedily. offline + homo version using ** CPU **
           The O(BS M^2) version
        Args:
            budget (int): budget constraint
            cores (int):  number of cores for parallelzation
        Return:
            (list): an element is [str, int, float],
                    where str is the list of subset_index, int is # of sensors, float is O_T
        '''
        plot_data = []
        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_index = [i for i in range(self.sen_num)] # S\T in the paper
        subset_to_compute = []
        dot_of_selected = np.zeros((len(self.transmitters), len(self.transmitters)))

        while cost < budget and complement_index:
            start = time.time()
            candidate_result = Parallel(n_jobs=cores)(delayed(self.o_t_approximate2)(dot_of_selected, candidate) for candidate in complement_index)

            best_candidate = complement_index[0]
            maximum = candidate_result[0]
            for i in range(len(candidate_result)):
                #print(complement_index[i], candidate_result[i])
                if candidate_result[i] > maximum:
                    maximum = candidate_result[i]
                    best_candidate = complement_index[i]
            self.update_dot_of_selected(dot_of_selected, best_candidate)
            print('cost = {}, # of batch = {}, time = {}, best = {}, o_t = {}'.format(cost+1, math.ceil(len(complement_index)/cores), time.time() - start, best_candidate, maximum))

            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            complement_index.remove(best_candidate)
            cost += 1
            subset_to_compute.append(copy.deepcopy(subset_index))
            plot_data.append([len(subset_index), maximum, 0]) # don't compute real o_t now, delay to after all the subsets are selected

            if maximum > 0.999999999:
                break

        subset_results = Parallel(n_jobs=cores)(delayed(self.o_t_cpu)(subset_index) for subset_index in subset_to_compute)

        for i in range(len(subset_results)):
            plot_data[i][2] = subset_results[i]
        return plot_data


    def select_offline_greedy_p_lazy_cpu(self, budget, cores):
        '''(Parallel + Lazy greedy) Select a subset of sensors greedily. offline + homo version using ** CPU **
           The O(BS M^2) version
        Attributes:
            budget (int): budget constraint
            cores (int): number of cores for parallelzation
        Return:
            (list): an element is [str, int, float],
                    where str is the list of subset_index, int is # of sensors, float is O_T
        '''
        counter = 0
        base_ot_approx = 1 - 0.5*len(self.transmitters)
        plot_data = []
        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_sensors = copy.deepcopy(self.sensors)    # S\T in the paper
        subset_to_compute = []
        dot_of_selected = np.zeros((len(self.transmitters), len(self.transmitters)))

        while cost < budget and complement_sensors:
            best_candidate = -1
            best_sensor = None
            complement_sensors.sort()   # sorting the gain descendingly
            new_base_ot_approx = 0
            update, max_gain = 0, 0
            start = time.time()
            while update < len(complement_sensors):
                update_end = update+cores if update+cores <= len(complement_sensors) else len(complement_sensors)
                candidiate_index = []
                for i in range(update, update_end):
                    candidiate_index.append(complement_sensors[i].index)
                counter += 1

                candidate_results = Parallel(n_jobs=cores)(delayed(self.o_t_approximate2)(dot_of_selected, candidate) for candidate in candidiate_index)

                for i, j in zip(range(update, update_end), range(0, cores)):  # the two range might be different, if the case, follow the first range
                    complement_sensors[i].gain_up_bound = candidate_results[j] - base_ot_approx  # update the upper bound of gain
                    if complement_sensors[i].gain_up_bound > max_gain:
                        max_gain = complement_sensors[i].gain_up_bound
                        best_candidate = candidiate_index[j]
                        best_sensor = complement_sensors[i]
                        new_base_ot_approx = candidate_results[j]

                if update_end < len(complement_sensors) and max_gain > complement_sensors[update_end].gain_up_bound:   # where the lazy happens
                    #print('\n***LAZY!***\n', cost, (update, update_end), len(complement_sensors), '\n')
                    break
                update += cores
            self.update_dot_of_selected(dot_of_selected, best_candidate)
            print('cost = {}, time = {}, best = {}, o_t = {}'.format(cost+1, time.time()-start, best_candidate, new_base_ot_approx))
            base_ot_approx = new_base_ot_approx             # update the base o_t_approx for the next iteration
            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            subset_to_compute.append(copy.deepcopy(subset_index))
            plot_data.append([len(subset_index), base_ot_approx, 0]) # don't compute real o_t now, delay to after all the subsets are selected
            complement_sensors.remove(best_sensor)
            cost += 1
            if base_ot_approx > 0.9999999999999:
                break
        print('number of o_t_approx', counter)
        #return # for scalability test, we don't need to compute the real Ot in the scalability test.
        subset_results = Parallel(n_jobs=len(plot_data))(delayed(self.o_t_cpu)(subset_index) for subset_index in subset_to_compute)

        for i in range(len(subset_results)):
            plot_data[i][2] = subset_results[i]

        return plot_data


    #@profile
    def select_offline_greedy_lazy_gpu(self, budget, cores, cuda_kernal):
        '''(Parallel + Lazy greedy) Select a subset of sensors greedily. offline + homo version using ** GPU **
           The O(BS M^2) implementation + lookup table
        Args:
            budget (int): budget constraint
            cores (int): number of cores for parallelzation
            cuda_kernal (cuda_kernals.o_t_approx_kernal2 or o_t_approx_dist_kernal2): the O_{aux} in the paper
        Return:
            (list): an element is [str, int, float],
                    where str is the list of subset_index, int is # of sensors, float is O_T
        '''
        print('Start sensor selection...')
        start1 = time.time()
        base_ot_approx = 0
        if cuda_kernal == o_t_approx_kernal2:
            base_ot_approx = 1 - 0.5*len(self.transmitters)
        elif cuda_kernal == o_t_approx_dist_kernal2:
            largest_dist = (self.grid_len-1)*math.sqrt(2)
            max_gain_up_bound = 0.5*len(self.transmitters)*largest_dist   # the default bound is for non-distance
            for sensor in self.sensors:                                   # need to update the max gain upper bound for o_t_approx with distance
                sensor.gain_up_bound = max_gain_up_bound
            base_ot_approx = (1 - 0.5*len(self.transmitters))*largest_dist

        plot_data = []
        cost = 0                                             # |T| in the paper
        subset_index = []                                    # T   in the paper
        complement_sensors = copy.deepcopy(self.sensors)     # S\T in the paper
        subset_to_compute = []
        n_h = len(self.transmitters)                         # number of hypotheses/transmitters
        dot_of_selected   = np.zeros((n_h, n_h))
        d_dot_of_selected = cuda.to_device(dot_of_selected)  # ValueError: ctypes objects containing pointers cannot be pickled
        d_covariance      = cuda.to_device(self.covariance)  # transfer only once
        d_meanvec         = cuda.to_device(self.meanvec_array)
        d_results         = cuda.device_array(n_h*n_h, np.float64)
        d_lookup_table    = cuda.to_device(self.lookup_table_q)

        #logger = open('dataSplat/log', 'w')
        while cost < budget and complement_sensors:
            best_candidate = complement_sensors[0].index    # init as the first sensor
            best_sensor = complement_sensors[0]
            complement_sensors.sort()                       # sorting the gain descendingly
            new_base_ot_approx = 0
            max_gain = 0
            start = time.time()
            for i in range(len(complement_sensors)):
                candidate = complement_sensors[i].index

                candidate_result = self.o_t_approx_host(d_dot_of_selected, candidate, d_covariance, d_meanvec, d_results, cuda_kernal, d_lookup_table)

                #print(i, (complement_sensors[i].x, complement_sensors[i].y), candidate_result, file=logger)
                complement_sensors[i].gain_up_bound = candidate_result - base_ot_approx
                if complement_sensors[i].gain_up_bound > max_gain:
                    max_gain = complement_sensors[i].gain_up_bound
                    best_candidate = candidate
                    best_sensor = complement_sensors[i]
                    new_base_ot_approx = candidate_result

                if i+1 < len(complement_sensors) and max_gain > complement_sensors[i+1].gain_up_bound:   # where the lazy happens
                    #print('LAZY! ', cost, i, 'saves', len(complement_sensors) - i)
                    break

            self.update_dot_of_selected_host(d_dot_of_selected, best_candidate, d_covariance, d_meanvec)

            #print('cost = {}, time = {}, best = {}, ({}, {}), o_t = {}'.format(\
            #    cost+1, time.time()-start, best_candidate, best_sensor.x, best_sensor.y, new_base_ot_approx))
            #print(best_sensor.x, best_sensor.y, file=logger)
            base_ot_approx = new_base_ot_approx             # update the base o_t_approx for the next iteration
            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            subset_to_compute.append(copy.deepcopy(subset_index))
            plot_data.append([len(subset_index), base_ot_approx, 0, copy.copy(subset_index)]) # don't compute real o_t now, delay to after all the subsets are selected
            complement_sensors.remove(best_sensor)
            if base_ot_approx > 0.9999999999999:
                break
            cost += 1
        #return # test speed for pure selection
        #logger.close()
        print('Totel time of selection:', time.time() - start1)
        subset_results = Parallel(n_jobs=cores, max_nbytes=None)(delayed(self.o_t_host)(subset_index) for subset_index in subset_to_compute)

        for i in range(len(subset_results)):
            plot_data[i][2] = subset_results[i]

        return plot_data


    def select_offline_GA(self, budget, cores):
        '''Using the Ot real during selection, not submodular, no proformance guarantee
        Args:
            budget (int): budget constraint
            cores (int): number of cores for parallelzation
        Return:
            (list): an element is [str, int, float],
                    where str is the list of subset_index, int is # of sensors, float is O_T
        '''
        print('Start GA selection (homo)')
        plot_data = []
        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_index = [i for i in range(self.sen_num)] # S\T in the paper
        while cost < budget and complement_index:
            start = time.time()
            candidate_results = Parallel(n_jobs=cores, max_nbytes=None)(delayed(self.inner_greedy_real)(subset_index, candidate) for candidate in complement_index)

            best_candidate = candidate_results[0][0]   # an element of candidate_results is a tuple - (int, float, list)
            maximum = candidate_results[0][1]          # where int is the candidate, float is the O_T, list is the subset_list with new candidate
            for candidate in candidate_results:
                if candidate[1] > maximum:
                    best_candidate = candidate[0]
                    maximum = candidate[1]

            print('cost = {}, time = {}, best = {}, ({}, {}), o_t = {}'.format(\
                cost+1, time.time()-start, best_candidate, self.sensors[best_candidate].x, self.sensors[best_candidate].y, maximum))

            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            complement_index.remove(best_candidate)
            cost += 1
            plot_data.append([len(subset_index), maximum, copy.copy(subset_index)])

            if maximum > 0.999999999:
                break

        return plot_data


    def select_offline_GA_hetero(self, budget, cores):
        '''Offline selection when the sensors are heterogeneous
           Two pass method: first do a homo pass, then do a hetero pass, choose the best of the two
        Args:
            budget (int): budget we have for the heterogeneous sensors
            cores (int): number of cores for parallelization
        '''
        print('Start GA selection (hetero)')
        cost = 0                                             # |T| in the paper
        subset_index = []                                    # T   in the paper
        complement_index = [i for i in range(self.sen_num)]  # S\T in the paper
        maximum = 0
        first_pass_plot_data = []
        while cost < budget and complement_index:
            sensor_delete = []
            for index in complement_index:
                if cost + self.sensors[index].cost > budget: # over budget
                    sensor_delete.append(index)
            for sensor in sensor_delete:
                complement_index.remove(sensor)
            if not complement_index:                         # if there are no sensors that can be selected, then break
                break

            candidate_results = Parallel(n_jobs=cores, max_nbytes=None)(delayed(self.inner_greedy_real)(subset_index, candidate) for candidate in complement_index)

            best_candidate = candidate_results[0][0]   # an element of candidate_results is a tuple - (int, float, list)
            maximum = candidate_results[0][1]          # where int is the candidate, float is the O_T, list is the subset_list with new candidate
            for candidate in candidate_results:
                if candidate[1] > maximum:
                    best_candidate = candidate[0]
                    maximum = candidate[1]

            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            complement_index.remove(best_candidate)
            cost += self.sensors[best_candidate].cost
            first_pass_plot_data.append([cost, maximum, copy.copy(subset_index)])           # Y value is real o_t
            print(best_candidate, maximum, cost)

        print('end of the first homo pass and start of the second hetero pass')

        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_index = [i for i in range(self.sen_num)] # S\T in the paper
        base_ot = 0                                         # O_T from the previous iteration
        second_pass_plot_data = []
        while cost < budget and complement_index:
            sensor_delete = []
            for index in complement_index:
                if cost + self.sensors[index].cost > budget: # over budget
                    sensor_delete.append(index)
            for sensor in sensor_delete:
                complement_index.remove(sensor)
            if not complement_index:                         # if there are no sensors that can be selected, then break
                break

            candidate_results = Parallel(n_jobs=cores, max_nbytes=None)(delayed(self.inner_greedy_real)(subset_index, candidate) for candidate in complement_index)

            best_candidate = candidate_results[0][0]                       # an element of candidate_results is a tuple - (int, float, list)
            cost_of_candiate = self.sensors[best_candidate].cost
            new_base_ot = candidate_results[0][1]
            maximum = (candidate_results[0][1]-base_ot)/cost_of_candiate   # where int is the candidate, float is the O_T, list is the subset_list with new candidate
            for candidate in candidate_results:
                incre = candidate[1] - base_ot
                cost_of_candiate = self.sensors[candidate[0]].cost
                incre_cost = incre/cost_of_candiate     # increment of O_T devided by cost
                #print(candidate[2], candidate[1], incre, cost_of_candiate, incre_cost)
                if incre_cost > maximum:
                    best_candidate = candidate[0]
                    maximum = incre_cost
                    new_base_ot = candidate[1]
            base_ot = new_base_ot
            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            complement_index.remove(best_candidate)
            cost += self.sensors[best_candidate].cost
            second_pass_plot_data.append([cost, base_ot, copy.copy(subset_index)])           # Y value is real o_t
            print(best_candidate, base_ot, cost)

        if second_pass_plot_data[-1][1] > first_pass_plot_data[-1][1]:
            print('second pass is selected')
            return second_pass_plot_data
        else:
            print('first pass is selected')
            return first_pass_plot_data


    def select_offline_optimal(self, budget, cores):
        '''brute force all possible subsets in a small input such as 10 x 10 grid
        Args:
            budget (int): budget constraint
            cores  (int): number of cores for parallelzation
        Return:
            (list): an element is [int, float, list],
                    where str is int is # of sensors, float is O_T, list of subset_index
        '''
        start = time.time()
        subset_to_compute = list(combinations(range(len(self.sensors)), budget))
        print('cost = {}, # Ot = {},'.format(budget, len(subset_to_compute)), end=' ')
        results = Parallel(n_jobs=cores, max_nbytes=None)(delayed(self.o_t_host)(subset_index) for subset_index in subset_to_compute)

        results = np.array(results)
        best_subset = results.argmax()
        best_ot = results.max()
        print('time = {}, best subset = {}, best Ot = {}'.format( \
               time.time()-start, subset_to_compute[best_subset], best_ot))

        return budget, best_ot


    def inner_greedy_real(self, subset_index, candidate):
        '''Inner loop for selecting candidates of GA
        Args:
            subset_index (list):
            candidate (int):
        Return:
            (tuple): (index, o_t_approx, new subset_index)
        '''
        subset_index2 = copy.deepcopy(subset_index)
        ordered_insert(subset_index2, candidate)     # guarantee subset_index always be sorted here
        o_t = self.o_t_host(subset_index2)
        return (candidate, o_t, subset_index2)


    def select_offline_random(self, number, cores):
        '''Select a subset of sensors randomly
        Args:
            number (int): number of sensors to be randomly selected
            cores (int): number of cores for parallelization
        Return:
            (list): results to be plotted. each element is (str, int, float),
                    where str is the list of selected sensors, int is # of sensor, float is O_T
        '''
        print('Start random sensor selection (homo)')
        random.seed()
        subset_index = []
        plot_data = []
        sequence = [i for i in range(self.sen_num)]
        i = 1

        subset_to_compute = []
        while i <= number:
            select = random.choice(sequence)
            ordered_insert(subset_index, select)
            subset_to_compute.append(copy.deepcopy(subset_index))
            sequence.remove(select)
            i += 1

        results = Parallel(n_jobs=cores, max_nbytes=None)(delayed(self.o_t_host)(subset_index) for subset_index in subset_to_compute)

        for subset, result in zip(subset_to_compute, results):
            plot_data.append([len(subset), result, subset])

        return plot_data


    def select_offline_random_hetero(self, budget, cores):
        '''Offline selection when the sensors are heterogeneous
        Args:
            budget (int): budget we have for the heterogeneous sensors
            cores (int): number of cores for parallelization
        '''
        print('Start random sensor selection (hetero)')
        random.seed(0)    # though algorithm is random, the results are the same every time

        self.subset = {}
        subset_index = []
        sequence = [i for i in range(self.sen_num)]
        cost = 0
        cost_list = []
        subset_to_compute = []
        while cost < budget:
            option = []
            for index in sequence:
                temp_cost = self.sensors[index].cost
                if cost + temp_cost <= budget:  # a sensor can be selected if adding its cost is under budget
                    option.append(index)
            if not option:                      # if there are no sensors that can be selected, then break
                break
            select = random.choice(option)
            ordered_insert(subset_index, select)
            subset_to_compute.append(copy.deepcopy(subset_index))
            sequence.remove(select)
            cost += self.sensors[select].cost
            cost_list.append(cost)

        results = Parallel(n_jobs=cores, max_nbytes=None)(delayed(self.o_t_host)(subset_index) for subset_index in subset_to_compute)

        plot_data = []
        for cost, result, subset in zip(cost_list, results, subset_to_compute):
            plot_data.append([cost, result, subset])

        return plot_data


    def select_offline_coverage(self, budget, cores):
        '''A coverage-based baseline algorithm
        '''
        print('start coverage-based selection (homo)')
        random.seed(0)
        center = (int(self.grid_len/2), int(self.grid_len/2))
        min_dis = 99999
        first_index, i = 0, 0
        first_sensor = None
        for sensor in self.sensors:        # select the first sensor that is closest to the center of the grid
            temp_dis = distance.euclidean([center[0], center[1]], [sensor.x, sensor.y])
            if temp_dis < min_dis:
                min_dis = temp_dis
                first_index = i
                first_sensor = sensor
            i += 1
        subset_index = [first_index]
        subset_to_compute = [copy.deepcopy(subset_index)]
        complement_index = [i for i in range(self.sen_num)]
        complement_index.remove(first_index)

        radius = self.compute_coverage_radius(first_sensor, subset_index) # compute the radius
        print('radius', radius)
        coverage = np.zeros((self.grid_len, self.grid_len), dtype=int)
        self.add_coverage(coverage, first_sensor, radius)
        cost = 1
        while cost < budget and complement_index:  # find the sensor that has the least overlap
            least_overlap = 99999
            best_candidate = []
            best_sensor = []
            for candidate in complement_index:
                sensor = self.index_to_sensor(candidate)
                overlap = self.compute_overlap(coverage, sensor, radius)
                if overlap < least_overlap:
                    least_overlap = overlap
                    best_candidate = [candidate]
                    best_sensor = [sensor]
                elif overlap == least_overlap:
                    best_candidate.append(candidate)
                    best_sensor.append(sensor)
            choose = random.choice(range(len(best_candidate)))
            ordered_insert(subset_index, best_candidate[choose])
            complement_index.remove(best_candidate[choose])
            self.add_coverage(coverage, best_sensor[choose], radius)
            subset_to_compute.append(copy.deepcopy(subset_index))
            cost += 1

        results = Parallel(n_jobs=cores, max_nbytes=None)(delayed(self.o_t_host)(subset_index) for subset_index in subset_to_compute)

        plot_data = []
        for subset, result in zip(subset_to_compute, results):
            plot_data.append([len(subset), result, subset])

        return plot_data


    def select_offline_coverage_hetero(self, budget, cores):
        '''A coverage-based baseline algorithm (heterogeneous version)
        '''
        print('Start coverage-based sensor selection (hetero)')

        random.seed(0)

        center = (int(self.grid_len/2), int(self.grid_len/2))
        min_dis = 99999
        first_index, i = 0, 0
        first_sensor = None
        for sensor in self.sensors:        # select the first sensor that is closest to the center of the grid
            temp_dis = distance.euclidean([center[0], center[1]], [sensor.x, sensor.y])
            if temp_dis < min_dis:
                min_dis = temp_dis
                first_index = i
                first_sensor = sensor
            i += 1
        subset_index = [first_index]
        subset_to_compute = [copy.deepcopy(subset_index)]
        complement_index = [i for i in range(self.sen_num)]
        complement_index.remove(first_index)

        radius = self.compute_coverage_radius(first_sensor, subset_index) # compute the radius
        print('radius', radius)

        coverage = np.zeros((self.grid_len, self.grid_len), dtype=int)
        self.add_coverage(coverage, first_sensor, radius)
        cost = self.sensors[first_index].cost
        cost_list = [cost]

        while cost < budget and complement_index:
            option = []
            for index in complement_index:
                temp_cost = self.sensors[index].cost
                if cost + temp_cost <= budget:  # a sensor can be selected if adding its cost is under budget
                    option.append(index)
            if not option:                      # if there are no sensors that can be selected, then break
                break

            min_overlap_cost = 99999   # to minimize overlap*cost
            best_candidate = []
            best_sensor = []
            for candidate in option:
                sensor = self.index_to_sensor(candidate)
                overlap = self.compute_overlap(coverage, sensor, radius)
                temp_cost = self.sensors[candidate].cost
                overlap_cost = (overlap+0.001)*temp_cost
                if overlap_cost < min_overlap_cost:
                    min_overlap_cost = overlap_cost
                    best_candidate = [candidate]
                    best_sensor = [sensor]
                elif overlap_cost == min_overlap_cost:
                    best_candidate.append(candidate)
                    best_sensor.append(sensor)
            choose = random.choice(range(len(best_candidate)))
            ordered_insert(subset_index, best_candidate[choose])
            complement_index.remove(best_candidate[choose])
            self.add_coverage(coverage, best_sensor[choose], radius)
            subset_to_compute.append(copy.deepcopy(subset_index))
            cost += self.sensors[best_candidate[choose]].cost
            cost_list.append(cost)

        print(len(subset_to_compute), subset_to_compute)
        results = Parallel(n_jobs=cores, max_nbytes=None)(delayed(self.o_t_host)(subset_index) for subset_index in subset_to_compute)

        plot_data = []
        for cost, result, subset in zip(cost_list, results, subset_to_compute):
            plot_data.append([cost, result, subset])

        return plot_data


    def select_offline_greedy_hetero(self, budget, cores, cuda_kernal):
        '''(Lazy) Offline selection when the sensors are heterogeneous
           Two pass method: first do a homo pass, then do a hetero pass, choose the best of the two
        Args:
            budget (int): budget we have for the heterogeneous sensors
            cores (int): number of cores for parallelization
            cost_filename (str): file that has the cost of sensors
        '''
        print('Start sensor selection (hetero)')
        base_ot_approx = 1 - 0.5*len(self.transmitters)
        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_sensors = copy.deepcopy(self.sensors)    # S\T in the paper
        first_pass_plot_data = []

        n_h = len(self.transmitters)
        dot_of_selected   = np.zeros((n_h, n_h))
        d_dot_of_selected = cuda.to_device(dot_of_selected)
        d_covariance      = cuda.to_device(self.covariance)
        d_meanvec         = cuda.to_device(self.meanvec_array)
        d_results         = cuda.device_array(n_h*n_h, np.float64)
        d_lookup_table    = cuda.to_device(self.lookup_table_q)

        while cost < budget and complement_sensors:
            sensor_delete = []                  # sensors that will lead to over budget
            for sensor in complement_sensors:
                if cost + sensor.cost > budget: # over budget
                    sensor_delete.append(sensor)
            for sensor in sensor_delete:
                complement_sensors.remove(sensor)
            complement_sensors.sort()           # sort the sensors by gain upper bound descendingly
            if not complement_sensors:          # if there are no sensors that can be selected, then break
                break

            best_candidate = complement_sensors[0].index
            best_sensor = complement_sensors[0]
            new_base_ot_approx = 0
            max_gain = 0

            for i in range(len(complement_sensors)):
                candidate = complement_sensors[i].index
                candidate_result = self.o_t_approx_host(d_dot_of_selected, candidate, d_covariance, d_meanvec, d_results, cuda_kernal, d_lookup_table)

                complement_sensors[i].gain_up_bound = candidate_result - base_ot_approx
                if complement_sensors[i].gain_up_bound > max_gain:
                    max_gain = complement_sensors[i].gain_up_bound
                    best_candidate = candidate
                    best_sensor = complement_sensors[i]
                    new_base_ot_approx = candidate_result

                if i+1 < len(complement_sensors) and max_gain > complement_sensors[i+1].gain_up_bound:   # where the lazy happens
                    #print('\n******LAZY! cost, (update, update_end), len(complement_sensors)')
                    break

            self.update_dot_of_selected_host(d_dot_of_selected, best_candidate, d_covariance, d_meanvec)

            base_ot_approx = new_base_ot_approx
            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            complement_sensors.remove(best_sensor)
            cost += self.sensors[best_candidate].cost
            first_pass_plot_data.append([cost, 0, copy.copy(subset_index)])           # Y value is real o_t
            #print(best_candidate, base_ot_approx, cost)

        #print('Homo pass ends, hetero pass starts', end=' ')

        lowest_cost = 1
        for sensor in self.sensors:
            if sensor.cost < lowest_cost:
                lowest_cost = sensor.cost
        max_gain_up_bound = 0.5*len(self.transmitters)/lowest_cost
        for sensor in self.sensors:
            sensor.gain_up_bound = max_gain_up_bound

        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_sensors = copy.copy(self.sensors)        # S\T in the paper
        base_ot_approx = 1 - 0.5*len(self.transmitters)
        second_pass_plot_data = []
        dot_of_selected   = np.zeros((n_h, n_h))
        d_dot_of_selected = cuda.to_device(dot_of_selected)

        while cost < budget and complement_sensors:
            sensor_delete = []                  # sensors that will lead to over budget
            for sensor in complement_sensors:
                if cost + sensor.cost > budget: # over budget
                    sensor_delete.append(sensor)
            for sensor in sensor_delete:
                complement_sensors.remove(sensor)
            complement_sensors.sort()           # sort the sensors by gain upper bound descendingly
            if not complement_sensors:          # if there are no sensors that can be selected, then break
                break

            best_candidate = complement_sensors[0].index
            best_sensor = complement_sensors[0]
            new_base_ot_approx = 0
            max_gain = 0

            for i in range(len(complement_sensors)):
                candidate = complement_sensors[i].index
                candidate_result = self.o_t_approx_host(d_dot_of_selected, candidate, d_covariance, d_meanvec, d_results, cuda_kernal, d_lookup_table)

                complement_sensors[i].gain_up_bound = (candidate_result - base_ot_approx)/complement_sensors[i].cost  # takes cost into account
                if complement_sensors[i].gain_up_bound > max_gain:
                    max_gain = complement_sensors[i].gain_up_bound
                    best_candidate = candidate
                    best_sensor = complement_sensors[i]
                    new_base_ot_approx = candidate_result

                if i+1 < len(complement_sensors) and max_gain > complement_sensors[i+1].gain_up_bound:   # where the lazy happens
                    #print('\n******LAZY! cost, (update, update_end), len(complement_sensors)')
                    break

            self.update_dot_of_selected_host(d_dot_of_selected, best_candidate, d_covariance, d_meanvec)
            base_ot_approx = new_base_ot_approx
            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            complement_sensors.remove(best_sensor)
            cost += self.sensors[best_candidate].cost
            second_pass_plot_data.append([cost, 0, copy.copy(subset_index)])           # Y value is real o_t
            #print(best_candidate, base_ot_approx, cost)

        first_pass = []
        for data in first_pass_plot_data:
            first_pass.append(data[2])
        second_pass = []
        for data in second_pass_plot_data:
            second_pass.append(data[2])

        first_pass_o_ts = Parallel(n_jobs=cores, max_nbytes=None)(delayed(self.o_t_host)(subset_index) for subset_index in first_pass)
        second_pass_o_ts = Parallel(n_jobs=cores, max_nbytes=None)(delayed(self.o_t_host)(subset_index) for subset_index in second_pass)

        if second_pass_o_ts[-1] > first_pass_o_ts[-1]:
            print('Second pass is selected')
            for i in range(len(second_pass_o_ts)):
                second_pass_plot_data[i][1] = second_pass_o_ts[i]
            return second_pass_plot_data
        else:
            print('First pass is selected')
            for i in range(len(first_pass_o_ts)):
                first_pass_plot_data[i][1] = first_pass_o_ts[i]
            return first_pass_plot_data


    def update_battery(self, selected, energy=1):
        '''Update the battery of sensors
        Args:
            energy (int):    energy consumption amount
            selected (list): list of index of selected sensors
        '''
        for select in selected:
            self.sensors[select].update_battery(energy)


    def index_to_sensor(self, index):
        '''A temporary solution for the inappropriate data structure for self.sensors
        '''
        i = 0
        for sensor in self.sensors:
            if i == index:
                return sensor
            else:
                i += 1


    def compute_coverage_radius(self, first_sensor, subset_index):
        '''Compute the coverage radius for the coverage-based selection algorithm
        Args:
            first_sensor (tuple): sensor that is closest to the center
            subset_index (list):
        '''
        return 3
        sub_cov = self.covariance_sub(subset_index)
        sub_cov_inv = np.linalg.inv(sub_cov)        # inverse
        radius = 1
        for i in range(1, int(self.grid_len/2)):    # compute 'radius'
            transmitter_i = self.transmitters[(first_sensor.x - i)*self.grid_len + first_sensor.y] # 2D index --> 1D index
            i_x, i_y = transmitter_i.x, transmitter_i.y
            if i_x < 0:
                break
            transmitter_i.set_mean_vec_sub(subset_index)
            prob_i = []
            for transmitter_j in self.transmitters:
                j_x, j_y = transmitter_j.x, transmitter_j.y
                if i_x == j_x and i_y == j_y:
                    continue
                transmitter_j.set_mean_vec_sub(subset_index)
                pj_pi = transmitter_j.mean_vec_sub - transmitter_i.mean_vec_sub
                prob_i.append(1 - norm.sf(0.5 * math.sqrt(np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi))))
            product = 1
            for prob in prob_i:
                product *= prob
            print(i, product)
            if product > 0.00001:     # set threshold
                radius = i
            else:
                break
        return radius


    def compute_overlap(self, coverage, sensor, radius):
        '''Compute the overlap between selected sensors and the new sensor
        Args:
            coverage (2D array)
            sensor (Sensor)
            radius (int)
        '''
        x_low = sensor.x - radius if sensor.x - radius >= 0 else 0
        x_high = sensor.x + radius if sensor.x + radius <= self.grid_len-1 else self.grid_len-1
        y_low = sensor.y - radius if sensor.y - radius >= 0 else 0
        y_high = sensor.y + radius if sensor.y + radius <= self.grid_len-1 else self.grid_len-1

        overlap = 0
        for x in range(x_low, x_high+1):
            for y in range(y_low, y_high):
                if distance.euclidean([x, y], [sensor.x, sensor.y]) <= radius:
                    overlap += coverage[x][y]
        return overlap


    def add_coverage(self, coverage, sensor, radius):
        '''When seleted a sensor, add coverage by 1
        Args:
            coverage (2D array): each element is a counter for coverage
            sensor (Sensor): (x, y)
            radius (int): radius of a sensor
        '''
        x_low = sensor.x - radius if sensor.x - radius >= 0 else 0
        x_high = sensor.x + radius if sensor.x + radius <= self.grid_len-1 else self.grid_len-1
        y_low = sensor.y - radius if sensor.y - radius >= 0 else 0
        y_high = sensor.y + radius if sensor.y + radius <= self.grid_len-1 else self.grid_len-1

        for x in range(x_low, x_high+1):
            for y in range(y_low, y_high+1):
                if distance.euclidean([x, y], [sensor.x, sensor.y]) <= radius:
                    coverage[x][y] += 1



    def update_hypothesis(self, true_transmitter, subset_index):
        '''Use Bayes formula to update P(hypothesis): from prior to posterior
           After we add a new sensor and get a larger subset, the larger subset begins to observe data from true transmitter
           An important update from update_hypothesis to update_hypothesis_2 is that we are not using attribute transmitter.multivariant_gaussian. It saves money
        Args:
            true_transmitter (Transmitter)
            subset_index (list)
        '''
        true_x, true_y = true_transmitter.x, true_transmitter.y
        #np.random.seed(true_x*self.grid_len + true_y*true_y)  # change seed here
        data = []                          # the true transmitter generate some data
        for index in subset_index:
            sensor = self.sensors[index]
            mean = self.means[self.grid_len*true_x + true_y, sensor.index]
            std = self.stds[self.grid_len*true_x + true_y, sensor.index]
            data.append(np.random.normal(mean, std))
        for trans in self.transmitters:
            trans.set_mean_vec_sub(subset_index)
            cov_sub = self.covariance[np.ix_(subset_index, subset_index)]
            likelihood = multivariate_normal(mean=trans.mean_vec_sub, cov=cov_sub).pdf(data)
            print('Likelihood = ', likelihood)
            self.grid_posterior[trans.x][trans.y] = likelihood * self.grid_priori[trans.x][trans.y]
        denominator = self.grid_posterior.sum()

        try:
            #self.grid_posterior = self.grid_posterior/denominator
            self.grid_priori = copy.deepcopy(self.grid_posterior)   # the posterior in this iteration will be the prior in the next iteration
        except Exception as e:
            print(e)
            print('denominator', denominator)


    def collect_sensors_in_radius(self, size_R, sensor, given_sensors = None, num_sensors = None):
        '''Returns a subset of sensors that are within a radius of given sensor'''
        if given_sensors is None:
            given_sensors = self.sensors
        subset_sensors = []
        for cur_sensor in given_sensors:
            if (cur_sensor.x > sensor.x - size_R) and (cur_sensor.x < sensor.x + size_R) and (cur_sensor.y > sensor.y - size_R) and (cur_sensor.y < sensor.y + size_R):
                distance_euc = math.sqrt((cur_sensor.x - sensor.x)**2 + (cur_sensor.y - sensor.y)**2)
                if (distance_euc < size_R):
                    subset_sensors.append(cur_sensor.index)
        return subset_sensors


    def delete_transmitter(self, trans_pos, power, sensor_subset, sensor_outputs):
        '''Remove a transmitter and change sensor outputs accordingly'''
        trans_index = trans_pos[0] * self.grid_len + trans_pos[1]
        for sen_index in sensor_subset:
            #sensor_output = db_2_amplitude(sensor_outputs[sen_index])
            #sensor_output_from_transmitter = db_2_amplitude(self.means[trans_index, sen_index])
            #sensor_output -= sensor_output_from_transmitter
            #sensor_outputs[sen_index] = amplitude_2_db(sensor_output)
            sensor_output = db_2_power_(sensor_outputs[sen_index])
            #if sen_index == 6:
            #    print(sensor_output)
            sensor_output_from_transmitter = db_2_power_(self.means[trans_index, sen_index] + power)
            sensor_output -= sensor_output_from_transmitter
            sensor_outputs[sen_index] = power_2_db_(sensor_output)
            #if sen_index == 182:
            #    print('-', trans_pos, power, sensor_output_from_transmitter, sensor_output, sensor_outputs[sen_index])
        sensor_outputs[np.isnan(sensor_outputs)] = -120


    def set_intruders(self, true_indices, powers, randomness = False):
        '''Create intruders and return sensor outputs accordingly
        Args:
            true_indices (list): a list of integers (transmitter index)
        '''
        true_transmitters = []
        for i in true_indices:
            true_transmitters.append(self.transmitters[i])

        sensor_outputs = np.zeros(len(self.sensors))
        for i in range(len(true_transmitters)):
            tran_x = true_transmitters[i].x
            tran_y = true_transmitters[i].y
            power = powers[i]                                # varies power
            for sen_index in range(len(self.sensors)):
                if randomness:
                    dBm = db_2_power_(np.random.normal(self.means[tran_x * self.grid_len + tran_y, sen_index] + power, self.sensors[sen_index].std))
                else:
                    dBm = db_2_power_(self.means[tran_x * self.grid_len + tran_y, sen_index] + power)
                sensor_outputs[sen_index] += dBm
                #if sen_index == 182:
                #    print('+', (tran_x, tran_y), power, dBm, sensor_outputs[sen_index])
        sensor_outputs = power_2_db_(sensor_outputs)
        return (true_transmitters, sensor_outputs)


    def get_cluster_localization(self, intruders, sensor_outputs, num_of_intruders):
        '''A baseline clustering localization method
        Args:
            intruders (list): a list of integers (transmitter index)
            sensor_outputs (list): a list of float (RSSI)
            num_of_intruders:
        Return:
            (list): a list of locations [ [x1, y1], [x2, y2], ... ]
        '''
        threshold = int(0.25*len(sensor_outputs))       # threshold: instead of a specific value, it is a percentage of sensors
        arg_decrease = np.flip(np.argsort(sensor_outputs))
        threshold = sensor_outputs[arg_decrease[threshold]]
        threshold = threshold if threshold > -75 else -75
        #visualize_sensor_output(self.grid_len, intruders, sensor_outputs, self.sensors, threshold)

        sensor_to_cluster = []
        for index, output in enumerate(sensor_outputs):
            if output > threshold:
                sensor_to_cluster.append((self.sensors[index].x, self.sensors[index].y))
        k = 1
        inertias = []
        upper_bound = min(2*num_of_intruders, num_of_intruders+5)
        while k <= len(sensor_to_cluster) and k <= upper_bound:    # try all K, up to # of sensors above a threshold
            kmeans = KMeans(n_clusters=k).fit(sensor_to_cluster)
            inertias.append(kmeans.inertia_)  # inertia is the sum of squared distances of samples to their closest cluster center
            #visualize_cluster(self.grid_len, intruders, sensor_to_cluster, kmeans.labels_)
            k += 1

        k = find_elbow(inertias, num_of_intruders)              # the elbow point is the best K
        print('Real K = {}, clustered K = {}'.format(len(intruders), k))
        kmeans = KMeans(n_clusters=k).fit(sensor_to_cluster)
        #visualize_cluster(self.grid_len, intruders, sensor_to_cluster, kmeans.labels_)

        localize = kmeans.cluster_centers_
        for i in range(len(localize)):
            localize[i][0] = round(localize[i][0])
            localize[i][1] = round(localize[i][1])
        return localize


    def compute_error2(self, true_locations, pred_locations):
        '''Given the true location and localization location, computer the error
           Comparing to compute_error, this one do not include error, ** used for SPLOT and clustering **
        Args:
            true_locations (list): an element is a tuple (true transmitter 2D location)
            pred_locations (list): an element is a tuple (predicted transmitter 2D location)
        Return:
            (tuple): (distance error, miss, false alarm)
        '''
        if len(pred_locations) == 0:
            return -1, 1, 0
        distances = np.zeros((len(true_locations), len(pred_locations)))
        for i in range(len(true_locations)):
            for j in range(len(pred_locations)):
                distances[i, j] = np.sqrt((true_locations[i][0] - pred_locations[j][0]) ** 2 + (true_locations[i][1] - pred_locations[j][1]) ** 2)

        k = 0
        matches = []
        misses = list(range(len(true_locations)))
        falses = list(range(len(pred_locations)))
        while k < min(len(true_locations), len(pred_locations)):
            min_error = np.min(distances)
            min_error_index = np.argmin(distances)
            i = min_error_index // len(pred_locations)
            j = min_error_index %  len(pred_locations)
            matches.append((i, j, min_error))
            distances[i, :] = np.inf
            distances[:, j] = np.inf
            k += 1

        tot_error = 0              # distance error
        detected = 0
        threshold = self.grid_len / 5
        for match in matches:
            error = match[2]
            if error <= threshold:
                tot_error += error
                detected += 1
                misses.remove(match[0])
                falses.remove(match[1])

        print('\nPred:', end=' ')
        for match in matches:
            print(str(pred_locations[match[1]]).ljust(9), end='; ')
        print('\nTrue:', end=' ')
        for match in matches:
            print(str(true_locations[match[0]]).ljust(9), end='; ')
        print('\nMiss:', end=' ')
        for miss in misses:
            print(true_locations[miss], end=' ')
        print('\nFalse Alarm:', end=' ')
        for false in falses:
            print(pred_locations[false], end=' ')
        print()
        try:
            return tot_error / detected, (len(true_locations) - detected) / len(true_locations), (len(pred_locations) - detected) / len(true_locations)
        except:
            return 0, 0, 0


    def compute_error(self, true_locations, true_powers, pred_locations, pred_powers):
        '''Given the true location and localization location, computer the error
        Args:
            true_locations (list): an element is a tuple (true transmitter 2D location)
            true_powers (list):    an element is a float 
            pred_locations (list): an element is a tuple (predicted transmitter 2D location)
            pred_powers (list):    an element is a float
        Return:
            (tuple): (distance error, miss, false alarm, power error)
        '''
        if len(pred_locations) == 0:
            return -1, 1, 0, -1
        distances = np.zeros((len(true_locations), len(pred_locations)))
        for i in range(len(true_locations)):
            for j in range(len(pred_locations)):
                distances[i, j] = np.sqrt((true_locations[i][0] - pred_locations[j][0]) ** 2 + (true_locations[i][1] - pred_locations[j][1]) ** 2)

        k = 0
        matches = []
        misses = list(range(len(true_locations)))
        falses = list(range(len(pred_locations)))
        while k < min(len(true_locations), len(pred_locations)):
            min_error = np.min(distances)
            min_error_index = np.argmin(distances)
            i = min_error_index // len(pred_locations)
            j = min_error_index %  len(pred_locations)
            power_error = pred_powers[j] - true_powers[i]
            matches.append((i, j, min_error, power_error))
            distances[i, :] = np.inf
            distances[:, j] = np.inf
            k += 1

        tot_error = 0              # distance error
        tot_power_error = 0        # power error
        detected = 0
        threshold = self.grid_len / 5
        for match in matches:
            error = match[2]
            if error <= threshold:
                tot_error += error
                tot_power_error += match[3]
                detected += 1
                misses.remove(match[0])
                falses.remove(match[1])

        print('\nPred:', end=' ')
        for match in matches:
            print(str(pred_locations[match[1]]).ljust(9) + str(round(pred_powers[match[1]], 3)).ljust(6), end='; ')
        print('\nTrue:', end=' ')
        for match in matches:
            print(str(true_locations[match[0]]).ljust(9) + str(round(true_powers[match[0]], 3)).ljust(6), end='; ')
        print('\nMiss:', end=' ')
        for miss in misses:
            print(true_locations[miss], true_powers[miss], end=';  ')
        print('\nFalse Alarm:', end=' ')
        for false in falses:
            print(pred_locations[false], pred_powers[false], end=';  ')
        print()
        try:
            return tot_error / detected, (len(true_locations) - detected) / len(true_locations), (len(pred_locations) - detected) / len(true_locations), tot_power_error / detected
        except:
            return 0, 0, 0, 0


    def ignore_boarders(self, edge):
        '''
        Args:
            edge (int): this amount of edge is ignored at the boarders
        '''
        self.grid_priori[0:self.grid_len*edge] = self.grid_priori[-self.grid_len*edge:-1] = 0      # horizontal edge
        for i in range(edge):
            self.grid_priori[np.ix_(range(i, self.grid_len * self.grid_len, self.grid_len))] = 0  # vertical edge
            self.grid_priori[np.ix_(range(self.grid_len - 1 - i, self.grid_len * self.grid_len, self.grid_len))] = 0


    def get_q_threshold(self, radius):
        '''Different number of sensors (expected) get a different thershold
        '''
        inside = self.sen_num * 3.14159 * radius**2 / len(self.transmitters)
        outside = self.sen_num - inside
        #q = np.power(norm(0, 1).pdf(3), inside) * np.power(0.2, outside) * np.power(3., self.sen_num)
        q = np.power(norm(0, 1).pdf(2.77), inside)
        q *= np.power(0.6, outside)  # 0.6 = 0.2 x 3
        q *= np.power(3, inside)
        return q


    def get_q_threshold_custom(self, inside, radius):
        '''Different number of sensors (real) get a different thershold
        Args:
            inside (int): real number of sensors inside radius R
        Return:
            (float): the customized q threshold
        '''
        prior = 1./(3.14 * radius**2)
        outside = self.sen_num - inside
        q = np.power(norm(0, 1).pdf(2), inside) * prior  # [1.5, 2] change smaller because of change of db - power ratio function
        q *= np.power(0.6, outside)
        q *= np.power(3, inside)
        return q


    def prune_hypothesis(self, hypotheses, sensor_outputs, radius):
        '''Prune hypothesis who has less than 2 sensors with RSS > -80 in radius
        Args:
            transmitters (list): a list of candidate transmitter (hypothesis, location)
            sensor_outputs (list)
            radius (int)
        Return:
            (list): an element is a transmitter index (int)
        '''
        prunes = []
        for tran in hypotheses:
            counter = 0
            x = tran // self.grid_len
            y = tran % self.grid_len
            for sensor, output in enumerate(sensor_outputs):
                if output > -80:
                    dist = math.sqrt((x - self.sensors[sensor].x)**2 + (y - self.sensors[sensor].y)**2)
                    if dist < radius:
                        counter += 1
                if counter == 3:
                    break
            else:
                prunes.append(tran)
        for prune in prunes:
            hypotheses.remove(prune)


    def ignore_screwed_sensor(self, subset_sensors, previous_identified, min_dist):
        '''When a sensor is close to a identified intruder, it is likely to get screwed due to deleting signals with a wrong power
           Ignore the potential screwed sensor whose distance <= min_dist
        Args:
            subset_sensors (list)
            previous_identified (list): an element is a 2D index of intuder
            min_dist (int): threshold of being a likely screwed sensor
        '''
        screwed = []
        for sen in subset_sensors:
            sen_x = self.sensors[sen].x
            sen_y = self.sensors[sen].y
            for intru in previous_identified:
                dist = math.sqrt((sen_x - intru[0])**2 + (sen_y - intru[1])**2)
                if dist <= min_dist:
                    screwed.append(sen)
                    break
        for screw in screwed:
            subset_sensors.remove(screw)


    def mle_closedform(self, sensor_outputs, mean_vec, variance):
        '''Solve the vaires power issue: from discrete values to continueous values
        Args:
            sensor_outputs (np.array): sensor outputs, the data D={x1, x2, ... , xn} of MLE
            mean_vec (np.array):       the mean of the guassian distributions
            variance (np.array):       the variance of sensors (inside a circle)
        Return:
            delta_p (float):  the power
        '''
        prod = np.prod(variance)
        tmp = prod/variance
        delta_p = np.sum(tmp*(sensor_outputs - mean_vec))/(np.sum(tmp))   # closed form solution by doing derivatation on the MLE expresstion
        if delta_p > 2:
            delta_p = 2
        elif delta_p < -2:
            delta_p = -2
        return delta_p


    #@profile
    def posterior_iteration(self, hypotheses, radius, sensor_outputs, fig, previous_identified, subset_index = None):
        '''
        Args:
            hypothesis (list): an element is potential hypothesis
            radius (int): the transmission radius
            sensor_outputs (list): a list of residual RSS of each sensor
            fig (int): for plotting
            previous_identified (list): an element is a 2D index, identified intruder in previous
            subset_index (list): a list of sensor index
        Return:
            posterior (np.array): 1D array of posterior
            H_0 (bool): whether H_0 is the largest likelihood or not
            q (np.array): 2D array of Q
            power_grid (np.array): 2D array of power
        '''
        position_to_check = [(24, 42)]
        self.grid_posterior = np.zeros(self.grid_len * self.grid_len + 1)
        power_grid = np.zeros((self.grid_len, self.grid_len))
        out_prob = 0.2 # probability of sensor outside the radius
        constant = 3
        self.prune_hypothesis(hypotheses, sensor_outputs, radius)
        for trans in self.transmitters: #For each location, first collect sensors in vicinity
            if self.grid_priori[trans.x * self.grid_len + trans.y] == 0 or trans.hypothesis not in hypotheses:
                self.grid_posterior[trans.x * self.grid_len + trans.y] = 0
                continue
            if (trans.x, trans.y) in position_to_check:
                print(trans.x, trans.y)
            my_sensor = Sensor(trans.x, trans.y, 1, 1, gain_up_bound=1, index=0)
            subset_sensors = self.collect_sensors_in_radius(radius, my_sensor)
            self.ignore_screwed_sensor(subset_sensors, previous_identified, min_dist=2)
            subset_sensors = np.array(subset_sensors)
            all_sensors = np.arange(0, len(self.sensors), 1).astype(int)
            remaining_sensors = np.setdiff1d(all_sensors, subset_sensors, assume_unique=True)
            if len(subset_sensors) < 3:
                likelihood = 0
                #power_max = 0
                delta_p = 0
            else:
                sensor_outputs_copy = np.copy(sensor_outputs)  # change copy to np.array
                sensor_outputs_copy = sensor_outputs_copy[subset_sensors]
                mean_vec = np.copy(trans.mean_vec)
                mean_vec = mean_vec[subset_sensors]
                variance = np.diagonal(self.covariance)[subset_sensors]
                delta_p = self.mle_closedform(sensor_outputs_copy, mean_vec, variance)
                mean_vec = mean_vec + delta_p  # add the delta of power
                stds = np.sqrt(np.diagonal(self.covariance)[subset_sensors])
                array_of_pdfs = self.get_pdfs(mean_vec, stds, sensor_outputs_copy)
                likelihood = np.prod(array_of_pdfs)

                '''
                likelihood_max = 0
                power_max = 0

                for power in trans.powers:                       # varies power
                    sensor_outputs_copy = np.copy(sensor_outputs)
                    sensor_outputs_copy = sensor_outputs_copy[subset_sensors]
                    mean_vec = np.copy(trans.mean_vec)
                    mean_vec = mean_vec[subset_sensors] + power  # add the delta of power
                    stds = np.sqrt(np.diagonal(self.covariance)[subset_sensors])
                    array_of_pdfs = self.get_pdfs(mean_vec, stds, sensor_outputs_copy)
                    likelihood = np.prod(array_of_pdfs)
                    if likelihood > likelihood_max:
                        likelihood_max = likelihood
                        power_max = power
                    if len(np.unique(trans.powers)) == 1:        # no varying power
                        break
                likelihood = likelihood_max
                '''

            likelihood *= np.power(out_prob*constant, len(remaining_sensors)) * np.power(constant, len(subset_sensors))

            self.grid_posterior[trans.x * self.grid_len + trans.y] = likelihood * self.grid_priori[trans.x * self.grid_len + trans.y]# don't care about
            power_grid[trans.x][trans.y] = delta_p
            #power_grid[trans.x][trans.y] = power_max

        # Also check the probability of no transmitter to avoid false alarms
        mean_vec = np.full(len(sensor_outputs), -80)
        sensor_outputs_copy = copy.copy(sensor_outputs)
        sensor_outputs_copy[sensor_outputs_copy < -80] = -80
        array_of_pdfs = self.get_pdfs(mean_vec, np.sqrt(np.diagonal(self.covariance)), sensor_outputs_copy)
        likelihood = np.prod(array_of_pdfs) * np.power(2., len(self.sensors))
        self.grid_posterior[self.grid_len * self.grid_len] = likelihood * self.grid_priori[-1]
        # check if H_0's likelihood*prior is one of the largest
        if self.grid_posterior[len(self.transmitters)] == self.grid_posterior[np.argmax(self.grid_posterior)]:
            H_0 = True
        else:
            H_0 = False

        q = copy.copy(self.grid_posterior)
        #visualize_q(self.grid_len, q, fig)

        grid_posterior_copy = np.copy(self.grid_posterior)
        for trans in self.transmitters:
            if self.grid_posterior[trans.x * self.grid_len + trans.y] == 0:
                continue
            if (trans.x, trans.y) in position_to_check:
                pass#print(self.grid_posterior[trans.x * self.grid_len + trans.y])
            min_x = int(max(0, trans.x - radius))
            max_x = int(min(trans.x + radius, self.grid_len - 1))
            min_y = int(max(0, trans.y - radius))
            max_y = int(min(trans.y + radius, self.grid_len - 1))
            den = np.sum(np.array([self.grid_posterior[x * self.grid_len + y] for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)
                                                                              if math.sqrt((x-trans.x)**2 + (y-trans.y)**2) < radius]))
            grid_posterior_copy[trans.x * self.grid_len + trans.y] /= den

        grid_posterior_copy = np.nan_to_num(grid_posterior_copy)
        self.grid_posterior = grid_posterior_copy
        return self.grid_posterior, H_0, q, power_grid


    def our_localization(self, sensor_outputs, intruders, fig):
        '''Our localization, reduce R procedure 1 + procedure 2
        '''
        identified   = []
        pred_power   = []
        proc_1_count = 0
        print('Procedure 1')
        hypotheses = list(range(len(self.transmitters)))
        R_list = [8, 6, 4]
        for R in R_list:
            identified_R, pred_power_R, counter_R = self.procedure1(hypotheses, sensor_outputs, intruders, fig, R, identified)
            identified.extend(identified_R)
            pred_power.extend(pred_power_R)
        
        hypotheses = list(range(len(self.transmitters)))
        R_list = [6]
        for R in R_list:
            identified_R, pred_power_R, counter_R = self.procedure1(hypotheses, sensor_outputs, intruders, fig, R, identified)
            identified.extend(identified_R)
            pred_power.extend(pred_power_R)

        proc_1_count = len(identified)

        print('Procedure 2')
        identified2, pred_power2 = self.procedure2(sensor_outputs, intruders, fig, R=6, previous_identified=identified)
        identified.extend(identified2)
        pred_power.extend(pred_power2)

        return identified, pred_power, float(proc_1_count)/len(identified)


    def procedure2(self, sensor_outputs, intruders, fig, R, previous_identified):
        '''Our hypothesis-based localization algorithm's procedure 2
        Args:
            sensor_outputs (np.array)
            intruders (list): for plotting
            fig (int)       : for plotting
            R (int)
        Return:
            (list, list)
        '''
        visualize_sensor_output(self.grid_len, intruders, sensor_outputs, self.sensors, -80, fig)
        detected, power = [], []
        center_list = []
        center = self.get_center_sensor(sensor_outputs, R, center_list)
        combination_checked = {0}
        while center != -1:
            center_list.append(center)
            sensor_subset = self.collect_sensors_in_radius(R, self.sensors[center])  # sensor in R, hypothesis in R
            self.ignore_screwed_sensor(sensor_subset, previous_identified, min_dist=3)
            hypotheses = [h for h in range(len(self.transmitters)) \
                          if math.sqrt((self.transmitters[h].x - self.sensors[center].x)**2 + (self.transmitters[h].y - self.sensors[center].y)**2) < R ]
            for t in range(2, 4):
                print('t =', t)
                hypotheses_combination = list(combinations(hypotheses, t))
                hypotheses_combination = [x for x in hypotheses_combination if x not in combination_checked] # prevent checking the same combination again
                if len(hypotheses_combination) == 0:
                    break
                q_threshold = np.power(norm(0, 1).pdf(2), len(sensor_subset)) * (1./len(hypotheses_combination))
                combination_checked = combination_checked.union(set(hypotheses_combination))     # union of all combinations checked
                print('q-threshold = {}, inside = {}'.format(q_threshold, len(sensor_subset)))
                #posterior, H_0, Q, power = self.procedure2_iteration(hypotheses_combination, sensor_outputs, sensor_subset)
                posterior, Q = self.procedure2_iteration(hypotheses_combination, sensor_outputs, sensor_subset)
                print('combination = {}; max Q = {}; posterior = {}'.format([ (hypo//self.grid_len, hypo%self.grid_len) for hypo in \
                       hypotheses_combination[np.argmax(Q)] ], np.max(Q), np.max(posterior)))
                if np.max(Q) > q_threshold and np.max(posterior) > 0.1:  # 
                    print('** Intruder! **')
                    hypo_comb = hypotheses_combination[np.argmax(Q)]
                    for hypo in hypo_comb:
                        x = hypo//self.grid_len
                        y = hypo%self.grid_len
                        detected.append((x, y))
                        power.append(0)
                        self.delete_transmitter((x, y), 0, range(len(self.sensors)), sensor_outputs)
                    visualize_sensor_output(self.grid_len, intruders, sensor_outputs, self.sensors, -80, fig)
                    break
            center = self.get_center_sensor(sensor_outputs, R, center_list)
            center_list.append(center)
        return detected, power


    #@profile
    def procedure2_iteration(self, hypotheses_combination, sensor_outputs, sensor_subset):
        '''MAP over combinaton of hypotheses
        Args:
            hypotheses_combination (list): an element is a tuple of transmitter index, i.e. (t1, t2)
            sensor_outputs (np.array)
        Return:
            posterior (np.array): 1D array of posterior
            H_0 (bool): whether H_0 is the largest likelihood or not
            q (np.array): 2D array of Q
            power_grid (np.array): 2D array of power
        '''
        # Note try single power first
        posterior = np.zeros(len(hypotheses_combination))
        prior = 1./len(hypotheses_combination)
        for i in range(len(hypotheses_combination)):
            combination = hypotheses_combination[i]
            #if combination == (7*50+34, 9*50+32):
            #    print(combination)
            mean_vec = np.zeros(len(sensor_subset))
            for hypo in combination:
                mean_vec += db_2_power_(self.means[hypo][sensor_subset])
            mean_vec = power_2_db_(mean_vec)
            stds = np.sqrt(np.diagonal(self.covariance)[sensor_subset])
            array_of_pdfs = self.get_pdfs(mean_vec, stds, sensor_outputs[sensor_subset])
            likelihood = np.prod(array_of_pdfs)
            posterior[i] = likelihood * prior
        return posterior/np.sum(posterior), posterior  # question: is the denometer the summation of everything?


    def get_pdfs(self, mean_vec, stds, sensor_outputs):
        ''' Replace:
            norm(mean_vec, stds).pdf(sensor_outputs[sensor_subset])
            from 0.7 ms down to 0.013 ms
        '''
        sensor_outputs = np.abs((sensor_outputs - mean_vec) / stds)
        index = [i if i<390000 else 389999 for i in np.array(sensor_outputs * 10000, np.int)]
        return self.lookup_table_norm[index]


    def get_center_sensor(self, sensor_outputs, R, center_list):
        '''Check a center for procedure 2, if no centers, return -1
           Return the index of sensor with highest residual received power
        Args:
            sensor_outputs (np.array)
            R (int)
            center_list (list): a sensor cannot be center twice
        Return:
            (int)
        '''
        sensor_outputs = np.copy(sensor_outputs)
        flag = True
        while flag:
            sen_descent = np.flip(np.argsort(sensor_outputs))
            for c in sen_descent:
                if c not in center_list:
                    center = c                       # the first sensor that hasn't been a center before
                    break
            if sensor_outputs[center] < -65:         # center's RSS has to > -65
                center = -1
                flag = False
                break
            center_sensor = self.sensors[center]
            counter = 1
            for sen_index in range(len(self.sensors)):
                if sensor_outputs[sen_index] > -75:  # inaccurate residual power during deleting intruders
                    dist = math.sqrt((self.sensors[sen_index].x - center_sensor.x)**2 + (self.sensors[sen_index].y - center_sensor.y)**2)
                    if dist >=1 and dist < R:
                        counter += 1
                    if counter == 3:                 # need three "strong" sensor
                        flag = False
                        print('\ncenter =', (self.sensors[center].x, self.sensors[center].y), 'RSS =', sensor_outputs[center])
                        break
            else:
                sensor_outputs[center] = -80         # bug here ... shouldn't change the origin sensor output
        return center


    #@profile
    def procedure1(self, hypotheses, sensor_outputs, intruders, fig, radius, previous_identified):
        '''Our hypothesis-based localization algorithm's procedure 1
        Args:
            hypotheses (list): transmitters (1D index) that has 2 or more sensors in radius with RSS > -80 
            sensor_outputs (np.array)
            intruders (list): for plotting
            fig (int):        for plotting
            radius (int) 
            previous_identified (list): an element is a 2D index, identified intruder in previous
        Return:
            (list, list, int)
        '''
        num_cells = self.grid_len * self.grid_len + 1
        self.grid_priori = np.full(num_cells, 1.0 / (3.14*radius**2))  # modify priori to whatever himanshu likes
        #for intruder in previous_identified:
        #    self.grid_priori[intruder[0]*self.grid_len + intruder[1]] = 0 # set prior of previous identified location to zero
        self.ignore_boarders(edge=2)
        identified = []
        pred_power = []
        detected = True
        print('R = {}'.format(radius))
        offset = 0 #0.74 for synthetic, 0.5 for splat
        counter = 0
        while detected:
            counter += 1
            visualize_sensor_output(self.grid_len, intruders, sensor_outputs, self.sensors, -80, fig)
            detected = False
            previous_identified = list(set(previous_identified).union(set(identified)))
            posterior, H_0, Q, power = self.posterior_iteration(hypotheses, radius, sensor_outputs, fig, previous_identified)

            if H_0:
                print('H_0 is most likely')
                continue

            posterior = np.reshape(posterior[:-1], (self.grid_len, self.grid_len))
            visualize_q_prime(posterior, fig)
            indices = peak_local_max(posterior, 2, threshold_abs=0.8, exclude_border = False)  # change 2?
            sensor_subset = range(len(self.sensors))

            if len(indices) == 0:
                print("No Q' peaks...")
                continue

            for index in indices:  # 2D index
                print('detected peak =', index, "; Q' =", round(posterior[index[0]][index[1]], 3), end='; ')
                q = Q[index[0]*self.grid_len + index[1]]
                subset_sensors = self.collect_sensors_in_radius(radius, Sensor(index[0], index[1], 1))
                self.ignore_screwed_sensor(subset_sensors, previous_identified, min_dist=2)
                sen_inside = len(subset_sensors)
                q_threshold = self.get_q_threshold_custom(sen_inside, radius)
                print('Q =', q, end='; ')
                print('q-threshold = {}, inside = {}'.format(q_threshold, sen_inside), end=' ')
                if q > q_threshold:
                    print(' **Intruder!**')
                    detected = True
                    p = power[index[0]][index[1]] - offset  # deduct a power offset, 0.74 value is from 100 exeriments
                    self.delete_transmitter(index, p, sensor_subset, sensor_outputs)
                    identified.append(tuple(index))
                    pred_power.append(p)
                else:
                    #pass
                    print()
            print('---')
        return identified, pred_power, counter


    def get_confined_area(self, sensor, R):
        '''Get the confined area described in MobiCom'17
        Args:
            sensor (Sensor)
            R (int)
        Return:
            (list): an element is (int, (int, int)) -- (1D index, 2D index)
        '''
        confined_area = []
        min_x = sensor.x - R
        min_y = sensor.y - R
        for x in range(min_x, min_x + 2*R):
            if x < 0 or x >= self.grid_len:
                continue
            for y in range(min_y, min_y + 2*R):
                if y < 0 or y >= self.grid_len:
                    continue
                dist = math.sqrt((x - sensor.x)**2 + (y - sensor.y)**2)
                if dist < R:
                    confined_area.append((x, y))
        return confined_area


    def euclidean(self, location1, location2, minPL):
        '''
        Args:
            location1 (tuple) (x, y)
            location2 (tuple) (x, y)
            minPL     (float) minimum path length
        Return:
            float
        '''
        dist = np.sqrt( (location1[0] - location2[0])**2 + (location1[1] - location2[1])**2 )
        dist *= 1             #SPLAT data = 25; IPSN data = 1; representing the size of grid
        dist = minPL if dist < minPL else dist
        return dist


    def compute_path_loss(self, sensor_outputs):
        distance_vector = np.zeros(2000)
        path_loss_vector = np.zeros(2000)
        i = 0
        for trans in self.transmitters:
            for sensor in self.sensors:
                path_loss_vector[i] = 30 - sensor_outputs[sensor.index]
                distance_vector[i] = self.euclidean((sensor.x, sensor.y), (trans.x, trans.y), minPL=1)
                i += 1
                if i == 2000:
                    break
            if i == 2000:
                break
        #print(distance_vector)
        distance_vector = np.log(distance_vector)
        #power_vector = db_2_amplitude(power_vector)
        #print(np.isnan(power_vector).any())
        #print(np.isinf(power_vector).any())
        A = np.vstack([distance_vector, np.ones(len(path_loss_vector))]).T
        #print(A.shape, power_vector.shape)
        #print('Dist = ', distance_vector)
        W, n = nnls(A, path_loss_vector)[0]
        #print('Power = ', path_loss_vector, 'W = ', W, 'n = ', n)

        # import matplotlib.pyplot as plt
        # plt.plot(distance_vector, path_loss_vector, 'o', label='Original data', markersize=10)
        # plt.plot(distance_vector, W * distance_vector + n, 'r', label='Fitted line')
        #
        # plt.legend()
        # plt.show()

        return (W, n)


    def splot_localization(self, sensor_outputs, intruders, fig, R1, R2, threshold=None):
        sigma_x_square = 0.5
        delta_c        = 1
        n_p            = 2  # 2.46
        minPL          = 0.5   # For SPLOT 1.5, for Ridge and LASSO 1.0
        delta_N_square = 1     # no specification in MobiCom'17 ?
        R1             = 8
        R2             = 8     # larger R might help for ridge regression
        threshold      = -70

        visualize_sensor_output(self.grid_len, intruders, sensor_outputs, self.sensors, -80, fig)
        weight_global  = np.zeros((self.grid_len, self.grid_len))
        sensor_sorted_index = np.flip(np.argsort(sensor_outputs))

        if threshold is None:
            threshold = int(0.3*len(sensor_outputs))       # threshold: instead of a specific value, it is a percentage of sensors
            sensor_sorted_index = np.flip(np.argsort(sensor_outputs))  # decrease
            threshold = sensor_outputs[sensor_sorted_index[threshold]]
            threshold = threshold if threshold > -75 else -75
        #gradient, noise = self.compute_path_loss(sensor_outputs)
        detected_intruders = []
        sensor_outputs_copy = np.copy(sensor_outputs)
        local_maximum_list = []

        for i in range(len(sensor_outputs_copy)): #Obtain local maximum within radius size_R
            current_sensor = self.sensors[sensor_sorted_index[i]]
            current_sensor_output = sensor_outputs_copy[current_sensor.index]
            if current_sensor_output < threshold:
                continue
            sensor_subset = self.collect_sensors_in_radius(R1, current_sensor)
            local_maximum_list.append(current_sensor.index)
            for sen_num in sensor_subset:
                sensor_outputs_copy[sen_num] = -85

        #Obtained local maximum list; now compute intruder location
        detected_intruders = []
        for sen_local_max in local_maximum_list:
            sensor_subset = self.collect_sensors_in_radius(R2, self.sensors[sen_local_max])
            confined_area = self.get_confined_area(self.sensors[sen_local_max], R2)
            total_voxel = len(confined_area)   # the Q in the paper
            W_matrix = np.zeros((len(sensor_subset), total_voxel))
            for i, sen_index in enumerate(sensor_subset):
                sensor = self.sensors[sen_index]
                for q, voxel in enumerate(confined_area):
                    dist = self.euclidean((sensor.x, sensor.y), voxel, minPL)
                    W_matrix[i, q] = (dist/0.5) ** (-n_p)

            
            W_transpose = np.transpose(W_matrix)
            y = np.zeros(len(sensor_subset))
            for i in range(len(sensor_subset)):
                y[i] = db_2_power(sensor_outputs[sensor_subset[i]])
            #'''
            Cx = np.zeros((total_voxel, total_voxel))
            for j in range(total_voxel):
                voxel_j = confined_area[j]
                for l in range(total_voxel):
                    voxel_l = confined_area[l]
                    dist = self.euclidean(voxel_j, voxel_l, minPL)
                    Cx[j][l] = sigma_x_square * np.power(np.e, -dist/delta_c)
            Cx = delta_N_square * Cx

            try:
                X1 = np.matmul(W_transpose, W_matrix)
                X1 = X1 + Cx
                X2 = np.linalg.inv(X1)
            except Exception as e:
                print(e)
            X = np.matmul(X2, W_transpose)
            X = np.matmul(X, y)               # shape of X: (total_voxel, 1)
            weight_local = np.zeros((self.grid_len, self.grid_len))
            for i, x in enumerate(X):
                voxel = confined_area[i]
                weight_local[voxel[0]][voxel[1]] = x
                weight_global[voxel[0]][voxel[1]] = x
            #visualize_splot(weight_local, 'splot', str(fig)+'-'+str(self.sensors[sen_local_max].x)+'-'+str(self.sensors[sen_local_max].y))

            index = np.argmax(X)
            detected_intruders.append(confined_area[index])
            #'''

            '''
            from sklearn.linear_model import Lasso, Ridge
            linear = Ridge(alpha=0.1)
            #linear = Lasso(alpha=0.0000001)   # 0.00001 for synthetic
            linear.fit(W_matrix, y)
            X = linear.coef_
            weight_local = np.zeros((self.grid_len, self.grid_len))
            for i, x in enumerate(X):
                voxel = confined_area[i]
                weight_local[voxel[0]][voxel[1]] = x
                weight_global[voxel[0]][voxel[1]] = x
            visualize_splot(weight_local, 'splot-ridge', str(fig)+'-'+str(self.sensors[sen_local_max].x)+'-'+str(self.sensors[sen_local_max].y))
            index = np.argmax(X)
            detected_intruders.append(confined_area[index])
            '''

        #visualize_splot(weight_global, 'splot-ridge', fig)
        return detected_intruders


    def weighted_distance_priori(self, complement_index):
        '''Compute the weighted distance priori according to the priori distribution for every sensor in
           the complement index list and return the all the distances
        Args:
            complement_index (list)
        Return:
            (np.ndarray) - index
        '''
        distances = []
        for index in complement_index:
            sensor = self.sensors[index]
            weighted_distance = 0
            for transmitter in self.transmitters:
                tran_x, tran_y = transmitter.x, transmitter.y
                dist = distance.euclidean([sensor.x, sensor.y], [tran_x, tran_y])
                dist = dist if dist >= 1 else 0.5                                 # sensor very close by with high priori should be selected
                weighted_distance += 1/dist * self.grid_priori[tran_x][tran_y]    # so the metric is priori/disctance

            distances.append(weighted_distance)
        return np.array(distances)


    def transmitters_to_array(self):
        '''transform the transmitter objects to numpy array, for the sake of CUDA
        '''
        mylist = []
        for transmitter in self.transmitters:
            templist = []
            for mean in transmitter.mean_vec:
                templist.append(mean)
            mylist.append(templist)
        self.meanvec_array = np.array(mylist)  # TODO replace this with sels.means_all?


    #@profile
    def o_t_approx_host(self, d_dot_of_selected, candidate, d_covariance, d_meanvec, d_results, cuda_kernal, d_lookup_table):
        '''host code for o_t_approx.
            TYPE = "numba.cuda.cudadrv.devicearray.DeviceNDArray", which cannot be pickled --> cannot exist before using joblib
        Args:
            d_dot_of_selected (TYPE): stores the np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi)) of sensors already selected
                                      in previous iterations. shape=(m, m) where m is the number of hypothesis (grid_len^2)
            candidate (int)         : a candidate sensor index
            d_covariance (TYPE)     : covariance matrix
            d_meanvec (TYPE)        : contains the mean vector of every transmitter
            d_results (TYPE)        : save the results for each (i, j) pair of transmitter and sensor's error
            cuda_kernal (cuda_kernals.o_t_approx_kernal2 or o_t_approx_dist_kernal2)
            d_lookup_table (TYPE)   : trade space for time
        Return:
            (float): o_t_approx
        '''
        n_h = len(self.transmitters)
        threadsperblock = (self.TPB, self.TPB)
        blockspergrid_x = math.ceil(n_h/threadsperblock[0])
        blockspergrid_y = math.ceil(n_h/threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        cuda_kernal[blockspergrid, threadsperblock](d_meanvec, d_dot_of_selected, candidate, d_covariance, self.grid_priori[0][0], d_lookup_table, d_results)

        summation = sum_reduce(d_results)

        return 1 - summation


    def update_dot_of_selected_host(self, d_dot_of_selected, best_candidate, d_covariance, d_meanvec):
        '''Host code for updating dot_of_selected after a new sensor is seleted
           TYPE = "numba.cuda.cudadrv.devicearray.DeviceNDArray", which cannot be pickled --> cannot exist before using joblib
        Args:
            d_dot_of_selected (TYPE): stores the np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi)) of sensors already selected
                                      in previous iterations. shape=(m, m) where m is the number of hypothesis (grid_len^2)
            best_candidate (int)    : the best candidate sensor selected in the iteration
            d_covariance (TYPE)     : covariance matrix
            d_meanvec (TYPE)        : contains the mean vector of every transmitter
        '''
        n_h = len(self.transmitters)
        threadsperblock = (self.TPB, self.TPB)
        blockspergrid_x = math.ceil(n_h/threadsperblock[0])
        blockspergrid_y = math.ceil(n_h/threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        update_dot_of_selected_kernal[blockspergrid, threadsperblock](d_meanvec, d_dot_of_selected, best_candidate, d_covariance)


    #@profile
    def o_t_host(self, subset_index):
        '''host code for o_t.
        Args:
            subset_index (np.ndarray, n=1): index of some sensors
        '''
        n_h = len(self.transmitters)   # number of hypotheses/transmitters
        sub_cov = self.covariance_sub(subset_index)
        sub_cov_inv = np.linalg.inv(sub_cov)           # inverse
        d_meanvec_array = cuda.to_device(self.meanvec_array)
        d_subset_index = cuda.to_device(subset_index)
        d_sub_cov_inv = cuda.to_device(sub_cov_inv)
        d_results = cuda.device_array((n_h, n_h), np.float64)

        threadsperblock = (self.TPB, self.TPB)
        blockspergrid_x = math.ceil(n_h/threadsperblock[0])
        blockspergrid_y = math.ceil(n_h/threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        o_t_kernal[blockspergrid, threadsperblock](d_meanvec_array, d_subset_index, d_sub_cov_inv, d_results)

        results = d_results.copy_to_host()
        return np.sum(results.prod(axis=1)*self.grid_priori[0][0])


    def convert_to_pos(self, true_indices):
        list = []
        for index in true_indices:
            x = index // self.grid_len
            y = index % self.grid_len
            list.append((x, y))
        return list


def main3():
    '''main 3: interpolation
    '''
    selectsensor = SelectSensor(grid_len=40)
    selectsensor.init_data('dataSplat/1600-50/cov', 'dataSplat/1600-50/sensors', 'dataSplat/1600-50/hypothesis-25')
    selectsensor.interpolate_loc(scale=4, hypo_file='dataSplat/1600-50/hypothesis-25-scale-4', sensor_file='dataSplat/1600-50/sensors-scale-4')


def main1():
    '''main 1: synthetic data + SPLOT
    '''
    selectsensor = SelectSensor(grid_len=50)
    #selectsensor.init_data('data50/homogeneous-100/cov', 'data50/homogeneous-100/sensors', 'data50/homogeneous-100/hypothesis')
    selectsensor.init_data('data50/homogeneous-200/cov', 'data50/homogeneous-200/sensors', 'data50/homogeneous-200/hypothesis')
    true_powers = [-2, -1, 0, 1, 2]
    #true_powers = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    #true_powers = [0, 0, 0, 0, 0]   # no varing power
    selectsensor.vary_power(true_powers)

    repeat = 10
    errors = []
    misses = []
    false_alarms = []
    start = time.time()
    for i in range(0, repeat):
        print('\n\nTest ', i)
        random.seed(i)
        np.random.seed(i)
        true_indices, true_powers = generate_intruders(grid_len=selectsensor.grid_len, edge=2, num=5, min_dist=20, powers=true_powers)
        #true_indices, true_powers = generate_intruders_2(grid_len=selectsensor.grid_len, edge=2, min_dist=16, max_dist=5, intruders=true_indices, powers=true_powers, cluster_size=3)
        #true_indices = [x * selectsensor.grid_len + y for (x, y) in true_indices]

        intruders, sensor_outputs = selectsensor.set_intruders(true_indices=true_indices, powers=true_powers, randomness=False)

        r1 = 8
        r2 = 5
        threshold = -65
        pred_locations = selectsensor.splot_localization(sensor_outputs, intruders, fig=i, R1=r1, R2=r2, threshold=threshold)
        true_locations = selectsensor.convert_to_pos(true_indices)

        try:
            error, miss, false_alarm = selectsensor.compute_error2(true_locations, pred_locations)
            if error >= 0:
                errors.append(error)
            misses.append(miss)
            false_alarms.append(false_alarm)
            print('error/miss/false/power = {}/{}/{}'.format(error, miss, false_alarm) )
            visualize_localization(selectsensor.grid_len, true_locations, pred_locations, i)
        except Exception as e:
            print(e)

    try:
        print('(mean/max/min) error=({}/{}/{}), miss=({}/{}/{}), false_alarm=({}/{}/{})'.format(round(sum(errors)/len(errors), 3), round(max(errors), 3), round(min(errors), 3), \
              round(sum(misses)/repeat, 3), max(misses), min(misses), round(sum(false_alarms)/repeat, 3), max(false_alarms), min(false_alarms)) )
        print('Ours! time = ', time.time()-start)
    except:
        print('Empty list!')


def main2():
    '''main 2: synthetic data + Our localization
    '''
    selectsensor = SelectSensor(grid_len=50)
    #selectsensor.init_data('data50/homogeneous-100/cov', 'data50/homogeneous-100/sensors', 'data50/homogeneous-100/hypothesis')
    #selectsensor.init_data('data50/homogeneous-150-2/cov', 'data50/homogeneous-150-2/sensors', 'data50/homogeneous-150-2/hypothesis')
    #selectsensor.init_data('data50/homogeneous-156/cov', 'data50/homogeneous-156/sensors', 'data50/homogeneous-156/hypothesis')
    selectsensor.init_data('data50/homogeneous-200/cov', 'data50/homogeneous-200/sensors', 'data50/homogeneous-200/hypothesis')
    num_of_intruders = 5
    #true_powers = [-2, -1, 0, 1, 2]
    #true_powers = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    #true_powers = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # no varing power
    #selectsensor.vary_power(true_powers)
    #selectsensor.init_data('data50/homogeneous-625/cov', 'data50/homogeneous-625/sensors', 'data50/homogeneous-625/hypothesis')
    #selectsensor.init_data('data50/homogeneous-75-4/cov', 'data50/homogeneous-75-4/sensors', 'data50/homogeneous-75-4/hypothesis')

    a, b = 0, 50
    errors = []
    misses = []
    false_alarms = []
    power_errors = []
    proc_1_ratio = 0
    start = time.time()
    for i in range(a, b):
        print('\n\nTest ', i)
        random.seed(i)
        true_powers = [random.uniform(-2, 2) for i in range(num_of_intruders)]
        #print(true_powers)
        random.seed(i)
        np.random.seed(i)
        true_indices, true_powers = generate_intruders(grid_len=selectsensor.grid_len, edge=2, num=num_of_intruders, min_dist=1, powers=true_powers)
        #true_indices, true_powers = generate_intruders_2(grid_len=selectsensor.grid_len, edge=2, min_dist=16, max_dist=5, intruders=true_indices, powers=true_powers, cluster_size=3)
        #true_indices = [x * selectsensor.grid_len + y for (x, y) in true_indices]

        intruders, sensor_outputs = selectsensor.set_intruders(true_indices=true_indices, powers=true_powers, randomness=True)

        pred_locations, pred_power, ratio = selectsensor.our_localization(sensor_outputs, intruders, i)
        #pred_locations = selectsensor.get_cluster_localization(intruders, sensor_outputs)
        proc_1_ratio += ratio
        true_locations = selectsensor.convert_to_pos(true_indices)

        try:
            error, miss, false_alarm, power_error = selectsensor.compute_error(true_locations, true_powers, pred_locations, pred_power)
            if error >= 0:
                errors.append(error)
                power_errors.append(abs(power_error))
            misses.append(miss)
            false_alarms.append(false_alarm)
            print('error/miss/false/power = {}/{}/{}/{}'.format(error, miss, false_alarm, power_error) )
            visualize_localization(selectsensor.grid_len, true_locations, pred_locations, i)
        except Exception as e:
            print(e)

    try:
        print('(mean/max/min) error=({}/{}/{}), miss=({}/{}/{}), false_alarm=({}/{}/{}), power=({}/{}/{})'.format(round(sum(errors)/len(errors), 3), round(max(errors), 3), round(min(errors), 3), \
              round(sum(misses)/(b-a), 3), max(misses), min(misses), round(sum(false_alarms)/(b-a), 3), max(false_alarms), min(false_alarms), round(sum(power_errors)/len(power_errors), 3), round(max(power_errors), 3), round(min(power_errors), 3) ) )
        print('Ours! time = ', round(time.time()-start, 3), '; proc 1 ratio =', round(proc_1_ratio/(b-a), 3))
    except Exception as e:
        print(e)
    print('true power continuous, during localization continuous power, have noise')


def main4():
    '''main 4: SPLAT data + Our localization
    '''
    selectsensor = SelectSensor(grid_len=40)
    #selectsensor.init_data('dataSplat/homogeneous-100/cov', 'dataSplat/homogeneous-100/sensors', 'dataSplat/homogeneous-100/hypothesis')
    #selectsensor.init_data('dataSplat/homogeneous-150/cov', 'dataSplat/homogeneous-150/sensors', 'dataSplat/homogeneous-150/hypothesis')
    selectsensor.init_data('dataSplat/homogeneous-200/cov', 'dataSplat/homogeneous-200/sensors', 'dataSplat/homogeneous-200/hypothesis')
    #selectsensor.init_data('dataSplat/homogeneous-250/cov', 'dataSplat/homogeneous-250/sensors', 'dataSplat/homogeneous-250/hypothesis')
    #selectsensor.init_data('dataSplat/homogeneous-300/cov', 'dataSplat/homogeneous-300/sensors', 'dataSplat/homogeneous-300/hypothesis')
    #true_powers = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]) * 2     # 10 intruders
    #true_powers = [-2, -1.4, -0.8, -0.2, 0.4, 1, 1.6]                                  # 7 intruders
    true_powers = [-2, -1, 0, 1, 2]                                                    # 5 intruders
    #true_powers = [-2, 0, 2]                                                           # 3 intruders
    #true_powers = [0]                                                                  # 1 intruders
    #true_powers = [0, 0, 0, 0, 0]   # no varing power
    selectsensor.vary_power(true_powers)

    repeat = 1
    errors = []
    misses = []
    false_alarms = []
    power_errors = []
    proc_1_ratio = 0
    start = time.time()
    for i in range(0, repeat):
        print('\n\nTest ', i)
        random.seed(i)
        np.random.seed(i)
        true_indices, true_powers = generate_intruders(grid_len=selectsensor.grid_len, edge=2, num=5, min_dist=1, powers=true_powers)
        #true_indices, true_powers = generate_intruders_2(grid_len=selectsensor.grid_len, edge=2, min_dist=16, max_dist=5, intruders=true_indices, powers=true_powers, cluster_size=3)
        #true_indices = [x * selectsensor.grid_len + y for (x, y) in true_indices]

        intruders, sensor_outputs = selectsensor.set_intruders(true_indices=true_indices, powers=true_powers, randomness=True)

        pred_locations, pred_power, ratio = selectsensor.our_localization(sensor_outputs, intruders, i)
        #pred_locations = selectsensor.get_cluster_localization(intruders, sensor_outputs)
        proc_1_ratio += ratio
        true_locations = selectsensor.convert_to_pos(true_indices)

        try:
            error, miss, false_alarm, power_error = selectsensor.compute_error(true_locations, true_powers, pred_locations, pred_power)
            if error >= 0:
                errors.append(error)
                power_errors.append(abs(power_error))
            misses.append(miss)
            false_alarms.append(false_alarm)
            print('error/miss/false/power = {}/{}/{}/{}'.format(error, miss, false_alarm, power_error) )
            visualize_localization(selectsensor.grid_len, true_locations, pred_locations, i)
        except Exception as e:
            print(e)

    try:
        print('(mean/max/min) error=({}/{}/{}), miss=({}/{}/{}), false_alarm=({}/{}/{}), power=({}/{}/{})'.format(round(sum(errors)/len(errors), 3), round(max(errors), 3), round(min(errors), 3), \
              round(sum(misses)/repeat, 3), max(misses), min(misses), round(sum(false_alarms)/repeat, 3), max(false_alarms), min(false_alarms), round(sum(power_errors)/len(power_errors), 3), round(max(power_errors), 3), round(min(power_errors), 3) ) )
        print('Ours! time = ', round(time.time()-start, 3), '; proc 1 ratio =', round(proc_1_ratio/repeat, 3))
    except:
        print('Empty list!')
    
    print('300 sensors')


def main5():
    '''main 5: SPLAT data + SPLOT localization
    '''
    selectsensor = SelectSensor(grid_len=40)
    #selectsensor.init_data('dataSplat/homogeneous-100/cov', 'dataSplat/homogeneous-100/sensors', 'dataSplat/homogeneous-100/hypothesis')
    #selectsensor.init_data('dataSplat/homogeneous-150/cov', 'dataSplat/homogeneous-150/sensors', 'dataSplat/homogeneous-150/hypothesis')
    #selectsensor.init_data('dataSplat/homogeneous-200/cov', 'dataSplat/homogeneous-200/sensors', 'dataSplat/homogeneous-200/hypothesis')
    #selectsensor.init_data('dataSplat/homogeneous-250/cov', 'dataSplat/homogeneous-250/sensors', 'dataSplat/homogeneous-250/hypothesis')
    selectsensor.init_data('dataSplat/homogeneous-300/cov', 'dataSplat/homogeneous-300/sensors', 'dataSplat/homogeneous-300/hypothesis')
    true_powers = [-2, -1, 0, 1, 2]                                                    # 5 intruders
    #true_powers = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]) * 2    # 10 intruders
    #true_powers = [-2, -1.4, -0.8, -0.2, 0.4, 1, 1.6]                                  # 7 intruders
    #true_powers = [-2, 0, 2]                                                           # 3 intruders
    #true_powers = [0]                                                                   # 1 intruders
    #true_powers = [0, 0, 0, 0, 0]   # no varing power
    selectsensor.vary_power(true_powers)

    repeat = 30
    errors = []
    misses = []
    false_alarms = []
    start = time.time()
    for i in range(0, repeat):
        print('\n\nTest ', i)
        random.seed(i)
        np.random.seed(i)
        true_indices, true_powers = generate_intruders(grid_len=selectsensor.grid_len, edge=2, num=5, min_dist=1, powers=true_powers)
        #true_indices, true_powers = generate_intruders_2(grid_len=selectsensor.grid_len, edge=2, min_dist=16, max_dist=5, intruders=true_indices, powers=true_powers, cluster_size=3)
        #true_indices = [x * selectsensor.grid_len + y for (x, y) in true_indices]

        intruders, sensor_outputs = selectsensor.set_intruders(true_indices=true_indices, powers=true_powers, randomness=False)

        r1 = 8
        r2 = 5
        threshold = -65
        pred_locations = selectsensor.splot_localization(sensor_outputs, intruders, fig=i, R1=r1, R2=r2, threshold=threshold)
        true_locations = selectsensor.convert_to_pos(true_indices)

        try:
            error, miss, false_alarm = selectsensor.compute_error2(true_locations, pred_locations)
            if error >= 0:
                errors.append(error)
            misses.append(miss)
            false_alarms.append(false_alarm)
            print('error/miss/false/power = {}/{}/{}'.format(error, miss, false_alarm) )
            visualize_localization(selectsensor.grid_len, true_locations, pred_locations, i)
        except Exception as e:
            print(e)

    try:
        print('(mean/max/min) error=({}/{}/{}), miss=({}/{}/{}), false_alarm=({}/{}/{})'.format(round(sum(errors)/len(errors), 3), round(max(errors), 3), round(min(errors), 3), \
              round(sum(misses)/repeat, 3), max(misses), min(misses), round(sum(false_alarms)/repeat, 3), max(false_alarms), min(false_alarms)) )
        print('Ours! time = ', time.time()-start)
    except:
        print('Empty list!')



if __name__ == '__main__':
    #main1()
    main2()
    #main3()
    #main4()
    #main5()
