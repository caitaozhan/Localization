'''
Some useful utilities
'''

import os
import json
import numpy as np
import pandas as pd
import random
import scipy
import math
import shutil
import matplotlib.pyplot as plt


def read_config(filename):
    '''Read json file into json object
    '''
    with open(filename, 'r') as handle:
        dictdump = json.loads(handle.read(), encoding='utf-8')
    return dictdump


def ordered_insert(sensor_index, index):
    '''Insert index into sensor_index and guarantee sensor_index is sorted from small to large.
    Attributes:
        sensor_index (list)
        index (int)
    '''
    size = len(sensor_index)
    for i in range(size):
        if index < sensor_index[i]:
            sensor_index.insert(i, index)
            break
    else:
        sensor_index.insert(size, index)


def print_results(results):
    '''print the results array copied from device
    '''
    for i in range(results.shape[0]):
        for j in range(results[i].shape[0]):
            print(results[i, j], end=' ')
        print()


def power_2_db(abso):
    '''Transform the power (dBm) into dB
       y = 10*log10(x)
       where y is dB and x is the power (dBm)
    '''
    try:
        val = 10*np.log10(abso)
    except:
        pass
    try:
        val[np.isnan(val)] = -80
        val[val < -80] = -80
    except:
        if val < -80 or val is np.nan:
            val = -80
    return val


def db_2_power(db):
    '''Transform dB into power (dBm)
       x = 10^(y/10)
       where y is power and x is the absolute value of iq samples
    '''
    try:
        if db <= -80:
            return 0
    except:
        pass
    val = np.power(10, np.array(db)/10)
    try:
        val[val <= 0.00000001] = 0
    except:
        if val <= 0.00000001:  # noise floor
            val = 0
    return val


def power_2_db_(value, utah=False):
    '''Transform the power of signals into decibal signal strength
       y = 20*log10(x)
    '''
    if utah == False:
        return 10*np.log10(value)  # here value is power
    else:
        return 20*np.log10(value)  # here value is amplitude (for utah data)


def db_2_power_(db, utah=False):
    '''Transform the decibal signal strength into power
       x = 10^(y/20)
    '''
    if utah == False:
        return np.power(10, np.array(db)/10)  # returning power
    else:
        return np.power(10, np.array(db)/20)  # returning amplitude (for utah data)


def find_elbow(inertias, num_intruder):
    '''Find the elbow point of kmeans clustering, to determine the K
    Args:
        inertias (list): a list of float
    Return:
        (int): the K
    '''
    #deltas = []
    #for i in range(len(inertias)-1):
    #    deltas.append(inertias[i] - inertias[i+1])
    #if not deltas:  # there is only one inertia
    #    return 1

    #print('ratio1 = ', deltas[9]/inertias[0])
    #print('ratio2 = ', inertias[9]/inertias[0])


    '''
    param = { # lognormal
        1:1.1,
        3:0.15,
        5:0.06,
        7:0.04,
        10:0.03,
    }
    '''
    param = {  # splat
        1:0.5,
        3:0.15,
        5:0.06,
        7:0.05,
        10:0.03,
    }
    i = 0
    while i < len(inertias):
        if inertias[i] < param[num_intruder]*inertias[0] or inertias[i] < 5: # after elbow point: slope is small
            break      # 0.5 for {1, 2}
        i += 1         # 0.15 for {}
    return i+1# 0.0176 for 10~20


def random_secondary():
    '''Generate some random secondary from a pool
    Return:
        (list): a subset from a pool total_secondary
    '''
    total_secondary = [149, 456, 590, 789, 889, 999, 1248, 1500]
    num = random.randint(1, 4)
    secondary = random.sample(total_secondary, num)
    secondary = [149, 1248]
    print('The secondary are: ', secondary)
    return secondary


def random_intruder(grid_len):
    '''Generate some random intruder
    Return:
        (list): a list of integers (transmitter index)
    '''
    intruders = list(range(grid_len*grid_len))
    num = random.randint(3, 4)
    intruder = random.sample(intruders, num)
    print('The new intruder are: ', intruder)
    return intruder


def get_distance(grid_len, index1, index2):
    '''Convert 1D index to 2D index, then return distance of the 2D index
    '''
    x1, y1 = index1//grid_len, index1%grid_len
    x2, y2 = index2//grid_len, index2%grid_len
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


def generate_intruders_utah(grid_len, locations, lt, num, min_dist=0, max_dist=20):
    '''
    Args:
        locations (np.ndarray, n=2): the 44 real number locations
        num       (int): number of intruders
        min_dist  (int): minimum distance between intruders
    Return:
        list<(float, float)>: a list of real number locations
        list<int>           : a list of 1D index of grid locations
    '''
    intruders_real = []
    intruders_grid = []
    counter   = 0
    trials    = 0
    num_loc   = len(locations)
    while counter < num:
        trials += 1
        if trials < 1000:
            tmp = random.sample(range(num_loc), 1)[0]
            x, y = locations[tmp]
            for intruder in intruders_real:
                dist = math.sqrt((x - intruder[0])**2 + (y - intruder[1])**2)
                if dist < min_dist or dist > max_dist:
                    break
            else:
                intruders_real.append(locations[tmp])
                cell = lt.real_2_gridcell(locations[tmp])
                intruders_grid.append(cell[0]*grid_len + cell[1])
                counter += 1
        else:
            intruders_real, intruders_grid = generate_intruders_utah(grid_len, locations, lt, num, min_dist)
    return intruders_real, intruders_grid


def set_intruders_utah(true_indices, powers, means, grid_loc, ll, randomness=False):
    '''
    '''
    true_transmitters = []
    for i in true_indices:
        true_transmitters.append(ll.transmitters[i])
    sensor_outputs = np.zeros(len(ll.sensors))
    for i in range(len(true_transmitters)):
        tran_x = true_transmitters[i].x
        tran_y = true_transmitters[i].y
        indx = 0
        for j, loc in enumerate(grid_loc):
            if (tran_x, tran_y) == tuple(loc):
                indx = j
                break
        power = powers[i]                                # varies power
        for sen_index in range(len(ll.sensors)):
            if randomness:
                dBm = db_2_power_(np.random.normal(means[indx, sen_index] + power, ll.sensors[sen_index].std), utah=ll.utah)
            else:
                dBm = db_2_power_(means[indx, sen_index] + power, utah=ll.utah)
            sensor_outputs[sen_index] += dBm
            #if sen_index == 182:
            #    print('+', (tran_x, tran_y), power, dBm, sensor_outputs[sen_index])
    sensor_outputs = power_2_db_(sensor_outputs, utah=ll.utah)
    return (true_transmitters, sensor_outputs)


def generate_intruders(grid_len, edge, num, min_dist, powers):
    '''
    Args:
        grid_len (int):
        edge (int): intruders cannot be at the edge
        num (int): number of intruders
        min_dist (int): minimum distance between intruders
        powers (list): an element is float, denoting power
    Return:
        (list): a list of 1D index
    '''
    #random.shuffle(powers)
    #powers = [np.random.choice(powers) for _ in range(num)]

    intruders = []
    counter = 0
    trials = 0
    while counter < num:
        trials += 1
        if trials < 1000:           # could get stuck ... when everywhere is covered according to the min_dist constraint
            tmp = random.sample(range(grid_len * grid_len), 1)
            tmp = tmp[0]
            x = tmp//grid_len
            y = tmp%grid_len
            if x < edge or x >= grid_len-edge:
                continue
            if y < edge or y >= grid_len-edge:
                continue
            flag = True
            for intruder in intruders:
                dist = get_distance(grid_len, intruder, tmp)
                if dist < min_dist:  # new location violates the min distance constraint
                    flag = False
                    break
            if flag:
                intruders.append(tmp)
                counter += 1
        else:
            #print('Oops!')
            intruders, powers = generate_intruders(grid_len, edge, num, min_dist, powers) # when stuck, just restart again
            break
    return intruders, powers


def generate_intruders_2(grid_len, edge, min_dist, max_dist, intruders, powers, cluster_size):
    '''generate intruders for testing procedure 2
    Args:
        grid_len (int):
        edge (int): intruders cannot be at the edge
        min_dist (int): minimum distance between intruders in different cluster
        max_dist (int): maximum distance between intruders in the same cluster
        powers (list): an element is float, denoting power
    Return:
        (list): a list of 1D index
    '''
    counter = 0
    num = len(intruders)
    new_intruders = []
    size = 1                              # size of a cluster of intruders
    while counter < num:
        c_intruder = intruders[counter]   # a cluster center
        c_x = c_intruder//grid_len
        c_y = c_intruder%grid_len
        trial = 0
        if trial < 100:
            trial += 1
            rand = np.random.randint(low=-max_dist, high=max_dist+1, size=2)
            x = c_x + rand[0]
            y = c_y + rand[1]
            if x < edge or x >= grid_len-edge:
                continue
            if y < edge or y >= grid_len-edge:
                continue
            tmp = x*grid_len + y
            if get_distance(grid_len, c_intruder, tmp) >= max_dist:
                continue
            for intruder in intruders:
                if intruder != c_intruder and get_distance(grid_len, intruder, tmp) < min_dist:
                    break
            else:
                size += 1
                new_intruders.append(tmp)
                if size == cluster_size:
                    counter += 1
                    size = 1
                    continue
        else:
            print('Oooops!')
            new_intruders, powers = generate_intruders_2(grid_len, edge, min_dist, max_dist, intruders, powers, cluster_size)
            break
    intruders.extend(new_intruders)
    return intruders, powers


def generate_intruders_real():
    pass # TODO


def read_utah_data(path='dataUtah'):
    '''Read utah data
    Return:
        mean (np.ndarray, n=2)
        stds (np.ndarray, n=1)
        locations (np.ndarray, n=2)
    '''
    savedSig = scipy.io.loadmat(path + '/savedSig/savedSig.mat')
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
    
    locations = scipy.io.loadmat(path + '/savedSig/deviceLocs.mat')
    locations = locations['deviceLocs']
    locations = np.array(locations)

    with open(path + '/wall.txt', 'r') as f:
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

    return mean, np.mean(stds, 0), locations, wall


def distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def plot_cdf(x):
    x, y = sorted(x), np.arange(len(x)) / len(x)
    plt.plot(x, y, linewidth=3)
    plt.show()


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
            (int)
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


def my_local_max(posterior, radius, threshold_abs):
    '''skimage.feature.peak_local_max()'s behavior is strange, not what I expected.
    Args:
        posterior -- np.ndrray, n=2
        radius    -- float
        threshold_abs -- float
    Return:
        list<tuple<int, int>> -- a list of 2D index of the local maximum
    '''
    local_max = []
    size = len(posterior)
    for i in range(size):
        for j in range(size):
            if posterior[i, j] < threshold_abs:
                continue
            is_local_max = True
            for x in range(i-int(radius), i+int(radius)+1):
                for y in range(j-int(radius), j+int(radius)+1):
                    if x >= 0 and y >= 0 and x < size and y < size and distance((x, y), (i, j)) < radius:
                        if posterior[x][y] > posterior[i][j]:
                            is_local_max = False
            if is_local_max:
                local_max.append((i, j))
    return local_max


def guarantee_dir(directory):
    '''Gurantee that a directory exists
    Args:
        directory -- str
    '''
    if os.path.exists(directory) is False:
        os.mkdir(directory)


def clean_itwom(itwom, fspl):
    '''itwom has strange pathloss. eg. pathloss = 0 when distance between tx and rx is small (0 ~ 200m)
       Use fspl to replace the strange fspl
    Args:
        itwom -- np.1darray
        fslp  -- np.1darray
    '''
    if len(itwom) != len(fspl):
        print('itwom and fslp length not equal')
        return

    for i in range(len(itwom)):
        if itwom[i] <= 0.:
            itwom[i] = fspl[i]


def relocate_sensors(random_sensors, grid_len):
    '''Relocate sensors that are side by side
    '''
    new_random_sensors = []
    need_to_relocate = []
    ocupy_grid = np.zeros((grid_len, grid_len), dtype=int)
    random_sensors.sort()
    for sen in random_sensors:
        s_x = sen // grid_len
        s_y = sen % grid_len
        if ocupy_grid[s_x][s_y] == 1:
            need_to_relocate.append(sen)
        else:
            new_random_sensors.append(sen)
            for x, y in [(0, 0), (-1, 0), (0, -1), (0, 1), (1, 0)]:
                try:
                    ocupy_grid[s_x + x][s_y + y] = 1
                except:
                    pass
    available = []
    for x in range(grid_len):
        for y in range(grid_len):
            if ocupy_grid[x][y] == 0:
                available.append(x*grid_len + y)

    relocated = random.sample(available, len(need_to_relocate))
    new_random_sensors.extend(relocated)
    return new_random_sensors


def subsample_from_full(train, grid_len, sensor_density, transmit_power):
    random.seed(sensor_density)
    s = train.cov
    full_training_dir = s[:s.find('_')]
    subsample_dir     = s[:s.rfind('/')]
    sen_density       = s[s.find('_')+1:s.rfind('/')]
    guarantee_dir(subsample_dir)

    # Phase 1: subsample the interpolated data
    # step 1: get a subset of sensors
    all_sensors = list(range(grid_len * grid_len))
    random_sensors_subset = random.sample(all_sensors, sensor_density)
    random_sensors_subset = relocate_sensors(random_sensors_subset, grid_len)
    random_sensors_subset = relocate_sensors(random_sensors_subset, grid_len)
    random_sensors_subset.sort()
    with open(full_training_dir + '/sensors', 'r') as full_f, open(train.sensors, 'w') as sub_f:
        all_lines = full_f.readlines()
        for sen_index in random_sensors_subset:
            sub_line = all_lines[sen_index]
            sub_f.write(sub_line)

    # step 2: deal with the subsample cov
    cov = pd.read_csv(full_training_dir + '/cov', header=None, delimiter=' ')
    cov = cov.values
    cov = cov[np.ix_(random_sensors_subset, random_sensors_subset)]
    np.savetxt(subsample_dir + '/cov', cov, delimiter='', fmt='%1.3f ')

    # step 3: deal with hypothesis file
    power = transmit_power["T1"]
    with open(full_training_dir + '/hypothesis', 'r') as full_f, open(subsample_dir + '/hypothesis', 'w') as sub_f:
        lines = full_f.readlines()
        num_all_sensors = len(all_sensors)
        for i in range(num_all_sensors):
            for sen_index in random_sensors_subset:
                index = i*num_all_sensors + sen_index
                # print(index, lines[index])
                line = lines[index].split(' ')
                pathloss = float(line[4])
                rss = power - pathloss
                line[4] = str(rss)
                sub_f.write(' '.join(line))

    with open(subsample_dir + '/train_power', 'w') as f:
        f.write(json.dumps(transmit_power))

    with open(subsample_dir + '/sensors') as f_sen, open(subsample_dir + '/hostname_loc', 'w') as f_hl:
        index = 0   # index is the hostname
        for line in f_sen:
            line = line.split(' ')
            x, y = line[0], line[1]
            f_hl.write('{}: ({}, {})\n'.format(index, x, y))
            index += 1

    # Phase 2: subsample the full data
    full_truth_training_dir = '../mysplat/output8'
    sub_truth_training_dir  = full_truth_training_dir + '_{}'.format(sen_density)
    # if os.path.exists(sub_truth_training_dir) is True:
    #     return
    guarantee_dir(sub_truth_training_dir)
    # step 1: use the same set of sensors
    shutil.copy(subsample_dir + '/sensors', sub_truth_training_dir + '/sensors')

    # step 2: use the same cov
    shutil.copy(subsample_dir + '/cov', sub_truth_training_dir + '/cov')

    # step 3: hypothesis file
    with open(sub_truth_training_dir + '/hypothesis', 'w') as f:
        for hypo in range(grid_len*grid_len):
            t_x = hypo // grid_len
            t_y = hypo % grid_len
            hypo4 = '{:04}'.format(hypo)
            output = np.loadtxt(full_truth_training_dir + '/{}'.format(hypo4), delimiter=',')
            fspl, itwom = output[0], output[1]
            clean_itwom(itwom, fspl)
            for sen_1dindex in random_sensors_subset:
                s_x = sen_1dindex // grid_len
                s_y = sen_1dindex % grid_len
                rss = power - itwom[sen_1dindex]
                std = 1
                f.write('{} {} {} {} {:.2f} {}\n'.format(t_x, t_y, s_x, s_y, rss, std))


if __name__ == '__main__':
    for _ in range(20):
        print('intruders = ', generate_intruders(grid_len=50, edge=2, num=6, min_dist=20, powers=[-4, -2, 0, 2, 4]))
