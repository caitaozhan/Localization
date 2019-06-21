'''
Some useful utilities
'''
import json
import numpy as np
import random
import scipy


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


def power_2_db_(abso):
    '''Transform the decibal signal strength into absolute value of iq samples
       y = 20*log10(x)
       where y is power in dB and x is the absolute value of iq samples
    '''
    return 10*np.log10(abso)


def db_2_power_(db):
    '''Transform the decibal signal strength into absolute value of iq samples
       x = 10^(y/20)
       where y is power in dB and x is the absolute value of iq samples
    '''
    return np.power(10, np.array(db)/10)


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
        1:1.1,
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
    for _ in range(20):
        print('intruders = ', generate_intruders(grid_len=50, edge=2, num=6, min_dist=20, powers=[-4, -2, 0, 2, 4]))

