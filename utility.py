'''
Some useful utilities
'''
import json
import numpy as np
import random

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


def amplitude_2_db(abso):
    '''Transform the decibal signal strength into absolute value of iq samples
       y = 20*log10(x)
       where y is power in dB and x is the absolute value of iq samples
    '''
    try:
        val = 20*np.log10(abso)
    except:
        pass
    try:
        val[np.isnan(val)] = -80
        val[val < -80] = -80
    except:
        if val < -80 or val is np.nan:
            val = -80
    return val


def db_2_amplitude(db):
    '''Transform the decibal signal strength into absolute value of iq samples
       x = 10^(y/20)
       where y is power in dB and x is the absolute value of iq samples
    '''
    try:
        if db <= -80:
            return 0
    except:
        pass
    val = np.power(10, np.array(db)/20)
    try:
        val[val <= 0.0001] = 0
    except:
        if val <= 0.0001:  # noise floor
            val = 0
    return val


def amplitude_2_db_(abso):
    '''Transform the decibal signal strength into absolute value of iq samples
       y = 20*log10(x)
       where y is power in dB and x is the absolute value of iq samples
    '''
    return 20*np.log10(abso)


def db_2_amplitude_(db):
    '''Transform the decibal signal strength into absolute value of iq samples
       x = 10^(y/20)
       where y is power in dB and x is the absolute value of iq samples
    '''
    return np.power(10, np.array(db)/20)


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

    param ={1:0.5,
            2:0.5,
            4:0.15,
            8:0.08,
            16:0.02,
            24:0.006,
            30:0.003
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
    '''
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
    Return:
        (list): a list of 1D index
    '''
    np.random.shuffle(powers)
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


if __name__ == '__main__':
    for _ in range(20):
        print('intruders = ', generate_intruders(grid_len=50, edge=2, num=6, min_dist=20, powers=[-4, -2, 0, 2, 4]))

