import numpy as np
import pandas as pd
import sys
import time
import random
sys.path.append('../..')
from localize import Localization
from utility_ipsn import generate_intruders


# SelectSensor version on May 31, 2019
ll = Localization(grid_len=50)
ll.init_data('../../data50/homogeneous-200/cov', '../../data50/homogeneous-200/sensors', '../../data50/homogeneous-200/hypothesis')

num_of_intruders = 1

a, b = 0, 50
errors = []
misses = []
false_alarms = []
power_errors = []
start = time.time()
for i in range(a, b):
    print('\n\nTest ', i)
    random.seed(i)
    true_powers = [random.uniform(-2, 2) for i in range(num_of_intruders)]
    random.seed(i)
    np.random.seed(i)
    true_indices, true_powers = generate_intruders(grid_len=ll.grid_len, edge=2, num=num_of_intruders, min_dist=1, powers=true_powers)
    intruders, sensor_outputs = ll.set_intruders(true_indices=true_indices, powers=true_powers, randomness=True)

    pred_locations = ll.cluster_localization(intruders, sensor_outputs, num_of_intruders)

    true_locations = ll.convert_to_pos(true_indices)

    try:
        error, miss, false_alarm = ll.compute_error2(true_locations, pred_locations)
        if len(error) != 0:
            errors.extend(error)
        misses.append(miss)
        false_alarms.append(false_alarm)
        print('error/miss/false/power = {}/{}/{}'.format(np.array(error).mean(), miss, false_alarm) )
    except Exception as e:
        print(e)

try:
    errors = np.array(errors)
    power_errors = np.array(power_errors)
    np.savetxt('{}-cluster-error.txt'.format(num_of_intruders), errors, delimiter=',')
    np.savetxt('{}-cluster-miss.txt'.format(num_of_intruders), misses, delimiter=',')
    np.savetxt('{}-cluster-false.txt'.format(num_of_intruders), false_alarms, delimiter=',')
    np.savetxt('{}-cluster-time.txt'.format(num_of_intruders), [(time.time()-start)/(b-a)], delimiter=',')
    print('(mean/max/min) error=({}/{}/{}), miss=({}/{}/{}), false_alarm=({}/{}/{})'.format(round(errors.mean(), 3), round(errors.max(), 3), round(errors.min(), 3), \
          round(sum(misses)/(b-a), 3), max(misses), min(misses), round(sum(false_alarms)/(b-a), 3), max(false_alarms), min(false_alarms) ) )
    print('Ours! time = ', round(time.time()-start, 3))
except Exception as e:
    print(e)