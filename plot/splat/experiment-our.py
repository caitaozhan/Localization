import numpy as np
import pandas as pd
import sys
import time
import random
sys.path.append('../..')
from select_sensor import SelectSensor
from utility import generate_intruders


# SelectSensor version on May 31, 2019
selectsensor = SelectSensor(grid_len=40)
selectsensor.init_data('../../dataSplat/homogeneous-200/cov', '../../dataSplat/homogeneous-200/sensors', '../../dataSplat/homogeneous-200/hypothesis')

num_of_intruders = 10

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
    random.seed(i)
    np.random.seed(i)
    true_indices, true_powers = generate_intruders(grid_len=selectsensor.grid_len, edge=2, num=num_of_intruders, min_dist=1, powers=true_powers)

    intruders, sensor_outputs = selectsensor.set_intruders(true_indices=true_indices, powers=true_powers, randomness=True)
    pred_locations, pred_power, ratio = selectsensor.our_localization(sensor_outputs, intruders, i)
    proc_1_ratio += ratio
    true_locations = selectsensor.convert_to_pos(true_indices)

    try:
        error, miss, false_alarm, power_error = selectsensor.compute_error(true_locations, true_powers, pred_locations, pred_power)
        if len(error) != 0:
            errors.extend(error)
            power_errors.extend(power_error)
        misses.append(miss)
        false_alarms.append(false_alarm)
        print('error/miss/false/power = {}/{}/{}/{}'.format(np.array(error).mean(), miss, false_alarm, np.array(power_error).mean()) )
    except Exception as e:
        print(e)

try:
    errors = np.array(errors)
    power_errors = np.array(power_errors)
    np.savetxt('{}-ours-error.txt'.format(num_of_intruders), errors, delimiter=',')
    np.savetxt('{}-ours-miss.txt'.format(num_of_intruders), misses, delimiter=',')
    np.savetxt('{}-ours-false.txt'.format(num_of_intruders), false_alarms, delimiter=',')
    np.savetxt('{}-ours-power.txt'.format(num_of_intruders), power_errors, delimiter=',')
    np.savetxt('{}-ours-time.txt'.format(num_of_intruders), [selectsensor.time_1/(b-a), selectsensor.time_2/(b-a)], delimiter=',')
    print('(mean/max/min) error=({}/{}/{}), miss=({}/{}/{}), false_alarm=({}/{}/{}), power=({}/{}/{})'.format(round(errors.mean(), 3), round(errors.max(), 3), round(errors.min(), 3), \
          round(sum(misses)/(b-a), 3), max(misses), min(misses), round(sum(false_alarms)/(b-a), 3), max(false_alarms), min(false_alarms), round(power_errors.mean(), 3), round(power_errors.max(), 3), round(power_errors.min(), 3) ) )
    print('Ours! time = ', round(time.time()-start, 3), '; proc 1 ratio =', round(proc_1_ratio/(b-a), 3))
except Exception as e:
    print(e)