import numpy as np
import sys
import random

from utility import read_utah_data, distance, plot_cdf, generate_intruders_utah
from location_transform import LocationTransform
from localize import Localization
from plots import visualize_localization


if __name__ == '__main__':
    means, stds, locations = read_utah_data(path='dataUtah')
    lt = LocationTransform(locations, cell_len=1)
    
    ll = Localization(grid_len=lt.grid_len, debug=False)
    ll.init_utah(means, stds, locations, lt, percentage=1.)
    
    num_of_intruders = 2
    a, b = 0, 33

    errors = []
    misses = []
    false_alarms = []
    power_errors = []
    ll.counter.num_exper = b-a
    ll.counter.time_start()
    for i in range(a, b):
        print('\n\nTest ', i)
        random.seed(i)
        np.random.seed(i)
        true_powers = [random.uniform(-0, 0) for i in range(num_of_intruders)]
        intruders_real, true_indices = generate_intruders_utah(grid_len=ll.grid_len, locations=locations, lt=lt, num=num_of_intruders, min_dist=10)
        intruders, sensor_outputs = ll.set_intruders(true_indices=true_indices, powers=true_powers, randomness=True)

        pred_locations, pred_power = ll.our_localization(sensor_outputs, intruders, i)

        true_locations = ll.convert_to_pos(true_indices)
        pred_locations_real = [lt.gridcell_2_real(cell) for cell in pred_locations]

        try:
            error, miss, false_alarm, power_error = ll.compute_error(intruders_real, true_powers, pred_locations_real, pred_power)
            if len(error) != 0:
                errors.extend(error)
                power_errors.extend(power_error)
            misses.append(miss)
            false_alarms.append(false_alarm)
            print('\nerror/miss/false/power = {:.3f}/{}/{}/{:.3f}'.format(np.array(error).mean(), miss, false_alarm, np.array(power_error).mean()) )
            if ll.debug:
                visualize_localization(ll.grid_len, true_locations, pred_locations, i)
        except Exception as e:
            print(e)

    try:
        plot_cdf(errors)
        errors = np.array(errors)
        power_errors = np.array(power_errors)
        print('(mean/max/min) error=({:.3f}/{:.3f}/{:.3f}), miss=({:.3f}/{}/{}), false_alarm=({:.3f}/{}/{}), power=({:.3f}/{:.3f}/{:.3f})'.format(errors.mean(), errors.max(), errors.min(), \
              sum(misses)/(b-a), max(misses), min(misses), sum(false_alarms)/(b-a), max(false_alarms), min(false_alarms), power_errors.mean(), power_errors.max(), power_errors.min() ) )
        ll.counter.time_end()
        ratios = ll.counter.procedure_ratios()
        print(ratios)
        print('Proc-1 time = {:.3f}, Proc-1.1 = {:.3f}， Proc-2-2 time = {:.3f}, Proc-2-3 time = {:.3f}'.format(ll.counter.time1_average(), ll.counter.time2_average(), ll.counter.time3_average(), ll.counter.time4_average()))
    except Exception as e:
        print(e)


    # print(mean)
    # print(stds)
    # print(locations)
    # print(lt)

    # errors = []
    # for real_loc in lt.real_location:
    #     gridcell = lt.real_2_gridcell(real_loc)
    #     real_loc2 = lt.gridcell_2_real(gridcell)
    #     error = distance(real_loc, real_loc2)
    #     errors.append(error)
    #     print('({:5.2f}, {:5.2f}) -> ({:2d}, {:2d}) -> ({:5.2f}, {:5.2f}); error = {:3.2f}'.format(real_loc[0], real_loc[1], gridcell[0], gridcell[1], real_loc2[0], real_loc2[1], error))
    # plot_cdf(errors)