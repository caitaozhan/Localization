import numpy as np
import sys
import random

from utility_ipsn import read_utah_data, distance, plot_cdf, generate_intruders_utah, set_intruders_utah
from location_transform import LocationTransform
from localize import Localization
from plots import visualize_localization

def main0():
    '''Read the Utah data, location transform
    '''
    means, stds, locations, wall = read_utah_data(path='dataUtah')

    lt = LocationTransform(locations, cell_len=1)

    print(means)
    print(stds)
    print(locations)
    print(wall)
    print(lt)

    errors = []
    for real_loc in lt.real_location:
        gridcell = lt.real_2_gridcell(real_loc)
        real_loc2 = lt.gridcell_2_real(gridcell)
        error = distance(real_loc, real_loc2)
        errors.append(error)
        print('({:5.2f}, {:5.2f}) -> ({:2d}, {:2d}) -> ({:5.2f}, {:5.2f}); error = {:3.2f}'.format(real_loc[0], real_loc[1], gridcell[0], gridcell[1], real_loc2[0], real_loc2[1], error))
    plot_cdf(errors)


def main1():
    '''Our localization
    '''
    random.seed(0)
    np.random.seed(0)
    means, stds, locations, wall = read_utah_data(path='dataUtah')
    lt = LocationTransform(locations, cell_len=1)

    ll = Localization(grid_len=lt.grid_len, case='utah', debug=False)
    ll.init_utah(means, stds, locations, lt, wall, interpolate=True, percentage=.8)

    num_of_intruders = 3
    a, b = 0, 100

    errors = []
    misses = []
    false_alarms = []
    power_errors = []
    ll.counter.num_exper = b-a
    ll.counter.time_start()
    for i in range(a, b):
    # for i in [27, 50]:
        if i == 29 or i == 74 or i == 77 or i == 84:
            i += 100
        print('\n\nTest ', i)
        random.seed(i)
        np.random.seed(i)
        true_powers = [random.uniform(-0, 0) for i in range(num_of_intruders)]
        intruders_real, true_indices = generate_intruders_utah(grid_len=ll.grid_len, locations=locations, lt=lt, num=num_of_intruders, min_dist=1, max_dist=3)
        intruders, sensor_outputs = set_intruders_utah(true_indices=true_indices, powers=true_powers, means=means, grid_loc=lt.grid_location, ll=ll, randomness=True)
        print('True: ' + ''.join(map(lambda x:str(x), intruders)))

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
        # plot_cdf(errors)
        # np.savetxt('visualize/utah/0.7.txt', errors, delimiter=',')
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


def main2():
    '''SPLOT
    '''
    random.seed(0)
    np.random.seed(0)
    means, stds, locations, wall = read_utah_data(path='dataUtah')
    lt = LocationTransform(locations, cell_len=1)

    ll = Localization(grid_len=lt.grid_len, case='utah', debug=False)
    ll.init_utah(means, stds, locations, lt, wall, interpolate=False)

    a, b = 0, 100
    num_of_intruders = 2
    errors = []
    misses = []
    false_alarms = []
    for i in range(a, b):
    # for i in [89]:
        if i == 29 or i == 74 or i == 77 or i == 84:
            i += 100
        print('\n\nTest ', i)
        random.seed(i)
        np.random.seed(i)
        true_powers = [random.uniform(-0, 0) for i in range(num_of_intruders)]
        intruders_real, true_indices = generate_intruders_utah(grid_len=ll.grid_len, locations=locations, lt=lt, num=num_of_intruders, min_dist=1, max_dist=3)

        intruders, sensor_outputs = set_intruders_utah(true_indices=true_indices, powers=true_powers, means=means, grid_loc=lt.grid_location, ll=ll, randomness=True)
        
        r1 = 10
        r2 = 6
        threshold = -51
        pred_locations = ll.splot_localization(sensor_outputs, intruders, fig=i, R1=r1, R2=r2, threshold=threshold)

        print('True', end=' ')
        for intru in intruders:
            print(intru)
        true_locations = ll.convert_to_pos(true_indices)
        pred_locations_real = [lt.gridcell_2_real(cell) for cell in pred_locations]

        try:
            error, miss, false_alarm = ll.compute_error2(intruders_real, pred_locations_real)
            if len(error) != 0:
                errors.extend(error)
            misses.append(miss)
            false_alarms.append(false_alarm)
            print('\nerror/miss/false = {:.3f}/{}/{}'.format(np.array(error).mean(), miss, false_alarm) )
            if ll.debug:
                visualize_localization(ll.grid_len, true_locations, pred_locations, i)
        except Exception as e:
            print(e)

    try:
        # plot_cdf(errors)
        # np.savetxt('visualize/utah/ridge.txt', errors, delimiter=',')
        errors = np.array(errors)
        print('(mean/max/min) error=({:.3f}/{:.3f}/{:.3f}), miss=({:.3f}/{}/{}), false_alarm=({:.3f}/{}/{})'.format(errors.mean(), errors.max(), errors.min(), \
              sum(misses)/(b-a), max(misses), min(misses), sum(false_alarms)/(b-a), max(false_alarms), min(false_alarms) ) )
    except Exception as e:
        print(e)

if __name__ == '__main__':
    # main0()
    main1()   # Ours
    # main2()   # SPLOT