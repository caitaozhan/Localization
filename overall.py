'''The overall approach
'''

from localize import Localization
import time
import plots
import numpy as np
import line_profiler
import random
from utility import random_intruder, random_secondary


def overall():
    '''Overall approach
    '''
    config              = 'config/splat_config_40.json'
    cov_file            = 'dataSplat/1600/cov'
    sensor_file         = 'dataSplat/1600/sensors'
    intruder_hypo_file  = 'dataSplat/1600/hypothesis'
    primary_hypo_file   = 'dataSplat/1600/hypothesis_primary'
    secondary_hypo_file = 'dataSplat/1600/hypothesis_secondary'
    primary = [123, 1357]  # fixed

    selectsensor = Localization(config)
    selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)             # offline step: init intruder hypo
    selectsensor.setup_primary_transmitters(primary, primary_hypo_file)           # offline step: construct primary
    selectsensor.add_primary(primary_hypo_file)                                   # offline step: construct intruder + primary

    counter = 0
    while counter < 3:
        time.sleep(1)
        # if counter % 1 == 0:                                                      # Change secondary
        #     secondary = random_secondary()
        #     selectsensor.setup_secondary_transmitters(secondary, secondary_hypo_file)    # online step: construct secondary
        #     selectsensor.add_secondary(secondary_hypo_file)  # online step: construct primary + intruder + secondary
        #     selectsensor.rescale_all_hypothesis()
        #     selectsensor.transmitters_to_array()
        #     #results = selectsensor.select_offline_greedy_lazy_gpu(25, 10, o_t_approx_kernal2)
        #     results = selectsensor.select_offline_greedy_hetero(25, 10, o_t_approx_kernal2)
        #     print(results[-1])
        #     #plots.visualize_selection(counter, selectsensor.grid_len, primary, secondary, results[-1], selectsensor.sensors)

        '''
        intruders, sensor_outputs = selectsensor.set_intruders(true_indices=random_intruder(selectsensor.grid_len))
        pred_positions = selectsensor.get_posterior_localization_subset(intruders, sensor_outputs, results)
        #localize = selectsensor.get_splot_localization(intruders, sensor_outputs)  # works as expected
        error, misses = selectsensor.compute_error(intruders, pred_positions)
        print(error, misses)
        selectsensor.update_battery(results[-1][-1], energy=3)
        counter += 1
        '''

if __name__ == '__main__':
    '''main
    '''
    overall()

