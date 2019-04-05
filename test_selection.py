'''Testing
'''

from select_sensor import SelectSensor
try:
    from cuda_kernals import o_t_approx_kernal, o_t_kernal, o_t_approx_dist_kernal, \
                             o_t_approx_kernal2, o_t_approx_dist_kernal2
except Error:
    pass
import time
import plots
import numpy as np
import line_profiler


def test_ipsn_homo():
    '''2019 IPSN version
    '''
    selectsensor = SelectSensor('config/ipsn_50.json')
    selectsensor.init_data('data50/homogeneous-150-2/cov', 'data50/homogeneous-150-2/sensors', 'data50/homogeneous-150-2/hypothesis')
    # CPU version
    #selectsensor.select_offline_greedy(40)
    #selectsensor.select_offline_greedy2(5)
    #selectsensor.select_offline_greedy_p(20, 16)
    #selectsensor.select_offline_greedy_p_lazy_cpu(20, 16)
    #selectsensor.select_offline_greedy_p_lazy_gpu(1, 12, o_t_approx_kernal)

    # GPU version
    selectsensor.transmitters_to_array()

    #start = time.time()
    #selectsensor.select_offline_greedy_p_lazy_gpu(15, 12, o_t_approx_kernal)
    #print('time = {}'.format(time.time()-start))

    results = selectsensor.select_offline_greedy_lazy_gpu(150, 12, o_t_approx_kernal2)
    for r in results:
        print(r[:-1])

    #plots.figure_1a(selectsensor, None)


def test_ipsn_hetero():
    '''2019 IPSN version
    '''
    selectsensor = SelectSensor('config/ipsn_config.json')
    selectsensor.init_data('data16/heterogeneous/cov', 'data16/heterogeneous/sensors', 'data16/heterogeneous/hypothesis')
    selectsensor.select_offline_greedy_hetero(5, 12, o_t_approx_kernal2)
    selectsensor.select_offline_greedy_hetero(5, 12, o_t_approx_kernal2)


def test_utah():
    '''2019 Mobicom version
    '''
    selectsensor = SelectSensor('config/utah_config.json')
    selectsensor.init_data('dataUtah/cov', 'dataUtah/sensors', 'dataUtah/hypothesis')
    #selectsensor.init_utah_legal_transmiters()                       # turn some sensors into legal transmitters
    #selectsensor.update_utah_settings('dataUtah/sensors_update')     # update the settings: priori and sensor list
    #selectsensor.add_legal_and_intruder_hypothesis('dataUtah/hypothesis_legal', 'dataUtah/hypothesis', 'dataUtah/hypothesis_add') # add the hypothesis of legal tansmitters and intruder
    #selectsensor.init_data('dataUtah/cov', 'dataUtah/sensors_update', 'dataUtah/hypothesis_add')  # init again using new sensors and new hypothesis
    #selectsensor.rescale_hypothesis_add()
    selectsensor.select_offline_greedy_p_lazy_cpu(5, 4)
    #selectsensor.locate_intruder_accuracy()
    #plots.figure_1a(selectsensor, None)


#@profile
def test_splat(large=True, flag=1):
    '''2019 Mobicom version using data generated from SPLAT
    Args:
        large (bool): True for 4096 hypothesis, False for 1600 hypothesis
        flag (int):   1 for just intruders, 2 for complete steps, 3 for directly using added hypothesis
    '''
    if large is False:
        config              = 'config/splat_config_40.json'
        cov_file            = 'dataSplat/1600/cov'
        sensor_file         = 'dataSplat/1600/sensors'
        intruder_hypo_file  = 'dataSplat/1600/hypothesis-25'
        primary_hypo_file   = 'dataSplat/1600/hypothesis_primary'
        secondary_hypo_file = 'dataSplat/1600/hypothesis_secondary'

        if flag == 1:      # just the intruders
            selectsensor = SelectSensor(config)
            selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
            selectsensor.rescale_intruder_hypothesis()
            selectsensor.transmitters_to_array()
            results = selectsensor.select_offline_greedy_lazy_gpu(50, 12, o_t_approx_kernal2)
            plots.save_data_AGA(results, 'plot_data_splat/fig1-homo/AGA')
            for r in results:
                print(r)
        if flag == 2:      # the complete process: intruder --> primary --> secondary
            selectsensor = SelectSensor(config)
            selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)                # init intruder hypo
            selectsensor.setup_primary_transmitters([123, 1357], primary_hypo_file)          # setup primary
            selectsensor.add_primary(primary_hypo_file)
            selectsensor.setup_secondary_transmitters([456, 789, 1248], secondary_hypo_file) # setup secondary
            selectsensor.add_secondary(secondary_hypo_file)
            selectsensor.rescale_all_hypothesis()
            selectsensor.transmitters_to_array()
            results = selectsensor.select_offline_greedy_lazy_gpu(20, 10, o_t_approx_kernal2)
            for r in results:
                print(r)
        if flag == 3:      # use added hypo directly
            selectsensor = SelectSensor(config)
            selectsensor.init_data(cov_file, sensor_file, all_hypo_file)
            selectsensor.rescale_intruder_hypothesis()
            selectsensor.transmitters_to_array()
            results = selectsensor.select_offline_greedy_lazy_gpu(20, 10, o_t_approx_kernal2)
            for r in results:
                print(r)

    else: # large is True
        config              = 'config/splat_config_64.json'
        cov_file            = 'dataSplat/4096/cov'
        sensor_file         = 'dataSplat/4096/sensors'
        intruder_hypo_file  = 'dataSplat/4096/hypothesis'
        primary_hypo_file   = 'dataSplat/4096/hypothesis_primary'
        secondary_hypo_file = 'dataSplat/4096/hypothesis_secondary'

        if flag == 1:      # just the intruders
            selectsensor = SelectSensor(config)
            selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
            selectsensor.rescale_intruder_hypothesis()
            selectsensor.transmitters_to_array()
            results = selectsensor.select_offline_greedy_lazy_gpu(20, 12, o_t_approx_kernal2)
            for r in results:
                print(r)
        if flag == 2:      # the complete process
            selectsensor = SelectSensor(config)
            selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)                 # init intruder hypo
            selectsensor.setup_primary_transmitters([479, 3456], primary_hypo_file)           # setup primary
            selectsensor.add_primary(primary_hypo_file)
            selectsensor.setup_secondary_transmitters([789, 1357, 2345], secondary_hypo_file) # setup secondary
            selectsensor.add_secondary(secondary_hypo_file)
            selectsensor.rescale_all_hypothesis()
            selectsensor.transmitters_to_array()
            results = selectsensor.select_offline_greedy_lazy_gpu(20, 10, o_t_approx_kernal2)
            for r in results:
                print(r)
        if flag == 3:      # use added hypo directly
            selectsensor = SelectSensor(config)
            selectsensor.init_data(cov_file, sensor_file, all_hypo_file)
            selectsensor.rescale_intruder_hypothesis()
            selectsensor.transmitters_to_array()
            results = selectsensor.select_offline_greedy_lazy_gpu(20, 10, o_t_approx_kernal2)
            for r in results:
                print(r)


def test_splat_baseline(flag):
    '''The baseline (GA, random, coverage), without background, homogeneous, 40 x 40 grid
    '''

    config              = 'config/splat_config_40.json'
    cov_file            = 'dataSplat/1600/cov'
    sensor_file         = 'dataSplat/1600/sensors'
    intruder_hypo_file  = 'dataSplat/1600/hypothesis-25'
    selectsensor = SelectSensor(config)
    selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
    selectsensor.rescale_intruder_hypothesis()
    selectsensor.transmitters_to_array()        # for GPU

    if flag == 0 or flag == 1:  # GA
        selectsensor.transmitters_to_array()
        results = selectsensor.select_offline_GA(50, 10)
        plots.save_data(results, 'plot_data_splat/fig1-homo/GA')

    if flag == 0 or flag == 2:  # Random
        results = selectsensor.select_offline_random(100, 10)
        plots.save_data(results, 'plot_data_splat/fig1-homo/random')

    if flag == 0 or flag == 3:  # Coverage
        results = selectsensor.select_offline_coverage(100, 10)
        plots.save_data(results, 'plot_data_splat/fig1-homo/coverage')


def test_splat_opt():
    '''Comparing AGA to the optimal and baselines, without background, homogeneous, small grid 10 x 10
    '''
    config              = 'config/splat_config_10.json'
    cov_file            = 'dataSplat/100/cov{}'
    sensor_file         = 'dataSplat/100/sensors{}'
    intruder_hypo_file  = 'dataSplat/100/hypothesis{}'

    for i in range(1, 11):
        print('\ncase {}'.format(i))
        selectsensor = SelectSensor(config)
        selectsensor.init_data(cov_file.format(i), sensor_file.format(i), intruder_hypo_file.format(i))
        selectsensor.rescale_intruder_hypothesis()
        selectsensor.transmitters_to_array()        # for GPU

        results = selectsensor.select_offline_greedy_lazy_gpu(10, 12, o_t_approx_kernal2)
        plots.save_data_AGA(results, 'plot_data_splat/fig2-homo-small/AGA{}'.format(i))

        results = selectsensor.select_offline_GA(10, 10)
        plots.save_data(results, 'plot_data_splat/fig2-homo-small/GA{}'.format(i))

        results = selectsensor.select_offline_coverage(10, 10)
        plots.save_data(results, 'plot_data_splat/fig2-homo-small/coverage{}'.format(i))

        results = selectsensor.select_offline_random(10, 10)
        plots.save_data(results, 'plot_data_splat/fig2-homo-small/random{}'.format(i))

        plot_data = []
        for budget in range(1, 11):
            budget, ot = selectsensor.select_offline_optimal(budget, 12)
            plot_data.append([budget, ot])
        plots.save_data(plot_data,'plot_data_splat/fig2-homo-small/optimal{}'.format(i))


def test_splat_total_sensors():
    '''AGA against varies total # of sensors
    '''
    config              = 'config/splat_config_40-50.json'
    cov_file            = 'dataSplat/1600-50/cov'
    sensor_file         = 'dataSplat/1600-50/sensors'
    intruder_hypo_file  = 'dataSplat/1600-50/hypothesis'
    selectsensor = SelectSensor(config)
    selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
    selectsensor.rescale_intruder_hypothesis()
    selectsensor.transmitters_to_array()
    results = selectsensor.select_offline_greedy_lazy_gpu(20, 12, o_t_approx_kernal2)
    plots.save_data_AGA(results, 'plot_data_splat/fig4-homo-total-sensors/50-sensors')

    config              = 'config/splat_config_40-100.json'
    cov_file            = 'dataSplat/1600-100/cov'
    sensor_file         = 'dataSplat/1600-100/sensors'
    intruder_hypo_file  = 'dataSplat/1600-100/hypothesis'
    selectsensor = SelectSensor(config)
    selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
    selectsensor.rescale_intruder_hypothesis()
    selectsensor.transmitters_to_array()
    results = selectsensor.select_offline_greedy_lazy_gpu(20, 12, o_t_approx_kernal2)
    plots.save_data_AGA(results, 'plot_data_splat/fig4-homo-total-sensors/100-sensors')

    config              = 'config/splat_config_40.json'
    cov_file            = 'dataSplat/1600/cov'
    sensor_file         = 'dataSplat/1600/sensors'
    intruder_hypo_file  = 'dataSplat/1600/hypothesis'
    selectsensor = SelectSensor(config)
    selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
    selectsensor.rescale_intruder_hypothesis()
    selectsensor.transmitters_to_array()
    results = selectsensor.select_offline_greedy_lazy_gpu(20, 12, o_t_approx_kernal2)
    plots.save_data_AGA(results, 'plot_data_splat/fig4-homo-total-sensors/200-sensors')

    config              = 'config/splat_config_40-400.json'
    cov_file            = 'dataSplat/1600-400/cov'
    sensor_file         = 'dataSplat/1600-400/sensors'
    intruder_hypo_file  = 'dataSplat/1600-400/hypothesis'
    selectsensor = SelectSensor(config)
    selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
    selectsensor.rescale_intruder_hypothesis()
    selectsensor.transmitters_to_array()
    results = selectsensor.select_offline_greedy_lazy_gpu(20, 12, o_t_approx_kernal2)
    plots.save_data_AGA(results, 'plot_data_splat/fig4-homo-total-sensors/400-sensors')

    config              = 'config/splat_config_40-800.json'
    cov_file            = 'dataSplat/1600-800/cov'
    sensor_file         = 'dataSplat/1600-800/sensors'
    intruder_hypo_file  = 'dataSplat/1600-800/hypothesis'
    selectsensor = SelectSensor(config)
    selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
    selectsensor.rescale_intruder_hypothesis()
    selectsensor.transmitters_to_array()
    results = selectsensor.select_offline_greedy_lazy_gpu(20, 12, o_t_approx_kernal2)
    plots.save_data_AGA(results, 'plot_data_splat/fig4-homo-total-sensors/800-sensors')


def test_splat_hetero(flag):
    '''The baseline (GA, random, coverage), without background, heterogeneous, 40 x 40 grid
    '''
    config              = 'config/splat_config_40.json'
    cov_file            = 'dataSplat/1600-hetero/cov'
    sensor_file         = 'dataSplat/1600-hetero/sensors'
    intruder_hypo_file  = 'dataSplat/1600-hetero/hypothesis-25'
    selectsensor = SelectSensor(config)
    selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
    selectsensor.rescale_intruder_hypothesis()
    selectsensor.transmitters_to_array()        # for GPU

    if flag == 0 or flag == 1:  # AGA
        results = selectsensor.select_offline_greedy_hetero(30, 12, o_t_approx_kernal2)
        plots.save_data(results, 'plot_data_splat/fig3-hetero/AGA')

    if flag == 0 or flag == 2:  # Random
        results = selectsensor.select_offline_random_hetero(50, 12)
        plots.save_data(results, 'plot_data_splat/fig3-hetero/random')

    if flag == 0 or flag == 3:  # Coverage
        results = selectsensor.select_offline_coverage_hetero(50, 12)
        plots.save_data(results, 'plot_data_splat/fig3-hetero/coverage')

    if flag == 0 or flag == 4:  # GA
        results = selectsensor.select_offline_GA_hetero(30, 12)
        plots.save_data(results, 'plot_data_splat/fig3-hetero/GA')


def test_ipsn2():
    '''2019 Mobicom version using data generated from IPSN
    '''

    cov_file           = 'dataSplat/1600/cov'
    sensor_file        = 'dataSplat/1600/sensors'
    intruder_hypo_file = 'dataSplat/1600/hypothesis'
    legal_hypo_file    = 'dataSplat/1600/hypothesis_legal'
    add_hypo_file      = 'dataSplat/1600/hypothesis_add'

    selectsensor = SelectSensor('config/splat_config.json')
    #selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
    selectsensor.init_data(cov_file, sensor_file, add_hypo_file)
    #selectsensor.setup_legal_transmitters([123, 456, 789, 1357, 1248], legal_hypo_file)
    #selectsensor.add_legal_and_intruder_hypothesis(legal_hypo_file, intruder_hypo_file, add_hypo_file)
    #selectsensor.rescale_add_hypothesis()
    selectsensor.rescale_intruder_hypothesis()
    selectsensor.transmitters_to_array()
    print(selectsensor.select_offline_greedy_lazy_gpu(20, 12, o_t_approx_kernal2))

def test_splat_localization_single_intruder():
    config              = 'config/splat_config_40.json'
    cov_file            = 'dataSplat/1600/cov'
    sensor_file         = 'dataSplat/1600/sensors'
    intruder_hypo_file  = 'dataSplat/1600/hypothesis'
    primary_hypo_file   = 'dataSplat/1600/hypothesis_primary'
    intr_pri_hypo_file  = 'dataSplat/1600/hypothesis_intru_pri'
    secondary_hypo_file = 'dataSplat/1600/hypothesis_secondary'
    all_hypo_file       = 'dataSplat/1600/hypothesis_all'

    selectsensor = SelectSensor(config)
    selectsensor.init_data(cov_file, sensor_file, intruder_hypo_file)
    selectsensor.rescale_intruder_hypothesis()
    selectsensor.transmitters_to_array()
    results = selectsensor.localize(10, -1)
    for r in results:
        print(r)



if __name__ == '__main__':
    #test_map()
    test_ipsn_homo()
    #test_ipsn_hetero()
    #test_splat(False, 1)
    #test_splat(False, 2)
    #test_splat(False, 3)
    #test_splat(True, 1)
    #test_splat(True, 2)
    #test_splat_localization_single_intruder()
    #select_online_random(self, budget, cores, true_index=-1)
    #test_splat(True, 3)
    #test_splat_baseline(0)
    #test_splat_opt()
    #test_splat_total_sensors()
    #test_splat_hetero(0)
