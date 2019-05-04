'''Localization testing
'''
from select_sensor import SelectSensor
import time
import random


def splat_splot(sensor_num=200, num_intruders=None):
    '''localization. SPLAT scale 4 version of hypothesis-25
       SPLOT
    '''
    file_handle = open('plot_data_splat/fig5-sensor-num/splot-{}'.format(sensor_num), 'w')

    selectsensor = SelectSensor('config/splat_config_40_scale4.json')
    selectsensor.init_data('dataSplat/1600-{}/cov'.format(sensor_num), 'dataSplat/1600-{}/sensors-scale-4'.format(sensor_num), 'dataSplat/1600-{}/hypothesis-25-scale-4'.format(sensor_num))
    repeat = 30
    errors = []
    misses = []
    false_alarms = []

    for _ in range(0, repeat):
        num_intruders = random.randint(10, 20)
        true_indices = random.sample(range(selectsensor.grid_len * selectsensor.grid_len), num_intruders)

        intruders, sensor_outputs = selectsensor.set_intruders(true_indices=true_indices)

        r1 = 25
        r2 = 12
        threshold = -68
        pred_location = selectsensor.splot_localization(intruders, sensor_outputs, R1=r1, R2=r2, threshold=threshold)

        true_positions = selectsensor.convert_to_pos(true_indices)
        print(true_positions)
        print(pred_location)
        try:
            error, miss, false_alarm = selectsensor.compute_error(true_positions, pred_location)
            errors.append(error)
            misses.append(miss)
            false_alarms.append(false_alarm)
            print(error, miss, false_alarm, '\n')
        except:
            print('except')
            errors.append(0)
            misses.append(1)
            false_alarms.append(0)
    print('(mean/max/min) error=({}/{}/{}), miss=({}/{}/{}), false_alarm=({}/{}/{})'.format(\
            sum(errors)/len(errors), max(errors), min(errors), sum(misses)/repeat, max(misses), min(misses), \
            sum(false_alarms)/len(false_alarms), max(false_alarms), min(false_alarms)))

    print('{},{},{}'.format(sum(errors)/len(errors), sum(misses)/repeat, sum(false_alarms)/len(false_alarms)), file=file_handle)

    print('SPLOT! # of intruder = {}, threshold = {}'.format(num_intruders, threshold))
    
    file_handle.close()


def splat_clustering(sensor_num=200, num_intruders=None):
    '''localization. SPLAT scale 4 version of hypothesis-25
       clustering
    '''
    file_handle = open('plot_data_splat/fig5-sensor-num/cluster-{}'.format(sensor_num), 'w')

    selectsensor = SelectSensor('config/splat_config_40_scale4.json')
    selectsensor.init_data('dataSplat/1600-{}/cov'.format(sensor_num), 'dataSplat/1600-{}/sensors-scale-4'.format(sensor_num), 'dataSplat/1600-{}/hypothesis-25-scale-4'.format(sensor_num))
    repeat = 30
    errors = []
    misses = []
    false_alarms = []

    for _ in range(0, repeat):
        #num_intruders = random.randint(10, 20)
        true_indices = random.sample(range(selectsensor.grid_len * selectsensor.grid_len), num_intruders)

        intruders, sensor_outputs = selectsensor.set_intruders(true_indices=true_indices)

        pred_location = selectsensor.get_cluster_localization(intruders, sensor_outputs, num_intruders)

        true_positions = selectsensor.convert_to_pos(true_indices)
        #print(true_positions)
        #print(pred_location)
        try:
            error, miss, false_alarm = selectsensor.compute_error(true_positions, pred_location)
            errors.append(error)
            misses.append(miss)
            false_alarms.append(false_alarm)
            #   print(error, miss, false_alarm, '\n')
        except:
            print('except')
    print('(mean/max/min) error=({}/{}/{}), miss=({}/{}/{}), false_alarm=({}/{}/{})'.format(\
            sum(errors)/repeat, max(errors), min(errors), sum(misses)/repeat, max(misses), min(misses), \
            sum(false_alarms)/repeat, max(false_alarms), min(false_alarms)))

    print('{},{},{}'.format(sum(errors)/len(errors), sum(misses)/repeat, sum(false_alarms)/len(false_alarms)), file=file_handle)
    print('clustering!')


def splat_ours(sensor_num=None, num_intruders=None):
    '''localization. SPLAT scale 4 version of hypothesis-25
       ours
    '''
    file_handle = open('plot_data_splat/fig5-sensor-num/ours-{}-'.format(sensor_num), 'w')

    selectsensor = SelectSensor('config/splat_config_40_scale4.json')
    selectsensor.init_data('dataSplat/1600-{}/cov'.format(sensor_num), 'dataSplat/1600-{}/sensors-scale-4'.format(sensor_num), 'dataSplat/1600-{}/hypothesis-25-scale-4'.format(sensor_num))
    repeat = 10
    errors = []
    misses = []
    false_alarms = []

    for _ in range(0, repeat):
        num_intruders = random.randint(10, 20)
        true_indices = random.sample(range(selectsensor.grid_len * selectsensor.grid_len), num_intruders)
        start = time.time()

        intruders, sensor_outputs = selectsensor.set_intruders(true_indices=true_indices)

        pred_location = selectsensor.procedure1(intruders, sensor_outputs)
        print('time = ', time.time()-start)

        true_positions = selectsensor.convert_to_pos(true_indices)
        #print(true_positions)
        #print(pred_location)
        try:
            error, miss, false_alarm = selectsensor.compute_error(true_positions, pred_location)
            errors.append(error)
            misses.append(miss)
            false_alarms.append(false_alarm)
            print(error, miss, false_alarm, '\n')
        except:
            print('except')
    print('(mean/max/min) error=({}/{}/{}), miss=({}/{}/{}), false_alarm=({}/{}/{})'.format(\
            sum(errors)/repeat, max(errors), min(errors), sum(misses)/repeat, max(misses), min(misses), \
            sum(false_alarms)/repeat, max(false_alarms), min(false_alarms)))
    print('Ours! Num of intruders = {}'.format(num_intruders))

    file_handle = open('plot_data_splat/fig5-sensor-num/ours-{}'.format(sensor_num), 'w')
    print('{},{},{}'.format(sum(errors)/len(errors), sum(misses)/repeat, sum(false_alarms)/len(false_alarms)), file=file_handle)
    file_handle.close()    


def varies_intruder():
    ''' varies # of intruder
    '''
    #mylist = [1, 2, 4, 8, 16, 24, 30]
    mylist = [30]
    for i in mylist:
        #splat_clustering(num_intruders=i)
        #splat_ours(num_intruders=i)
        splat_splot(num_intruders=i)


if __name__ == '__main__':
    #varies_intruder()
    #splat_splot(sensor_num=100)

    splat_ours(sensor_num=100)
