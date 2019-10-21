'''Localization server
'''
import os
import time
import argparse
import sys
import argparse
import numpy as np
from flask import Flask, request
from loc_default_config import TrainingInfo
from client import Default
from localize import Localization
from input_output import Input, Output
from utility import subsample_from_full
from plots import visualize_localization


try:
    sys.path.append('../rtl-testbed/')
    from default_config import OutdoorMap, IndoorMap, SplatMap
except:
    print('Import error')

app = Flask(__name__)


@app.route('/on', methods=['GET'])
def on():
    '''See whether the server is turned on'''
    return 'on'


@app.route('/localize', methods=['POST'])
def localize():
    '''process the POST request
    '''
    # step 0: parse the request data
    myinput = Input.from_json_dict(request.get_json())  # get_json() return a dict
    myinput.train_percent = train.train_percent

    # step 1: set up sensor data
    try:
        sensor_data = myinput.sensor_data
        sensor_outputs = np.zeros(len(sensor_data))
        for hostname, rss in sensor_data.items():
            index = server_support.get_index(hostname)
            sensor_outputs[index] = rss
    except Exception as e:
        print(e)
        print('most probability a few sensors did not send its data')
        print(sensor_data)
        print(hostname, index)
        return 'Bad Request'

    # step 2: set up ground truth
    ground_truth = myinput.ground_truth
    true_locations, true_powers, intruders = server_support.parse_ground_truth(ground_truth, ll)

    # step 3: do the localization
    print('\n****\nNumber =', myinput.experiment_num)
    outputs = []
    if 'our' in myinput.methods:
        start = time.time()
        pred_locations, pred_power = ll.our_localization(np.copy(sensor_outputs), intruders, myinput.experiment_num)
        end = time.time()
        pred_locations = server_support.pred_loc_to_center(pred_locations)
        # visualize_localization(40, true_locations, pred_locations, myinput.experiment_num)
        errors, miss, false_alarm, power_errors = ll.compute_error(true_locations, true_powers, pred_locations, pred_power)
        outputs.append(Output('our', errors, false_alarm, miss, power_errors, end-start, pred_locations))
    if 'splot' in myinput.methods:
        start = time.time()
        pred_locations = ll.splot_localization(np.copy(sensor_outputs), intruders, myinput.experiment_num)
        end = time.time()
        pred_locations = server_support.pred_loc_to_center(pred_locations)
        errors, miss, false_alarm = ll.compute_error2(true_locations, pred_locations)
        outputs.append(Output('splot', errors, false_alarm, miss, [0], end-start, pred_locations))
    if 'cluster' in myinput.methods:
        start = time.time()
        pred_locations = ll.cluster_localization_range(intruders, np.copy(sensor_outputs), num_of_intruders=int(myinput.num_intruder))
        end = time.time()
        pred_locations = server_support.pred_loc_to_center(pred_locations)
        errors, miss, false_alarm = ll.compute_error2(true_locations, pred_locations)
        outputs.append(Output('cluster', errors, false_alarm, miss, [0], end-start, pred_locations))


    # step 4: log the input and output
    server_support.log(myinput, outputs)

    return 'Hello world'


class ServerSupport:
    '''Misc things to support the server running
    '''
    def __init__(self, sensors_hostname, output_dir, output_file, tx_calibrate):
        '''
        Args:
            sensors_hostname -- str -- a file name
            output_dir       -- str -- a directory name
            output_file      -- str -- a file name
            tx_calibrate     -- {str:int} -- the calibrated power for different transmitters
        '''
        self.hostname_2_index = {}
        self.init_hostname_2_index(sensors_hostname)
        self.output = self.init_output(output_dir, output_file)
        self.tx_calibrate = tx_calibrate


    def init_hostname_2_index(self, sensors_hostname):
        '''init a dictionary'''
        with open(sensors_hostname, 'r') as f:
            counter = 0
            for line in f:
                line = line.split(':')
                self.hostname_2_index[line[0]] = counter
                counter += 1

    def init_output(self, output_dir, output_file):
        '''set up output file
        Args:
            output_dir  -- str
            output_file -- str
        Return:
            io.TextIOWrapper
        '''
        if os.path.exists(output_dir) is False:
            os.mkdir(output_dir)
        return open(output_dir + '/' + output_file, 'a')

    def get_index(self, hostname):
        index = self.hostname_2_index.get(hostname)
        if index is not None:
            return index
        else:
            raise Exception('hostname {} do not exist'.format(hostname))

    def parse_ground_truth(self, ground_truth, ll):
        '''parse the ground truth from the client
        Args:
            ground_truth -- {...} eg. {'T1': {'location': [9.5, 5.5], 'gain': '50'}}
            ll           -- Localization
        Return:
            true_locations -- list<(float, float)>
            true_powers    -- list<float>
            intruders      -- list<Transmitter>
        '''
        grid_len = ll.grid_len
        true_locations, true_powers = [], []
        intruders = []
        for tx, truth in sorted(ground_truth.items()):
            for key, value in truth.items():
                if key == 'location':
                    one_d_index = (int(value[0])*grid_len + int(value[1]))
                    two_d_index = (value[0], value[1])
                    true_locations.append(two_d_index)
                    intruders.append(ll.transmitters[one_d_index])
                elif key == 'gain':
                    train_power = self.tx_calibrate[tx]
                    true_powers.append(float(value) - train_power)
                else:
                    raise Exception('key = {} invalid!'.format(key))
        return true_locations, true_powers, intruders

    def pred_loc_to_center(self, pred_locations):
        '''Make the predicted locations be the center of the predicted grid
        Args:
            pred_locations -- list<tuple<int, int>>
        Return:
            list<tuple<int, int>>
        '''
        pred_center = []
        for pred in pred_locations:
            center = (pred[0] + 0.5, pred[1] + 0.5)
            pred_center.append(center)
        return pred_center

    def log(self, myinput, outputs):
        '''log the results
        Args:
            myinput -- Input
            outputs -- {str:Output}
        '''
        self.output.write(myinput.log())
        for output in outputs:
            self.output.write(output.log())
        self.output.write('\n')
        self.output.flush()




if __name__ == 'server':
    ########## Testbed ############
    # data_source = 'testbed-indoor'         # 1
    # training_data = '9.26.inter-sub-2'     # 2
    # result_date = '10.16'                  # 3
    # train_percent = 37                     # 4
    # output_dir  = 'results/{}'.format(result_date)
    # output_file = 'log.indoor'                    # 5
    # train = TrainingInfo.naive_factory(data_source, training_data, train_percent)
    # print(train)
    # server_support = ServerSupport(train.hostname_loc, output_dir, output_file, train.tx_calibrate)
    # ll = Localization(grid_len=10, case=data_source, debug=True)
    # if data_source == 'testbed-indoor':
    #     MAP = IndoorMap
    # elif data_source == 'testbed-outdoor':
    #     MAP = OutdoorMap
    # ll.init_data(train.cov, train.sensors, train.hypothesis, MAP)

    ######## Splat ############
    grid_len       = 40
    data_source    = 'splat'
    gran           = 12                                   # 1   [6, 8, 10, 12, 14, 16, 18]
    sensor_density = 240                                 # 2   [80, 160, 240, 320, 400]
    transmit_power = {"T1":30}                           # 3
    full_training_data = 'inter-' + str(gran)
    sub_training_data  = full_training_data + '_{}'.format(sensor_density)     # 4

    result_date = '10.20-num'                                # 5
    train_percent = int(gran*gran/(40*40)*100)                  # 6
    output_dir  = 'results/{}'.format(result_date)
    output_file = 'log'                                  # 7
    train = TrainingInfo.naive_factory(data_source, sub_training_data, train_percent)
    # subsample_from_full(train, grid_len, sensor_density, transmit_power)       # 8

    print(train)
    server_support = ServerSupport(train.hostname_loc, output_dir, output_file, train.tx_calibrate)
    ll = Localization(grid_len=grid_len, case=data_source, debug=True)
    ll.init_data(train.cov, train.sensors, train.hypothesis, SplatMap)



if __name__ == '__main__':

    hint = 'python server.py -gran 12'
    parser = argparse.ArgumentParser(description='server side of experiments. | hint ' + hint)
    parser.add_argument('-gran', '--granularity', type=int, nargs=1, default=[None], help='granularity of the training coarse grid')
    parser.add_argument('-num', '--num_intruder', type=int, nargs=1, default=[None], help='number of intruders')
    parser.add_argument('-sen', '--sensor_density', type=int, nargs=1, default=[None], help='sensor density')
    args = parser.parse_args()

    gran        = args.granularity[0]                    # 1 [6, 8, 10, 12, 14, 16, 18]
    num_intru   = args.num_intruder[0]                   # 2 [80, 160, 240, 320, 400]
    sensor_density = args.sensor_density[0]

    if gran is not None:
        num_intru = Default.num_intruder
        sensor_density = Default.sen_density
        port = int(gran)
        output_file = 'log-gran-{}'.format(gran)
    elif num_intru is not None:
        gran = Default.training_gran
        sensor_density = Default.sen_density
        port = int(num_intru)
        output_file = 'log-num-{}'.format(num_intru)
    elif sensor_density is not None:
        gran = Default.training_gran
        num_intru = Default.num_intruder
        port = int(sensor_density)
        output_file = 'log-sen-{}'.format(sensor_density)
    else:
        raise Exception('argument mistakes!')

    print('granularity = {}\nnum intruder = {}\nsensor density = {}\n'.format(gran, num_intru, sensor_density))

    grid_len       = 40
    data_source    = 'splat'
    transmit_power = {"T1":30}                           # 3
    full_training_data = 'inter-' + str(gran)
    sub_training_data  = full_training_data + '_{}'.format(sensor_density)     # 4

    result_date = '10.21-2'                                # 5
    train_percent = int(gran*gran/(40*40)*100)           # 6
    output_dir  = 'results/{}'.format(result_date)
    train = TrainingInfo.naive_factory(data_source, sub_training_data, train_percent)
    subsample_from_full(train, grid_len, sensor_density, transmit_power)       # 8

    print(train)
    server_support = ServerSupport(train.hostname_loc, output_dir, output_file, train.tx_calibrate)
    ll = Localization(grid_len=grid_len, case=data_source, debug=False)
    ll.init_data(train.cov, train.sensors, train.hypothesis, SplatMap)

    app.run(host="0.0.0.0", port=5000 + int(port), debug=False)
