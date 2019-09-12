'''Localization server
'''
import time
import argparse
import numpy as np
from flask import Flask, request
from loc_default_config import TrainingPath
from localize import Localization
from input_output import Input, Output

import sys
sys.path.append('rtl-testbed/')
try:
    from default_config import OutdoorMap, IndoorMap
except:
    print('Import error')

app = Flask(__name__)

@app.route('/localize', methods=['POST'])
def localize():
    '''process the POST request'''
    data = request.get_json()   # type(data) = dict
    for key, value in data.items():
        print('key   :', key)
        print('value :', value)

    sensor_data = data["sensor_data"]
    sensor_output = np.zeros(len(sensor_data))
    for hostname, rss in sensor_data.items():
        index = server_support.get_index(hostname)
        sensor_output[index] = rss

    ground_truth = data['ground_truth']
    true_locations, true_powers, intruders = server_support.parse_ground_truth(ground_truth, ll)

    i = data['experiment_num']

    start = time.time()
    pred_locations, pred_power = ll.our_localization(sensor_output, intruders, i)
    end = time.time()
    errors, miss, false_alarm, power_errors = ll.compute_error(true_locations, true_powers, pred_locations, pred_power)
    output = Output(errors, false_alarm, miss, power_errors, end-start)
    print(output)
    return str(data) + '\n'


class ServerSupport:
    '''Misc things to support the server running
    '''
    def __init__(self, sensors_hostname):
        self.hostname_2_index = {}
        self.init_hostname_2_index(sensors_hostname)

    def init_hostname_2_index(self, sensors_hostname):
        '''init a dictionary'''
        with open(sensors_hostname, 'r') as f:
            counter = 0
            for line in f:
                line = line.split(':')
                self.hostname_2_index[line[0]] = counter
                counter += 1

    def get_index(self, hostname):
        index = self.hostname_2_index.get(hostname)
        if index is not None:
            return index
        else:
            raise Exception('hostname {} do not exist'.format(hostname))


    def parse_ground_truth(self, ground_truth, ll, train_power=50):
        '''parse the ground truth from the client
        Args:
            ground_truth -- {...} eg. {'T1': {'location': [9.5, 5.5], 'gain': '50'}}
            ll           -- Localization
        Return:
            true_locations -- list<(int, int)>
            true_powers    -- list<float>
            intruders      -- list<Transmitter>
        '''
        grid_len = ll.grid_len
        true_locations, true_powers = [], []
        intruders = []
        for _, truth in ground_truth.items():
            for key, value in truth.items():
                if key == 'location':
                    one_d_index = (int(value[0])*grid_len + int(value[1]))
                    two_d_index = (int(value[0]), int(value[1]))
                    true_locations.append(two_d_index)
                    intruders.append(ll.transmitters[one_d_index])
                elif key == 'gain':
                    true_powers.append(train_power - float(value))
                else:
                    raise Exception('key = {} invalid!'.format(key))
        return true_locations, true_powers, intruders


data_source = 'testbed-indoor'
output_file = 'output/results'
train = TrainingPath.naive_factory(data_source)
server_support = ServerSupport(train.sensors_hostname)
ll = Localization(grid_len=10, case=data_source, debug=True)
ll.init_data(train.cov, train.sensors, train.hypothesis, IndoorMap)  # improve map


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='localization server')
    parser.add_argument('-src', '--data_source', type=str, nargs=1, default=['testbed-indoor'], help='data source: testbed-indoor, testbed-outdoor')
    parser.add_argument('-of', '--output_file', type=str, nargs=1, default=['output/results'], help='the localization results')
    args = parser.parse_args()
    data_source = args.data_source[0]
    output_file = args.output_file[0]
    train = TrainingPath.naive_factory(data_source)
    server_support = ServerSupport(train.sensors_hostname)
    ll = Localization(grid_len=10, case=data_source, debug=True)
    ll.init_data(train.cov, train.sensors, train.hypothesis, IndoorMap)  # improve map

    app.run(debug=True)
