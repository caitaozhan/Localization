'''Localization server
'''
import os
import time
import argparse
import sys
import numpy as np
from flask import Flask, request
from loc_default_config import TrainingInfo
from localize import Localization
from input_output import Input, Output

try:
    sys.path.append('../rtl-testbed/')
    from default_config import OutdoorMap, IndoorMap
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
        pred_locations = ll.cluster_localization(intruders, np.copy(sensor_outputs), num_of_intruders=int(myinput.num_intruder))
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


data_source = 'testbed-outdoor'        # 1
training_data = '10.6.inter-idw+-sub'  # 2
result_date = '10.16'                  # 3
train_percent = 18                     # 4
output_dir  = 'results/{}'.format(result_date)
output_file = 'log'                    # 5
train = TrainingInfo.naive_factory(data_source, training_data, train_percent)
print(train)
server_support = ServerSupport(train.hostname_loc, output_dir, output_file, train.tx_calibrate)
ll = Localization(grid_len=10, case=data_source, debug=True)
if data_source == 'testbed-indoor':
    MAP = IndoorMap
elif data_source == 'testbed-outdoor':
    MAP = OutdoorMap
ll.init_data(train.cov, train.sensors, train.hypothesis, MAP)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='localization server')
    parser.add_argument('-src', '--data_source', type=str, nargs=1, default=['testbed-indoor'], help='data source: testbed-indoor, testbed-outdoor')
    parser.add_argument('-od', '--output_dir', type=str, nargs=1, default=['results/9.14'], help='the localization results')
    parser.add_argument('-of', '--output_file', type=str, nargs=1, default=['log'], help='the localization results')
    parser.add_argument('-td', '--training_date', type=str, nargs=1, default=[None], help='the date when trainig data is trained')
    args = parser.parse_args()

    data_source = args.data_source[0]
    output_file = args.output_file[0]
    training_date = args.training_date[0]

    train = TrainingInfo.naive_factory(data_source, training_date, 100)
    server_support = ServerSupport(train.hostname_loc, output_dir, output_file, train.tx_calibrate)
    ll = Localization(grid_len=10, case=data_source, debug=True)
    ll.init_data(train.cov, train.sensors, train.hypothesis, IndoorMap)  # improve map

    app.run(host="0.0.0.0", debug=True)
