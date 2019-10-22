'''
The client side, send localization request to the server
'''

import argparse
import random
import numpy as np
import requests
from subprocess import PIPE, Popen

from input_output import Input, Output
from localize import Localization
from utility import generate_intruders
from loc_default_config import Default



class Client:
    @staticmethod
    def test_server(ip, port):
        url = 'http://{}:{}/on'.format(ip, port)
        requests.get(url=url)

    @staticmethod
    def prepare_sensor_output(sensor_outputs):
        sensordata = {}
        for i, sen_output in enumerate(sensor_outputs):
            sensordata[str(i)] = round(sen_output, 2)            # hostname is sensor index
        return sensordata

    @staticmethod
    def prepare_ground_truth(intruders, true_powers):
        tx_data = {}
        for i, intru in enumerate(intruders):
            t_x = round(intru.x + random.uniform(0, 1), 2)
            t_y = round(intru.y + random.uniform(0, 1), 2)
            power = round(Default.trained_power + true_powers[i], 2)
            tx_data[str(i)] = {
                               "location":(t_x, t_y),
                               "gain":str(power)
                              }
        return tx_data


# ** Only intruders **

# if __name__ == '__main__':

#     hint = 'python client.py -gran 12 -num 5 -sen 240 -met our -rep 10 -p 5012'

#     parser = argparse.ArgumentParser(description='client side | hint: ' + hint)
#     parser.add_argument('-gran', '--training_gran', type=int, nargs=1 ,default=[Default.training_gran], help='Training granularity')
#     parser.add_argument('-num', '--num_intruder', type=int, nargs=1, default=[Default.num_intruder], help='Number of intruders')
#     parser.add_argument('-sen', '--sen_density', type=int, nargs=1, default=[Default.sen_density], help='Sensor density')
#     parser.add_argument('-met', '--methods', type=str, nargs='+', default=Default.methods, help='Methods to compare')
#     parser.add_argument('-rep', '--repeat', type=int, nargs=1, default=[Default.repeat], help='Number of experiments to repeat')
#     parser.add_argument('-p',   '--port', type=int, nargs=1, default=[5012], help='Different port of the server holds different data')
#     args = parser.parse_args()

#     training_gran = args.training_gran[0]
#     num_intruder  = args.num_intruder[0]
#     sen_density   = args.sen_density[0]
#     methods       = args.methods
#     repeat        = args.repeat[0]
#     port          = args.port[0]

#     # Client.test_server(Default.server_ip, port)

#     myinput = Input(num_intruder=num_intruder, data_source='splat', methods=methods, sen_density=sen_density)

#     # initialize a Localization object with the ground truth, use it to generate read data
#     ll = Localization(grid_len=40, case='splat', debug=False)
#     true_data_path = Default.true_data_path.format(sen_density)
#     cov_file = true_data_path + '/cov'
#     sensor_file = true_data_path + '/sensors'
#     hypothesis_file = true_data_path + '/hypothesis'
#     print('client side true data: \n{}\n{}\n{}\n'.format(cov_file, sensor_file, hypothesis_file))
#     ll.init_data(cov_file, sensor_file, hypothesis_file)
#     ll.init_truehypo(hypothesis_file)

#     if repeat > 0:
#         myrange = range(repeat)
#     if repeat <= 0:
#         myrange = range(-repeat, -(repeat-1))
#     print('myrange is:', myrange)
#     for i in myrange:
#         random.seed(i)
#         np.random.seed(i)

#         # generate testing data and the ground truth
#         true_powers = [random.uniform(-2, 2) for i in range(num_intruder)]
#         true_indices, true_powers = generate_intruders(grid_len=ll.grid_len, edge=2, num=num_intruder, min_dist=1, powers=true_powers)
#         intruders, sensor_outputs = ll.set_intruders(true_indices=true_indices, powers=true_powers, randomness=True, truemeans=True)

#         # set up myinput
#         myinput.experiment_num = i
#         myinput.train_percent = int(training_gran**2/Default.grid_len**2 * 100)
#         myinput.sensor_data = Client.prepare_sensor_output(sensor_outputs)
#         myinput.ground_truth = Client.prepare_ground_truth(intruders, true_powers)
#         print(myinput.experiment_num, '\n', myinput.ground_truth, '\n')

#         curl = "curl -d \'{}\' -H \'Content-Type: application/json\' -X POST http://{}:{}/localize"
#         command = curl.format(myinput.to_json_str(), Default.server_ip, port)
#         p = Popen(command, stdout=PIPE, shell=True)
#         p.wait()



# ** authorized users + intruders **

if __name__ == '__main__':

    hint = 'python client.py -gran 12 -num 5 -sen 240 -met our -rep 10 -p 5012'

    parser = argparse.ArgumentParser(description='client side | hint: ' + hint)
    parser.add_argument('-gran', '--training_gran', type=int, nargs=1 ,default=[Default.training_gran], help='Training granularity')
    parser.add_argument('-num', '--num_intruder', type=int, nargs=1, default=[Default.num_intruder], help='Number of intruders')
    parser.add_argument('-sen', '--sen_density', type=int, nargs=1, default=[Default.sen_density], help='Sensor density')
    parser.add_argument('-met', '--methods', type=str, nargs='+', default=Default.methods, help='Methods to compare')
    parser.add_argument('-rep', '--repeat', type=int, nargs=1, default=[Default.repeat], help='Number of experiments to repeat')
    parser.add_argument('-p',   '--port', type=int, nargs=1, default=[5012], help='Different port of the server holds different data')
    args = parser.parse_args()

    training_gran = args.training_gran[0]
    num_intruder  = args.num_intruder[0]
    sen_density   = args.sen_density[0]
    methods       = args.methods
    repeat        = args.repeat[0]
    port          = args.port[0]

    # Client.test_server(Default.server_ip, port)

    myinput = Input(num_intruder=num_intruder, data_source='splat', methods=methods, sen_density=sen_density)

    # initialize a Localization object with the ground truth, use it to generate read data
    ll = Localization(grid_len=40, case='splat', debug=False)
    true_data_path = Default.true_data_path.format(sen_density)
    cov_file = true_data_path + '/cov'
    sensor_file = true_data_path + '/sensors'
    hypothesis_file = true_data_path + '/hypothesis'
    print('client side true data: \n{}\n{}\n{}\n'.format(cov_file, sensor_file, hypothesis_file))
    ll.init_data(cov_file, sensor_file, hypothesis_file)
    ll.init_truehypo(hypothesis_file)

    if repeat > 0:
        myrange = range(repeat)
    if repeat <= 0:
        myrange = range(-repeat, -(repeat-1))
    print('myrange is:', myrange)
    for i in myrange:
        random.seed(i)
        np.random.seed(i)

        # generate testing data and the ground truth
        true_powers = [random.uniform(-2, 2) for i in range(num_intruder)]
        true_indices, true_powers = generate_intruders(grid_len=ll.grid_len, edge=2, num=num_intruder, min_dist=1, powers=true_powers)

        # TODO: add authorized users

        intruders, sensor_outputs = ll.set_intruders(true_indices=true_indices, powers=true_powers, randomness=True, truemeans=True)

        # set up myinput
        myinput.experiment_num = i
        myinput.train_percent = int(training_gran**2/Default.grid_len**2 * 100)
        myinput.sensor_data = Client.prepare_sensor_output(sensor_outputs)
        myinput.ground_truth = Client.prepare_ground_truth(intruders, true_powers)
        print(myinput.experiment_num, '\n', myinput.ground_truth, '\n')

        curl = "curl -d \'{}\' -H \'Content-Type: application/json\' -X POST http://{}:{}/localize_ss"
        command = curl.format(myinput.to_json_str(), Default.server_ip, port)
        p = Popen(command, stdout=PIPE, shell=True)
        p.wait()
