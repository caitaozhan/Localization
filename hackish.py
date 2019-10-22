'''
Some hackish stuff to get results
'''

import numpy as np
import random
from collections import defaultdict
from input_output import Input, Output

class Hackish:

    @staticmethod
    def read_data(logs):
        '''Neglect Splot's Output'''
        data = []   # an element is: Input -> Output
        for log in logs:
            f = open(log, 'r')
            while True:
                inputline = f.readline()
                if inputline == '':
                    break
                myinput = Input.from_json_str(inputline)
                outputline = f.readline()
                while outputline != '' and outputline != '\n':
                    output = Output.from_json_str(outputline)
                    if output.method == 'our':  # only our method can predict power
                        data.append((myinput, output))
                    outputline = f.readline()
        return data


    @staticmethod
    def indoor_power(logs):
        data = Hackish.read_data(logs)

    @staticmethod
    def outdoor_power(logs):
        '''Need to adjust power, then find the error ratio of fix power and not fix power
        '''
        ground_truth = {"T1":59, "T2":58, "T3":29}
        data = Hackish.read_data(logs)
        p_errors = defaultdict(list)
        for inpt, output in data:
            if inpt.num_intruder != 1:  # only considering the single intruder case
                continue
            try:
                p_error = output.power[0]
            except IndexError:
                continue
            for tx, truth in inpt.ground_truth.items():
                true_delta = float(truth['gain']) - ground_truth[tx]
            pred_delta = true_delta + p_error
            if pred_delta > 1:
                pred_delta = 1
            if pred_delta < -1:
                pred_delta = -1
            p_error = pred_delta - true_delta
            p_errors[true_delta].append(p_error)
        p_errors2 = {}
        all_errors = []
        for delta, errors in p_errors.items():
            error = np.mean(np.absolute(errors))
            print(delta, error, len(errors))
            p_errors2[delta] = error
            all_errors.extend(list(np.absolute(errors)))
        print('ratio of power error true_delta is 1 of -1 to true_delta is 0 = ', np.mean([p_errors2[1.0], p_errors2[-1.0]]) / p_errors2[0.0])
        print('all mean = {}'.format(np.mean(all_errors)))

def main1():
    '''power error'''
    logs = ['results/9.27-inter/log', 'results/9.28/log']
    Hackish.indoor_power(logs)

    logs = ['results/10.13/log']
    Hackish.outdoor_power(logs)

def main2():
    '''random power'''
    a = 2
    errors = []
    i = 0
    while i < 1000000:
        true = random.uniform(-a, a)
        pred = random.uniform(0, 0)
        error = abs(pred - true)
        errors.append(error)
        i += 1
    print(np.mean(errors), np.std(errors))

def main3():
    '''Interpolation for outdoors'''
    pass

if __name__ == '__main__':
    # main1()
    main2()