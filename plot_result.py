'''
Generate plots that go into the paper
'''

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from input_output import IOUtility
import tabulate


class PlotResult:
    '''Class for plotting results
    '''
    plt.rcParams['font.size'] = 60
	# plt.rcParams['font.weight'] = 'bold'
	# plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['lines.linewidth'] = 10
    plt.rcParams['lines.markersize'] = 15

    @staticmethod
    def reduce_avg(vals):
        vals = [val for val in vals if np.isnan(val)==False]
        return np.mean(vals)

    @staticmethod
    def error_numintru(data, src, train_percent):
        '''Plot the error
            y-axis: error (m)
            x-axis: # intruders
        Args:
            data -- [ (Input, {str: Output}), ... ]
            src  -- str -- source of data
            train_percent -- int -- training data percentage
        '''
        metric = 'error'            # y-axis
        methods = ['our', 'splot']  # tha bars
        reduce_f = PlotResult.reduce_avg
        table = defaultdict(list)
        for myinput, output_by_method in data:
            if myinput.data_source == src and myinput.train_percent == train_percent:
                table[myinput.num_intruder].append({method: output.get_metric(metric) for method, output in output_by_method.items()})

        print_table = [[x] + [reduce_f([y_by_method[method] for y_by_method in list_of_y_by_method]) for method in methods] for x, list_of_y_by_method in sorted(table.items())]
        print(tabulate.tabulate(print_table, headers = ['NUM INTRU'] + [method + ' ' + metric for method in methods]), '\n')


    @staticmethod
    def missfalse_numintru(data, src, train_percent):
        '''Plot the miss and false alarm
            y-axis: miss and false stacked together
            x-axis: # intruders
        Args:
            data -- [ (Input, {str: Output}), ... ]
            src  -- str -- source of data
            train_percent -- int -- training data percentage
        '''
        metrics = ['miss', 'false_alarm']            # y-axis
        methods = ['our', 'splot']     # tha bars
        reduce_f = PlotResult.reduce_avg
        table = defaultdict(list)
        for metric in metrics:
            for myinput, output_by_method in data:
                if myinput.data_source == src and myinput.train_percent == train_percent:
                    table[myinput.num_intruder].append({method: output.get_metric(metric) for method, output in output_by_method.items()})

            print_table = [[x] + [reduce_f([y_by_method[method] for y_by_method in list_of_y_by_method]) for method in methods] for x, list_of_y_by_method in sorted(table.items())]
            print(tabulate.tabulate(print_table, headers = ['NUM INTRU'] + [method + ' ' + metric for method in methods]), '\n')



if __name__ == '__main__':

    logs = ['results/9.20/log', 'results/9.23/log', 'results/9.24/log', 'results/9.26/log']
    data = IOUtility.read_logs(logs)

    PlotResult.error_numintru(data, src='testbed-indoor', train_percent=100)
    PlotResult.missfalse_numintru(data, src='testbed-indoor', train_percent=100)

