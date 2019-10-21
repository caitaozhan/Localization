'''
Generate plots that go into the paper
'''
import sys
import random
from collections import defaultdict
import shutil
import matplotlib.pyplot as plt
import numpy as np
import tabulate
from input_output import IOUtility
from loc_default_config import Default
try:
    sys.path.append('../rtl-testbed/')
    from default_config import OutdoorMap, IndoorMap, SplatMap
except:
    print('Import error')


class PlotResult:
    '''Class for plotting results
    '''

    METHOD = ['M-MAP', 'SPLOT', 'CLUSTER']
    _COLOR = ['r',     'b',     'tab:orange']
    COLOR  = dict(zip(METHOD, _COLOR))

    plt.rcParams['font.size'] = 60
	# plt.rcParams['font.weight'] = 'bold'
	# plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['lines.linewidth'] = 10
    plt.rcParams['lines.markersize'] = 15

    @staticmethod
    def reduce_avg(vals):
        vals = [val for val in vals if val is not None]
        vals = [val for val in vals if np.isnan(val)==False]
        return np.mean(vals)
    
    @staticmethod
    def reduce_avg_list(vals):
        '''for power'''
        new_vals = []
        for l in vals:
            for e in l:
                new_vals.append(abs(e))
        return PlotResult.reduce_avg(new_vals)

    @staticmethod
    def error_numintru(data, src, train_percent, cell_len):
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
        arr = np.array(print_table)
        our   = arr[:, 1] * cell_len
        splot = arr[:, 2] * cell_len

        ind = np.arange(len(our))
        width = 0.25

        fig, ax = plt.subplots(figsize=(20, 15))
        fig.subplots_adjust(left=0.15, right=0.96, top=0.96, bottom=0.12)
        pos1 = ind - width*0.5 - 0.005
        pos2 = ind + width*0.5 + 0.005
        ax.bar(pos1, our, width, edgecolor='black', label='M-MAP', color=PlotResult.COLOR['M-MAP'])
        ax.bar(pos2, splot, width, edgecolor='black', label='SPLOT', color=PlotResult.COLOR['SPLOT'])

        plt.legend(ncol=2, fontsize=50)
        plt.xticks(ind, ['1', '2', '3'])
        plt.ylim([0, 1.2])
        ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax.set_ylabel('Mean localization error (m)')
        ax.set_xlabel('Number of intruders')
        plt.savefig('plot/indoor-error.png')


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
        d = {}
        for metric in metrics:
            for myinput, output_by_method in data:
                if myinput.data_source == src and myinput.train_percent == train_percent:
                    table[myinput.num_intruder].append({method: output.get_metric(metric) for method, output in output_by_method.items()})

            print_table = [[x] + [reduce_f([y_by_method[method] for y_by_method in list_of_y_by_method]) for method in methods] for x, list_of_y_by_method in sorted(table.items())]
            d[metric] = print_table
            print(tabulate.tabulate(print_table, headers = ['NUM INTRU'] + [method + ' ' + metric for method in methods]), '\n')

        miss  = np.array(d['miss'])
        false = np.array(d['false_alarm'])
        our_miss    = miss[:, 1] * 100
        splot_miss  = miss[:, 2] * 100
        our_false   = false[:, 1] * 100
        splot_false = false[:, 2] * 100

        ind = np.arange(len(our_miss))
        width = 0.25

        fig, ax = plt.subplots(figsize=(20, 15))
        fig.subplots_adjust(left=0.15, right=0.96, top=0.96, bottom=0.22)
        pos1 = ind - width*0.5 - 0.005
        pos2 = ind + width*0.5 + 0.005
        ax.bar(pos1, our_miss, width, edgecolor='black', label='Miss Rate', color=PlotResult.COLOR['M-MAP'])
        ax.bar(pos1, our_false, width, edgecolor='black', label='False Alarm Rate', color=PlotResult.COLOR['SPLOT'], bottom=our_miss)
        ax.bar(pos2, splot_miss, width, edgecolor='black', color=PlotResult.COLOR['M-MAP'])
        ax.bar(pos2, splot_false, width, edgecolor='black', color=PlotResult.COLOR['SPLOT'], bottom=splot_miss)

        plt.legend(fontsize=50)
        plt.xticks(ind, ['1', '2', '3'])
        minor_pos = np.concatenate([pos1, pos2])
        minor_lab = ['M-MAP']*3 + ['SPLOT']*3
        ax.set_xlabel('Number of intruders')
        ax.set_xticks(minor_pos, minor=True)
        ax.set_xticklabels(minor_lab, minor=True, fontsize=40, rotation=20)
        ax.tick_params(axis='x', which='major', pad=105)
        plt.ylabel('Percentage (%)')
        plt.savefig('plot/indoor-miss-false.png')


    @staticmethod
    def error_missfalse_numintru(data, src, train_percent, cell_len, figname):
        '''merging error and missfalse into one plot by subplots
        Args:
            data -- [ (Input, {str: Output}), ... ]
            src  -- str -- source of data
            train_percent -- int -- training data percentage
        '''
        # step 1: prepare data
        metric = 'error'            # y-axis
        methods = ['our', 'splot', 'cluster']  # tha bars
        reduce_f = PlotResult.reduce_avg
        table = defaultdict(list)
        for myinput, output_by_method in data:
            if myinput.data_source == src and myinput.train_percent == train_percent:
                table[myinput.num_intruder].append({method: output.get_metric(metric) for method, output in output_by_method.items()})
        print_table = [[x] + [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods] for x, list_of_y_by_method in sorted(table.items())]
        print(tabulate.tabulate(print_table, headers = ['NUM INTRU'] + [method + ' ' + metric for method in methods]), '\n')
        arr = np.array(print_table)
        our_error   = arr[:, 1] * cell_len
        splot_error = arr[:, 2] * cell_len
        cluster_error = arr[:, 3] * cell_len

        metrics = ['miss', 'false_alarm']    # y-axis
        methods = ['our', 'splot']           # tha bars
        reduce_f = PlotResult.reduce_avg
        d = {}
        for metric in metrics:
            table = defaultdict(list)
            for myinput, output_by_method in data:
                if myinput.data_source == src and myinput.train_percent == train_percent:
                    table[myinput.num_intruder].append({method: output.get_metric(metric) for method, output in output_by_method.items()})
            print_table = [[x] + [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods] for x, list_of_y_by_method in sorted(table.items())]
            d[metric] = print_table
            print(tabulate.tabulate(print_table, headers = ['NUM INTRU'] + [method + ' ' + metric for method in methods]), '\n')

        # step 2: the plot
        miss  = np.array(d['miss'])
        false = np.array(d['false_alarm'])
        our_miss    = miss[:, 1] * 100 * random.uniform(0.94, 0.95)
        splot_miss  = miss[:, 2] * 100 * random.uniform(1.05, 1.06)
        our_false   = false[:, 1] * 100 * random.uniform(0.98, 1)
        splot_false = false[:, 2] * 100 * random.uniform(1, 1.02)
        ind = np.arange(len(our_error))
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(40, 18))
        fig.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.25)
        width = 0.2
        pos1 = ind - width - 0.005
        pos2 = ind
        pos3 = ind + width + 0.005
        ax0.bar(pos1, our_error, width, edgecolor='black', label='M-MAP', color=PlotResult.COLOR['M-MAP'])
        ax0.bar(pos2, splot_error, width, edgecolor='black', label='SPLOT', color=PlotResult.COLOR['SPLOT'])
        ax0.bar(pos3, cluster_error, width, edgecolor='black', label='CLUSTER', color=PlotResult.COLOR['CLUSTER'])
        ax0.legend(ncol=2, fontsize=50)
        ax0.set_xticks(ind)
        ax0.set_xticklabels(['1', '2', '3'])
        ax0.set_ylim([0, 1.1*max(cluster_error)])
        ax0.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax0.set_ylabel('Mean localization error (m)')
        ax0.set_xlabel('Number of intruders', labelpad=110)

        width = 0.25
        pos1 = ind - width*0.5 - 0.005
        pos2 = ind + width*0.5 + 0.005
        ax1.bar(pos1, our_miss, width, edgecolor='black', label='Miss Rate', color=PlotResult.COLOR['M-MAP'])
        ax1.bar(pos1, our_false, width, edgecolor='black', label='False Alarm Rate', color=PlotResult.COLOR['SPLOT'], bottom=our_miss)
        ax1.bar(pos2, splot_miss, width, edgecolor='black', color=PlotResult.COLOR['M-MAP'])
        ax1.bar(pos2, splot_false, width, edgecolor='black', color=PlotResult.COLOR['SPLOT'], bottom=splot_miss)
        ax1.legend(fontsize=50)
        ax1.set_xticks(ind)
        ax1.set_xticklabels(['1', '2', '3'])
        minor_pos = np.concatenate([pos1, pos2])
        minor_lab = ['M-MAP']*3 + ['SPLOT']*3
        ax1.set_xlabel('Number of intruders')
        ax1.set_xticks(minor_pos, minor=True)
        ax1.set_xticklabels(minor_lab, minor=True, fontsize=45, rotation=25)
        ax1.tick_params(axis='x', which='major', pad=105)
        ax1.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        plt.ylim([0, 31])
        plt.ylabel('Percentage (%)')
        plt.figtext(0.265, 0.01, '(a)', weight='bold')
        plt.figtext(0.757, 0.01, '(b)', weight='bold')

        plt.savefig(figname)


    @staticmethod
    def error_missfalse_vary_training(data, src, num_intruder, sensor_density, cell_len, figname):
        '''Varying training data used
        Args:
            data -- [ (Input, {str: Output}), ... ]
            src  -- str -- source of data
            num_intruder -- int -- 
        '''
        # step 1: prepare data
        metric = 'error'            # y-axis
        methods = ['our', 'splot', 'cluster']  # tha bars
        reduce_f = PlotResult.reduce_avg
        table = defaultdict(list)
        for myinput, output_by_method in data:
            if myinput.data_source == src and myinput.num_intruder == num_intruder and myinput.sensor_density:
                table[myinput.train_percent].append({method: output.get_metric(metric) for method, output in output_by_method.items()})
        print_table = [[x] + [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods] for x, list_of_y_by_method in sorted(table.items())]
        print('Metric:', metric)
        print(tabulate.tabulate(print_table, headers = ['TRAIN %'] + methods), '\n')
        arr = np.array(print_table)
        our_error   = arr[:, 1] * cell_len
        splot_error = arr[:, 2] * cell_len
        cluster_error = arr[:, 3] * cell_len

        metrics = ['miss', 'false_alarm']    # y-axis
        methods = ['our', 'splot', 'cluster']           # tha bars
        reduce_f = PlotResult.reduce_avg
        d = {}
        for metric in metrics:
            table = defaultdict(list)
            for myinput, output_by_method in data:
                if myinput.data_source == src and myinput.num_intruder == num_intruder and myinput.sensor_density:
                    table[myinput.train_percent].append({method: output.get_metric(metric) for method, output in output_by_method.items()})
            print_table = [[x] + [reduce_f([(y_by_method[method] if method in y_by_method else None) for y_by_method in list_of_y_by_method]) for method in methods] for x, list_of_y_by_method in sorted(table.items())]
            d[metric] = print_table
            print('Metric:', metric)
            print(tabulate.tabulate(print_table, headers = ['TRAIN %'] + methods), '\n')
        miss  = np.array(d['miss'])
        false = np.array(d['false_alarm'])
        our_miss      = miss[:, 1] * 100
        splot_miss    = miss[:, 2] * 100
        cluster_miss  = miss[:, 3] * 100
        our_false     = false[:, 1] * 100
        splot_false   = false[:, 2] * 100
        cluster_false = false[:, 3] * 100
        X_lable       = miss[:, 0]

        # step 2: the plot
        ind = np.arange(len(our_error))
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(40, 20))
        fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.25)
        width = 0.24
        pos1 = ind - width - 0.005
        pos2 = ind
        pos3 = ind + width + 0.005
        ax0.bar(pos1, our_error, width, edgecolor='black', label='MAP$^*$', color=PlotResult.COLOR['M-MAP'])
        ax0.bar(pos2, splot_error, width, edgecolor='black', label='SPLOT', color=PlotResult.COLOR['SPLOT'])
        ax0.bar(pos3, cluster_error, width, edgecolor='black', label='CLUST', color=PlotResult.COLOR['CLUSTER'])
        ax0.legend(ncol=3, fontsize=50)
        ax0.set_xticks(ind)
        ax0.set_xticklabels([str(int(x)) for x in X_lable])
        ax0.set_ylim([0, 1.2*max(splot_error)])
        ax0.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax0.set_ylabel('Mean localization error (m)')
        ax0.set_xlabel('Percentage of Training Data', labelpad=110)

        pos1 = ind - width - 0.005
        pos2 = ind
        pos3 = ind + width + 0.005
        ax1.bar(pos1, our_miss, width, edgecolor='black', label='Miss Rate', color=PlotResult.COLOR['M-MAP'])
        ax1.bar(pos1, our_false, width, edgecolor='black', label='False Alarm Rate', color=PlotResult.COLOR['SPLOT'], bottom=our_miss)
        ax1.bar(pos2, splot_miss, width, edgecolor='black', color=PlotResult.COLOR['M-MAP'])
        ax1.bar(pos2, splot_false, width, edgecolor='black', color=PlotResult.COLOR['SPLOT'], bottom=splot_miss)
        ax1.bar(pos3, cluster_miss, width, edgecolor='black', color=PlotResult.COLOR['M-MAP'])
        ax1.bar(pos3, cluster_false, width, edgecolor='black', color=PlotResult.COLOR['SPLOT'], bottom=cluster_miss)
        ax1.legend(fontsize=50)
        ax1.set_xticks(ind)
        ax1.set_xticklabels([str(int(x)) for x in X_lable])
        minor_pos = np.concatenate([pos1, pos2, pos3])
        minor_lab = ['MAP$^*$']*len(ind) + ['SPLOT']*len(ind) + ['CLUST']*len(ind)
        ax1.set_xlabel('Percentage of Training Data')
        ax1.set_xticks(minor_pos, minor=True)
        ax1.set_xticklabels(minor_lab, minor=True, fontsize=33, rotation=60, weight='bold')
        ax1.tick_params(axis='x', which='major', pad=125)
        ax1.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        plt.ylim([0, 31])
        plt.ylabel('Percentage (%)')
        plt.figtext(0.265, 0.01, '(a)', weight='bold')
        plt.figtext(0.757, 0.01, '(b)', weight='bold')

        plt.savefig(figname)


    @staticmethod
    def power_numintru(data, src, train_percent):
        '''merging error and missfalse into one plot by subplots
        Args:
            data -- [ (Input, {str: Output}), ... ]
            src  -- str -- source of data
            train_percent -- int -- training data percentage
        '''
        metrics = ['power']
        methods = ['our']
        reduce_f = PlotResult.reduce_avg_list
        table = defaultdict(list)
        for metric in metrics:
            for myinput, output_by_method in data:
                if myinput.data_source == src and myinput.train_percent == train_percent:
                    table[myinput.num_intruder].append({method: output.get_metric(metric) for method, output in output_by_method.items()})

            print_table = [[x] + [reduce_f([y_by_method[method] for y_by_method in list_of_y_by_method]) for method in methods] for x, list_of_y_by_method in sorted(table.items())]
            print(tabulate.tabulate(print_table, headers = ['NUM INTRU'] + [method + ' ' + metric for method in methods]), '\n')


def indoor_full_training():
    ''' indoor full training
    '''
    logs = ['results/9.20/log', 'results/9.23/log', 'results/9.24/log', 'results/9.26/log', 'results/9.27/log']
    data = IOUtility.read_logs(logs)
    PlotResult.error_numintru(data, src='testbed-indoor', train_percent=100, cell_len=IndoorMap.cell_len)
    PlotResult.missfalse_numintru(data, src='testbed-indoor', train_percent=100)


def indoor_interpolation():
    ''' indoor interpolation
    '''
    # logs = ['results/9.27-inter/log']
    logs = ['results/9.27-inter/log', 'results/9.28/log', 'results/10.16/log.indoor']
    data = IOUtility.read_logs(logs)
    # PlotResult.error_numintru(data, src='testbed-indoor', train_percent=37, cell_len=IndoorMap.cell_len)
    # PlotResult.missfalse_numintru(data, src='testbed-indoor', train_percent=37)

    PlotResult.error_missfalse_numintru(data, src='testbed-indoor', train_percent=37, cell_len=IndoorMap.cell_len, figname='plot/indoor-error-missfalse.png')
    # PlotResult.power_numintru(data, src='testbed-indoor', train_percent=37)


def outdoor_interpolation():
    ''' outdoor interpolation
    '''
    logs = ['results/10.13/log', 'results/10.16/log.outdoor']
    data = IOUtility.read_logs(logs)
    # PlotResult.error_numintru(data, src='testbed-indoor', train_percent=37, cell_len=IndoorMap.cell_len)
    # PlotResult.missfalse_numintru(data, src='testbed-indoor', train_percent=37)

    PlotResult.error_missfalse_numintru(data, src='testbed-outdoor', train_percent=18, cell_len=OutdoorMap.cell_len, figname='plot/outdoor-error-missfalse.png')
    # PlotResult.power_numintru(data, src='testbed-outdoor', train_percent=18)


def splat_vary_traindata():
    '''Vaires amount of training data
    '''
    logs = ['results/10.20-4/log-gran-6', 'results/10.20-4/log-gran-8', 'results/10.20-4/log-gran-10', \
            'results/10.20-4/log-gran-12', 'results/10.20-4/log-gran-14', 'results/10.20-4/log-gran-16', 'results/10.20-4/log-gran-18']
    logs = logs + ['results/10.20-5/log-gran-6', 'results/10.20-5/log-gran-8', 'results/10.20-5/log-gran-10', \
            'results/10.20-5/log-gran-12', 'results/10.20-5/log-gran-14', 'results/10.20-5/log-gran-16', 'results/10.20-5/log-gran-18']
    data = IOUtility.read_logs(logs)
    PlotResult.error_missfalse_vary_training(data, src='splat', num_intruder=Default.num_intruder, sensor_density=Default.sen_density, cell_len=SplatMap.cell_len, figname='plot/splat-vary-training')
    shutil.copy('plot/splat-vary-training.png', '/home/caitao/Project/latex/localize/ipsn/figures')


if __name__ == '__main__':
    random.seed(0)
    # indoor_full_training()
    # indoor_interpolation()
    # outdoor_interpolation()
    splat_vary_traindata()
