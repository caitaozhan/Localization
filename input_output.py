'''
Encapsulate the Input and Output of the localization system
'''

import numpy as np
import json   # json.dumps(): dict --> str;  json.loads(): str --> dict

class Default:
    num_intruder = 5
    data_source  = 'splat'    # splat, utah, testbed-indoor, testbed-outdoor
    methods      = ['splot', 'our', 'cluster']



class Input:
    '''Encapsulate the input of the algorithm
    '''
    def __init__(self, num_intruder = Default.num_intruder,  # int
                       data_source  = Default.data_source,   # str
                       methods      = Default.methods):      # list<str>
        self.num_intruder   = num_intruder
        self.data_source    = data_source
        self.train_percent  = -1
        self.methods        = methods
        self.experiment_num = -1
        self.sensor_data    = None
        self.ground_truth   = None


    def to_json_str(self):
        '''return json formated string
        Return:
            str
        '''
        inputdict = {'num_intruder':self.num_intruder,
                     'data_source':self.data_source,
                     'train_percent':self.train_percent,
                     'methods':self.methods,
                     'experiment_num':self.experiment_num,
                     'ground_truth':self.ground_truth,
                     'sensor_data':self.sensor_data
                     }
        return json.dumps(inputdict)


    @classmethod
    def from_json_str(cls, json_str):
        '''Init an Input object from json string
        Args:
            json_str -- str
        Return:
            Input
        '''
        inputdict = json.loads(json_str)
        return cls.from_json_dict(inputdict)


    @classmethod
    def from_json_dict(cls, json_dict):
        '''Init an Input object from json dictionary
        Args:
            json_dict -- dict
        Return:
            Input
        '''
        myinput = cls(0, 0, 0)
        myinput.num_intruder   = json_dict['num_intruder']
        myinput.data_source    = json_dict['data_source']
        myinput.train_percent  = json_dict['train_percent']
        myinput.methods        = json_dict['methods']
        myinput.experiment_num = json_dict['experiment_num']
        myinput.sensor_data    = json_dict['sensor_data']
        myinput.ground_truth   = json_dict['ground_truth']
        return myinput


    def log(self):
        return self.to_json_str() + '\n'


class Output:
    '''Encapsulate the output of the algorithms
    '''
    def __init__(self, method = None,        # str
                       error  = [],          #
                       false_alarm = None,   # float
                       miss   = None,        # float
                       power  = None,        # []
                       time   = None,         # float
                       preds  = []):
        self.method = method
        self.error = [round(e, 3) for e in error]
        self.false_alarm = round(false_alarm, 3)
        self.miss = round(miss, 3)
        self.power = [round(p, 3) for p in power]
        self.time = round(time, 3)
        self.preds  = preds


    def get_avg_error(self):
        '''The average error
        '''
        return np.mean(self.error)


    def get_metric(self, metric):
        '''
        Args:
            metric -- str
        Return:
            float
        '''
        if metric == 'error':
            return self.get_avg_error()
        if metric == 'miss':
            return self.miss
        if metric == 'false_alarm':
            return self.false_alarm
        if metric == 'power':
            return self.power


    def to_json_str(self):
        '''return json formated string
        Return:
            str
        '''
        outputdict = {
            "method":self.method,
            "error":self.error,
            "false_alarm":self.false_alarm,
            "miss":self.miss,
            "power":self.power,
            "time":self.time,
            "preds":self.preds
        }
        return json.dumps(outputdict)

    def log(self):
        return self.to_json_str() + '\n'


    @classmethod
    def from_json_str(cls, json_str):
        '''Init an Output object from json
        Args:
            json_str -- str
        Return:
            Output
        '''
        outputdict = json.loads(json_str)
        return cls.from_json_dict(outputdict)


    @classmethod
    def from_json_dict(cls, json_dict):
        '''Init an Output object from json dictionary
        Args:
            json_dict -- dict
        Return:
            Output
        '''
        method = json_dict['method']
        error = json_dict['error']
        false_alarm = json_dict['false_alarm']
        miss = json_dict['miss']
        power = json_dict['power']
        time = json_dict['time']
        preds = json_dict['preds']
        return cls(method, error, false_alarm, miss, power, time, preds)


class IOUtility:

    @staticmethod
    def read_logs(logs):
        '''Read logs
        Args:
            logs -- [str, ...] -- a list of filenames
        Return:
            data -- [ (Input, {str: Output}), ... ] -- data to plot
        '''
        data = []
        for log in logs:
            f = open(log, 'r')
            while True:
                inputline = f.readline()
                if inputline == '':
                    break
                myinput = Input.from_json_str(inputline)
                output_by_method = {}
                outputline = f.readline()
                while outputline != '' and outputline != '\n':
                    output = Output.from_json_str(outputline)
                    output_by_method[output.method] = output
                    outputline = f.readline()
                data.append((myinput, output_by_method))
        return data


if __name__ == '__main__':
    input_json_dict = {
            'num_intruder':1,
            'data_source':'testbed-indoor',
            'train_percent':100,
            'methods':['our', 'splot'],
            'experiment_num':16,
            'sensor_data':{'host-101': -43.908313, 'host-102': -44.45974, 'host-104': -43.888960999999995, 'host-105': -44.3658905, 'host-110': -44.532863500000005, 'host-120': -44.250901, 'host-130': -44.527539000000004, 'host-140': -44.170377, 'host-150': -44.052081, 'host-158': -41.056728, 'host-160': -43.658579, 'host-166': -43.570872, 'host-170': -44.408883, 'host-180': -43.859264499999995},
            'ground_truth':{'T1': {'gain': '50', 'location': [9.5, 5.5]}}
            }
    ouroutput = Output('our', [0.5], 0.0, 0.0, 1.2, 2.0)
    splotoutput = Output('splot', [1], 0.0, 0.0, None, 0.5)
    myinput = Input.from_json_dict(input_json_dict)

    f = open('localize_log', 'w')
    f.write(myinput.log())
    f.write(ouroutput.log())
    f.write(splotoutput.log())
    f.write('\n')

    f.write(myinput.log())
    f.write(ouroutput.log())
    f.write(splotoutput.log())
    f.write('\n')

    f.write(myinput.log())
    f.write(ouroutput.log())
    f.write(splotoutput.log())
    f.write('\n')

    f.close()

    f = open('localize_log', 'r')
    data = []
    while True:
        inputline = f.readline()
        if inputline == '':
            break
        myinput = Input.from_json_str(inputline)
        output_by_method = {}
        outputline = f.readline()
        while outputline != '' and outputline != '\n':
            output = Output.from_json_str(outputline)
            output_by_method[output.method] = output
            outputline = f.readline()
        data.append((myinput, output_by_method))
