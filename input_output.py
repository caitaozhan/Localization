'''
Encapsulate the Input and Output of the localization system
'''

import json

class Default:
    num_intruder = 5
    data_source  = 'splat'    # splat, utah, testbed-indoor, testbed-outdoor
    methods      = ['splot', 'our', 'cluster']



class Input:
    counter = 0
    def __init__(self, num_intruder = Default.num_intruder,  # int
                       data_source  = Default.data_source,   # str
                       methods      = Default.methods):      # list<str>
        self.num_intruder   = num_intruder
        self.data_source    = data_source
        self.methods        = methods
        self.experiment_num = Input.counter
        self.sensor_data    = None
        self.ground_truth   = None
        Input.counter += 1
    
    def to_json(self):
        '''return string in json format
        '''
        inputdict = {'num_intruder':self.num_intruder,
                     'data_source':self.data_source,
                     'methods':self.methods,
                     'experiment_num':self.experiment_num,
                     'sensor_data':self.sensor_data,
                     'ground_truth':self.ground_truth}
        return json.dumps(inputdict)


class Output:
    def __init__(self, error = None,         # float
                       false_alarm = None,   # 
                       miss = None, 
                       time = None):
        self.error = error
        self.false_alarm = false_alarm
        self.miss = miss
        self.time = time
