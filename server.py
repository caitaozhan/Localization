'''Localization server
'''
import time
import argparse
from flask import Flask, request
from loc_default_config import TrainingPath
from localize import Localization

import sys
sys.path.append('rtl-testbed/')
from default_config import OutdoorMap, IndoorMap

app = Flask(__name__)

@app.route('/localize', methods=['POST'])
def localize():
    '''process the POST request'''
    data = request.get_json()   # type(data) = dict
    for key, value in data.items():
        print(key)
        print(value)
        print()
    # ll.our_localization()
    data['caitao'] = 'zhan'
    return str(data) + '\n'



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='localization server')
    parser.add_argument('-src', '--data_source', type=str, nargs=1, default=['testbed-indoor'], help='data source: testbed-indoor, testbed-outdoor')
    args = parser.parse_args()

    data_source = args.data_source[0]
    print(data_source)
    train = TrainingPath.naive_factory(data_source)
    ll = Localization(grid_len=10, case='testbed-indoor', debug=True)
    ll.init_data(train.cov, train.sensors, train.hypothesis, IndoorMap)  # improve map



    app.run(debug=True)
