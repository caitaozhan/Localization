'''Naive Factory for thresholds and filepath
'''

import json


class Default:
    grid_len       = 40
    training_gran  = 12    # [6, 8, 10, 12, 14, 16, 18]
    num_intruder   = 5     # [1, 3, 5, 7, 10]
    sen_density    = 240   # [80, 160, 240, 320, 400]
    num_authorized = 5     # [0 or 5] NOTE: watchout!
    repeat         = 10    # repeating experiments
    methods        = ['our', 'splot', 'cluster', 'our-ss']
    true_data_path = '../mysplat/output8_{}'
    trained_power  = 30
    server_ip      = '0.0.0.0'


class Config:
    def __init__(self, q_threshold_1, q_threshold_2, q_prime_threshold_1, q_prime_threshold_2,\
                       r_list, r_2, edge, noise_floor_prune, center_threshold, surround_threshold, error_threshold, delta_threshold):
        '''
        Configurations for our algorithm
        Args:
            q_threshold_1 (float): the q threshold in procedure 1. how far away from the mean?
            q_threshold_2 (float): the q threshold in procedure 2. how far away from the mean?
            q_prime_threshold_1 (float): the q prime threshold in procedure 1. how far away from the mean?
            q_prime_threshold_2 (float): the q prime threshold in procedure 2. how far away from the mean?
            r_list (list<int>)   : the R in procedure 1
            r_2    (int)         : the R in procedure 2
            edge (int)           : ignore the borders
            noise_floor_prune (int) : the noise floor threshold for pruning
            center_threshold (int)  : the threshold for center sensor in procedure 2
            surround_threshold (int): the threshold for surronding sensor in procedure 2
            error_threshold (float) : a fraction of the grid length
        '''
        self.Q        = q_threshold_1
        self.Q2       = q_threshold_2
        self.Q_prime1 = q_prime_threshold_1
        self.Q_prime2 = q_prime_threshold_2
        self.R_list   = r_list
        self.R2       = r_2
        self.edge     = edge
        self.noise_floor_prune  = noise_floor_prune
        self.center_threshold   = center_threshold
        self.surround_threshold = surround_threshold
        self.error_threshold    = error_threshold
        self.delta_threshold    = delta_threshold


    def __str__(self):
        return 'Q      = {}\n'.format(self.Q) + \
               'Q2     = {}\n'.format(self.Q2) + \
               'Q\' 1   = {}\n'.format(self.Q_prime1) + \
               'Q\' 2   = {}\n'.format(self.Q_prime2) + \
               'R_list = {}\n'.format(self.R_list) + \
               'R2     = {}\n'.format(self.R2) + \
               'edge   = {}\n'.format(self.edge) + \
               'noise floor prune     = {}\n'.format(self.noise_floor_prune) + \
               'center threshold      = {}\n'.format(self.center_threshold) + \
               'surround threshold    = {}\n'.format(self.surround_threshold) + \
               'error threshold       = {}\n'.format(self.error_threshold) + \
               'power delta threshold = {}\n'.format(self.delta_threshold)

    @classmethod
    def naive_factory(cls, case):
        if case == 'lognormal':
            q        = 1.9
            q2       = 2.
            q_prime1 = 0.8
            q_prime2 = 0.1
            r        = [8, 6, 5, 4]
            r2       = 6
            e        = 2
            nf_p     = -80
            c_thre   = -65
            s_thre   = -75
            e_thre   = 0.2
            d_thre   = 2    # power delta

            return cls(q_threshold_1=q, q_threshold_2=q2, q_prime_threshold_1=q_prime1, q_prime_threshold_2=q_prime2,\
                       r_list=r, r_2=r2, edge=e, noise_floor_prune=nf_p, center_threshold=c_thre, surround_threshold=s_thre, error_threshold = e_thre, delta_threshold=d_thre)

        elif case == 'splat':
            q        = 2.6
            q2       = 4
            q_prime1 = 0.4
            q_prime2 = 0.1
            r        = [8, 6, 5, 4, 3, 2]
            r2       = 6
            e        = 2
            nf_p     = -70
            c_thre   = -50
            s_thre   = -60
            e_thre   = 0.5
            d_thre   = 2

            return cls(q_threshold_1=q, q_threshold_2=q2, q_prime_threshold_1=q_prime1, q_prime_threshold_2=q_prime2,\
                       r_list=r, r_2=r2, edge=e, noise_floor_prune=nf_p, center_threshold=c_thre, surround_threshold=s_thre, error_threshold = e_thre, delta_threshold=d_thre)

        elif case == 'utah':
            q        = 3.
            q2       = 2.
            q_prime1 = 0.8
            q_prime2 = 0.1
            r        = [5, 4, 3]
            r2       = 4
            e        = 0
            nf_p     = -65
            c_thre   = -50
            s_thre   = -60
            e_thre   = 1
            d_thre   = 2

            return cls(q_threshold_1=q, q_threshold_2=q2, q_prime_threshold_1=q_prime1, q_prime_threshold_2=q_prime2,\
                       r_list=r, r_2=r2, edge=e, noise_floor_prune=nf_p, center_threshold=c_thre, surround_threshold=s_thre, error_threshold = e_thre, delta_threshold=d_thre)

        elif case == 'testbed-indoor':
            q        = 1.5
            q2       = 2.4
            q_prime1 = 0.3
            q_prime2 = 0.05
            r        = [3.1, 2.3]
            r2       = 2.3
            e        = 0
            nf_p     = -47.5
            c_thre   = -40
            s_thre   = -45
            e_thre   = 1
            d_thre   = 1.5

            return cls(q_threshold_1=q, q_threshold_2=q2, q_prime_threshold_1=q_prime1, q_prime_threshold_2=q_prime2,\
                       r_list=r, r_2=r2, edge=e, noise_floor_prune=nf_p, center_threshold=c_thre, surround_threshold=s_thre, error_threshold = e_thre, delta_threshold=d_thre)

        elif case == 'testbed-outdoor':
            q        = 1.9
            q2       = 1.9
            q_prime1 = 0.3
            q_prime2 = 0.05
            r        = [4.1, 3.1]
            r2       = 4.1
            e        = 0
            nf_p     = -47.5
            c_thre   = -42
            s_thre   = -45
            e_thre   = 1
            d_thre   = 2

            return cls(q_threshold_1=q, q_threshold_2=q2, q_prime_threshold_1=q_prime1, q_prime_threshold_2=q_prime2,\
                       r_list=r, r_2=r2, edge=e, noise_floor_prune=nf_p, center_threshold=c_thre, surround_threshold=s_thre, error_threshold = e_thre, delta_threshold=d_thre)

        else:
            print('unknown case', case)



class ConfigSplot:
    '''Configurations for SPLOT
    '''
    def __init__(self, R1, R2, localmax_threshold, sigma_x_square, delta_c, n_p, minPL, delta_N_square):
        self.R1 = R1                                    # radius for local maximal
        self.R2 = R2                                    # radius for localizing Tx
        self.localmax_threshold = localmax_threshold    # threshold for local maximal
        self.sigma_x_square = sigma_x_square
        self.delta_c = delta_c
        self.n_p = n_p
        self.minPL = minPL
        self.delta_N_square = delta_N_square


    @classmethod
    def naive_factory(cls, case):
        if case == 'testbed-indoor':
            R1 = 3
            R2 = 3
            localmax_threshold = -43
            sigma_x_square = 0.5
            delta_c = 1
            n_p = 2
            minPL = 1
            delta_N_square = 1

            return cls(R1, R2, localmax_threshold, sigma_x_square, delta_c, n_p, minPL, delta_N_square)

        elif case == 'testbed-outdoor':
            R1 = 3
            R2 = 3
            localmax_threshold = -42
            sigma_x_square = 0.5
            delta_c = 1
            n_p = 2
            minPL = 1
            delta_N_square = 1

            return cls(R1, R2, localmax_threshold, sigma_x_square, delta_c, n_p, minPL, delta_N_square)

        else:
            R1 = 8
            R2 = 8
            localmax_threshold = -60
            sigma_x_square = 0.5
            delta_c = 1
            n_p = 2
            minPL = 1.5
            delta_N_square = 1

            return cls(R1, R2, localmax_threshold, sigma_x_square, delta_c, n_p, minPL, delta_N_square)


class TrainingInfo:
    '''the information of training data
    '''
    def __init__(self, cov, sensors, hypothesis, hostname_loc, train_percent, train_power):
        self.cov = cov
        self.sensors = sensors
        self.hypothesis = hypothesis
        self.hostname_loc = hostname_loc
        self.train_percent = train_percent
        self.train_power = train_power
        self.tx_calibrate = None   # human calibration, so that these Tx transmite at similar power (during training)
        self.init_tx_calibrate()

    def init_tx_calibrate(self):
        '''init the tx calibration. Note that Tx are not homogeneous
        '''
        try:                                # for testbed
            lines = open(self.train_power, 'r').readlines()
            train_power = json.loads(lines[0])
            train_power = list(train_power.items())[0]
            if train_power[0] == 'T1':
                if train_power[1] == 53.0:
                    self.tx_calibrate = {"T1":53, "T2":53, "T3":26, "T5":23}
                elif train_power[1] == 45.0:
                    self.tx_calibrate = {"T1":45, "T2":45, "T3":15, "T5":17}
            elif train_power[0] == 'T2':
                if train_power[1] == 58.0:
                    self.tx_calibrate = {"T1":59, "T2":58, "T3":29, "T5":27}
            if self.tx_calibrate is None:
                raise Exception('error durring train power calibration ')
        except:
            self.tx_calibrate = {}
            for i in range(15):
                self.tx_calibrate[str(i)] = 30   # for splat assume all the Tx are well calibrated


    @classmethod
    def naive_factory(cls, data_source, data, train_percent):
        '''a naive factory function for training data path
        Args:
            data_source   -- str,
            data          -- str, eg. "9.15"
            train_percent -- int
        Return:
            TrainingInfo
        '''
        if data_source == 'testbed-indoor' or data_source == 'testbed-outdoor':
            cov = '../rtl-testbed/training/{}/cov'.format(data)
            sensors = '../rtl-testbed/training/{}/sensors'.format(data)
            hypothesis = '../rtl-testbed/training/{}/hypothesis'.format(data)
            hostname_loc = '../rtl-testbed/training/{}/hostname_loc'.format(data)
            train_power = '../rtl-testbed/training/{}/train_power'.format(data)
            return cls(cov, sensors, hypothesis, hostname_loc, train_percent, train_power)
        elif data_source == 'splat':
            cov = '../mysplat/ipsn/{}/cov'.format(data)
            sensors = '../mysplat/ipsn/{}/sensors'.format(data)
            hypothesis = '../mysplat/ipsn/{}/hypothesis'.format(data)
            hostname_loc = '../mysplat/ipsn/{}/hostname_loc'.format(data)
            train_power = '../mysplat/ipsn/{}/train_power'.format(data)
            return cls(cov, sensors, hypothesis, hostname_loc, train_percent, train_power)
        else:
            raise Exception('data source {} invalid'.format(data_source))

    def __str__(self):
        return 'Training data info:\ncov = {}\nsensors = {}\nhypothesis = {}\nsensors_hostname = {}\ntrain_power = {}\n'.format(\
                self.cov, self.sensors, self.hypothesis, self.hostname_loc, self.train_power)
