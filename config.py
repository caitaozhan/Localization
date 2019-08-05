'''Naive Factory for thresholds
'''

class Config:
    def __init__(self, q_threshold_1, q_threshold_2, q_prime_threshold_1, q_prime_threshold_2,\
                       r_list, r_2, edge, noise_floor_prune, center_threshold, surround_threshold, error_threshold):
        '''
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

    
    def __str__(self):
        return 'Q      = {}\n'.format(self.Q) + \
               'Q2     = {}\n'.format(self.Q2) + \
               'Q\' 1   = {}\n'.format(self.Q_prime1) + \
               'Q\' 2   = {}\n'.format(self.Q_prime2) + \
               'R_list = {}\n'.format(self.R_list) + \
               'R2     = {}\n'.format(self.R2) + \
               'edge   = {}\n'.format(self.edge) + \
               'noise floor prune  = {}\n'.format(self.noise_floor_prune) + \
               'center threshold   = {}\n'.format(self.center_threshold) + \
               'surround threshold = {}\n'.format(self.surround_threshold) + \
               'error threshold    = {}\n'.format(self.error_threshold)

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

            return cls(q_threshold_1=q, q_threshold_2=q2, q_prime_threshold_1=q_prime1, q_prime_threshold_2=q_prime2,\
                       r_list=r, r_2=r2, edge=e, noise_floor_prune=nf_p, center_threshold=c_thre, surround_threshold=s_thre, error_threshold = e_thre)

        if case == 'splat':
            q        = 2.3
            q2       = 2.
            q_prime1 = 0.5
            q_prime2 = 0.1
            r        = [8, 6, 5, 4]
            r2       = 6
            e        = 2
            nf_p     = -75
            c_thre   = -65
            s_thre   = -75
            e_thre   = 0.2

            return cls(q_threshold_1=q, q_threshold_2=q2, q_prime_threshold_1=q_prime1, q_prime_threshold_2=q_prime2,\
                       r_list=r, r_2=r2, edge=e, noise_floor_prune=nf_p, center_threshold=c_thre, surround_threshold=s_thre, error_threshold = e_thre)
        
        if case == 'utah':
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

            return cls(q_threshold_1=q, q_threshold_2=q2, q_prime_threshold_1=q_prime1, q_prime_threshold_2=q_prime2,\
                       r_list=r, r_2=r2, edge=e, noise_floor_prune=nf_p, center_threshold=c_thre, surround_threshold=s_thre, error_threshold = e_thre)
        
        else:
            print('unknown case', case)
        
