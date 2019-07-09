'''Naive Factory for thresholds
'''

class Config:
    def __init__(self, q_threshold, r_list, edge, noise_floor_prune, center_threshold, surround_threshold, error_threshold):
        '''
        Args:
            q_threshold (float): how far away from the mean?
            r_list (list<int>) : the R in the algo
            edge (int)         : ignore the borders
            noise_floor_prune (int) : the noise floor threshold for pruning
            center_threshold (int)  : the threshold for center sensor in procedure 2
            surround_threshold (int): the threshold for surronding sensor in procedure 2
            error_threshold (float) : a fraction of the grid length
        '''
        self.Q      = q_threshold
        self.R_list = r_list
        self.edge   = edge
        self.noise_floor_prune  = noise_floor_prune
        self.center_threshold   = center_threshold
        self.surround_threshold = surround_threshold
        self.error_threshold    = error_threshold
        # self.
    
    def __str__(self):
        return 'Q      = {}\n'.format(self.Q) + \
               'R_list = {}\n'.format(self.R_list) + \
               'edge   = {}\n'.format(self.edge) + \
               'noise floor prune  = {}\n'.format(self.noise_floor_prune) + \
               'center threshold   = {}\n'.format(self.center_threshold) + \
               'surround threshold = {}\n'.format(self.surround_threshold) + \
               'error threshold    = {}\n'.format(self.error_threshold)

    @classmethod
    def naive_factory(cls, case):
        if case == 'lognormal' or case == 'splat':
            q = 1.9
            r = [8, 6, 5, 4]
            e = 2
            nf_p   = -80
            c_thre = -65
            s_thre = -75
            e_thre = 0.2

            return cls(q_threshold=q, r_list=r, edge=e, noise_floor_prune=nf_p, \
                       center_threshold=c_thre, surround_threshold=s_thre, error_threshold = e_thre)
        
        if case == 'utah':
            q = 3.5
            r = [5, 4, 3]
            e = 0
            nf_p   = -65
            c_thre = -55
            s_thre = -65
            e_thre = 1

            return cls(q_threshold=q, r_list=r, edge=e, noise_floor_prune=nf_p, \
                       center_threshold=c_thre, surround_threshold=s_thre, error_threshold = e_thre)

