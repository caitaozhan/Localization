'''
Predefines the authorized users.
Either from random generation, or from outdoor testbed data trace driven
'''

import random
from collections import defaultdict

class Authorized:

    def __init__(self, case, grid_len, edge):
        self.primaries = []                  # do not change per experiment
        self.secondaries = defaultdict(list)     # change per experiment
        self.init_authorized(case, grid_len, edge)

    def init_authorized(self, case, grid_len, edge):
        '''
        Args:
            case -- str
            grid_len -- int
            edge -- int -- no intruders at the edge
        '''
        if case == 'splat':
            self.primaries = [(5, 5), (34, 5), (34, 34), (5, 34)]  # the 4 primaries are fixed
            random.seed(0)
            for i in range(100):                                   # predefine the secondaries for 100 test cases
                for _ in range(3):                                 # 3 secondaries in each test case
                    x = random.randint(edge, grid_len-edge-1)
                    y = random.randint(edge, grid_len-edge-1)
                    self.secondaries[i].append((x, y))
        if case == 'outdoor-testbed':
            pass

    def __str__(self):
        return 'Primary = {}'.format(str(self.primaries)) + '\nSecondary = {}'.format(str(self.secondaries))


if __name__ == '__main__':
    authorized = Authorized('splat', 40, 2)
    print(authorized)
