'''
Predefines the authorized users.
Either from random generation, or from outdoor testbed data trace driven
'''

import random
from collections import defaultdict

class Authorized:

    def __init__(self, grid_len, edge, case, num):
        '''
        Args:
            grid_len -- int
        '''
        self.grid_len = grid_len
        self.primaries = []                  # do not change per experiment
        self.primaries_power = []
        self.secondaries = defaultdict(list)     # change per experiment
        self.secondaries_power = defaultdict(list)
        self.init_authorized(grid_len, edge, case, num)

    def init_authorized(self, grid_len, edge, case, num):
        '''
        Args:
            grid_len -- int
            edge -- int -- no intruders at the edge
            case -- str
            num  -- int -- number of authorized users.
        '''
        if case == 'splat':
            if num == 0:
                return
            elif num == 2:
                self.primaries = [(5, 5)]                              # the one primary is fixed
                random.seed(0)
                self.primaries_power = [2]
                random.seed(num)
                for i in range(100):                                   # predefine the secondaries for 100 test cases
                    for _ in range(1):                                 # 1 secondaries in each test case
                        x = random.randint(edge, grid_len-edge-1)
                        y = random.randint(edge, grid_len-edge-1)
                        self.secondaries[i].append((x, y))
                        power = 2
                        self.secondaries_power[i].append(power)
            elif num == 4:
                self.primaries = [(5, 5), (34, 5)]                     # the 2 primaries are fixed
                random.seed(0)
                self.primaries_power = [2 for _ in range(2)]
                random.seed(num)
                for i in range(100):                                   # predefine the secondaries for 100 test cases
                    for _ in range(2):                                 # 2 secondaries in each test case
                        x = random.randint(edge, grid_len-edge-1)
                        y = random.randint(edge, grid_len-edge-1)
                        self.secondaries[i].append((x, y))
                        power = 2
                        self.secondaries_power[i].append(power)
            elif num == 5:
                self.primaries = [(5, 5), (34, 34)]                    # 2 primaries at two opposite corners
                self.primaries_power = [2, 2]
                random.seed(0)
                for i in range(100):
                    for j in range(3):                                 # 3 secondaries randomly spread out
                        self.secondaries[i].append(self.random_secondary(j, edge))
                        self.secondaries_power[i].append(2)
            elif num == 6:
                self.primaries = [(5, 5), (34, 5), (34, 34)]           # the 3 primaries are fixed
                random.seed(0)
                self.primaries_power = [2 for _ in range(3)]
                random.seed(num)
                for i in range(100):                                   # predefine the secondaries for 100 test cases
                    for _ in range(3):                                 # 3 secondaries in each test case
                        x = random.randint(edge, grid_len-edge-1)
                        y = random.randint(edge, grid_len-edge-1)
                        self.secondaries[i].append((x, y))
                        power = 2
                        self.secondaries_power[i].append(power)
            elif num == 8:
                self.primaries = [(5, 5), (34, 5), (34, 34), (5, 34)]  # the 4 primaries are fixed
                random.seed(0)
                self.primaries_power = [2 for _ in range(4)]
                random.seed(num)
                for i in range(100):                                   # predefine the secondaries for 100 test cases
                    for _ in range(4):                                 # 4 secondaries in each test case
                        x = random.randint(edge, grid_len-edge-1)
                        y = random.randint(edge, grid_len-edge-1)
                        self.secondaries[i].append((x, y))
                        power = 2
                        self.secondaries_power[i].append(power)
            else:
                raise Exception('Oops! Invalid num = {}'.format(num))
        if case == 'outdoor-testbed':
            pass

    def random_secondary(self, j, edge):
        '''
        Args:
            j -- int -- [0, 2]
        Return:
            (int, int) -- a 2D location for secondaries
        '''
        if j == 0:
            x = random.randint(edge+2, edge+5)
            y = random.randint(self.grid_len-edge-5, self.grid_len-edge-2)
        elif j == 1:
            x = random.randint(int(self.grid_len/2)-2, int(self.grid_len/2)+3)
            y = random.randint(int(self.grid_len/2)-2, int(self.grid_len/2)+3)
        elif j == 2:
            x = random.randint(self.grid_len-edge-5, self.grid_len-edge-2)
            y = random.randint(edge+2, edge+5)
        return (x, y)


    def __str__(self):
        return 'Primary = {}\n'.format(str(self.primaries)) + 'Primary power = {}\n'.format(str(self.primaries_power)) + \
               '\nSecondary = {}\n'.format(str(self.secondaries)) + '\nSecondary power = {}\n'.format(str(self.secondaries_power))


    def add_authorized_users(self, true_indices, true_powers, exp_num):
        '''Add authorized users to the ground truth
        Args:
            true_indices -- [int ...] -- 1d index of the truth intruders
            true_powers -- [int ...] -- truth intruders' power
            exp_num     -- int       -- experiment number
        '''
        all_authorized = self.primaries + self.secondaries[exp_num]
        all_power      = self.primaries_power + self.secondaries_power[exp_num]
        all_authorized = [a[0]*self.grid_len + a[1] for a in all_authorized]
        true_indices.extend(all_authorized)
        true_powers.extend(all_power)


if __name__ == '__main__':
    authorized = Authorized(40, 2, 'splat', 5)
    print(authorized)
