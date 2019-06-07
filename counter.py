'''A counter that uses property, instead of setters and getters
   Common use cases: time counter
'''

import time

class Counter:

    def __init__(self):
        self.__time  = 0     # total time
        self.__time1 = 0     # procedure 1        (total time of several experiments)
        self.__time2 = 0     # procedure 1.1      (total time of several experiments)
        self.__time3 = 0     # procedure 2, t=2   (total time of several experiments)
        self.__time4 = 0     # procedure 2, t=3   (total time of several experiments)
        self.__num_exper = 1 # number of experiments
        self.__time1_start = 0
        self.__time2_start = 0
        self.__time3_start = 0
        self.__time4_start = 0
        self.__proc_1   = 0     # number of intruders indentified by procedure 1
        self.__proc_1_1 = 0     # number of intruders indentified by procedure 1.1
        self.__proc_2_2 = 0     # number of intruders indentified by procedure 2, t=2
        self.__proc_2_3 = 0     # number of intruders indentified by procedure 2, t=3

    def reset(self):
        self.__time = self.__time1 = self.__time2 = self._time3 = self.__time4 = 0
        self.__proc_1 = self.__proc_1_1 = self.__proc_2_2 = self.__proc_2_3 = 0

    @property
    def time(self):
        return self.__time

    @property
    def time1(self):
        return self.__time1
    
    @property
    def time2(self):
        return self.__time2
    
    @property
    def time3(self):
        return self.__time3
    
    @property
    def time4(self):
        return self.__time4

    @property
    def num_exper(self):
        return self.__num_exper

    @property
    def proc_1(self):
        return self.__proc_1

    @property
    def proc_1_1(self):
        return self.__proc_1_1

    @property
    def proc_2_2(self):
        return self.__proc_2_2

    @property
    def proc_2_3(self):
        return self.__proc_2_3

    @time.setter
    def time(self, time):
        if time > 0:
            self.__time = time
        else:
            print('Input error:', time)

    @time1.setter
    def time1(self, time):
        if time > 0:
            self.__time1 = time
        else:
            print('Input error:', time)
    
    @time2.setter
    def time2(self, time):
        if time > 0:
            self.__time2 = time
        else:
            print('Input error:', time)
    
    @time3.setter
    def time3(self, time):
        if time > 0:
            self.__time3 = time
        else:
            print('Input error:', time)
    
    @time4.setter
    def time4(self, time):
        if time > 0:
            self.__time4 = time
        else:
            print('Input error:', time)

    @num_exper.setter
    def num_exper(self, num):
        if num > 0 and isinstance(num, int):
            self.__num_exper = num
        else:
            print('Input error:', num)

    @proc_1.setter
    def proc_1(self, num):
        if num >= 0 and isinstance(num, int):
            self.__proc_1 = num
        else:
            print('Input error:', num)

    @proc_1_1.setter
    def proc_1_1(self, num):
        if num >= 0 and isinstance(num, int):
            self.__proc_1_1 = num
        else:
            print('Input error:', num)

    @proc_2_2.setter
    def proc_2_2(self, num):
        if num >= 0 and isinstance(num, int):
            self.__proc_2_2 = num
        else:
            print('Input error:', num)

    @proc_2_3.setter
    def proc_2_3(self, num):
        if num >= 0 and isinstance(num, int):
            self.__proc_2_3 = num
        else:
            print('Input error:', num)

    def time_start(self):
        self.__time = time.time()

    def time1_start(self):
        self.__time1_start = time.time()
    
    def time2_start(self):
        self.__time2_start = time.time()

    def time3_start(self):
        self.__time3_start = time.time()
    
    def time4_start(self):
        self.__time4_start = time.time()

    def time_end(self):
        self.__time = time.time() - self.__time
        print('Total time: {:.3f}'.format(self.__time))

    def time1_end(self):
        print('Proceduer 1   time:', round(time.time()-self.__time1_start, 3))
        self.__time1 += (time.time()-self.__time1_start)
    
    def time2_end(self):
        print('Proceduer 1.1 time:', round(time.time()-self.__time2_start, 3))
        self.__time2 += (time.time()-self.__time2_start)
    
    def time3_end(self):
        print('Proceduer 2,2 time:', round(time.time()-self.__time3_start, 3))
        self.__time3 += (time.time()-self.__time3_start)
    
    def time4_end(self):
        print('Proceduer 2,3 time:', round(time.time()-self.__time4_start, 3), '\n')
        self.__time4 += (time.time()-self.__time4_start)
    
    def time1_average(self):
        return self.__time1 / self.__num_exper

    def time2_average(self):
        return self.__time2 / self.__num_exper

    def time3_average(self):
        return self.__time3 / self.__num_exper

    def time4_average(self):
        return self.__time4 / self.__num_exper
    
    def procedure_ratios(self):
        total = self.__proc_1 + self.proc_1_1 + self.proc_2_2 + self.proc_2_3
        print('Proc-1 = {:.3f}; Proc-1.1 = {:.3f}; Proc-2-2 = {:.3f}; Proc-2-3 = {:.3f}'.format(\
            self.__proc_1/total, self.__proc_1_1/total, self.__proc_2_2/total, self.__proc_2_3/total))


if __name__ == '__main__':
    c = Counter()
    c.num_exper = 3
    print(c.time1)

    c.time1_start()
    c.time1_end()
    print(c.time1)

    c.time1_start()
    c.time1_end()
    print(c.time1)

    c.time1_start()
    c.time1_end()
    print(c.time1)

    print(c.time1_average())


