from random import sample
import numpy as np


class ReplayMemory:
    def __init__(self, memory_cap, batch_size, state_length, trace_length, network, gamma):

        state_shape = (memory_cap, state_length)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(memory_cap, dtype=np.int32)
        self.r = np.zeros(memory_cap, dtype=np.float32)
        self.d = np.zeros(memory_cap, dtype=np.int32)

        self.memory_cap = memory_cap
        self.batch_size = batch_size
        self.trace_length = trace_length
        self.begin = 0
        self.end = 0
        self.k = 20
        self.t = 0
        self.delta = 0
        self.delta_ = 0
        self.n_old = 50
        self.network = network
        self.gamma = gamma

    def add_transition(self, s1, a, r, s2, d, is_init=False):

        self.s1[self.end, :] = s1
        self.a[self.end] = a
        self.r[self.end] = r
        self.s2[self.end, :] = s2
        self.d[self.end] = d
        self.end += 1

        if self.end >= self.memory_cap:
            for i in range(self.begin, self.end):
                self.s1[i-self.begin, :] = self.s1[i, :]
                self.a[i-self.begin] = self.a[i]
                self.r[i-self.begin] = self.r[i]
                self.s2[i-self.begin, :] = self.s2[i, :]
                self.d[i-self.begin] = self.d[i]
            self.end = self.end - self.begin
            self.begin = 0

        if is_init==False:
            self.t += 1
            if self.t == self.k:
                self.t = 0
                if self.check():
                    self.begin += self.k * 2
                    self.delta = self.calculate()
                else:
                    self.delta = self.delta_
                print('mem length:',self.end-self.begin)


    def get_transition(self):
        indexes = []
        for _ in range(self.batch_size):
            accepted = False
            while not accepted:
                point = np.random.randint(self.begin, self.end - self.trace_length)
                accepted = True
                for i in range(self.trace_length-1):
                    if self.d[point+i] != 0:
                        accepted = False
                        break
                if accepted:
                    for i in range(self.trace_length):
                        indexes.append(point+i)

        return self.s1[indexes], self.a[indexes], self.r[indexes], self.s2[indexes], self.d[indexes]

    def check(self):
        self.delta_ = self.calculate()
        if self.delta_ > self.delta or self.end - self.begin <= self.k:
            return False
        else:
            return True
    
    def calculate(self):
        ans = 0.0
        state_in = (np.zeros([1, self.network.hidden_size]), np.zeros([1, self.network.hidden_size]))
        for i in range(self.n_old):
            q1 = np.max(self.network.get_1q(self.s2[self.begin+i].reshape(1,-1), state_in), axis=1)[0]
            q2 = self.network.get_1q(self.s1[self.begin+i].reshape(1,-1), state_in)[0][self.a[self.begin+i]]
            #print('q1:',q1,'q2:',q2)
            ans += abs(self.r[self.begin+i] + self.gamma*q1 - q2)
        print('ans:',ans)
        return ans
