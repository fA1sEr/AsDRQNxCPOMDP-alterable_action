from random import randint, random
import numpy as np
from Network import Network
from ReplayMemory_adaptive import ReplayMemory


class Agent:
    def __init__(self, memory_cap, batch_size, state_length, action_count, session,
                 lr, gamma, epsilon_min, epsilon_decay_steps, epsilon_max, trace_length, hidden_size):

        self.model = Network(session=session, action_count=action_count,
                             state_length=state_length, lr=lr, batch_size=batch_size,
                             trace_length=trace_length, hidden_size=hidden_size, scope='main')

        self.target_model = Network(session=session, action_count=action_count,
                                    state_length=state_length, lr=lr, batch_size=batch_size,
                                    trace_length=trace_length, hidden_size=hidden_size, scope='target')

        self.memory = ReplayMemory(memory_cap=memory_cap, batch_size=batch_size,
                                   state_length=state_length, trace_length=trace_length, network=self.model, gamma=gamma)

        self.batch_size = batch_size

        self.state_length = state_length
        self.action_count = action_count
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_max = epsilon_max
        self.hidden_size = hidden_size
        self.trace_length = trace_length

        self.epsilon = epsilon_max
        self.training_steps = 0

        self.epsilon_decrease = (epsilon_max-epsilon_min)/epsilon_decay_steps

        self.min_buffer_size = batch_size*trace_length

        self.state_in = (np.zeros([1, self.hidden_size]), np.zeros([1, self.hidden_size]))

    def add_transition(self, s1, a, r, s2, d, is_init=False):
        self.memory.add_transition(s1, a, r, s2, d, is_init)

    def __1to0(self, x):
        for i in x:
            if i==-1:
                i=0
        return x

    def learn_from_memory(self):

        if self.memory.end - self.memory.begin > self.min_buffer_size:
            state_in = (np.zeros([self.batch_size, self.hidden_size]), np.zeros([self.batch_size, self.hidden_size]))
            s1, a, r, s2, d = self.memory.get_transition()
            inputs = s1

            q = np.max(self.target_model.get_q(s2, state_in), axis=1)
            #print('q:',self.target_model.get_q(s2, state_in))
            targets = r + self.gamma * (1 - self.__1to0(d)) * q
            #print('target_q:',targets)
            self.model.learn(inputs, targets, state_in, a)

    def act(self, state, train=True):
        if train:
            self.epsilon = self.explore(self.epsilon)
            if random() < self.epsilon:
                a = self.random_action()
            else:
                a, self.state_in = self.model.get_best_action(state, self.state_in)
                a = a[0]
            #    print('a:',a)
        else:
            a, self.state_in = self.model.get_best_action(state, self.state_in)
            a = a[0]
            #print('a:',a)
        return a

    def explore(self, epsilon):
        return max(self.epsilon_min, epsilon-self.epsilon_decrease)

    def random_action(self):
        return randint(0, self.action_count - 1)

    def reset_cell_state(self):
        self.state_in = (np.zeros([1, self.hidden_size]), np.zeros([1, self.hidden_size]))
