import skimage.color
import skimage.transform
from random import choices
from random import randint
import numpy as np

class mit:
    def __init__(self):
        self.file = 'mit.POMDP'
        self.discount = 0.990000
        self.states = 204
        self.actions = 4
        self.observations = 28
        self.cur_state = 0
        self.cur_observation = 0
        self.state_list = [i for i in range(self.states)]
        self.obs_list = [i for i in range(self.observations)]
        self.T = np.zeros([self.actions, self.states, self.states], dtype=np.float32)
        self.O = np.zeros([self.actions, self.states, self.observations], dtype=np.float32)
        self.reward = np.zeros([self.actions, self.states], dtype=np.float32)
        self.p_state = [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
        self.total_steps = 0
        self.is_terminated = False
        self.total_reward = 0
        
    def init_game(self):
        file_reader = open(self.file)
        lines = file_reader.readlines()
        # 读取数据
        print('Reading file....')
        for line in lines:
            line = line.strip()
            # Transition Probabilities
            if line[0]=='T':
                items = line.split(':')
                action = int(items[1].strip())
                state = int(items[2].strip())
                items = items[3].strip().split(' ')
                n_state = int(items[0].strip())
                p_state = float(items[1].strip())
                self.T[action][state][n_state] = p_state
            # Observation Probabilities
            if line[0]=='O':
                items = line.split(':')
                action = int(items[1].strip())
                state = int(items[2].strip())
                items = items[3].strip().split(' ')
                obs = int(items[0].strip())
                p_state = float(items[1].strip())
                self.O[action][state][obs] = p_state
            # Rewards
            if line[0]=='R':
                items = line.split(':')
                action = int(items[1].strip())
                state = int(items[2].strip())
                items = items[4].strip().split(' ')
                r_state = float(items[1].strip())
                self.reward[action][state] = r_state

        print('Reading file done!')
        self.reset()
    
    def reset(self):
        print('Initailize start state...')
        self.cur_state = choices(self.state_list, self.p_state, k=1)[0]
        p_obs = self.O[randint(0,self.actions-1)][self.cur_state].flatten()
        self.cur_observation = choices(self.obs_list, p_obs, k=1)[0]
        self.total_reward = 0
        self.total_steps = 0
        self.is_terminated = False
        print('Initailize done! Init_state:', self.cur_state, ' Init_obs:', self.cur_observation)
                
    def make_action(self, action):
        self.total_steps += 1
        if self.total_steps >= 250:
            self.is_terminated = True
            
        if action<0 or action>=self.actions:
            print('There is no this action:', action)
                
        p_trans = self.T[action][self.cur_state].flatten()
        self.cur_state = choices(self.state_list, p_trans, k=1)[0]
        p_obs = self.O[action][self.cur_state].flatten()
        self.cur_observation = choices(self.obs_list, p_obs, k=1)[0]
        cur_reward = self.reward[action][self.cur_state]
        self.total_reward += cur_reward
        # 如果到达目标地点，奖励值为1，判为终止状态
        if cur_reward>0:
            self.is_terminated = True
            
        return self.cur_observation, cur_reward, self.is_terminated
    
    def get_total_reward(self):
        return self.total_reward

class GameSimulator:
    def __init__(self, frame_repeat=1):
        self.last_action = 0
        self.action_length = 4 # change two place [2]
        self.game = None
        self.frame_repeat = frame_repeat
        
    def initialize(self):
        # 初始化游戏，返回游戏的动作数目
        print("Initializing game...")
        self.game = mit()
        self.game.init_game()
        print("Game initialized.")
        return self.game.actions

    def get_state(self):
        # 获取当前游戏的画面，游戏结束则获得空
        obs_list = np.zeros([self.game.observations], dtype=np.int32).tolist()
        obs_list[self.game.cur_observation] = 1
        action_rep = self.action_length//self.game.actions
        action_remain = self.action_length%self.game.actions
        obs_list = obs_list + ([0]*action_remain)
        for i in range(self.game.actions):
            if i == self.last_action:
                obs_list = obs_list + ([1]*action_rep)
            else:
                obs_list = obs_list + ([0]*action_rep)
        return obs_list
    
    def get_action_size(self):
        # 获取动作数目
        return self.game.actions
    
    def __get_button_size(self):
        return self.game.actions
    
    def make_action(self, action):
        # 执行动作
        _, reward, done = self.game.make_action(action)
        new_state = self.get_state()
        self.last_action = action
        return new_state, reward, done
    
    def is_terminared(self):
        # 判断游戏是否终止
        return self.game.is_terminated
    
    def reset(self):
        # 重新开始游戏
        self.game.reset()
    
    def close(self):
        # 关闭游戏模拟器
        print('No this function.')
        
    def get_total_reward(self):
        return self.game.get_total_reward()
