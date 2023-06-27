import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions.normal import Normal 
import itertools
import random
import os
from logger import Logger
from collections import deque


class BufferQueue:
    def __init__(self, size = int(1e5)):
        self.state = deque(maxlen = size)
        self.reward = deque(maxlen = size)
        self.action = deque(maxlen = size)
        self.done = deque(maxlen = size)  
        self.next_state = deque(maxlen = size)
    
    def insert(self,state, reward, action, done, next_state):
        self.state.append(state)
        self.reward.append(reward)
        self.action.append(action)
        self.done.append(done)
        self.next_state.append(next_state)

    def sample(self,batch_size):
        index = random.sample(range(len(self.state)),batch_size)
        state_sample = [self.state[i] for i in index]
        
        reward_sample = [self.reward[i] for i in index]
        
        action_sample = [self.action[i] for i in index]
        
        done_sample = [self.done[i] for i in index]
        
        next_state_sample = [self.next_state[i] for i in index]

        
        return ( torch.tensor(np.array(state_sample)).float(), 
                torch.tensor(reward_sample).float(), 
                torch.tensor(np.array(action_sample)).float(), 
                torch.tensor(done_sample).float(), 
                torch.tensor(np.array(next_state_sample)).float())
     
    def __len__(self):
         return len(self.state)

    
class BufferList:
    def __init__(self, size = int(1e5)):
        self.state = []
        self.reward = []
        self.action = []
        self.done = []  
        self.next_state = []
    
    def insert(self,state, reward, action, done, next_state):
        self.state.append(state)
        self.reward.append(reward)
        self.action.append(action)
        self.done.append(done)
        self.next_state.append(next_state)

    def sample(self,batch_size, start):
        index = random.sample(range(max(start,0),len(self.state)),batch_size)
        state_sample = [self.state[i] for i in index]
        
        reward_sample = [self.reward[i] for i in index]
        
        action_sample = [self.action[i] for i in index]
        
        done_sample = [self.done[i] for i in index]
        
        next_state_sample = [self.next_state[i] for i in index]
        
        return ( torch.tensor(np.array(state_sample)).float(), 
                torch.tensor(reward_sample).float(), 
                torch.tensor(np.array(action_sample)).float(), 
                torch.tensor(done_sample).float(), 
                torch.tensor(np.array(next_state_sample)).float())
     
    def __len__(self):
         return len(self.state)