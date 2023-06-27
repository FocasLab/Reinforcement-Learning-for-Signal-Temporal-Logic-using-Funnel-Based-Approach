#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


import os
import numpy as np
import torch
import pickle
from config import config


def save_parameters(agent,path):
    torch.save(agent.actor.state_dict(),'./'+path+'/actor.pt')
    torch.save(agent.actor_target.state_dict(),'./'+path+'/actor_target.pt')
    torch.save(agent.actor.optimizer.state_dict(),'./'+path+'/actor_optim.pt')

def load_parameters(agent,path, device):
    agent.actor.load_state_dict(torch.load('./'+path+'/actor.pt', map_location = device))
    agent.actor_target.load_state_dict(torch.load('./'+path+'/actor_target.pt', map_location = device))
    agent.actor.optimizer.load_state_dict(torch.load('./'+path+'/actor_optim.pt', map_location = device))

class Logger(object):
    def __init__(self, log_dir = None, log_freq = 10000):
        self.data = {}
        if(log_dir is None):
           self.log_dir = './log'
        else:
           self.log_dir = log_dir
           
        
        if(not os.path.isdir(self.log_dir)):
            os.mkdir(self.log_dir)
            
        if(not os.path.isdir(config["log_best"])):
            os.mkdir(config["log_best"])
        
        self.pointer = 0
        self.log_freq = log_freq

    def to_numpy(self, v):
        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
        return v

    def add_scalar(self, tag, value, step=None):
        value = self.to_numpy(value)
        if(tag in self.data):
            self.data[tag].append(value)
        else:
            self.data[tag] = []
            self.data[tag].append(value)

        self.pointer +=1
        if(self.pointer % self.log_freq == 0):
            pointer = 0
            self.flush()

    def flush(self):
        file = open(self.log_dir + '/logfile.pkl','wb')
        pickle.dump(self.data, file)
        file.close()


