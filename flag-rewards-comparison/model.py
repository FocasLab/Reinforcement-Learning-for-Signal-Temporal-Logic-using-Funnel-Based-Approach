import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import itertools
import random
import os
from logger import Logger
from collections import deque

def build_mlp(n_in, hidden, n_out, act_fn):
    hidden.append(n_out)
    li = []
    li.append(nn.Linear(n_in,hidden[0]))
    for i in range(1,len(hidden)):
        li.append(act_fn(0.1))
        li.append(nn.Linear(hidden[i-1],hidden[i]))
    return nn.Sequential(*nn.ModuleList(li))


class Actor(nn.Module):
    def __init__(self, env, config):
        super(Actor,self).__init__()

        self.env = env
        
        self.device = config['device']

        self.state_dim = config['state_dim']
        
        self.action_dim = config['action_dim'] 
        
        self.q_net = build_mlp(self.state_dim, config['hidden_layer'].copy(), self.action_dim, config['act_fn'])

        self.optimizer = Adam(self.parameters(),lr = config['lr_q'])
        
        self.init_epsilon = config['init_epsilon']
        
        self.epsilon = self.init_epsilon
        
        self.final_epsilon = config['final_epsilon']
        
        self.anneal_limit = config['anneal_limit']
        
    def forward(self,state):
        #print("shape state", state.shape)
        state = state.reshape(-1,self.state_dim).to(self.device)
        return self.q_net(state).to('cpu')
            
    def get_action(self,state, deterministic = False):
        
        q_values = self(state)
        
        if(deterministic == False):
           if(len(q_values) == 1): 
              chance = np.random.uniform(0,1, size = 1)
           else :
              chance = np.random.uniform(0,1, size = len(q_values))
           
           chance = torch.tensor(chance)
           
           action = (chance <self.epsilon)*torch.tensor(np.random.randint(0,self.action_dim)) + (chance >=self.epsilon)*torch.argmax(q_values, dim = -1)
        else:    
           action = torch.argmax(q_values, dim = -1)
        
        #print("dim",q_values.shape," ",action.shape," ",action)
        return action , torch.gather(q_values,1,action.unsqueeze(-1))
    
    def get_qvalue(self,state, action):        
        q_values = self(state)
        action = action.long()
        return torch.gather(q_values,1,action.unsqueeze(-1))

    
    def anneal(self):
        self.epsilon = self.epsilon - (self.init_epsilon - self.final_epsilon)/self.anneal_limit

        
class Agent:
    def __init__(self,env,config):
        
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        random.seed(config["seed"])
        
        self.env = env
        
        self.device = config['device']

        self.log = config['log']
        self.log_name = config['log_name']
        
        self.logger = Logger(log_dir = self.log_name)
        
        self.state_dim = config['state_dim']
        
        self.action_dim = config['action_dim'] 
        
        self.actor = Actor(env, config)
        self.actor.to(self.device)
        self.actor_target = Actor(env, config)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.to(self.device)

        self.tau = config['tau']
        self.gamma = config['gamma']
        self.buffer_size = config['buffer_size']

                   
      
    def update(self, buffer,batch_size = 256, gradient_steps = 1, q_freq = 1):
        
        
        for n_updates in range(1,gradient_steps+1):                       

            state, reward, action , done, next_state = buffer.sample(batch_size,len(buffer) - self.buffer_size)
            
            if(n_updates % q_freq == 0):

                with torch.no_grad():
                    _ , next_q_val = self.actor_target.get_action(next_state, deterministic = True)                 
                    q_target = reward + (1-done)*self.gamma*(next_q_val.reshape(-1))  
                            
                q_val = self.actor.get_qvalue(state, action)
                
                #print("q_val shape",q_val.shape)
                actor_loss = F.mse_loss(q_val.reshape(-1),q_target)

                if(self.log):
                  with torch.no_grad():      
                    self.logger.add_scalar("actor loss", actor_loss.item())
                    self.logger.add_scalar("reward min", min(reward).item())
                    self.logger.add_scalar("reward max", max(reward).item())
                    self.logger.add_scalar("q_target min", min(q_target).item())
                    self.logger.add_scalar("next_q_val min", min(next_q_val).item())
                    self.logger.add_scalar("q_target max", max(q_target).item())
                    self.logger.add_scalar("next_q_val max", max(next_q_val).item())


                self.actor.optimizer.zero_grad()                
                actor_loss.backward()
                self.actor.optimizer.step()
            
                #with torch.no_grad():
                    #for params, params_target in zip(self.actor.parameters(), self.actor_target.parameters()):
                        #params_target.data.mul_(self.tau)
                        #params_target.data.add_((1 - self.tau)*params)
                        
    def update_target(self):
        network_state = self.actor.state_dict()
        self.actor_target.load_state_dict(network_state)

      
    def evaluate(self,epi_len= 1000, n_iter = 1, deterministic = True):
        
        self.actor.train(False)
        with torch.no_grad():
            total_reward = 0
            total_steps = 0
            for epi in range(n_iter):
                state = self.env.reset()
                done = False
                while(not done):
                    total_steps += 1
                    action = self.get_action(torch.tensor(state).float(), deterministic = deterministic)
                    #print("eval action",action)
                    next_state, reward, done, _ = self.env.step(int(action.detach().item()))
                    total_reward += reward
                    state = next_state                            
        
        return (total_reward/n_iter), (total_reward/total_steps) 
    
    def get_action(self, state, deterministic = False):
        state = state.reshape(-1, self.state_dim)
        #print("state shape",state.shape)
        action, _ = self.actor.get_action(state, deterministic)
        return action[0]      
    
    def anneal(self):
        self.actor.anneal()
    