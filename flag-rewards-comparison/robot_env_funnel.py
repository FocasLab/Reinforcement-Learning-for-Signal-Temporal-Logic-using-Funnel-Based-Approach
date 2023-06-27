#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 12:12:06 2022

@author: rbccps5
"""


#*********************Staying in a Region*********************************88

import numpy as np
import math

class RobotEnv:
    def __init__(self,config):
        
        self.x = 0
        self.y = 0
        self.theta = 0
        
        #self.theta_dot = config['theta_dot']
        self.time_int = config["time_int"]
        self.timestep = 0
        self.epi_len = config['epi_len']
        self.config = config
        
    

        self.action_dict = {}
        self.action_dict_inv = {}
        
        v = -5   
        key = 0
        for i in range(21):
            
            w = -2
            for j in range(9):
                
                self.action_dict[(v,w)] = key
                self.action_dict_inv[key] = (v,w)
                w += 0.5
                key += 1
                
            v += 0.5
            
        self.robusts = []
                


    def reset(self):
        self.timestep = 0
        while(np.linalg.norm((self.x-2)**2 + (self.y-2)**2)>1):
            self.x = np.random.uniform(0,6)
            self.y = np.random.uniform(0,6)
#         self.flag = 0
        self.theta = np.random.uniform(-3.13, 3.14)
        #self.theta_dot = self.config['theta_dot']
        return np.array([self.x,self.y,self.theta, (self.timestep-self.epi_len//2)/100])
        #return np.array([self.theta, self.theta_dot])
        
    def step(self,action):
        done = False
        self.timestep+=1
        if(self.timestep >= self.epi_len):
            done = True
        #print(action)
        v,w = self.action_dict_inv[action]
        
        self.x = min(6,max(0,self.x + self.time_int*v*np.cos(self.theta)))
        self.y = min(6,max(0,self.y + self.time_int*v*np.sin(self.theta)))
        
#        self.x = self.x + self.time_int*v*np.cos(self.theta)
#        self.y = self.y + self.time_int*v*np.sin(self.theta)
        
        self.theta = self.theta + w * self.time_int 
        
     
        if(self.theta > self.config['theta_max'] or self.theta < self.config['theta_min'] ):
           #self.theta = ((math.floor(self.theta*100)+313)%628-313)/100
           self.theta = ((self.theta*100 + 313)%628-313)/100
                
  
        
        distance = np.linalg.norm(np.array([self.x-2,self.y-2]))
       
        robustness = 1-distance 
        
        self.robusts.append(robustness)
        
        reward = robustness + 5.55 * np.exp(-(1/100)*1.82*self.timestep) - 0.9

        return np.array([self.x, self.y,self.theta, (self.timestep-self.epi_len//2)/100]), reward, done, None
        #return np.array([theta_, theta_dot_]), reward, done, None
    
    def get_action_dim(self):
        return len(self.action_dict)
    
    def get_state_dim(self):
        return 4