#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 12:12:06 2022

@author: rbccps5
"""


#**********************globally inside a big ring, avoid 3 obstacles, reach two goals***********************************88

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
        
        v = -0.2
        key = 0
        for i in range(9):
            
            w = -2
            for j in range(9):
                
                self.action_dict[(v,w)] = key
                self.action_dict_inv[key] = (v,w)
                w += 0.5
                key += 1
                
            v += 0.05
            v = round(v,3)
#        print("here")
                
                

                
    def reset(self):
        self.timestep = 0
        self.x = np.random.uniform(0,4)
        self.y = np.random.uniform(0,4)
#        self.x = 1
#        self.y = 1
#        print("ok")
        self.theta = np.random.uniform(-3.13, 3.14)
        #self.theta_dot = self.config['theta_dot']
        return np.array([self.x,self.y,self.theta, (self.timestep-self.epi_len//2)/10])
#        return np.array([self.x,self.y,self.theta, (self.timestep)/2500])
        #return np.array([self.theta, self.theta_dot])
        
    def step(self,action):
        done = False
        self.timestep+=1
        if(self.timestep >= self.epi_len):
            done = True
        #print(action)
        v,w = self.action_dict_inv[action]
        
        
        self.x = min(4,max(0,self.x + self.time_int*v*np.cos(self.theta) ))
        self.y = min(4,max(0,self.y + self.time_int*v*np.sin(self.theta) )) 
        
#        self.x = self.x + self.time_int*v*np.cos(self.theta) 
#        self.y = self.y + self.time_int*v*np.sin(self.theta) 
        
        self.theta = self.theta + w * self.time_int 
        
     
        if(self.theta > self.config['theta_max'] or self.theta < self.config['theta_min'] ):
           #self.theta = ((math.floor(self.theta*100)+313)%628-313)/100
           self.theta = ((self.theta*100 + 313)%628-313)/100
                
                
        reward  = 0
        
        
        
        distance1 = max(abs(self.x-2),abs(self.y-2))       #(0,0)  2,2
        distance2 = max(abs(self.x-2),abs(self.y-0.5))    #(0,-1.5)  2,0.5
        distance3 = max(abs(self.x-3),abs(self.y-3))         #(1,1)  3,3
        
        robust0 = 2 - distance1
        robust1 = distance1 - 0.5
        robust2 = distance2 - 0.5
        robust3 = distance3 - 0.5
        
        obstacle_robust = min(robust0,robust1,robust2,robust3)
        
        obstacle_reward = (obstacle_robust + 0.45*np.exp(-(1/10)*1.504*(self.timestep))- 0.1)         #0.75 IS BEST POSSIBLE ROBUSTNESS          
        
        
        
        reward = obstacle_reward

        if(self.timestep <= 50):
            goal_distance1 = np.linalg.norm(np.array([self.x-3,self.y-1]))    #(1,-1)
            
            
            
            goal_robust1 = 0.3 - goal_distance1
            #reward = robust
            goal_reward1 = goal_robust1 + 4.04*np.exp(-(1/50)*3.7*(self.timestep)) -  0.1   #follow up rewards 
            
            reward = min(obstacle_reward,goal_reward1)
            
        elif(self.timestep <= 90):
            goal_distance2 = np.linalg.norm(np.array([self.x-1,self.y-3]))   #(-1,1)
            goal_robust2 = 0.3 - goal_distance2
            #reward = robust
            goal_reward2 = goal_robust2 + 2.9*np.exp(-(1/40)*3.36*(self.timestep-50)) -  0.1 #follow up rewards 
            
            reward = min(obstacle_reward,goal_reward2)
            
            
#         if(self.timestep <=700):
#             distance1 = np.linalg.norm(np.array([self.x-25,self.y-25]))
#             robust = distance1 - 5
#             #reward = robust
#             if(robust>30.35):
#                 reward = 30.35 - robust
#             else:
#                 reward = (robust + 35.35*np.exp(-(1/200)*0.152*(self.timestep))-30.35)
#         else:
#             distance2 = np.linalg.norm(np.array([self.x-25,self.y-25]))
#             robust = 3 - distance2
#             #reward = robust
#             reward = (robust + 35.35*np.exp(-(1/900)*2.46*(self.timestep-700))-3)
            
#        return np.array([self.x, self.y,self.theta, (self.timestep)/2500]), reward, done, None
        return np.array([self.x, self.y,self.theta, (self.timestep-self.epi_len//2)/10]), reward, done, None

        #return np.array([theta_, theta_dot_]), reward, done, None
    
    def get_action_dim(self):
        return len(self.action_dict)
    
    def get_state_dim(self):
        return 4