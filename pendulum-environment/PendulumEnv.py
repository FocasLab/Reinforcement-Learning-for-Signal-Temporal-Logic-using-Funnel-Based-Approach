import numpy as np
import math

class PendulumEnv:
    def __init__(self,config):
        self.theta = config['theta']
        self.theta_dot = config['theta_dot']
        self.time_int = config["time_int"]
        self.timestep = 0
        self.epi_len = config['epi_len']
        self.config = config
        
        self.g = 9.8
        self.m = 0.15
        self.l = 0.5
        self.mu = 0.05        

        self.action_dict = {}
        self.action_dict_inv = {}
        
        self.total_robusts = []
        
        for index,i in enumerate(np.arange(config['action_min']*10**config['action_dis'],config['action_max']*10**config['action_dis']+1)):
            i_ = round(i/(10**config['action_dis']),config['action_dis'])
            self.action_dict[i_] = index
            self.action_dict_inv[index] = i_

                
    def reset(self):
        self.timestep = 0
        self.theta = 2.3
        #self.theta = np.random.uniform(-3.13, 3.14)
        self.theta_dot = self.config['theta_dot']
        return np.array([self.theta, self.theta_dot, (self.timestep-self.epi_len//2)/100])
        #return np.array([self.theta, self.theta_dot])
        
    def step(self,action):
        done = False
        self.timestep+=1
        if(self.timestep >= self.epi_len):
            done = True
        
        #print("theta before",self.theta)
        self.theta = self.theta + self.time_int*self.theta_dot
        #print("theta after",self.theta)
        if(self.theta > self.config['theta_max'] or self.theta < self.config['theta_min'] ):
           #self.theta = ((math.floor(self.theta*100)+313)%628-313)/100
           self.theta = ((self.theta*100 + 313)%628-313)/100
                
        #print("theta dot before", self.theta_dot)
        self.theta_dot = self.theta_dot + self.time_int*((self.g/self.l)*np.sin(self.theta) + self.action_dict_inv[action]/(self.m*self.l**2) - (self.mu*self.theta_dot)/(self.m*self.l**2))
        #print("theta dot after", self.theta_dot)
        #self.theta_dot = max(min(self.config['theta_dot_max'],self.theta_dot), self.config['theta_dot_min'])
                
        reward  = 0
        theta_ = ((self.theta*100 + 313)%628-313)/100
        #theta_dot_ = round(self.theta_dot, self.config['state_dis'])
        theta_dot_ = self.theta_dot
        
        if(self.timestep <=700):
#            robust = min((0.06 - abs(theta_)),(0.06 - abs(theta_dot_)))
            robust = 0.06 - abs(theta_)
            reward = (robust + 3.2* np.exp(-(1/400)*4.15*self.timestep)-0.05)
            
        elif(self.timestep >=700 and self.timestep <=1200):
#            robust = min((0.06 - abs(1.57 - theta_)),(0.06 - abs(theta_dot_)))
            robust = 0.06 - abs(1.57 - theta_)
            reward = (robust + 1.7*np.exp(-(1/300)*3.5*(self.timestep-700))-0.05)
        
        elif(self.timestep >=1200):
#            robust = min((0.06 - abs(-1.57-theta_)),(0.06 - abs(theta_dot_)))
            robust = 0.06 - abs(-1.57 - theta_)
            reward = (robust + 3.2*np.exp(-(1/500)*4.15*(self.timestep-1200))-0.05)
        
        
            
        self.total_robusts.append(robust)

        return np.array([theta_, theta_dot_, (self.timestep-self.epi_len//2)/100]), reward, done, None
        #return np.array([theta_, theta_dot_]), reward, done, None
    
    def get_action_dim(self):
        return len(self.action_dict)
    
    def get_state_dim(self):
        return 3