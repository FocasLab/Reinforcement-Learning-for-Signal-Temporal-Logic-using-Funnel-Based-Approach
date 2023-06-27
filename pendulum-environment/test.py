import numpy as np
import torch
import torch.nn as nn
from PendulumEnv import PendulumEnv
from model import Agent
from buffer import BufferList
import sys
from logger import load_parameters
from config import config
import matplotlib.pyplot as plt

torch.manual_seed(config['seed'])
#np.random.seed(config['seed'])

env_eval = PendulumEnv(config)

config['state_dim'] = env_eval.get_state_dim()
config["action_dim"] = env_eval.get_action_dim()
config['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


buffer = BufferList(config['buffer_size'])
agent = Agent(env_eval, config)

load_parameters(agent,'log_pendulum_3_angles',config['device'])
epi_len = config['epi_len_eval']

plot = []
n_iter = 1
for epi in range(n_iter):
    total_reward = 0
    state = env_eval.reset()
    done = False
    steps = 0    
    while not done:
        action = agent.get_action(torch.tensor(state).float(), deterministic=True).detach()
        next_state, reward, done, _ = env_eval.step(int(action))
        #print("action",action)
        plot.append(state)
        total_reward += reward
        state = next_state
        steps +=1
    print("Reward",total_reward)
    print("steps",steps)
    
robusts = env_eval.total_robusts
time_scale = np.arange(0,2000,1) 
funnel_up = [0.06]*2000
funnel1 = -3.2* np.exp(-(1/400)*4.15*time_scale[:700])
funnel2 = -1.74*np.exp(-(1/300)*3.52*time_scale[:500])
funnel3 = -3.2* np.exp(-(1/400)*4.15*time_scale[:800])

effective_funnel = list(funnel1) + list(funnel2) + list(funnel3)

time_plot = np.arange(0,20,0.01) 
print(len(time_plot))
#print(len(time_plot))
#print(time_plot)
#print(time_scale)
#
#print(funnel1,time_scale[:600])
#plt.plot(time_plot,robusts)
#plt.plot(robusts)
#plt.plot(funnel1)
#plt.plot(time_plot,effective_funnel,color = "orange")
#plt.plot(time_plot,funnel_up,color = "orange")
#plt.xlabel("Time",fontsize = 12)
#plt.ylabel("Robustness",fontsize = 12)
#plt.legend(["robustness","funnel"])
#plt.savefig("cartpole_robustness_3",dpi = 800)

plot = np.array(plot)

#print(plot[:,0])
#plt.plot(plot[:,0],color = "red")

fig,ax=plt.subplots(1,1)
ax.plot(plot[:,0],color='red')
ax.vlines(400,-3.2,3.2,linestyles='dotted',color='black')
ax.vlines(700,-3.2,3.2,linestyles='dotted',color='black')
ax.vlines(1000,-3.2,3.2,linestyles='dotted',color='black')
ax.vlines(1200,-3.2,3.2,linestyles='dotted',color='black')
ax.vlines(1700,-3.2,3.2,linestyles='dotted',color='black')
ax.vlines(2000,-3.2,3.2,linestyles='dotted',color='black')
#ax.hlines(1.57,0,12,linestyles='dotted',color='black')
#ax.hlines(-1.57,0,20,linestyles='dotted',color='black')
ax.hlines(0,0,2000,linestyles='dotted',color='black')
ax.set_xticks((0,400,700,1000,1200,1700,2000))
ax.set_xticklabels((0,400,700,1000,1200,1700,2000),fontsize = 11)
#ax.set_yticks((-3.13,-1.57,0,1.57,3.13))
#ax.set_yticklabels((-3.13,-1.57,0,1.57,3.13),fontsize = 11)


#plt.plot(plot[:,1])
plt.xlabel("Time",fontsize = 12)
plt.ylabel("Angular velocity",fontsize = 12)
#ax.legend(["robustness","funnel"])
plt.savefig("cartpole_angular_velocity_3",dpi = 800)


