import numpy as np
import torch
import torch.nn as nn
#from PendulumEnv import PendulumEnv
from model import Agent
from buffer import BufferList
import sys
from logger import load_parameters
from config import config
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
from robot_env_flags import RobotEnv as robot_env_flags
from robot_env_funnel import RobotEnv as robot_env_funnel


torch.manual_seed(config['seed'])
np.random.seed(config['seed'])

#env = robot_env(config)
env_eval = robot_env_flags(config)

config['state_dim'] = env_eval.get_state_dim()
config["action_dim"] = env_eval.get_action_dim()
config['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


buffer = BufferList(config['buffer_size'])
agent = Agent(env_eval, config)

load_parameters(agent,'log_flag_rewards',config['device'])
epi_len = config['epi_len_eval']

plot = []
n_iter = 1

#def plot_trajecory():
    

for epi in range(n_iter):
    total_reward = 0
    state = env_eval.reset()
    state = env_eval.reset()
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
    
robusts = np.array(env_eval.robusts)

print(min(robusts[100:]))

time_plot = np.arange(0,20,0.01)


plot = np.array(plot)


fig,ax=plt.subplots(1,1)


ax.plot(plot[:,0],plot[:,1])
circle1 = Circle((2,2),1,linestyle = '--',fill = True,color = 'orange')

ax.add_patch(circle1)


#
ax.set_xlim(0,6)
ax.set_ylim(0,6)


#plt.savefig("robo_roustness_final_3",dpi = 800)



