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
from robot_env import RobotEnv as robot_env

torch.manual_seed(config['seed'])
np.random.seed(config['seed'])

#env = robot_env(config)
env_eval = robot_env(config)

config['state_dim'] = env_eval.get_state_dim()
config["action_dim"] = env_eval.get_action_dim()
config['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


buffer = BufferList(config['buffer_size'])
agent = Agent(env_eval, config)

load_parameters(agent,'log_turtlebot_2_goal_3_obs_nobn_3layer_1discretize_100secs_square_obs',config['device'])
epi_len = config['epi_len_eval']

plot = []
n_iter = 1

#def plot_trajecory():
    

for epi in range(n_iter):
    total_reward = 0
#    state = env_eval.reset()
#    state = env_eval.reset()
    state = env_eval.reset()
    print(state)

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
    
time_scale = np.arange(0,2000,1) 


time_plot = np.arange(0,20,0.01)

plot = np.array(plot)

fig,ax=plt.subplots(1,1)

#ax.plot(plot[:,0],plot[:,1])
#circle1 = Circle((2,2),2,linestyle = '--',fill = False,color = 'orange')
#
#circle2 = Circle((2,2),0.5,linestyle = '--',fill = False,color = 'red')
#circle21 = Circle((2,2),0.2,linestyle = '--',fill = True,color = 'red')
#
#
#circle3 = Circle((2,0.5),0.5,linestyle = '--',fill = False,color = 'red')
#circle31 = Circle((2,0.5),0.2,linestyle = '--',fill = True,color = 'red')
#
#
#circle4 = Circle((3,3),0.5,linestyle = '--',fill = False,color = 'red')
#circle41 = Circle((3,3),0.2,linestyle = '--',fill = True,color = 'red')
#
#
#circle5 = Circle((3,1),0.3,linestyle = '--',fill = False,color = 'green')
#circle6 = Circle((1,3),0.3,linestyle = '--',fill = False,color = 'green')
#
#ax.plot(plot[:,0],plot[:,1])
##circle1 = Circle((1,1),1,linestyle = '--',fill = False,color = 'green')
##
##circle2 = Circle((3,3),1,linestyle = '--',fill = False,color = 'green')
#
#
#
#
#square = Rectangle((0,0),4,4,fill = False, color = "orange")
#
#ax.add_patch(square)
#ax.add_patch(circle2)
#ax.add_patch(circle1)
#ax.add_patch(circle21)
#ax.add_patch(circle31)
#ax.add_patch(circle41)
#
#ax.add_patch(circle3)
#ax.add_patch(circle4)
#ax.add_patch(circle5)
#ax.add_patch(circle6)




square2 = Rectangle((1.65,1.65),0.7,0.7,fill = True, color = "red")
square3 = Rectangle((1.65,0.15),0.7,0.7,fill = True, color = "red")
square4 = Rectangle((2.65,2.65),0.7,0.7,fill = True, color = "red")


circle5 = Circle((3,1),0.3,linestyle = '--',fill = False,color = 'green')
circle6 = Circle((1,3),0.3,linestyle = '--',fill = False,color = 'green')

ax.plot(plot[:,0],plot[:,1])
#circle1 = Circle((1,1),1,linestyle = '--',fill = False,color = 'green')
#
#circle2 = Circle((3,3),1,linestyle = '--',fill = False,color = 'green')



square = Rectangle((0,0),4,4,fill = False, color = "orange")

ax.add_patch(square)
ax.add_patch(square2)
ax.add_patch(square3)
ax.add_patch(square4)
ax.add_patch(circle5)
ax.add_patch(circle6)


ax.scatter(plot[0][0],plot[0][1],color = 'green')
