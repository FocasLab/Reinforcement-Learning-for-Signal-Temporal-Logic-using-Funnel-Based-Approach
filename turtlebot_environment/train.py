import numpy as np
import torch
import torch.nn as nn
from model import Agent
from buffer import BufferList
import sys
from logger import save_parameters
from config import config
import pickle
from robot_env import RobotEnv as robot_env

torch.manual_seed(config['seed'])
np.random.seed(config['seed'])

env = robot_env(config)
env_eval = robot_env(config)

config['state_dim'] = env.get_state_dim()
config["action_dim"] = env.get_action_dim()


# config['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
config['device'] = torch.device('cpu')



buffer = BufferList(config['buffer_size'])
agent = Agent(env_eval, config)
total_env_steps = config['total_env_steps']
batch_size = config['batch_size']
learning_start = config['warmup_samples']
eval_freq = config['eval_freq']
update_freq = config['update_freq']
target_update = config['target_update']


epi_len = config['epi_len']


n_steps = 0
count = 0
history_rho = 0

pre_best_rew = -float('inf')

print("self actor-critic training...")

while(count < total_env_steps//eval_freq):
    state = env.reset()
    done = False
    for epi_steps in range(1,epi_len+1):
        n_steps += 1
        
        if(epi_steps == epi_len):
             done  = True  

        if(len(buffer)<learning_start):
            action = np.random.randint(0,config['action_dim'])
        else:
            action = agent.get_action(torch.tensor(state).float()).detach()
            if(n_steps <= config['anneal_limit']):
                agent.anneal()
        
        
        next_state, reward, done, _ = env.step(int(action))
        buffer.insert(state,reward,action,done,next_state)
                     
        state = next_state        
        
        if(n_steps % eval_freq == 0):
            
            count+=1    
            temp = agent.evaluate(epi_len = config['epi_len_eval'], n_iter = config['n_iter_eval'])
        
            agent.logger.add_scalar('Reward',temp[0])
            agent.logger.add_scalar('rho_eval', temp[1])
          
            if(temp[0]>pre_best_rew):
                pre_best_rew = temp[0]
                save_parameters(agent,config['log_best_name'])
                
            save_parameters(agent,config['log_name'])
          
            #agent.update_target()
          
            print(count," Reward", temp[0], flush = True)
          
            history_rho = temp[1]
    
        if(len(buffer)>=batch_size and n_steps % update_freq == 0):
            agent.update(buffer, batch_size = batch_size, gradient_steps = update_freq, q_freq = config['q_freq'])
        
        if(n_steps % target_update == 0):
            agent.update_target()
            

agent.logger.flush()        