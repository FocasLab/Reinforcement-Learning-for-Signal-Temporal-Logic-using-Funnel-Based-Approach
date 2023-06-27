import torch.nn as nn
import numpy as np

config = {}

config['theta_max'] = 3.14
config['theta_min'] = -3.13
#config['theta_dot_max'] = 1.5
#config['theta_dot_min'] = -1.5
config['state_dis'] = 2

config['action_max'] = 2.5
config['action_min'] = -2.5
config['action_dis'] = 1

config['time_int'] = 0.01
config['theta'] = np.random.uniform(-3.13, 3.14)
config['theta_dot'] = 0

config['seed'] = 5
config['buffer_size'] = int(1e6)
config['total_env_steps'] = int(2e6)
config['batch_size'] = 256
config['warmup_samples'] = 5000
config['eval_freq'] = 5000
config['epi_len'] = 2000
config['n_iter_eval'] = 5
config['epi_len_eval'] = 2000
config['act_fn'] = nn.ReLU
config['tau'] = 0.995
config['log'] = True
config['log_name'] = "log"
config['log_best'] = "log_best"
config['update_freq'] = 1
config['q_freq'] = 1
config['gamma'] = 0.99
config['init_epsilon'] = 1
config['final_epsilon'] = 0.1
config['anneal_limit'] = int(10e5)
config['target_update'] = 10000


config['lr_q'] = 1e-4
config['hidden_layer'] = [256,256,64]