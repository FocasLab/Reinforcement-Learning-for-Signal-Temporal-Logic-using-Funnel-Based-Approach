import matplotlib.pyplot as plt
import pickle

path = './log_turtlebot_2_goal_3_obs_nobn_3layer_1discretize_100secs_square_obs/logfile.pkl'
file = open(path, 'rb')
data = pickle.load(file)

print(data.keys())

#if(len(data.keys()) == 1):
#    n_cols = 1
#    n_rows = 1
#elif(len(data.keys()) == 2):
#    n_cols = 2
#    n_rows = 1
#elif(len(data.keys()) == 3):
#    n_cols = 3
#    n_rows = 1
#elif(len(data.keys()) == 4):
#    n_cols = 2
#    n_rows = 2
#elif(len(data.keys()) == 5):
#    n_cols = 3
#    n_rows = 2
#elif(len(data.keys()) == 6):
#    n_cols = 3
#    n_rows = 2
#else:
#    n_cols = 3
#    n_rows = 3


#index = 1
#for key in data.keys():
#    plt.subplot(n_rows,n_cols,index)
#    plt.title(key)
#    plt.plot(data[key])
#    index+=1
#
#plt.savefig('results_3.png')

plt.plot(data["Reward"])
print(max(data["Reward"]))
