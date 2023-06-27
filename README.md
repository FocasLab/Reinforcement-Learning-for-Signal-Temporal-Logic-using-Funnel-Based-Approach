# Reinforcement-Learning-for-Signal-Temporal-Logic-using-Funnel-Based-Approach


# USAGE 

* Run train.py to train the network.
* Trained network will be stored in the path provided in config['log_name'] in the config.py file, config['log_best_name'] will store the instance where maximum reward obtained.
* Trained log files are provided in each example and can be used in test.py without training again.
* loadparameters(path_to_trained_log_file) is the function called from test.py to load trained network.
