import pandas as pd
import numpy as np
import pickle

class ExperiencePool:
    """
    Experience pool for collecting trajectories.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def add(self, state, action, reward, done):
        self.states.append(state)  # sometime state is also called obs (observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)




# Define the list of columns to include
columns_to_use = ["location","t",'bps','retransmits','snd_cwnd','snd_wnd','rtt','rttvar']


df = pd.read_csv("D:/Rakshitha De Silva/1LLM/exp-pool/all_locations_bbr_labeled.csv")
df_training = df.iloc[:1506]
df_testing = df.iloc[1506:]
print(df.head())
print(df.tail())
print(df.shape)


# print(df.columns.tolist())
exp_pool_training = ExperiencePool()
exp_pool_testing = ExperiencePool()



# Iterate through each row and update the global reward variable
for index, row in df_training.iterrows():
    # global_reward += row['current_queue_delay']
    state = np.array(row[columns_to_use])
    reward = 0
    action_pred = row['action']
    action = row['expected_action']
    #  Reward Logic
    if action_pred == action:
        reward = 1
    else:
        reward = -1
    exp_pool_training.add(state=state, action=row['action'], reward=reward, done=0)

pickle_save_path = "../llm_framework/data/exp_pools/bbr_exp_pool.pkl" # Training data
pickle.dump(exp_pool_training, open(pickle_save_path, 'wb'))
print(f"Done. Training Experience pool saved at:", pickle_save_path)


# Iterate through each row and update the global reward variable
for index, row in df_testing.iterrows():
    # global_reward += row['current_queue_delay']
    state = np.array(row[columns_to_use])
    reward = 0
    action_pred = row['action']
    action = row['expected_action']
    #  Reward Logic
    if action_pred == action:
        reward = 1
    else:
        reward = -1
    exp_pool_testing.add(state=state, action=row['action'], reward=reward, done=0)

pickle_save_path = "../llm_framework/data/exp_pools/bbr_exp_pool_test.pkl" # Testing data
pickle.dump(exp_pool_testing, open(pickle_save_path, 'wb'))
print(f"Done. Testing Experience pool saved at:", pickle_save_path)