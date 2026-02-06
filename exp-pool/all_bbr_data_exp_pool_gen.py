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
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)


# -----------------------
# Config
# -----------------------
TOKYO_LOCATION_ID = 26

# If you also want dataset_flag inside the state, change this list to include it.
columns_to_use = ["dataset_flag","location", "t", "bps", "retransmits", "snd_cwnd", "snd_wnd", "rtt", "rttvar"]

csv_path = r"D:\Rakshitha De Silva\1LLM\exp-pool\processed_bbr_all_datasets.csv"

train_pkl_path = r"../llm_framework/data/exp_pools/new_bbr_exp_pool.pkl"
test_pkl_path  = r"../llm_framework/data/exp_pools/new_bbr_exp_pool_test.pkl"


# -----------------------
# Load CSV
# -----------------------
df = pd.read_csv(csv_path)

# Basic sanity checks
required_cols = set(columns_to_use + ["best_action_reward_based", "reward_best", "location"])
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}\nCSV columns are: {df.columns.tolist()}")

print("CSV loaded:", df.shape)
print("Locations found:", sorted(df["location"].unique().tolist()))

# -----------------------
# Split: Tokyo = test, rest = train
# -----------------------
df_testing = df[df["location"] == TOKYO_LOCATION_ID].copy().reset_index(drop=True)
df_training = df[df["location"] != TOKYO_LOCATION_ID].copy().reset_index(drop=True)

print("Training shape:", df_training.shape)
print("Testing shape :", df_testing.shape)


# -----------------------
# Build training pool
# -----------------------
exp_pool_training = ExperiencePool()

for _, row in df_training.iterrows():
    state = np.array(row[columns_to_use], dtype=float)
    reward = float(row["reward_best"])
    action = int(row["best_action_reward_based"])
    exp_pool_training.add(state=state, action=action, reward=reward, done=0)

pickle.dump(exp_pool_training, open(train_pkl_path, "wb"))
print("Done. Training Experience pool saved at:", train_pkl_path)
print("Training pool size:", len(exp_pool_training))


# -----------------------
# Build testing pool
# -----------------------
exp_pool_testing = ExperiencePool()

for _, row in df_testing.iterrows():
    state = np.array(row[columns_to_use], dtype=float)
    reward = float(row["reward_best"])
    action = int(row["best_action_reward_based"])
    exp_pool_testing.add(state=state, action=action, reward=reward, done=0)

pickle.dump(exp_pool_testing, open(test_pkl_path, "wb"))
print("Done. Evaluating Experience pool saved at:", test_pkl_path)
print("Testing pool size:", len(exp_pool_testing))
