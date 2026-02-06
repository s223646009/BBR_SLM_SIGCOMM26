import pandas as pd
import numpy as np
import pickle
from pathlib import Path

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
DATASET_FLAGS = [1, 2, 3, 4]

# columns_to_use = ["location", "t", "bps", "retransmits", "snd_cwnd", "snd_wnd", "rtt", "rttvar"]
columns_to_use = ["dataset_flag","location", "t", "bps", "retransmits", "snd_cwnd", "snd_wnd", "rtt", "rttvar"]

csv_path = r"D:\Rakshitha De Silva\1LLM\exp-pool\processed_bbr_all_datasets.csv"

out_dir = Path(r"../llm_framework/data/exp_pools")
out_dir.mkdir(parents=True, exist_ok=True)

# Save ONLY test pools (Tokyo)
test_all_path = out_dir / "new_bbr_exp_pool_test_tokyo_allflags.pkl"
test_flag_path_tpl = out_dir / "new_bbr_exp_pool_test_tokyo_datasetflag_{flag}.pkl"


def build_and_save_pool(df_part: pd.DataFrame, save_path: Path):
    exp_pool = ExperiencePool()

    df_part = df_part.copy()
    df_part["retransmits"] = df_part["retransmits"].fillna(0.0)

    for _, row in df_part.iterrows():
        state = np.array(row[columns_to_use], dtype=float)
        reward = float(row["reward_random"])
        action = int(row["random_action"])
        exp_pool.add(state=state, action=action, reward=reward, done=0)

    with open(save_path, "wb") as f:
        pickle.dump(exp_pool, f)

    print(f"Saved: {save_path} | rows={len(df_part)} | pool_size={len(exp_pool)}")
    return exp_pool


# -----------------------
# Load CSV
# -----------------------
df = pd.read_csv(csv_path)

required_cols = set(columns_to_use + ["random_action", "reward_random", "location", "dataset_flag"])
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}\nCSV columns are: {df.columns.tolist()}")

print("CSV loaded:", df.shape)

# -----------------------
# TESTING = Tokyo only
# -----------------------
df_tokyo_all = df[df["location"] == TOKYO_LOCATION_ID].copy().reset_index(drop=True)
if df_tokyo_all.empty:
    raise ValueError(f"No Tokyo rows found (location={TOKYO_LOCATION_ID}) in the CSV.")

# Save 1) Tokyo all flags test pool
build_and_save_pool(df_tokyo_all, test_all_path)

# Save 2) Tokyo test pools split by dataset_flag
for flag in DATASET_FLAGS:
    df_tokyo_flag = df[(df["location"] == TOKYO_LOCATION_ID) & (df["dataset_flag"] == flag)].copy()
    df_tokyo_flag = df_tokyo_flag.reset_index(drop=True)

    save_path = Path(str(test_flag_path_tpl).format(flag=flag))

    if df_tokyo_flag.empty:
        print(f"[WARN] No rows for Tokyo (location={TOKYO_LOCATION_ID}) with dataset_flag={flag}. "
              f"Skipping {save_path.name}")
        continue

    build_and_save_pool(df_tokyo_flag, save_path)
