import json
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# Location → numeric ID mapping
# ---------------------------------------------------------
LOCATION_MAP = {
    "London": 21,
    "Mumbai": 22,
    "Ohio": 23,
    "SaoPaulo": 24,
    "Sydney": 25,
    "Tokyo": 26,
}

# ---------------------------------------------------------
# 1. Load iperf3 JSON
# ---------------------------------------------------------
def load_iperf_json(path):
    txt = Path(path).read_text()
    marker = "iperf_json_finish"
    idx = txt.find(marker)
    if idx != -1:
        txt = txt[:idx]
    return json.loads(txt)

# ---------------------------------------------------------
# 2. Convert iperf intervals → DataFrame
# ---------------------------------------------------------
def iperf_to_dataframe(js):
    rows = []
    for intr in js["intervals"]:
        s = intr["streams"][0]
        rows.append({
            "t": s["end"],
            "bps": s["bits_per_second"],
            "retransmits": s.get("retransmits", np.nan),
            "snd_cwnd": s.get("snd_cwnd", np.nan),
            "snd_wnd": s.get("snd_wnd", np.nan),
            "rtt": s.get("rtt", np.nan),
            "rttvar": s.get("rttvar", np.nan),
        })
    return pd.DataFrame(rows).sort_values("t").reset_index(drop=True)

# ---------------------------------------------------------
# 3. Macro-phase detector (ProbeBW_UP / ProbeBW_DOWN / CRUISE)
# ---------------------------------------------------------
def detect_macro_phases(df, smooth_win=10, std_factor=0.7):

    df = df.copy()
    df["rate_smooth"] = df["bps"].rolling(smooth_win, center=True, min_periods=1).mean()
    df["dev"] = df["bps"] - df["rate_smooth"]

    dev_std = df["dev"].std()
    if not np.isfinite(dev_std) or dev_std == 0:
        dev_std = 1.0

    up_thresh = std_factor * dev_std
    down_thresh = -std_factor * dev_std

    N = len(df)
    phase = np.full(N, 3, dtype=int)  # default CRUISE
    dev = df["dev"].to_numpy()

    def is_local_max(i): return dev[i] > dev[i-1] and dev[i] >= dev[i+1]
    def is_local_min(i): return dev[i] < dev[i-1] and dev[i] <= dev[i+1]

    i = 1
    while i < N - 2:
        if dev[i] > up_thresh and is_local_max(i):
            phase[i] = 1  # ProbeBW_UP
            j = i + 1
            found_down = False

            while j < N - 2:
                if dev[j] < down_thresh and is_local_min(j):
                    phase[j] = 2  # ProbeBW_DOWN
                    found_down = True
                    break
                j += 1

            if found_down:
                for k in range(j + 1, min(j + 7, N)):
                    phase[k] = 3  # CRUISE
                i = j + 7
            else:
                i += 1
        else:
            i += 1

    df["phase"] = phase
    return df

# ---------------------------------------------------------
# 4. Discrete pacing gains and mapping
# ---------------------------------------------------------
ACTION_TO_GAIN = {
    1: 1.05, 2: 1.10, 3: 1.15, 4: 1.20, 5: 1.25,
    6: 0.90, 7: 0.92, 8: 0.94, 9: 0.96, 10: 0.98,
    11: 1.00
}
GAIN_TO_ACTION = {v: k for k, v in ACTION_TO_GAIN.items()}

# Phase-specific allowed actions
ACTIONS_UP = [1, 2, 3, 4, 5]
ACTIONS_DOWN = [6, 7, 8, 9, 10]
ACTIONS_CRUISE = [11]

# ---------------------------------------------------------
# 5. Phase-constrained pacing gain selection
# ---------------------------------------------------------
def add_pacing_gains(df, link_capacity_bps):

    df = df.copy()
    df["U"] = df["bps"] / link_capacity_bps

    df["P_up_raw"] = 3 / (df["U"] + 2)
    df["P_down_raw"] = (df["U"] + 1) / 2

    S_up   = np.array([1.05, 1.10, 1.15, 1.20, 1.25])
    S_down = np.array([0.90, 0.92, 0.94, 0.96, 0.98])

    def snap(val, S):
        return S[np.argmin(np.abs(S - val))]

    df["P_up"]   = df["P_up_raw"]  .apply(lambda x: snap(x, S_up))
    df["P_down"] = df["P_down_raw"].apply(lambda x: snap(x, S_down))

    S_selected = []
    for ph, pup, pdown in zip(df["phase"], df["P_up"], df["P_down"]):
        if ph == 1:
            S_selected.append(pup)
        elif ph == 2:
            S_selected.append(pdown)
        else:
            S_selected.append(1.0)

    df["S_selected"] = S_selected
    df["best_action"] = df["S_selected"].map(GAIN_TO_ACTION)

    return df

# ---------------------------------------------------------
# 6. Random exploration action
# ---------------------------------------------------------
def generate_random_actions(best_actions):

    rng = np.random.default_rng()
    out = []

    for a in best_actions:
        if rng.uniform() < 0.5:
            out.append(a)
        else:
            choices = [x for x in range(1, 12) if x != a]
            out.append(rng.choice(choices))

    return pd.Series(out, index=best_actions.index)

# ---------------------------------------------------------
# 7. Reward for single action
# ---------------------------------------------------------
def compute_reward_single(df, i, action, lambda_penalty=0.1):

    gain = ACTION_TO_GAIN[action]

    bps_i     = df.loc[i, "bps"]
    retrans_i = df.loc[i, "retransmits"]
    B_ref_i   = df.loc[i, "B_ref"]
    R_ref_i   = df.loc[i, "R_ref"]
    U_i       = df.loc[i, "U_hybrid"]

    T_i = min(bps_i / B_ref_i, 1.0)
    C_i = np.tanh(retrans_i / R_ref_i)
    up_penalty = max(gain - 1.0, 0.0)

    return T_i - 0.5 * C_i - lambda_penalty * U_i * up_penalty

# ---------------------------------------------------------
# 8. Reward for random actions + hybrid utilization
# ---------------------------------------------------------
def compute_rewards(df, random_actions, lambda_penalty=0.1, window=20):

    df = df.copy()
    df["random_action"] = random_actions

    df["B_ref"] = df["bps"].rolling(window, min_periods=1).quantile(0.95).fillna(method="bfill")
    df["R_ref"] = df["retransmits"].rolling(window, min_periods=1).quantile(0.95).fillna(method="bfill") + 1.0

    T = np.minimum(df["bps"] / df["B_ref"], 1.0)
    C = np.tanh(df["retransmits"] / df["R_ref"])

    RTT_min = df["rtt"].min()
    queue_delay = np.maximum(df["rtt"] - RTT_min, 0)

    q_ref_raw = queue_delay.rolling(window, min_periods=1).quantile(0.95).fillna(method="bfill")
    q_bound = 0.15 * RTT_min
    q_ref = np.minimum(q_ref_raw, q_bound).replace(0, 1)

    U_delay = np.minimum(queue_delay / q_ref, 1.0)
    U_rate = T
    U = np.maximum(U_rate, U_delay)

    df["U_hybrid"] = U

    gains = df["random_action"].map(ACTION_TO_GAIN)
    up_penalty = np.maximum(gains - 1.0, 0.0)

    df["reward_random"] = T - 0.5 * C - lambda_penalty * U * up_penalty

    return df

# ---------------------------------------------------------
# 9. Reward-optimal action (PHASE-CONSTRAINED)
# ---------------------------------------------------------
def find_best_actions_reward_based(df, lambda_penalty=0.1):

    best_actions = []
    best_rewards = []
    best_gains = []

    for i in range(len(df)):
        ph = df.loc[i, "phase"]

        # Allowed actions for this phase
        if ph == 1:
            candidates = ACTIONS_UP
        elif ph == 2:
            candidates = ACTIONS_DOWN
        else:
            candidates = ACTIONS_CRUISE

        rewards = {a: compute_reward_single(df, i, a, lambda_penalty) for a in candidates}

        best_a = max(rewards, key=rewards.get)
        best_actions.append(best_a)
        best_rewards.append(rewards[best_a])
        best_gains.append(ACTION_TO_GAIN[best_a])

    df["best_action_reward_based"] = best_actions
    df["best_gain_reward_based"] = best_gains
    df["reward_best"] = best_rewards

    return df


# ---------------------------------------------------------
# MAIN: Process ALL BBR logs
# ---------------------------------------------------------
if __name__ == "__main__":

    folder = Path("D:/Rakshitha De Silva/starlink-iperf3-data/downlink")

    all_rows = []

    for file in folder.glob("bbr_*_REV.json"):

        name = file.stem.replace("bbr_", "").replace("_REV", "")
        location_id = LOCATION_MAP.get(name, -1)

        print(f"Processing {name} (ID={location_id}) ...")

        js = load_iperf_json(file)
        df = iperf_to_dataframe(js)

        df = detect_macro_phases(df)

        link_capacity = df["bps"].max()
        df = add_pacing_gains(df, link_capacity)

        random_actions = generate_random_actions(df["best_action"])

        df = compute_rewards(df, random_actions)

        df = find_best_actions_reward_based(df)

        df.insert(0, "location", location_id)

        all_rows.append(df)

    df_all = pd.concat(all_rows, ignore_index=True)

    cols = [
        "location", "t", "bps", "retransmits", "snd_cwnd", "snd_wnd",
        "rtt", "rttvar", "phase",
        "S_selected", "best_action",
        "random_action", "reward_random",
        "best_action_reward_based", "best_gain_reward_based",
        "reward_best"
    ]

    output_csv = "processed_bbr_data.csv"
    df_all[cols].to_csv(output_csv, index=False)

    print(f"\nSaved CSV: {output_csv}")
    print(df_all[cols].head())
