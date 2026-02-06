import json
import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------------------------------------------------
# Location → numeric mapping
# -------------------------------------------------------------------
LOCATION_MAP = {
    "London": 21,
    "Mumbai": 22,
    "Ohio": 23,
    "SaoPaulo": 24,
    "Sydney": 25,
    "Tokyo": 26,
}

# -------------------------------------------------------------------
# Safe JSON loader
# -------------------------------------------------------------------
def load_iperf_json(path):
    txt = Path(path).read_text()
    marker = "iperf_json_finish"
    idx = txt.find(marker)
    if idx != -1:
        txt = txt[:idx]
    return json.loads(txt)

# -------------------------------------------------------------------
# Convert iperf3 JSON interval data → dataframe
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Macro-phase detector (ProbeBW_UP / ProbeBW_DOWN / CRUISE)
# -------------------------------------------------------------------
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
    phase = np.full(N, 3, dtype=int)
    dev = df["dev"].to_numpy()

    def is_local_max(i):
        return dev[i] > dev[i-1] and dev[i] >= dev[i+1]

    def is_local_min(i):
        return dev[i] < dev[i-1] and dev[i] <= dev[i+1]

    i = 1
    while i < N - 2:
        if dev[i] > up_thresh and is_local_max(i):
            phase[i] = 1
            j = i + 1
            found_down = False

            while j < N - 2:
                if dev[j] < down_thresh and is_local_min(j):
                    phase[j] = 2
                    found_down = True
                    break
                j += 1

            if found_down:
                for k in range(j+1, min(j+7, N)):
                    phase[k] = 3
                i = j + 7
            else:
                i += 1
        else:
            i += 1

    df["phase"] = phase
    return df

# -------------------------------------------------------------------
# Deep-BBR pacing + BBRv3 rule + Action mapping
# -------------------------------------------------------------------
def add_pacing_gains(df, link_capacity_bps):
    df = df.copy()
    df["U"] = df["bps"] / link_capacity_bps

    df["P_up_raw"] = 3 / (df["U"] + 2)
    df["P_down_raw"] = (df["U"] + 1) / 2

    S_up = np.array([1.05, 1.1, 1.15, 1.2, 1.25])
    S_down = np.array([0.9, 0.92, 0.94, 0.96, 0.98, 1.0])

    def snap(x, S):
        return S[np.argmin(np.abs(S - x))]

    df["P_up"] = df["P_up_raw"].apply(lambda x: snap(x, S_up))
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

    ACTION_MAP = {
        1.05: 1,
        1.10: 2,
        1.15: 3,
        1.20: 4,
        1.25: 5,
        0.90: 6,
        0.92: 7,
        0.94: 8,
        0.96: 9,
        0.98: 10,
        1.00: 11
    }

    df["Action"] = df["S_selected"].map(ACTION_MAP)

    # ----------------------------------------------------------
    # ADD training_action column (new requirement)
    # ----------------------------------------------------------
    training_actions = []
    rng = np.random.default_rng()  # high-quality random generator

    for act in df["Action"]:
        if rng.uniform() < 0.5:
            # 50% chance: same as Action
            training_actions.append(act)
        else:
            # 50% chance: random action from 1–11 except act
            choices = [a for a in range(1, 12) if a != act]
            training_actions.append(rng.choice(choices))

    df["training_action"] = training_actions

    return df

# -------------------------------------------------------------------
# MAIN: Process all logs and save combined CSV
# -------------------------------------------------------------------
folder = Path(r"D:\Rakshitha De Silva\starlink-iperf3-data\downlink")
output_csv = "all_locations_bbr_labeled.csv"

all_rows = []

for json_file in folder.glob("bbr_*_REV.json"):
    name = json_file.stem.replace("bbr_", "").replace("_REV", "")
    location_id = LOCATION_MAP.get(name, -1)

    js = load_iperf_json(json_file)
    df = iperf_to_dataframe(js)
    df = detect_macro_phases(df)

    link_capacity = df["bps"].max()
    df = add_pacing_gains(df, link_capacity)

    df.insert(0, "location", location_id)
    all_rows.append(df)

df_all = pd.concat(all_rows, ignore_index=True)

cols = [
    "location", "t", "bps", "retransmits", "snd_cwnd", "snd_wnd",
    "rtt", "rttvar", "phase", "S_selected", "Action", "training_action"
]

df_all[cols].to_csv(output_csv, index=False)

print("Saved CSV:", output_csv)
