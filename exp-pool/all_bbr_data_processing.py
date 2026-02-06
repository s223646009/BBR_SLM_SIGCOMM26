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
# 1. Load iperf3 JSON (strip any trailing marker text)
# ---------------------------------------------------------
def load_iperf_json(path: Path):
    txt = path.read_text(encoding="utf-8", errors="ignore")
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
    for intr in js.get("intervals", []):
        # iperf3 typically has streams[0]
        streams = intr.get("streams", [])
        if not streams:
            continue
        s = streams[0]
        rows.append({
            "t": s.get("end", np.nan),
            "bps": s.get("bits_per_second", np.nan),
            "retransmits": s.get("retransmits", np.nan),
            "snd_cwnd": s.get("snd_cwnd", np.nan),
            "snd_wnd": s.get("snd_wnd", np.nan),
            "rtt": s.get("rtt", np.nan),
            "rttvar": s.get("rttvar", np.nan),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("t").reset_index(drop=True)
    return df

# ---------------------------------------------------------
# 3. Macro-phase detector (ProbeBW_UP / ProbeBW_DOWN / CRUISE)
#    phase: 1=UP, 2=DOWN, 3=CRUISE
# ---------------------------------------------------------
def detect_macro_phases(df, smooth_win=10, std_factor=0.7):
    df = df.copy()

    # guard
    if df.empty or "bps" not in df:
        df["phase"] = np.array([], dtype=int)
        return df

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

    def is_local_max(i): return dev[i] > dev[i - 1] and dev[i] >= dev[i + 1]
    def is_local_min(i): return dev[i] < dev[i - 1] and dev[i] <= dev[i + 1]

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

ACTIONS_UP = [1, 2, 3, 4, 5]
ACTIONS_DOWN = [6, 7, 8, 9, 10]
ACTIONS_CRUISE = [11]

# ---------------------------------------------------------
# 5. Phase-constrained pacing gain selection
# ---------------------------------------------------------
def add_pacing_gains(df, link_capacity_bps):
    df = df.copy()
    if df.empty:
        df["U"] = []
        df["P_up_raw"] = []
        df["P_down_raw"] = []
        df["P_up"] = []
        df["P_down"] = []
        df["S_selected"] = []
        df["best_action"] = []
        return df

    # avoid divide-by-zero
    link_capacity_bps = float(link_capacity_bps) if np.isfinite(link_capacity_bps) else 1.0
    if link_capacity_bps <= 0:
        link_capacity_bps = 1.0

    df["U"] = df["bps"] / link_capacity_bps
    df["P_up_raw"] = 3 / (df["U"] + 2)
    df["P_down_raw"] = (df["U"] + 1) / 2

    S_up = np.array([1.05, 1.10, 1.15, 1.20, 1.25])
    S_down = np.array([0.90, 0.92, 0.94, 0.96, 0.98])

    def snap(val, S):
        if not np.isfinite(val):
            return S[len(S)//2]
        return S[np.argmin(np.abs(S - val))]

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
    df["best_action"] = df["S_selected"].map(GAIN_TO_ACTION)

    return df

# ---------------------------------------------------------
# 6. Random exploration action
# ---------------------------------------------------------
def generate_random_actions(best_actions, seed=12345):
    rng = np.random.default_rng(seed)
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

    bps_i = df.loc[i, "bps"]
    retrans_i = df.loc[i, "retransmits"]
    B_ref_i = df.loc[i, "B_ref"]
    R_ref_i = df.loc[i, "R_ref"]
    U_i = df.loc[i, "U_hybrid"]

    # safe guards
    if not np.isfinite(B_ref_i) or B_ref_i <= 0:
        B_ref_i = 1.0
    if not np.isfinite(R_ref_i) or R_ref_i <= 0:
        R_ref_i = 1.0
    if not np.isfinite(retrans_i):
        retrans_i = 0.0
    if not np.isfinite(bps_i):
        bps_i = 0.0
    if not np.isfinite(U_i):
        U_i = 0.0

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

    # retransmits can be NaN in some logs; treat as 0 for stability
    df["retransmits"] = df["retransmits"].fillna(0.0)

    df["B_ref"] = df["bps"].rolling(window, min_periods=1).quantile(0.95).bfill()
    df["R_ref"] = (df["retransmits"].rolling(window, min_periods=1).quantile(0.95).bfill() + 1.0)

    T = np.minimum(df["bps"] / df["B_ref"].replace(0, 1.0), 1.0)
    C = np.tanh(df["retransmits"] / df["R_ref"].replace(0, 1.0))

    # rtt can be NaN in some logs
    rtt_series = df["rtt"].fillna(method="ffill").fillna(method="bfill")
    RTT_min = np.nanmin(rtt_series.to_numpy()) if len(rtt_series) else np.nan
    if not np.isfinite(RTT_min) or RTT_min <= 0:
        RTT_min = 1.0

    queue_delay = np.maximum(rtt_series - RTT_min, 0)

    q_ref_raw = queue_delay.rolling(window, min_periods=1).quantile(0.95).bfill()
    q_bound = 0.15 * RTT_min
    q_ref = np.minimum(q_ref_raw, q_bound).replace(0, 1.0)

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
# Helpers: parse location name from filename per dataset
# ---------------------------------------------------------
def parse_location_from_filename(file: Path, dataset_flag: int):
    stem = file.stem

    if dataset_flag == 1:
        # downlink: bbr_London__REV.json OR bbr_London_REV.json (handle both)
        s = stem.replace("bbr_", "").replace("__REV", "").replace("_REV", "")
        return s

    if dataset_flag == 2:
        # uplink: bbr_London_FWD.json
        s = stem.replace("bbr_", "").replace("_FWD", "")
        return s

    if dataset_flag == 3:
        # downlink competitive: bbr_London__REV.json (double underscore)
        s = stem.replace("bbr_", "").replace("__REV", "").replace("_REV", "")
        return s

    if dataset_flag == 4:
        # uplink competitive: iperf_London_bbr.json
        # format: iperf_<CITY>_bbr
        parts = stem.split("_")
        if len(parts) >= 3 and parts[0].lower() == "iperf":
            return parts[1]
        # fallback
        return stem

    return stem

# ---------------------------------------------------------
# MAIN: Process all datasets into one CSV
# ---------------------------------------------------------
if __name__ == "__main__":

    DATASETS = [
        {
            "flag": 1,
            "folder": Path("D:/Rakshitha De Silva/starlink-iperf3-data/downlink"),
            "pattern": "bbr_*_REV.json",
            "label": "downlink",
        },
        {
            "flag": 2,
            "folder": Path("D:/Rakshitha De Silva/starlink-iperf3-data/uplink"),
            "pattern": "bbr_*_FWD.json",
            "label": "uplink",
        },
        {
            "flag": 3,
            "folder": Path("D:/Rakshitha De Silva/starlink-iperf3-data/download-compitive"),
            "pattern": "bbr_*__REV.json",
            "label": "downlink_competitive",
        },
        {
            "flag": 4,
            "folder": Path("D:/Rakshitha De Silva/starlink-iperf3-data/uplink-compitive"),
            "pattern": "iperf_*_bbr.json",
            "label": "uplink_competitive",
        },
    ]

    all_rows = []

    for ds in DATASETS:
        flag = ds["flag"]
        folder = ds["folder"]
        pattern = ds["pattern"]
        label = ds["label"]

        if not folder.exists():
            print(f"[WARN] Folder not found, skipping: {folder}")
            continue

        files = sorted(folder.glob(pattern))
        if not files:
            print(f"[WARN] No files matched {pattern} in {folder}")
            continue

        print(f"\n=== Processing dataset {label} (flag={flag}) ===")
        print(f"Folder: {folder}")
        print(f"Files : {len(files)}")

        for file in files:
            loc_name = parse_location_from_filename(file, flag)
            location_id = LOCATION_MAP.get(loc_name, -1)

            print(f"  - {file.name}  -> {loc_name} (loc_id={location_id})")

            try:
                js = load_iperf_json(file)
            except Exception as e:
                print(f"    [ERROR] JSON load failed: {e}")
                continue

            df = iperf_to_dataframe(js)
            if df.empty:
                print("    [WARN] Empty intervals, skipping.")
                continue

            df = detect_macro_phases(df)

            link_capacity = df["bps"].max()
            if not np.isfinite(link_capacity) or link_capacity <= 0:
                link_capacity = 1.0

            df = add_pacing_gains(df, link_capacity)

            random_actions = generate_random_actions(df["best_action"], seed=12345)
            df = compute_rewards(df, random_actions)
            df = find_best_actions_reward_based(df)

            # Insert identifying columns (dataset_flag MUST be first column)
            df.insert(0, "dataset_flag", flag)
            df.insert(1, "location", location_id)

            all_rows.append(df)

    if not all_rows:
        raise SystemExit("No data processed. Check folder paths and filename patterns.")

    df_all = pd.concat(all_rows, ignore_index=True)

    cols = [
        "dataset_flag", "location",
        "t", "bps", "retransmits", "snd_cwnd", "snd_wnd",
        "rtt", "rttvar", "phase",
        "S_selected", "best_action",
        "random_action", "reward_random",
        "best_action_reward_based", "best_gain_reward_based",
        "reward_best"
    ]

    output_csv = "processed_bbr_all_datasets.csv"
    df_all[cols].to_csv(output_csv, index=False)

    print(f"\nSaved CSV: {output_csv}")
    print(df_all[cols].head(10))
