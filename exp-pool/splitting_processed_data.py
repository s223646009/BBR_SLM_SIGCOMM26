import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
INPUT_CSV = Path(r"D:\Rakshitha De Silva\1LLM\exp-pool\processed_bbr_all_datasets.csv")
OUTPUT_DIR = Path(r"D:\Rakshitha De Silva\1LLM\exp-pool\processed_data_for_eval")

TOKYO_LOCATION_ID = 26
DATASET_FLAGS = [1, 2, 3, 4]

# ---------------------------------------------------------
# Load CSV
# ---------------------------------------------------------
df = pd.read_csv(INPUT_CSV)

# ---------------------------------------------------------
# Filter and save per dataset flag (Tokyo only)
# ---------------------------------------------------------
for flag in DATASET_FLAGS:
    df_tokyo_flag = df[
        (df["location"] == TOKYO_LOCATION_ID) &
        (df["dataset_flag"] == flag)
    ]

    output_path = OUTPUT_DIR / f"processed_bbr_tokyo_dataset_flag_{flag}.csv"
    df_tokyo_flag.to_csv(output_path, index=False)

    print(f"Saved: {output_path} | Rows: {len(df_tokyo_flag)}")

print("✅ Tokyo dataset splitting completed.")
