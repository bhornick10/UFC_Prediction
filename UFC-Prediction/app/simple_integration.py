print("Starting integration...")

import pandas as pd
import numpy as np

print("Loading prediction data...")
prediction_df = pd.read_csv(r"c:\Users\18438\UFC all code\UFC-Prediction\app\FIGHTER_STAT.csv")
print(f"Prediction data shape: {prediction_df.shape}")

print("Adding enhanced columns...")
enhanced_df = prediction_df.copy()
enhanced_df['sig_str_land_pM'] = np.nan
enhanced_df['sig_str_def_pct'] = np.nan
enhanced_df['td_avg'] = np.nan

print("Saving enhanced data...")
output_path = r"c:\Users\18438\UFC all code\UFC-Prediction\app\FIGHTER_STAT_ENHANCED.csv"
enhanced_df.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")
print("Integration complete!")
