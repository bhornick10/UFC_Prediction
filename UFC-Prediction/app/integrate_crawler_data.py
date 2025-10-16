import pandas as pd
import numpy as np
from datetime import datetime

def integrate_crawler_data():
    """Integrate ufc-stats-crawler data into UFC-Prediction format"""

    print("Loading crawler data...")
    try:
        # Load latest crawler data
        crawler_path = r"c:\Users\18438\UFC all code\ufc-stats-crawler\data\fighter_stats\latest.csv"
        crawler_df = pd.read_csv(crawler_path)
        print(f"Crawler data loaded successfully, shape: {crawler_df.shape}")
    except Exception as e:
        print(f"Error loading crawler data: {e}")
        return

    # Load existing prediction data
    prediction_path = r"c:\Users\18438\UFC all code\UFC-Prediction\app\FIGHTER_STAT.csv"
    prediction_df = pd.read_csv(prediction_path)
    print(f"Prediction data loaded, shape: {prediction_df.shape}")

    # For now, let's create a simple enhanced version by adding some crawler features
    # to existing fighters where names match

    enhanced_df = prediction_df.copy()

    # Add new columns from crawler data
    new_columns = ['sig_str_land_pM', 'sig_str_abs_pM', 'sig_str_def_pct',
                   'sig_str_land_pct', 'td_avg', 'td_def_pct', 'td_land_pct', 'sub_avg']

    for col in new_columns:
        enhanced_df[col] = np.nan  # Initialize with NaN

    # Try to match some fighters and add their enhanced stats
    matched_count = 0
    for idx, row in crawler_df.iterrows():
        fighter_name = row['name']
        # Look for matching fighter in prediction data
        matches = enhanced_df[enhanced_df['fighter'].str.lower() == fighter_name.lower()]
        if not matches.empty:
            fighter_idx = matches.index[0]
            # Add enhanced stats
            enhanced_df.loc[fighter_idx, 'sig_str_land_pM'] = row['sig_str_land_pM']
            enhanced_df.loc[fighter_idx, 'sig_str_abs_pM'] = row['sig_str_abs_pM']
            enhanced_df.loc[fighter_idx, 'sig_str_def_pct'] = row['sig_str_def_pct']
            enhanced_df.loc[fighter_idx, 'sig_str_land_pct'] = row['sig_str_land_pct']
            enhanced_df.loc[fighter_idx, 'td_avg'] = row['td_avg']
            enhanced_df.loc[fighter_idx, 'td_def_pct'] = row['td_def_pct']
            enhanced_df.loc[fighter_idx, 'td_land_pct'] = row['td_land_pct']
            enhanced_df.loc[fighter_idx, 'sub_avg'] = row['sub_avg']
            matched_count += 1

    print(f"Matched {matched_count} fighters with enhanced stats")

    # Save enhanced fighter data
    output_path = r"c:\Users\18438\UFC all code\UFC-Prediction\app\FIGHTER_STAT_ENHANCED.csv"
    enhanced_df.to_csv(output_path, index=False)

    print(f"Enhanced fighter data saved to: {output_path}")
    print(f"Shape: {enhanced_df.shape}")
    print("New columns added:", new_columns)

    return enhanced_df

if __name__ == "__main__":
    integrate_crawler_data()
