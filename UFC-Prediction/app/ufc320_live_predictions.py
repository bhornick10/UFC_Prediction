import pandas as pd
import pickle
import os
import requests
from typing import List, Dict, Any
import numpy as np

class UFC_Live_Predictor:
    def __init__(self, crawler_data_path: str, model_path: str):
        """Initialize with live crawler data and prediction model"""
        self.crawler_data_path = crawler_data_path
        self.model_path = model_path
        self.crawler_df = None
        self.model = None
        self.load_system()

    def load_system(self):
        """Load latest crawler data and prediction model"""
        try:
            # Load latest fighter data from crawler
            latest_file = os.path.join(self.crawler_data_path, "latest.csv")
            if os.path.exists(latest_file):
                self.crawler_df = pd.read_csv(latest_file)
                print(f"✅ Loaded {len(self.crawler_df)} fighters from live crawler data")
            else:
                print("❌ No latest crawler data found")
                return False

            # Load prediction model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("✅ Loaded ensemble prediction model")

            # Clean and standardize fighter names
            self.crawler_df['name'] = self.crawler_df['name'].str.strip()
            self.crawler_df['name_search'] = self.crawler_df['name'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
            
            return True

        except Exception as e:
            print(f"❌ Error loading system: {e}")
            return False

    def find_fighter(self, fighter_name: str) -> pd.DataFrame:
        """Find fighter in crawler database with fuzzy matching"""
        if self.crawler_df is None:
            raise ValueError("Crawler data not loaded")

        # Clean search name
        search_name = fighter_name.lower().strip().replace(r'[^\w\s]', '')
        
        # Exact match first
        exact_match = self.crawler_df[self.crawler_df['name_search'] == search_name]
        if not exact_match.empty:
            return exact_match.iloc[0:1]

        # Partial match
        partial_matches = self.crawler_df[
            self.crawler_df['name_search'].str.contains(search_name.split()[0], na=False) |
            self.crawler_df['name'].str.contains(fighter_name.split()[0], case=False, na=False)
        ]
        
        if not partial_matches.empty:
            print(f"🔍 Found partial matches for '{fighter_name}': {partial_matches['name'].tolist()[:3]}")
            return partial_matches.iloc[0:1]

        raise ValueError(f"Fighter '{fighter_name}' not found in database")

    def convert_crawler_to_prediction_format(self, fighter_data: pd.DataFrame) -> pd.DataFrame:
        """Convert crawler data format to prediction model format"""
        
        # Extract basic stats
        converted = pd.DataFrame()
        
        # Basic info
        converted['fighter'] = fighter_data['name']
        converted['Height_cms'] = fighter_data['height'].str.extract(r"(\d+)' (\d+)").apply(
            lambda x: float(x[0]) * 30.48 + float(x[1]) * 2.54 if pd.notna(x[0]) and pd.notna(x[1]) else np.nan, axis=1
        )
        converted['Reach_cms'] = fighter_data['reach'].str.extract(r'(\d+)').astype(float) * 2.54
        converted['Weight_lbs'] = fighter_data['weight'].str.extract(r'(\d+)').astype(float)

        # Fight record
        converted['wins'] = fighter_data['n_win'].fillna(0)
        converted['losses'] = fighter_data['n_loss'].fillna(0)
        
        # Performance stats from crawler
        converted['sig_str_land_pM'] = fighter_data['sig_str_land_pM'].fillna(0)
        converted['sig_str_abs_pM'] = fighter_data['sig_str_abs_pM'].fillna(0)
        converted['sig_str_def_pct'] = fighter_data['sig_str_def_pct'].fillna(0)
        converted['sig_str_land_pct'] = fighter_data['sig_str_land_pct'].fillna(0)
        converted['td_avg'] = fighter_data['td_avg'].fillna(0)
        converted['td_def_pct'] = fighter_data['td_def_pct'].fillna(0)
        converted['td_land_pct'] = fighter_data['td_land_pct'].fillna(0)
        converted['sub_avg'] = fighter_data['sub_avg'].fillna(0)

        # Stance encoding
        converted['Stance_Orthodox'] = (fighter_data['stance'] == 'Orthodox').astype(int)
        converted['Stance_Southpaw'] = (fighter_data['stance'] == 'Southpaw').astype(int)
        converted['Stance_Switch'] = (fighter_data['stance'] == 'Switch').astype(int)
        converted['Stance_Open_Stance'] = (~fighter_data['stance'].isin(['Orthodox', 'Southpaw', 'Switch'])).astype(int)

        # Fill missing columns with defaults
        default_columns = [
            'current_lose_streak', 'current_win_streak', 'longest_win_streak',
            'total_rounds_fought', 'total_title_bouts', 'win_by_Decision_Majority',
            'win_by_Decision_Split', 'win_by_Decision_Unanimous', 'win_by_KO_TKO',
            'win_by_Submission', 'win_by_TKO_Doctor_Stoppage', 'age'
        ]
        
        for col in default_columns:
            if col not in converted.columns:
                converted[col] = 0

        return converted

    def predict_fight(self, fighter1: str, fighter2: str) -> Dict[str, Any]:
        """Predict fight outcome using live data"""
        try:
            # Find fighters in crawler data
            f1_data = self.find_fighter(fighter1)
            f2_data = self.find_fighter(fighter2)

            print(f"✅ Found {f1_data['name'].iloc[0]} vs {f2_data['name'].iloc[0]}")

            # Convert to prediction format
            f1_converted = self.convert_crawler_to_prediction_format(f1_data)
            f2_converted = self.convert_crawler_to_prediction_format(f2_data)

            # Create features for model (Blue vs Red format)
            blue_features = f1_converted.iloc[:, 1:].copy()  # Skip fighter name
            blue_features.columns = ['B_' + col for col in blue_features.columns]

            red_features = f2_converted.iloc[:, 1:].copy()
            red_features.columns = ['R_' + col for col in red_features.columns]

            # Combine features
            combined = pd.concat([blue_features.reset_index(drop=True),
                                 red_features.reset_index(drop=True)], axis=1)

            # Fill any remaining NaN values
            combined = combined.fillna(0)

            # Make prediction
            features = combined.values
            prediction = self.model.predict(features)
            winner_idx = prediction[0]  # 1 = Blue wins, 0 = Red wins

            winner = fighter1 if winner_idx == 1 else fighter2
            loser = fighter2 if winner_idx == 1 else fighter1

            # Get confidence
            confidence = "N/A"
            try:
                probabilities = self.model.predict_proba(features)[0]
                confidence = max(probabilities) * 100
            except:
                pass

            # Get fighter stats for analysis
            f1_stats = {
                'record': f"{f1_data['n_win'].iloc[0]}-{f1_data['n_loss'].iloc[0]}-{f1_data.get('n_draw', pd.Series([0])).iloc[0]}",
                'sig_str_pM': f1_data['sig_str_land_pM'].iloc[0],
                'td_avg': f1_data['td_avg'].iloc[0],
                'stance': f1_data['stance'].iloc[0]
            }
            
            f2_stats = {
                'record': f"{f2_data['n_win'].iloc[0]}-{f2_data['n_loss'].iloc[0]}-{f2_data.get('n_draw', pd.Series([0])).iloc[0]}",
                'sig_str_pM': f2_data['sig_str_land_pM'].iloc[0],
                'td_avg': f2_data['td_avg'].iloc[0],
                'stance': f2_data['stance'].iloc[0]
            }

            return {
                'winner': winner,
                'loser': loser,
                'confidence': confidence,
                'fighter1': fighter1,
                'fighter2': fighter2,
                'fighter1_stats': f1_stats,
                'fighter2_stats': f2_stats,
                'data_source': 'live_crawler'
            }

        except Exception as e:
            return {
                'error': f"Prediction failed: {str(e)}",
                'fighter1': fighter1,
                'fighter2': fighter2
            }

    def predict_ufc320_main_card(self):
        """Predict UFC 320 main card using live data"""
        
        print("🔥 UFC 320 LIVE PREDICTIONS 🔥")
        print("Using Live UFC Stats Crawler Data")
        print("=" * 60)

        # UFC 320 Main Card from image
        fights = [
            {
                "main_event": True,
                "fighter1": "Magomed Ankalaev",
                "fighter2": "Alex Pereira",
                "weight_class": "Light Heavyweight",
                "description": "MAIN EVENT - Light Heavyweight Championship"
            },
            {
                "main_event": False,
                "fighter1": "Merab Dvalishvili",
                "fighter2": "Cory Sandhagen", 
                "weight_class": "Bantamweight",
                "description": "CO-MAIN EVENT - Bantamweight Championship"
            },
            {
                "main_event": False,
                "fighter1": "Jiří Procházka",
                "fighter2": "Khalil Rountree",
                "weight_class": "Light Heavyweight",
                "description": "Light Heavyweight Bout"
            },
            {
                "main_event": False,
                "fighter1": "Josh Emmett",
                "fighter2": "Youssef Zalal",
                "weight_class": "Featherweight", 
                "description": "Featherweight Bout"
            }
        ]

        results = []
        successful_predictions = 0

        for fight in fights:
            fighter1 = fight['fighter1']
            fighter2 = fight['fighter2']

            print(f"\n{'🏆 MAIN EVENT' if fight['main_event'] else '🥊 FEATURED'}")
            print(f"{fight['description']}")
            print(f"{fighter1} vs {fighter2} ({fight['weight_class']})")
            print("-" * 60)

            result = self.predict_fight(fighter1, fighter2)
            
            if 'error' not in result:
                winner = result['winner']
                confidence = result['confidence']
                f1_stats = result['fighter1_stats']
                f2_stats = result['fighter2_stats']

                print(f"🎯 PREDICTION: {winner} defeats {result['loser']}")
                if confidence != "N/A":
                    print(f"   Confidence: {confidence:.1f}%")
                
                print(f"\n📊 FIGHTER ANALYSIS:")
                print(f"   {fighter1}: {f1_stats['record']} | {f1_stats['sig_str_pM']:.2f} sig str/min | {f1_stats['stance']} stance")
                print(f"   {fighter2}: {f2_stats['record']} | {f2_stats['sig_str_pM']:.2f} sig str/min | {f2_stats['stance']} stance")

                results.append(result)
                successful_predictions += 1
            else:
                print(f"❌ {result['error']}")

            print()

        # Summary
        print("📊 UFC 320 PREDICTION SUMMARY")
        print("=" * 60)
        for result in results:
            if 'error' not in result:
                print(f"🥊 {result['fighter1']} vs {result['fighter2']} → {result['winner']} wins")
                if result['confidence'] != "N/A":
                    print(f"   Confidence: {result['confidence']:.1f}%")

        print(f"\n✅ Successfully predicted {successful_predictions}/4 UFC 320 main card fights!")
        print("🎉 Powered by live UFC stats crawler data!")

        return results

def main():
    """Run UFC 320 predictions with live data"""
    crawler_data_path = r"c:\Users\18438\UFC all code\ufc-stats-crawler\data\fighter_stats"
    model_path = r"c:\Users\18438\UFC all code\UFC-Prediction\app\ens_method.sav"
    
    predictor = UFC_Live_Predictor(crawler_data_path, model_path)
    results = predictor.predict_ufc320_main_card()
    
    return results

if __name__ == "__main__":
    main()
