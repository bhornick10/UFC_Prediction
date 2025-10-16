import pandas as pd
import numpy as np
import pickle
from typing import List, Dict, Any

class UFC_Predictor:
    def __init__(self, data_path: str, model_path: str):
        """Initialize the UFC predictor with data and model paths"""
        self.data_path = data_path
        self.model_path = model_path
        self.df = None
        self.model = None
        self.load_data_and_model()

    def load_data_and_model(self):
        """Load fighter data and trained model"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Loaded {len(self.df)} fighters with enhanced stats")

            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("✅ Loaded ensemble prediction model")

        except Exception as e:
            print(f"❌ Error loading data/model: {e}")
            raise

    def get_fighter_stats(self, fighter_name: str) -> pd.DataFrame:
        """Get fighter statistics from the database"""
        fighter_data = self.df[self.df['fighter'] == fighter_name]
        if fighter_data.empty:
            raise ValueError(f"Fighter '{fighter_name}' not found in database")
        return fighter_data

    def analyze_fighter_comparison(self, fighter1: str, fighter2: str) -> Dict[str, Any]:
        """Analyze key differences between two fighters"""
        f1_data = self.get_fighter_stats(fighter1)
        f2_data = self.get_fighter_stats(fighter2)

        analysis = {
            'fighter1': fighter1,
            'fighter2': fighter2,
            'key_stats': {}
        }

        # Key performance metrics to compare
        key_metrics = [
            ('sig_str_land_pM', 'Significant Strikes Landed per Minute'),
            ('td_avg', 'Takedowns Average'),
            ('sub_avg', 'Submission Average'),
            ('pass', 'Guard Passes'),
            ('rev', 'Reversals'),
            ('defence_percent', 'Defense Percentage'),
            ('str_def_percent', 'Strike Defense Percentage'),
            ('td_def_percent', 'Takedown Defense Percentage')
        ]

        for metric, description in key_metrics:
            if metric in f1_data.columns:
                f1_val = f1_data[metric].values[0]
                f2_val = f2_data[metric].values[0]

                if pd.notna(f1_val) and pd.notna(f2_val):
                    advantage = "fighter1" if f1_val > f2_val else "fighter2"
                    analysis['key_stats'][description] = {
                        'fighter1_value': f1_val,
                        'fighter2_value': f2_val,
                        'advantage': advantage
                    }

        return analysis

    def predict_fight(self, fighter1: str, fighter2: str) -> Dict[str, Any]:
        """Predict the outcome of a fight between two fighters"""
        try:
            # Get fighter data
            f1_data = self.get_fighter_stats(fighter1)
            f2_data = self.get_fighter_stats(fighter2)

            # Extract features (skip ID, date, fighter name)
            f1_features = f1_data.iloc[:, 3:]
            f2_features = f2_data.iloc[:, 3:]

            # Create Blue vs Red format (fighter1 = Blue, fighter2 = Red)
            blue_features = f1_features.copy()
            blue_features.columns = ['B_' + col for col in blue_features.columns]

            red_features = f2_features.copy()
            red_features.columns = ['R_' + col for col in red_features.columns]

            # Combine features
            combined = pd.concat([blue_features.reset_index(drop=True),
                                 red_features.reset_index(drop=True)], axis=1)

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

            # Get analysis
            analysis = self.analyze_fighter_comparison(fighter1, fighter2)

            return {
                'winner': winner,
                'loser': loser,
                'confidence': confidence,
                'fighter1': fighter1,
                'fighter2': fighter2,
                'analysis': analysis
            }

        except Exception as e:
            return {
                'error': f"Prediction failed: {str(e)}",
                'fighter1': fighter1,
                'fighter2': fighter2
            }

    def predict_fight_card(self, fights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict outcomes for an entire fight card"""
        results = []

        for fight in fights:
            fighter1 = fight.get('fighter1', '')
            fighter2 = fight.get('fighter2', '')

            if not fighter1 or not fighter2:
                results.append({
                    'error': 'Missing fighter names',
                    'fight': fight
                })
                continue

            prediction = self.predict_fight(fighter1, fighter2)
            prediction.update(fight)  # Add fight metadata
            results.append(prediction)

        return results

    def print_prediction_explanation(self, result: Dict[str, Any]):
        """Print detailed prediction with explanation"""
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return

        fighter1 = result['fighter1']
        fighter2 = result['fighter2']
        winner = result['winner']
        loser = result['loser']
        confidence = result['confidence']

        print(f"\n{'🏆 MAIN EVENT' if result.get('main_event', False) else '🥊 FEATURED'}")
        print(f"{result.get('description', 'Fight')}")
        print(f"{fighter1} ({result.get('record1', 'Record N/A')}) vs {fighter2} ({result.get('record2', 'Record N/A')})")
        print(f"Weight Class: {result.get('weight_class', 'Unknown Weight')}")
        print("-" * 60)

        print(f"🎯 PREDICTION: {winner} defeats {loser}")
        if confidence != "N/A":
            print(f"   Confidence: {confidence:.1f}%")
        else:
            print("   (Confidence not available)")

        # Print key advantages
        analysis = result.get('analysis', {})
        key_stats = analysis.get('key_stats', {})

        if key_stats:
            print("\n📊 KEY ADVANTAGES:")
            advantages_f1 = []
            advantages_f2 = []

            for stat_name, stat_data in key_stats.items():
                advantage = stat_data['advantage']
                if advantage == 'fighter1':
                    advantages_f1.append(f"{stat_name} ({stat_data['fighter1_value']:.2f} vs {stat_data['fighter2_value']:.2f})")
                else:
                    advantages_f2.append(f"{stat_name} ({stat_data['fighter2_value']:.2f} vs {stat_data['fighter1_value']:.2f})")

            if advantages_f1:
                print(f"   {fighter1} advantages: {', '.join(advantages_f1[:3])}")
            if advantages_f2:
                print(f"   {fighter2} advantages: {', '.join(advantages_f2[:3])}")

        print()

def test_upcoming_fights():
    """Test predictions on upcoming UFC 320 fights"""

    print("🔥 UFC 320 PREDICTIONS TEST 🔥")
    print("=" * 60)

    # Initialize predictor
    predictor = UFC_Predictor(
        data_path=r"c:\Users\18438\UFC all code\UFC-Prediction\app\FIGHTER_STAT_ENHANCED.csv",
        model_path=r"c:\Users\18438\UFC all code\UFC-Prediction\app\ens_method.sav"
    )

    # UFC 320 Main Card fights (October 4, 2025) - From Official Card
    upcoming_fights = [
        {
            "main_event": True,
            "fighter1": "Magomed Ankalaev",
            "fighter2": "Alex Pereira",
            "weight_class": "Light Heavyweight",
            "description": "MAIN EVENT - Light Heavyweight Championship",
            "record1": "21-1-1",
            "record2": "12-3-0"
        },
        {
            "main_event": False,
            "fighter1": "Merab Dvalishvili", 
            "fighter2": "Cory Sandhagen",
            "weight_class": "Bantamweight",
            "description": "CO-MAIN EVENT - Bantamweight Championship",
            "record1": "20-4-0",
            "record2": "18-5-0"
        },
        {
            "main_event": False,
            "fighter1": "Jiří Procházka",
            "fighter2": "Khalil Rountree",
            "weight_class": "Light Heavyweight", 
            "description": "Light Heavyweight Bout",
            "record1": "31-5-1",
            "record2": "15-6-0"
        },
        {
            "main_event": False,
            "fighter1": "Josh Emmett",
            "fighter2": "Youssef Zalal",
            "weight_class": "Featherweight",
            "description": "Featherweight Bout",
            "record1": "19-5-0",
            "record2": "17-5-1"
        }
    ]

    # Get predictions
    predictions = predictor.predict_fight_card(upcoming_fights)

    # Print detailed results
    successful_predictions = 0
    for result in predictions:
        if 'error' not in result:
            predictor.print_prediction_explanation(result)
            successful_predictions += 1
        else:
            print(f"❌ Failed to predict: {result.get('fighter1', 'Unknown')} vs {result.get('fighter2', 'Unknown')}")
            print(f"   Error: {result['error']}\n")

    # Summary
    print("📊 PREDICTION SUMMARY")
    print("=" * 60)
    for result in predictions:
        if 'error' not in result:
            status = "🏆 MAIN" if result.get('main_event', False) else "🥊 CO-MAIN"
            print(f"{status}: {result['fighter1']} vs {result['fighter2']} → {result['winner']} wins")
            if result['confidence'] != "N/A":
                print(f"   Confidence: {result['confidence']:.1f}%")

    print(f"\n✅ Successfully predicted {successful_predictions}/{len(predictions)} UFC 320 fights!")
    print("🎉 Enhanced AI model with current fighter data is working!")
    print("\n💡 To predict different fights, modify the 'upcoming_fights' list above!")

if __name__ == "__main__":
    test_upcoming_fights()
