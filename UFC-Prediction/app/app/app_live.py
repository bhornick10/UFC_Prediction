import streamlit as st
import pandas as pd
import pickle
import requests
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional
import os

# Configure page
st.set_page_config(
    page_title="UFC Live Predictions",
    page_icon="🥊",
    layout="wide"
)

# Style CSS
st.markdown('''
        <style>
            @import url('https://fonts.googleapis.com/css?family=Open+Sans+Condensed:300&display=swap');

            * {
                font-size: 1.2rem;
                margin: 0;
                padding: 0;
                -webkit-box-sizing: border-box;
                box-sizing: border-box;
                font-family: 'Open Sans Condensed', sans-serif !important;
                text-align:center;
            }

            #general{
                font-size: 19.2px;
            }
            
            #rsubheader{
                text-align: center;
                margin:0px 0px 10px 0px;
                padding: 12px 0px 30px 0px;
                border-bottom: 1px solid #909090;
            }
            
            .reportview-container .main .block-container{
                padding: 2rem 1rem !important;
            }
            
            .reportview-container h1{
                padding:0;
                margin:0;
            }

            .reportview-container .main footer{
                display: none !important;
            }

        </style>
        ''',unsafe_allow_html=True)

class UFC_Live_API:
    """Interface to UFC Stats Crawler API for live data"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.crawler_data_path = r"c:\Users\18438\UFC all code\ufc-stats-crawler\data\fighter_stats\latest.csv"
        
    def get_fighter_data(self, fighter_name: str) -> Optional[Dict[str, Any]]:
        """Get fighter data from API or fallback to local file"""
        try:
            # Try API first
            response = requests.get(f"{self.api_base_url}/fighter", 
                                  params={"name": fighter_name, "limit": 1},
                                  timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    return data["results"][0]
        except:
            pass
            
        # Fallback to local crawler data
        try:
            if os.path.exists(self.crawler_data_path):
                df = pd.read_csv(self.crawler_data_path)
                matches = df[df['name'].str.contains(fighter_name, case=False, na=False)]
                if not matches.empty:
                    return matches.iloc[0].to_dict()
        except:
            pass
            
        return None
    
    def get_all_fighters(self) -> list:
        """Get list of all available fighters"""
        try:
            if os.path.exists(self.crawler_data_path):
                df = pd.read_csv(self.crawler_data_path)
                return sorted(df['name'].dropna().tolist())
        except:
            pass
        return []

    def convert_to_prediction_format(self, fighter_data: Dict[str, Any]) -> Dict[str, float]:
        """Convert API/crawler data to prediction model format"""
        converted = {}
        
        # Handle height conversion
        height_str = fighter_data.get('height', '')
        if isinstance(height_str, str) and "'" in height_str:
            try:
                feet, inches = height_str.replace('"', '').split("'")
                converted['Height_cms'] = float(feet) * 30.48 + float(inches.strip()) * 2.54
            except:
                converted['Height_cms'] = 180.0  # Default
        else:
            converted['Height_cms'] = 180.0
            
        # Handle reach conversion
        reach_str = fighter_data.get('reach', '')
        if isinstance(reach_str, str) and reach_str.replace('"', '').replace('-', '').strip().isdigit():
            converted['Reach_cms'] = float(reach_str.replace('"', '')) * 2.54
        else:
            converted['Reach_cms'] = converted['Height_cms'] * 1.1  # Estimate
            
        # Handle weight conversion
        weight_str = fighter_data.get('weight', '')
        if isinstance(weight_str, str):
            try:
                converted['Weight_lbs'] = float(''.join(filter(str.isdigit, weight_str)))
            except:
                converted['Weight_lbs'] = 170.0  # Default
        else:
            converted['Weight_lbs'] = 170.0

        # Fight record
        converted['wins'] = float(fighter_data.get('n_win', 0))
        converted['losses'] = float(fighter_data.get('n_loss', 0))
        
        # Performance stats
        converted['sig_str_land_pM'] = float(fighter_data.get('sig_str_land_pM', 0))
        converted['sig_str_abs_pM'] = float(fighter_data.get('sig_str_abs_pM', 0))
        converted['sig_str_def_pct'] = float(fighter_data.get('sig_str_def_pct', 0))
        converted['sig_str_land_pct'] = float(fighter_data.get('sig_str_land_pct', 0))
        converted['td_avg'] = float(fighter_data.get('td_avg', 0))
        converted['td_def_pct'] = float(fighter_data.get('td_def_pct', 0))
        converted['td_land_pct'] = float(fighter_data.get('td_land_pct', 0))
        converted['sub_avg'] = float(fighter_data.get('sub_avg', 0))

        # Stance encoding
        stance = fighter_data.get('stance', 'Orthodox')
        converted['Stance_Orthodox'] = 1.0 if stance == 'Orthodox' else 0.0
        converted['Stance_Southpaw'] = 1.0 if stance == 'Southpaw' else 0.0
        converted['Stance_Switch'] = 1.0 if stance == 'Switch' else 0.0
        converted['Stance_Open_Stance'] = 1.0 if stance not in ['Orthodox', 'Southpaw', 'Switch'] else 0.0

        # Default values for missing fields
        defaults = {
            'current_lose_streak': 0, 'current_win_streak': 0, 'longest_win_streak': 0,
            'total_rounds_fought': 0, 'total_title_bouts': 0, 'win_by_Decision_Majority': 0,
            'win_by_Decision_Split': 0, 'win_by_Decision_Unanimous': 0, 'win_by_KO_TKO': 0,
            'win_by_Submission': 0, 'win_by_TKO_Doctor_Stoppage': 0, 'age': 30
        }
        
        for key, default_val in defaults.items():
            converted[key] = float(fighter_data.get(key, default_val))

        return converted

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_model():
    """Load the prediction model"""
    try:
        with open("ens_method.sav", 'rb') as f:
            return pickle.load(f)
    except:
        st.error("❌ Could not load prediction model")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes  
def get_fighter_list():
    """Get list of available fighters"""
    api = UFC_Live_API()
    fighters = api.get_all_fighters()
    if not fighters:
        st.error("❌ Could not load fighter data")
        return []
    return fighters

def predict_fight(fighter1_name: str, fighter2_name: str, model) -> Dict[str, Any]:
    """Predict fight outcome using live data"""
    api = UFC_Live_API()
    
    # Get fighter data from live API/crawler
    f1_data = api.get_fighter_data(fighter1_name)
    f2_data = api.get_fighter_data(fighter2_name)
    
    if not f1_data or not f2_data:
        return {"error": "Fighter data not found"}
    
    try:
        # Convert to prediction format
        f1_features = api.convert_to_prediction_format(f1_data)
        f2_features = api.convert_to_prediction_format(f2_data)
        
        # Create feature vectors (Blue vs Red format)
        blue_features = {f'B_{k}': v for k, v in f1_features.items()}
        red_features = {f'R_{k}': v for k, v in f2_features.items()}
        
        # Combine features
        all_features = {**blue_features, **red_features}
        feature_vector = np.array([list(all_features.values())])
        
        # Make prediction
        prediction = model.predict(feature_vector)[0]
        winner = fighter1_name if prediction == 1 else fighter2_name
        loser = fighter2_name if prediction == 1 else fighter1_name
        
        # Get confidence
        try:
            probabilities = model.predict_proba(feature_vector)[0]
            confidence = max(probabilities) * 100
        except:
            confidence = None
            
        return {
            "winner": winner,
            "loser": loser, 
            "confidence": confidence,
            "fighter1_data": f1_data,
            "fighter2_data": f2_data
        }
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

def main():
    """Main Streamlit application"""
    
    st.markdown('''
    <h1 id="rtitle" style="margin-bottom: 20px; margin-top:0px !important; padding-top:0px !important;">
    🥊 UFC Live Predictions
    </h1>
    ''', unsafe_allow_html=True)

    # Try to load UFC image
    try:
        img = Image.open('ufc.jpg')
        st.image(img, width=500)
    except:
        st.write("🥊 UFC Prediction System")

    st.markdown('''
    <h3 id="rsubheader">🔥 Live UFC Fight Predictions powered by Real-time Data & AI!</h3>
    ''', unsafe_allow_html=True)

    st.markdown('''
        <div class="my-top-text">
            <p><b id="general">Live Data:</b> This app uses REAL-TIME fighter statistics from the UFC Stats Crawler API, 
            ensuring predictions are based on the most current fighter performance data available. 
            The AI model combines multiple machine learning algorithms trained on comprehensive UFC fight history.</p>
        </div>
    ''', unsafe_allow_html=True)

    # Load model and fighters
    model = load_model()
    if not model:
        st.stop()
        
    fighters = get_fighter_list()
    if not fighters:
        st.stop()

    st.success(f"✅ Loaded {len(fighters)} fighters with live data")

    # Fighter selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔵 Blue Corner")
        blue_fighter = st.selectbox("Select Blue Fighter", fighters, key="blue")
        
    with col2:
        st.markdown("### 🔴 Red Corner") 
        red_fighter = st.selectbox("Select Red Fighter", fighters, key="red")

    # Prediction button
    if st.button("🎯 Predict Fight Outcome", type="primary"):
        if blue_fighter == red_fighter:
            st.error("❌ Please choose 2 different fighters")
        else:
            with st.spinner("🔄 Analyzing live fighter data and making prediction..."):
                result = predict_fight(blue_fighter, red_fighter, model)
                
                if "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    winner = result["winner"]
                    confidence = result.get("confidence")
                    
                    # Display prediction
                    if winner == blue_fighter:
                        st.success(f"🏆 **{winner}** (Blue Corner) WINS!")
                    else:
                        st.success(f"🏆 **{winner}** (Red Corner) WINS!")
                    
                    if confidence:
                        st.info(f"🎯 Confidence: {confidence:.1f}%")
                    
                    # Display fighter stats
                    st.markdown("### 📊 Fighter Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        f1_data = result["fighter1_data"]
                        st.markdown(f"**{blue_fighter}** 🔵")
                        st.write(f"Record: {f1_data.get('n_win', 0)}-{f1_data.get('n_loss', 0)}")
                        st.write(f"Sig. Strikes/Min: {f1_data.get('sig_str_land_pM', 0):.2f}")
                        st.write(f"Takedowns/Fight: {f1_data.get('td_avg', 0):.2f}")
                        st.write(f"Stance: {f1_data.get('stance', 'Unknown')}")
                        
                    with col2:
                        f2_data = result["fighter2_data"]
                        st.markdown(f"**{red_fighter}** 🔴")
                        st.write(f"Record: {f2_data.get('n_win', 0)}-{f2_data.get('n_loss', 0)}")
                        st.write(f"Sig. Strikes/Min: {f2_data.get('sig_str_land_pM', 0):.2f}")
                        st.write(f"Takedowns/Fight: {f2_data.get('td_avg', 0):.2f}")
                        st.write(f"Stance: {f2_data.get('stance', 'Unknown')}")

    # UFC 320 Quick Predictions
    st.markdown("---")
    st.markdown("### 🔥 UFC 320 Main Card Predictions")
    
    if st.button("🚀 Predict UFC 320 Main Card"):
        ufc320_fights = [
            ("Magomed Ankalaev", "Alex Pereira", "Light Heavyweight Championship"),
            ("Merab Dvalishvili", "Cory Sandhagen", "Bantamweight Championship"), 
            ("Jiří Procházka", "Khalil Rountree", "Light Heavyweight"),
            ("Josh Emmett", "Youssef Zalal", "Featherweight")
        ]
        
        st.markdown("#### 🏆 UFC 320 - October 4, 2025")
        
        for i, (f1, f2, desc) in enumerate(ufc320_fights):
            with st.spinner(f"Predicting {f1} vs {f2}..."):
                result = predict_fight(f1, f2, model)
                
                if "error" not in result:
                    winner = result["winner"]
                    confidence = result.get("confidence", "N/A")
                    
                    emoji = "🏆" if i == 0 else "🥊"
                    conf_str = f" ({confidence:.1f}%)" if confidence != "N/A" else ""
                    
                    st.write(f"{emoji} **{desc}**: {f1} vs {f2} → **{winner} wins**{conf_str}")
                else:
                    st.write(f"❌ {f1} vs {f2}: {result['error']}")

    # Disclaimer
    st.markdown('''
        <div style="padding-top:50px;"> 
        <p><b id="general">Disclaimer:</b> Predictions are for entertainment purposes only. 
        Past performance does not guarantee future results. This app should not be used for betting. 
        The creators are not responsible for any losses incurred.</p>
        </div>
        ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
