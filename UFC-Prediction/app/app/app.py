import streamlit as st
import altair as alt
import pandas as pd
import pickle
from PIL import Image
# encode blue=1 & red=0

# style css
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
            #rtitle{

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


st.markdown('''
    <h1 id="rtitle" style="margin-bottom: 20px; margin-top:0px !important; padding-top:0px !important;">
    🥊 UFC Live Predictions (Real-time Data)
    </h1>
''', unsafe_allow_html=True)

img = Image.open('ufc.jpg')
st.image(img, width=500)

st.markdown('''
    <h3 id="rsubheader">🔥 Live UFC Fight Predictions powered by Real-time Data & AI!</h3>
 ''', unsafe_allow_html=True)

st.markdown('''
    <div class="my-top-text">
        <p><b id="general">Live Data & AI:</b> This app now uses REAL-TIME fighter statistics from the UFC Stats Crawler, 
        ensuring predictions are based on the most current fighter data available. The AI model combines multiple machine learning 
        algorithms (Random Forest, SVM, Logistic Regression, XGBoost) trained on comprehensive UFC fight history from 1993-2019. 
        <br> For technical details, please <a id="general" href="https://github.com/rezan21/UFC-Prediction">visit github repository</a>.</p>
    </div>
''', unsafe_allow_html=True)



# Helper functions for data conversion
def convert_height(height_str):
    """Convert height string to cm"""
    if isinstance(height_str, str) and "'" in height_str:
        try:
            feet, inches = height_str.replace('"', '').split("'")
            return float(feet) * 30.48 + float(inches.strip()) * 2.54
        except:
            return 180.0
    return 180.0

def convert_reach(reach_str):
    """Convert reach string to cm"""
    if isinstance(reach_str, str) and reach_str.replace('"', '').replace('-', '').strip().isdigit():
        return float(reach_str.replace('"', '')) * 2.54
    return 180.0

def convert_weight(weight_str):
    """Convert weight string to lbs"""
    if isinstance(weight_str, str):
        try:
            return float(''.join(filter(str.isdigit, weight_str)))
        except:
            return 170.0
    return 170.0

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_live_data():
    """Load fighter data from UFC Stats Crawler"""
    import os
    
    crawler_data_path = r"c:\Users\18438\UFC all code\ufc-stats-crawler\data\fighter_stats\latest.csv"
    
    try:
        if os.path.exists(crawler_data_path):
            df = pd.read_csv(crawler_data_path)
            
            # Create legacy-compatible dataframe
            legacy_df = pd.DataFrame()
            legacy_df['fighter'] = df['name']  # Map to 'fighter' column name
            
            # Map stats
            legacy_df['wins'] = df.get('n_win', 0)
            legacy_df['losses'] = df.get('n_loss', 0)
            legacy_df['Height_cms'] = df.apply(lambda row: convert_height(row.get('height', '')), axis=1)
            legacy_df['Reach_cms'] = df.apply(lambda row: convert_reach(row.get('reach', '')), axis=1)
            legacy_df['Weight_lbs'] = df.apply(lambda row: convert_weight(row.get('weight', '')), axis=1)
            
            # Performance stats
            legacy_df['sig_str_land_pM'] = df.get('sig_str_land_pM', 0).fillna(0)
            legacy_df['sig_str_abs_pM'] = df.get('sig_str_abs_pM', 0).fillna(0)
            legacy_df['sig_str_def_pct'] = df.get('sig_str_def_pct', 0).fillna(0)
            legacy_df['sig_str_land_pct'] = df.get('sig_str_land_pct', 0).fillna(0)
            legacy_df['td_avg'] = df.get('td_avg', 0).fillna(0)
            legacy_df['td_def_pct'] = df.get('td_def_pct', 0).fillna(0)
            legacy_df['td_land_pct'] = df.get('td_land_pct', 0).fillna(0)
            legacy_df['sub_avg'] = df.get('sub_avg', 0).fillna(0)
            
            # Stance encoding
            stance_col = df.get('stance', pd.Series(['Orthodox'] * len(df)))
            legacy_df['Stance_Orthodox'] = (stance_col == 'Orthodox').astype(int)
            legacy_df['Stance_Southpaw'] = (stance_col == 'Southpaw').astype(int)
            legacy_df['Stance_Switch'] = (stance_col == 'Switch').astype(int)
            legacy_df['Stance_Open_Stance'] = (~stance_col.isin(['Orthodox', 'Southpaw', 'Switch'])).astype(int)
            
            # Default missing columns
            default_cols = {
                'current_lose_streak': 0, 'current_win_streak': 0, 'longest_win_streak': 0,
                'total_rounds_fought': 0, 'total_title_bouts': 0, 'win_by_Decision_Majority': 0,
                'win_by_Decision_Split': 0, 'win_by_Decision_Unanimous': 0, 'win_by_KO_TKO': 0,
                'win_by_Submission': 0, 'win_by_TKO_Doctor_Stoppage': 0, 'age': 30
            }
            
            for col, default_val in default_cols.items():
                legacy_df[col] = default_val
            
            # Fill any remaining NaN values
            legacy_df = legacy_df.fillna(0)
            
            st.success(f"✅ Loaded {len(legacy_df)} fighters from live UFC Stats Crawler")
            return legacy_df
            
    except Exception as e:
        st.warning(f"⚠️ Live data failed: {e}")
        
    # Fallback to static CSV
    try:
        fallback_df = pd.read_csv("FIGHTER_STAT_ENHANCED.csv")
        st.info("📁 Using static fighter data as fallback")
        return fallback_df
    except:
        st.error("❌ Could not load any fighter data")
        return pd.DataFrame()

# Load data
df = load_live_data()
fighters = df["fighter"].tolist() if not df.empty else []
ens_method = pickle.load(open("ens_method.sav", 'rb'))

def predictEnsemble(sample):
    prediction = ens_method.predict(sample)
    return (prediction)

def predictMatchByID(B, R):
    blueFighter = df[df["ID"] == B].iloc[:,3:]
    blueFighter.columns = ['B_'+col for col in blueFighter.columns] #concat prefix B_ and rename columns

    redFighter = df[df["ID"] == R].iloc[:,3:]
    redFighter.columns = ['R_'+col for col in redFighter.columns]

    blueFighter.reset_index(drop=True,inplace=True)
    redFighter.reset_index(drop=True,inplace=True)

    toPredict = pd.concat([blueFighter,redFighter],axis=1).values
    return (toPredict)

def main():
    r_fighter = st.selectbox("Red Fighter", fighters)
    b_fighter = st.selectbox("Blue Fighter", fighters)
    submitBtn = st.button("Predict Match")

    if(submitBtn):
        if b_fighter == r_fighter:
            st.error("Please choose 2 distict fighters")
        else:
            try:
                players = {
                    1:str(b_fighter),
                    0:str(r_fighter)
                }

                b_id = df["ID"][df["fighter"]==b_fighter].values[0]
                r_id = df["ID"][df["fighter"]==r_fighter].values[0]
                #print(f"blue id: {b_id}, red id: {r_id}")
                sample = predictMatchByID(b_id, r_id)
                #print(sample)

                prediction = predictEnsemble(sample).tolist()[0]
                #st.title(prediction)

                st.success(players[prediction] + ' Wins')
            except Exception as e:
                #st.write(e)
                st.error("Something went wrong! Reload the page and try again.")

    # UFC 320 Quick Predictions Section
    st.markdown("---")
    st.markdown("### 🔥 UFC 320 Main Card Predictions")
    st.markdown("**UFC 320 - October 4, 2025**")
    
    if st.button("🚀 Predict UFC 320 Main Card"):
        ufc320_fights = [
            ("Magomed Ankalaev", "Alex Pereira", "🏆 Light Heavyweight Championship"),
            ("Merab Dvalishvili", "Cory Sandhagen", "🏆 Bantamweight Championship"), 
            ("Jiří Procházka", "Khalil Rountree", "🥊 Light Heavyweight"),
            ("Josh Emmett", "Youssef Zalal", "🥊 Featherweight")
        ]
        
        st.markdown("#### 🔥 Live Predictions:")
        
        for f1, f2, desc in ufc320_fights:
            try:
                # Check if both fighters exist in dataset
                if f1 in fighters and f2 in fighters:
                    b_id = df["ID"][df["fighter"]==f1].values[0]
                    r_id = df["ID"][df["fighter"]==f2].values[0]
                    sample = predictMatchByID(b_id, r_id)
                    prediction = predictEnsemble(sample).tolist()[0]
                    
                    winner = f1 if prediction == 1 else f2
                    st.write(f"**{desc}**: {f1} vs {f2} → **{winner} wins** ✅")
                else:
                    # Handle fighters not in dataset
                    missing = []
                    if f1 not in fighters:
                        missing.append(f1)
                    if f2 not in fighters:
                        missing.append(f2)
                    st.write(f"**{desc}**: {f1} vs {f2} → ⚠️ Missing data for: {', '.join(missing)}")
                    
            except Exception as e:
                st.write(f"**{desc}**: {f1} vs {f2} → ❌ Prediction failed")

    st.markdown('''
        <div style="padding-top:100px;"> 
        <p><b id="general">Disclaimer:</b> The performance represented is historical and past performance is not a reliable indicator of future results and investors may not recover the full amount invested.  I do not condone this app's use for betting. I am not responsible for any damage done or losses incurred by way of this app.</p>
        </div>
        ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    
