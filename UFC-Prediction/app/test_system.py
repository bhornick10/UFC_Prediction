import pandas as pd
import pickle

print("Testing UFC Prediction System...")

try:
    # Test data loading
    df = pd.read_csv(r"c:\Users\18438\UFC all code\UFC-Prediction\app\FIGHTER_STAT_ENHANCED.csv")
    print(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")

    # Test model loading
    model = pickle.load(open(r"c:\Users\18438\UFC all code\UFC-Prediction\app\ens_method.sav", 'rb'))
    print("✅ Model loaded successfully")

    # Test fighter lookup
    test_fighter = "Magomed Ankalaev"
    fighter_data = df[df['fighter'] == test_fighter]
    if not fighter_data.empty:
        print(f"✅ Test fighter '{test_fighter}' found in database")
    else:
        print(f"❌ Test fighter '{test_fighter}' not found")

    print("🎉 All systems ready for UFC 320 predictions!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
