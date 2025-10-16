import pandas as pd
import pickle

def check_ufc320_fighters():
    """Check if UFC 320 fighters are in the database"""
    df = pd.read_csv(r"c:\Users\18438\UFC all code\UFC-Prediction\app\FIGHTER_STAT_ENHANCED.csv")
    
    ufc320_fighters = [
        'Magomed Ankalaev', 'Alex Pereira', 
        'Merab Dvalishvili', 'Cory Sandhagen',
        'Jiří Procházka', 'Khalil Rountree',
        'Josh Emmett', 'Youssef Zalal'
    ]
    
    print("🔍 Checking UFC 320 fighters in database:")
    print("=" * 50)
    
    found_fighters = []
    missing_fighters = []
    
    for fighter in ufc320_fighters:
        if fighter in df['fighter'].values:
            print(f"✅ {fighter} - Found")
            found_fighters.append(fighter)
        else:
            print(f"❌ {fighter} - Not found")
            missing_fighters.append(fighter)
    
    print(f"\n📊 Summary: {len(found_fighters)}/{len(ufc320_fighters)} fighters found")
    
    if missing_fighters:
        print(f"\n🔍 Searching for similar names...")
        for missing in missing_fighters:
            # Search for partial matches
            similar = df[df['fighter'].str.contains(missing.split()[0], case=False, na=False)]['fighter'].tolist()
            if similar:
                print(f"   Possible matches for '{missing}': {similar[:3]}")
    
    return found_fighters, missing_fighters

if __name__ == "__main__":
    check_ufc320_fighters()
