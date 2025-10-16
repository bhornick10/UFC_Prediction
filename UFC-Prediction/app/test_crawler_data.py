import pandas as pd
import os

def test_crawler_data():
    """Test if we can access and analyze the crawler data"""
    
    print("🔍 Testing UFC Stats Crawler Data")
    print("=" * 50)
    
    try:
        # Load latest fighter data
        crawler_path = r"c:\Users\18438\UFC all code\ufc-stats-crawler\data\fighter_stats\latest.csv"
        df = pd.read_csv(crawler_path)
        
        print(f"✅ Loaded {len(df)} fighters from crawler")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Test searching for UFC 320 fighters
        ufc320_fighters = ["Magomed Ankalaev", "Alex Pereira", "Merab Dvalishvili", "Cory Sandhagen"]
        
        print(f"\n🔍 Searching for UFC 320 fighters:")
        found_fighters = []
        
        for fighter in ufc320_fighters:
            # Search by name
            matches = df[df['name'].str.contains(fighter, case=False, na=False)]
            if not matches.empty:
                found = matches.iloc[0]
                print(f"✅ {fighter} → Found: {found['name']} ({found['n_win']}-{found['n_loss']})")
                found_fighters.append(found['name'])
            else:
                # Try partial search
                first_name = fighter.split()[0]
                partial_matches = df[df['name'].str.contains(first_name, case=False, na=False)]
                if not partial_matches.empty:
                    print(f"🔍 {fighter} → Partial matches: {partial_matches['name'].head(3).tolist()}")
                else:
                    print(f"❌ {fighter} → Not found")
        
        print(f"\n📈 Found {len(found_fighters)}/{len(ufc320_fighters)} UFC 320 fighters in crawler data")
        
        # Show sample data structure
        if len(df) > 0:
            print(f"\n📋 Sample fighter data:")
            sample = df.iloc[0]
            for col in ['name', 'n_win', 'n_loss', 'sig_str_land_pM', 'td_avg', 'stance']:
                if col in sample:
                    print(f"   {col}: {sample[col]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_crawler_data()
