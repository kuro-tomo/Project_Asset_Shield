import sqlite3
import pandas as pd
import os

def show_waitlist():
    db_path = 'logs/audit_global.db'
    if not os.path.exists(db_path):
        print("❌ Database not found. Run Singapore Core first.")
        return

    conn = sqlite3.connect(db_path)
    try:
        query = "SELECT timestamp_utc, email FROM waitlist ORDER BY id DESC"
        df = pd.read_sql_query(query, conn)
        
        print("\n🐍 [JÖRMUNGANDR] ACQUIRED IDENTITIES (SINGAPORE AUDIT DB)")
        print("="*70)
        if df.empty:
            print("No identities locked yet.")
        else:
            # Format display
            print(df.to_string(index=False, justify='left'))
        print("="*70)
        print(f"Total: {len(df)} records found.\n")
    except Exception as e:
        print(f"❌ Error reading DB: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    show_waitlist()
