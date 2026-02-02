import sqlite3
import os

db_path = "data/jquants_cache.db"

if not os.path.exists(db_path):
    print("Database not found.")
    exit()

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    # print(f"Tables: {tables}")
    
    # Assuming table name is 'daily_quotes' or similar based on provider code (not read yet but guessing)
    # Let's check table schema if table exists
    table_name = 'daily_quotes' # Hypothesis
    
    # Verify table name from list
    found_table = None
    for t in tables:
        if 'quote' in t[0]:
            found_table = t[0]
            break
            
    if found_table:
        # Get latest prices for 9984, 6758, 7203
        target_codes = ['99840', '67580', '72030'] # J-Quants uses 5 digit codes usually
        
        print(f"Checking prices in table: {found_table}")
        
        for code in target_codes:
            # Try 4 digit too if 5 digit fails
            query = f"SELECT Date, Close FROM {found_table} WHERE Code LIKE '{code}%' ORDER BY Date DESC LIMIT 1"
            cursor.execute(query)
            row = cursor.fetchone()
            if row:
                print(f"Code {code}: Date={row[0]}, Price={row[1]}")
            else:
                # Try 4 digit
                short_code = code[:4]
                query = f"SELECT Date, Close FROM {found_table} WHERE Code = '{short_code}' ORDER BY Date DESC LIMIT 1"
                cursor.execute(query)
                row = cursor.fetchone()
                if row:
                    print(f"Code {short_code}: Date={row[0]}, Price={row[1]}")
                else:
                    print(f"Code {code}: No data found")

    else:
        print("No quotes table found.")
        
    conn.close()

except Exception as e:
    print(f"Error: {e}")
