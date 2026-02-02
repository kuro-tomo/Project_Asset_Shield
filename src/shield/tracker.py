import sqlite3
import json
from datetime import datetime
import hashlib

def log_event(ticker, step, data):
    """Logs financial events into the Audit DB for future $3B due diligence."""
    db_path = "logs/audit.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS audit_logs 
                     (ticker TEXT, step TEXT, timestamp TEXT, data_hash TEXT, data_json TEXT)''')
    
    data['recorded_at'] = datetime.now().isoformat()
    data_json = json.dumps(data, sort_keys=True)
    data_hash = hashlib.sha256(data_json.encode()).hexdigest()

    cursor.execute('INSERT INTO audit_logs VALUES (?, ?, ?, ?, ?)', 
                   (ticker, step, data['recorded_at'], data_hash, data_json))
    conn.commit()
    conn.close()
    return data_hash
