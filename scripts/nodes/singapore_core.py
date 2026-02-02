from fastapi import FastAPI
import uvicorn
import sqlite3
import os

app = FastAPI()

@app.on_event("startup")
def init_db():
    os.makedirs('logs', exist_ok=True)
    conn = sqlite3.connect('logs/audit_global.db')
    c = conn.cursor()
    # 必要なカラムを確実に用意
    c.execute('''CREATE TABLE IF NOT EXISTS audit_logs 
                 (id INTEGER PRIMARY KEY, timestamp_utc TEXT, node TEXT, entity TEXT, score REAL, llm_rationale TEXT)''')
    conn.commit()
    conn.close()

@app.post("/ingest")
async def ingest_data(data: dict):
    # 送信されたデータを取り出す
    metadata = data.get("metadata", {})
    payload = (
        metadata.get("timestamp_utc"),
        data.get("node"),
        data.get("entity"),
        data.get("score"),
        metadata.get("llm_rationale")
    )
    
    conn = sqlite3.connect('logs/audit_global.db')
    c = conn.cursor()
    c.execute("INSERT INTO audit_logs (timestamp_utc, node, entity, score, llm_rationale) VALUES (?, ?, ?, ?, ?)", payload)
    conn.commit()
    conn.close()
    print(f"✅ Inscribed to Audit DB: {data.get('entity')}")
    return {"status": "success", "message": "Inscribed"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
