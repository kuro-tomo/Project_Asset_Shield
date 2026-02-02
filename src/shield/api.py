from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sqlite3

app = FastAPI()

# Enable CORS for the frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/v1/sovereign/latest")
def get_latest_evidence():
    """Fetch the latest proof of logic for the institutional dashboard."""
    try:
        conn = sqlite3.connect("logs/audit.db")
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT 1").fetchone()
        conn.close()
        if row:
            return dict(row)
        return {"status": "Waiting for data...", "nexus_score": 0.0}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
