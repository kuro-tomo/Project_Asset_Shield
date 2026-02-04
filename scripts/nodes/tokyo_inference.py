import requests
from datetime import datetime, timezone

CORE_URL = "http://localhost:8000/ingest"

def run_prototype():
    payload = {
        "node": "TYO-INFERENCE-ENGINE",
        "score": 5.2,
        "entity": "1234.T",
        "metadata": {
            "llm_rationale": "[Institutional Analysis] Upward revision significantly exceeds expectations. Dividend increase following policy change is positive. Stimulates passive fund buying demand.",
            "timestamp_utc": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        }
    }
    try:
        response = requests.post(CORE_URL, json=payload, timeout=5)
        print(f"🚀 Status: {response.status_code} | Response: {response.json()}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_prototype()
