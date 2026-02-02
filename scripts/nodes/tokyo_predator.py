import requests
import time
import random

NODE_ID = "AWS-AP-NORTHEAST-1-TYO"
CORE_URL = "http://localhost:8000/ingest"

def analyze_and_send():
    simulated_score = random.uniform(-0.01, 0.01)
    payload = {"node": NODE_ID, "score": simulated_score}
    try:
        response = requests.post(CORE_URL, json=payload)
        if response.status_code == 200:
            print(f"🗼 [TYO-PREDATOR] Insight sent to Singapore. Score: {simulated_score:.6f}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    print(f"🐍 JÖRMUNGANDR Predator Node: {NODE_ID} Active.")
    while True:
        analyze_and_send()
        time.sleep(5)
