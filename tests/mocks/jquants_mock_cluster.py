import time
import random
import os
from multiprocessing import Process

def generate_data(asset_id, filename):
    price = random.uniform(2000, 40000)
    vol = price * 0.005
    while True:
        price += random.uniform(-vol, vol)
        if price < 100: price = 100
        with open(filename, "a") as f:
            f.write(f"{time.time()},{price}\n")
        time.sleep(random.uniform(0.3, 0.8))

if __name__ == "__main__":
    print("--- JÖRMUNGANDR DATA CLUSTER V8.2: ON-DEMAND MODE ---")
    active_files = set()
    try:
        while True:
            # カレントディレクトリを監視し、mock_*.log が求められていたら生成開始
            files = [f for f in os.listdir('.') if f.startswith('mock_') and f.endswith('.log')]
            for f in files:
                if f not in active_files:
                    asset_id = f.replace('mock_', '').replace('.log', '') + "_T"
                    p = Process(target=generate_data, args=(asset_id, f))
                    p.daemon = True
                    p.start()
                    active_files.add(f)
                    print(f"[*] Started On-Demand Generation for: {asset_id}")
            time.sleep(2)
    except KeyboardInterrupt: pass
