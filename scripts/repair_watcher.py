import os

def repair():
    watcher_v75_code = """import time
import os
import json
import logging
import sys
from multiprocessing import Process, Manager
from shield.brain import ShieldBrain
from shield.evolution import EvolutionEngine

# Audit log configuration
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, filename='sovereign_audit.log')

TARGETS = {
    "8035_T": "mock_8035.log",
    "9984_T": "mock_9984.log",
    "7203_T": "mock_7203.log"
}

def monitor_asset(target_id, log_filename, shared_dict):
    logger = logging.getLogger(target_id)
    brain = ShieldBrain(target_id=target_id)
    evolver = EvolutionEngine()
    prices_jpy = []

    session = {"balance": 3333333.33, "total_profit": 0.0}
    position, entry_price_jpy = 0, 0
    tick_count = 0

    while True:
        try:
            if not os.path.exists(log_filename):
                time.sleep(2)
                continue

            with open(log_filename, "r") as f:
                f.seek(0, os.SEEK_END)
                while True:
                    line = f.readline()
                    if not line:
                        time.sleep(0.5)
                        continue

                    try:
                        price_jpy = float(line.strip().split(",")[1])
                    except: continue

                    prices_jpy.append(price_jpy)
                    if len(prices_jpy) > 100: prices_jpy.pop(0)

                    if len(prices_jpy) >= 50:
                        conf = brain.calculate_confidence(prices_jpy)
                        # Self-evolution check
                        tick_count += 1
                        if tick_count % 100 == 0:
                            evo_result = evolver.evolve_brain(brain)
                            if "SUCCESS" in evo_result:
                                logging.info(f"🧬 [{target_id}] EVOLVED: {evo_result}")

                        shared_dict[target_id] = {
                            "price": price_jpy, "conf": conf,
                            "status": "GUARD" if position == 1 else "OBSV",
                            "pnl": session["total_profit"]
                        }
        except Exception as e:
            time.sleep(1)

def dashboard_renderer(shared_dict):
    start_time = time.time()
    while True:
        try:
            os.system('clear')
            print("="*95)
            print(f" SHIELD V7.5 - SOVEREIGN EVOLUTIONARY ENGINE")
            print(f" STATUS: ACTIVE | THEORY: REGENERATION, EVOLUTION, MULTIPLICATION")
            print("="*95)
            print(f"{'TICKER':<10} | {'PRICE (JPY)':<12} | {'CONF':<8} | {'STATUS':<8} | {'PnL (USD)':<12}")
            print("-" * 95)
            for t_id, data in shared_dict.items():
                print(f"{t_id:<10} | {data.get('price', 0):12.2f} | {data.get('conf', 0):+8.4f} | {data.get('status', 'INIT'):<8} | ${data.get('pnl', 0):+,.2f}")
            print("-" * 95)
            time.sleep(1)
        except KeyboardInterrupt: break

if __name__ == "__main__":
    with Manager() as manager:
        shared_data = manager.dict()
        for t_id in TARGETS: shared_data[t_id] = {"price":0,"conf":0,"status":"INIT","pnl":0}
        processes = [Process(target=monitor_asset, args=(t_id, l_file, shared_data)) for t_id, l_file in TARGETS.items()]
        for p in processes: p.daemon = True; p.start()
        dashboard_renderer(shared_data)
"""
    with open('watcher.py', 'w') as f:
        f.write(watcher_v75_code)
    print("[SUCCESS] watcher.py has been rebuilt to V7.5.")

if __name__ == "__main__":
    repair()
