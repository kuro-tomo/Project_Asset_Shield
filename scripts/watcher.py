"""
SHIELD Watcher V9.1.1 - Silent Genesis

Multi-process asset monitoring with integrated Silence Protocol.
Implements Dead Man's Switch for operator safety.
"""

import time
import os
import sys
import signal
from pathlib import Path
from multiprocessing import Process, Manager
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.shield.brain import ShieldBrain
from src.shield.money_management import MoneyManager
from src.shield.silence import DeadlySilence, SilenceConfig, TriggerType

TARGETS = ["8035_T", "9984_T", "7203_T"]
INITIAL_CAPITAL_USD = 10000

# Global silence protocol instance
_silence_protocol: DeadlySilence = None


def initialize_silence_protocol() -> DeadlySilence:
    """Initialize the silence protocol with watcher-specific settings"""
    config = SilenceConfig(
        heartbeat_interval_seconds=1800,  # 30 minutes for active trading
        grace_period_seconds=300,         # 5 minute grace
    )
    
    silence = DeadlySilence(config)
    
    # Register callback for emergency shutdown
    def on_silence_trigger(trigger_type: TriggerType):
        print(f"\n[!] SILENCE PROTOCOL TRIGGERED: {trigger_type.name}")
        print("[!] Initiating emergency shutdown...")
        # The silence protocol will handle file destruction
    
    silence.register_callback(on_silence_trigger)
    
    return silence


def send_heartbeat():
    """Send heartbeat to silence protocol"""
    global _silence_protocol
    if _silence_protocol:
        _silence_protocol.send_heartbeat()


def monitor_asset(target_id, shared_dict):
    log_file = f"mock_{target_id.split('_')[0]}.log"
    if not os.path.exists(log_file):
        open(log_file, 'a').close()
        
    brain = ShieldBrain(target_id=target_id)
    mm = MoneyManager()
    
    while True:
        try:
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    lines = f.readlines()
                    if lines:
                        prices = [float(line.strip().split(",")[1]) for line in lines[-100:]]
                        if len(prices) >= 5:
                            conf = brain.calculate_confidence(prices)
                            verdict = "NEUTRAL WATCH"
                            if conf > 0.05: verdict = "INSTITUTIONAL BUY"
                            elif conf > 0.02: verdict = "AGGRESSIVE BUY"
                            
                            pos = mm.get_position_size(INITIAL_CAPITAL_USD, verdict)
                            shared_dict[target_id] = {
                                "price": prices[-1], 
                                "conf": conf,
                                "saint": pos['ledgers']['SAINT_PUBLIC'],
                                "ghost": pos['ledgers']['GHOST_PRIVATE']
                            }
            time.sleep(1)
        except Exception:
            time.sleep(1)

def multiplication_scout(shared_dict):
    candidates = ["6758_T", "6501_T", "9432_T", "6098_T", "4063_T"]
    for ticker in candidates:
        time.sleep(15)
        if ticker not in shared_dict:
            p = Process(target=monitor_asset, args=(ticker, shared_dict))
            p.daemon = True
            p.start()

def dashboard(shared_dict, silence: DeadlySilence):
    """Main dashboard with integrated silence protocol heartbeat"""
    heartbeat_counter = 0
    
    while True:
        os.system('clear')
        
        # Send heartbeat every 60 iterations (approximately every minute)
        heartbeat_counter += 1
        if heartbeat_counter >= 60:
            silence.send_heartbeat()
            heartbeat_counter = 0
        
        # Check if silence protocol is still active
        silence_status = "ARMED" if silence.is_active else "STANDBY"
        
        print("="*105)
        print(" SHIELD V9.1.1 - SILENT GENESIS | THEORY: REGENERATION, EVOLUTION, MULTIPLICATION")
        print(f" ACTIVE PROCESSES: {len(shared_dict):<2} | MODE: STEALTH (15:85 SPLIT) | SILENCE: {silence_status}")
        print("="*105)
        print(f"{'TICKER':<10} | {'PRICE':<10} | {'CONF':<10} | {'SAINT (15%)':<15} | {'GHOST (85%)':<15}")
        print("-"*105)
        for t_id, data in sorted(list(shared_dict.items())):
            print(f"{t_id:<10} | {data.get('price', 0):10.2f} | {data.get('conf', 0):+8.4f} | "
                  f"JPY {data.get('saint', 0):>10.0f} | JPY {data.get('ghost', 0):>10.0f}")
        
        # Show heartbeat status
        print("-"*105)
        print(f" [HEARTBEAT] Last: {datetime.now().strftime('%H:%M:%S')} | Next in: {60 - heartbeat_counter}s")
        
        time.sleep(1)


def handle_panic_signal(signum, frame):
    """Handle SIGUSR1 as panic signal"""
    global _silence_protocol
    if _silence_protocol:
        print("\n[!] PANIC SIGNAL RECEIVED (SIGUSR1)")
        _silence_protocol.trigger_panic(_silence_protocol.config.panic_sequence)
    sys.exit(1)


if __name__ == "__main__":
    # Initialize silence protocol
    _silence_protocol = initialize_silence_protocol()
    
    # Register signal handlers for emergency triggers
    signal.signal(signal.SIGUSR1, handle_panic_signal)
    
    # Start silence protocol monitoring
    _silence_protocol.start_monitoring()
    print("[+] Silence Protocol V9.0 ARMED")
    print(f"[+] Heartbeat interval: {_silence_protocol.config.heartbeat_interval_seconds}s")
    
    with Manager() as manager:
        shared_data = manager.dict()
        processes = []
        
        for t in TARGETS:
            p = Process(target=monitor_asset, args=(t, shared_data))
            p.start()
            processes.append(p)
            
        p_scout = Process(target=multiplication_scout, args=(shared_data,))
        p_scout.start()
        processes.append(p_scout)
        
        try:
            dashboard(shared_data, _silence_protocol)
        except KeyboardInterrupt:
            print("\n[*] Graceful shutdown initiated...")
            _silence_protocol.stop_monitoring()
            for p in processes:
                p.terminate()
            print("[+] All processes terminated.")
