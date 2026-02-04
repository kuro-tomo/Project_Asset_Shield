import subprocess
import time

# Target list (Japan major 5 companies)
TARGET_TICKERS = [
    "7203.T",  # Toyota Motor
    "6758.T",  # Sony Group
    "9984.T",  # SoftBank Group
    "8306.T",  # Mitsubishi UFJ Financial
    "6861.T"   # Keyence
]

def run_batch():
    print("============================================================")
    print("🚀 TIR BATCH MODE: Starting Nightly Patrol...")
    print(f"Targets: {', '.join(TARGET_TICKERS)}")
    print("============================================================\n")

    for ticker in TARGET_TICKERS:
        print(f"📡 Next Target: {ticker}")
        try:
            # Execute main.py as external process
            result = subprocess.run(["python3", "main.py", ticker], capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✅ Mission Success for {ticker}")
            else:
                print(f"❌ Mission Failed for {ticker}")
                print(f"Error Log: {result.stderr}")

        except Exception as e:
            print(f"⚠️ Unexpected error while processing {ticker}: {e}")

        # Cooldown for server load reduction and rate limiting
        print(f"⏳ Cooling down for 5 seconds...")
        time.sleep(5)

    print("\n============================================================")
    print("🏁 ALL MISSIONS COMPLETE. Reports are ready in output/reports.")
    print("============================================================")

if __name__ == "__main__":
    run_batch()
