import subprocess
import time

# 巡回するターゲットリスト（日本の主要5社）
TARGET_TICKERS = [
    "7203.T",  # トヨタ自動車
    "6758.T",  # ソニーグループ
    "9984.T",  # ソフトバンクグループ
    "8306.T",  # 三菱UFJフィナンシャルG
    "6861.T"   # キーエンス
]

def run_batch():
    print("============================================================")
    print("🚀 TIR BATCH MODE: Starting Nightly Patrol...")
    print(f"Targets: {', '.join(TARGET_TICKERS)}")
    print("============================================================\n")

    for ticker in TARGET_TICKERS:
        print(f"📡 Next Target: {ticker}")
        try:
            # main.py を外部プロセスとして実行
            result = subprocess.run(["python3", "main.py", ticker], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Mission Success for {ticker}")
            else:
                print(f"❌ Mission Failed for {ticker}")
                print(f"Error Log: {result.stderr}")
        
        except Exception as e:
            print(f"⚠️ Unexpected error while processing {ticker}: {e}")
        
        # サーバーへの負荷軽減と検知回避のためのクールダウン
        print(f"⏳ Cooling down for 5 seconds...")
        time.sleep(5)

    print("\n============================================================")
    print("🏁 ALL MISSIONS COMPLETE. Reports are ready in output/reports.")
    print("============================================================")

if __name__ == "__main__":
    run_batch()