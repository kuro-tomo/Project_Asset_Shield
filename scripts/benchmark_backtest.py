import sys
import os
import time
import statistics
from datetime import datetime

# パス設定
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from shield.jquants_client import JQuantsClient
from shield.jquants_backtest_provider import create_jquants_provider
from scripts.run_jquants_backtest import run_backtest

def benchmark():
    print("=== 全銘柄バックテスト所要時間算出ベンチマーク ===")
    
    # プロバイダー初期化
    provider = create_jquants_provider()
    status = provider.get_status()
    is_mock = status.get('plan') == 'mock' or 'mock' in str(status.get('plan', '')).lower()
    
    # 1. 全銘柄数の取得
    print("\n[Step 1] 全銘柄情報の取得中...")
    client = JQuantsClient()
    listed_info = client.get_listed_info()
    
    total_stocks = 0
    if is_mock:
        print("  ※ Mockモードのため、全銘柄数を概算値(4,000)として計算します。")
        total_stocks = 4000
    else:
        if listed_info:
            total_stocks = len(listed_info)
            print(f"  取得完了: 全 {total_stocks} 銘柄")
        else:
            print("  銘柄情報の取得に失敗しました。概算値(4,000)を使用します。")
            total_stocks = 4000
    
    # 2. サンプル銘柄の選定
    sample_codes = ["7203", "9984", "6758", "8035", "9432"] # トヨタ, SBG, ソニー, 東エレ, NTT
    
    # 3. バックテスト実行 & 計測
    print(f"\n[Step 2] サンプル {len(sample_codes)} 銘柄のバックテスト実行中...")
    
    execution_times = []
    
    # 1銘柄ずつ計測する
    # ログ出力を抑制したいが、loggerの設定を変えるのは複雑なので、標準出力を一時的に無効化する方法もあるが、
    # 進行状況が見えたほうがいいのでそのままにする。
    
    for code in sample_codes:
        print(f"  Testing {code}...", end=" ", flush=True)
        start_time = time.time()
        
        try:
            # バックテスト実行
            # 注意: run_backtest は内部でログ出力を行うため、画面が流れる
            run_backtest(
                provider=provider,
                codes=[code],
                initial_capital=100_000_000,
                phases=None 
            )
            duration = time.time() - start_time
            execution_times.append(duration)
            print(f"-> 完了 ({duration:.2f}秒)")
            
        except Exception as e:
            print(f"-> 失敗 ({e})")
    
    if not execution_times:
        print("有効な計測データがありません。")
        return

    # 4. 結果算出
    avg_time = statistics.mean(execution_times)
    total_estimated_seconds = avg_time * total_stocks
    
    print("\n" + "="*40)
    print(" BENCHMARK RESULT")
    print("="*40)
    print(f"  環境: {'MOCK MODE' if is_mock else 'PRODUCTION MODE'}")
    print(f"  サンプル数: {len(execution_times)} 銘柄")
    print(f"  1銘柄平均処理時間: {avg_time:.4f} 秒")
    print(f"  全銘柄数 (想定): {total_stocks} 銘柄")
    print("-" * 40)
    
    # 時間計算
    hours = total_estimated_seconds / 3600
    minutes = (total_estimated_seconds % 3600) / 60
    
    print(f"  推定総所要時間: {int(hours)}時間 {int(minutes)}分")
    print(f"  (約 {total_estimated_seconds / 60:.0f} 分)")
    
    if hours > 24:
        days = hours / 24
        print(f"  (日数換算: 約 {days:.1f} 日)")
        
    print("="*40)

if __name__ == "__main__":
    benchmark()
