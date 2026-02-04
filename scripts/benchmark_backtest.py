import sys
import os
import time
import statistics
from datetime import datetime

# Path setup
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from shield.jquants_client import JQuantsClient
from shield.jquants_backtest_provider import create_jquants_provider
from scripts.run_jquants_backtest import run_backtest

def benchmark():
    print("=== Full Universe Backtest Time Estimation Benchmark ===")

    # Initialize provider
    provider = create_jquants_provider()
    status = provider.get_status()
    is_mock = status.get('plan') == 'mock' or 'mock' in str(status.get('plan', '')).lower()

    # 1. Get total stock count
    print("\n[Step 1] Fetching stock universe...")
    client = JQuantsClient()
    listed_info = client.get_listed_info()

    total_stocks = 0
    if is_mock:
        print("  * Mock mode detected - using estimated value (4,000 stocks)")
        total_stocks = 4000
    else:
        if listed_info:
            total_stocks = len(listed_info)
            print(f"  Completed: {total_stocks} stocks total")
        else:
            print("  Failed to fetch stock info. Using estimated value (4,000)")
            total_stocks = 4000

    # 2. Select sample stocks
    sample_codes = ["7203", "9984", "6758", "8035", "9432"]  # Toyota, SBG, Sony, TEL, NTT

    # 3. Execute backtest & measure
    print(f"\n[Step 2] Running backtest on {len(sample_codes)} sample stocks...")

    execution_times = []

    # Measure each stock individually
    # Note: Keeping log output visible for progress monitoring

    for code in sample_codes:
        print(f"  Testing {code}...", end=" ", flush=True)
        start_time = time.time()

        try:
            # Execute backtest
            # Note: run_backtest outputs logs internally
            run_backtest(
                provider=provider,
                codes=[code],
                initial_capital=100_000_000,
                phases=None
            )
            duration = time.time() - start_time
            execution_times.append(duration)
            print(f"-> Done ({duration:.2f}s)")

        except Exception as e:
            print(f"-> Failed ({e})")

    if not execution_times:
        print("No valid measurement data available.")
        return

    # 4. Calculate results
    avg_time = statistics.mean(execution_times)
    total_estimated_seconds = avg_time * total_stocks

    print("\n" + "="*40)
    print(" BENCHMARK RESULT")
    print("="*40)
    print(f"  Environment: {'MOCK MODE' if is_mock else 'PRODUCTION MODE'}")
    print(f"  Sample size: {len(execution_times)} stocks")
    print(f"  Avg time per stock: {avg_time:.4f} sec")
    print(f"  Total stocks (estimated): {total_stocks}")
    print("-" * 40)

    # Time calculation
    hours = total_estimated_seconds / 3600
    minutes = (total_estimated_seconds % 3600) / 60

    print(f"  Estimated total time: {int(hours)}h {int(minutes)}min")
    print(f"  (approx. {total_estimated_seconds / 60:.0f} min)")

    if hours > 24:
        days = hours / 24
        print(f"  (approx. {days:.1f} days)")

    print("="*40)

if __name__ == "__main__":
    benchmark()
