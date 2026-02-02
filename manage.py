#!/usr/bin/env python3
import sys
import os
import argparse
import logging

# Ensure src is in python path for development execution
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from shield.pipeline import ProductionPipeline
    from shield.node_manager import main as node_main
except ImportError as e:
    print(f"Error importing TIR Core: {e}")
    print("Please ensure you have installed dependencies: pip install -r requirements.txt")
    print("And that 'src' is accessible.")
    sys.exit(1)

def run_pipeline(args):
    """Run the main production pipeline"""
    print(f"Starting TIR Pipeline for {args.ticker}...")
    pipeline = ProductionPipeline()
    pipeline.run(args.ticker, args.capital, dry_run=args.dry_run)

def run_node(args):
    """Run the node manager"""
    print("Starting Node Manager...")
    node_main()

def main():
    parser = argparse.ArgumentParser(description="TIR Quantitative Engine Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Pipeline Command
    pipeline_parser = subparsers.add_parser("run-pipeline", help="Run the trading pipeline")
    pipeline_parser.add_argument("--ticker", type=str, default="7203", help="Target ticker symbol")
    pipeline_parser.add_argument("--capital", type=float, default=100000, help="Initial capital (JPY)")
    pipeline_parser.add_argument("--dry-run", action="store_true", help="Execute in dry-run mode (no orders)")

    # Node Command
    node_parser = subparsers.add_parser("node", help="Run in Node/Agent mode")

    # Test Command
    test_parser = subparsers.add_parser("test", help="Run test suite")

    args = parser.parse_args()

    if args.command == "run-pipeline":
        run_pipeline(args)
    elif args.command == "node":
        run_node(args)
    elif args.command == "test":
        print("Running tests...")
        os.system("pytest tests/")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
