#!/usr/bin/env python3
"""
Asset Shield V2 - M&A Package Generator
Executes final due diligence checks and generates transfer package.
"""

import os
import sys
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

from shield.backtest_framework import (
    create_mock_data_provider,
    create_mock_strategy,
    MultiPhaseBacktester,
    BacktestPhase
)
from shield.asset_decoupling import AssetDecouplingManager
from shield.security_governance import IntegrityHashManager, BurnProtocol

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MA_PACKAGE")

def main():
    print("=" * 60)
    print("Asset Shield V2 - M&A Package Generator")
    print("Target: M&A Negotiation / Due Diligence")
    print("=" * 60)
    
    # Create package directory
    package_dir = Path("output/ma_package")
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)
    
    print(f"\n[1/4] Executing Technical Due Diligence (Backtest)...")
    # ---------------------------------------------------------
    try:
        data_provider = create_mock_data_provider()
        strategy = create_mock_strategy()
        
        backtester = MultiPhaseBacktester(
            strategy=strategy,
            data_provider=data_provider,
            initial_capital=100_000_000
        )
        
        summary = backtester.run_all_phases()
        audit_report = backtester.generate_audit_report()
        
        # Save report
        report_path = package_dir / "TECHNICAL_DD_REPORT.json"
        with open(report_path, 'w') as f:
            json.dump(audit_report, f, indent=2)
            
        print(f"  > Backtest Complete: Avg Return {summary['aggregate']['avg_annual_return']}")
        print(f"  > Report Saved: {report_path}")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        sys.exit(1)
        
    print(f"\n[2/4] Generating Asset Manifest (Decoupling)...")
    # ---------------------------------------------------------
    try:
        manager = AssetDecouplingManager(project_root=".")
        manifest = manager.scan_project()
        
        # Generate audit report
        audit_report = manager.generate_audit_report()
        
        # Save manifest and report
        manifest_path = package_dir / "ASSET_MANIFEST.json"
        audit_path = package_dir / "ASSET_AUDIT_REPORT.json"
        
        manager.save_manifest(str(manifest_path))
        with open(audit_path, 'w') as f:
            json.dump(audit_report, f, indent=2)
            
        print(f"  > Assets Scanned: {len(manifest.entries)} files")
        print(f"  > Version-J (Transferable): {audit_report['summary']['version_j_count']}")
        print(f"  > Version-Q (Retained IP): {audit_report['summary']['version_q_count']}")
        print(f"  > Manifest Saved: {manifest_path}")
        
    except Exception as e:
        logger.error(f"Decoupling failed: {e}")
        sys.exit(1)
        
    print(f"\n[3/4] Certifying System Integrity (Governance)...")
    # ---------------------------------------------------------
    try:
        integrity_mgr = IntegrityHashManager(project_root=".")
        cert = integrity_mgr.generate_certificate()
        
        # Save certificate
        cert_path = package_dir / "INTEGRITY_CERTIFICATE.json"
        integrity_mgr.save_certificate(cert, str(cert_path))
        
        print(f"  > Certificate ID: {cert.certificate_id}")
        print(f"  > Root Hash: {cert.root_hash[:16]}...")
        print(f"  > Certificate Saved: {cert_path}")
        
        # Burn Protocol Simulation
        burn = BurnProtocol(project_root=".")
        burn.register_access("api_key", "active_api_key_placeholder")
        burn.schedule_burn()
        burn_report = burn.execute_burn(dry_run=True)
        burn_cert = burn.generate_burn_certificate(burn_report)
        
        burn_path = package_dir / "BURN_PROTOCOL_CERTIFICATE.json"
        with open(burn_path, 'w') as f:
            json.dump(burn_cert, f, indent=2)
            
        print(f"  > Burn Protocol Verified: {burn_cert['certificate_id']}")
        
    except Exception as e:
        logger.error(f"Governance failed: {e}")
        sys.exit(1)

    print(f"\n[4/4] Finalizing Package...")
    # ---------------------------------------------------------
    # Copy Negotiation Materials & Valuation Evidence
    try:
        # Main Negotiation Deck
        src_doc = Path("docs/MA_NEGOTIATION_MATERIALS_JA.md")
        if src_doc.exists():
            shutil.copy2(src_doc, package_dir / "NEGOTIATION_MATERIALS.md")
            print(f"  > Materials Copied: NEGOTIATION_MATERIALS.md")
            
        # Cost Allocation Strategy
        cost_doc = Path("docs/DD_COST_AVOIDANCE_STRATEGY_JA.md")
        if cost_doc.exists():
            shutil.copy2(cost_doc, package_dir / "DD_COST_AVOIDANCE_STRATEGY.md")
            print(f"  > Materials Copied: DD_COST_AVOIDANCE_STRATEGY.md")

        # --- Valuation Simulation Artifacts ---
        
        # 1. Simulation Result JSON
        val_result = Path("output/valuation_simulation_result.json")
        if val_result.exists():
            shutil.copy2(val_result, package_dir / "VALUATION_SIMULATION_RESULT.json")
            print(f"  > Valuation Evidence: VALUATION_SIMULATION_RESULT.json")
            
        # 2. Simulation Report MD
        val_report = Path("output/VALUATION_SIMULATION_REPORT.md")
        if val_report.exists():
            shutil.copy2(val_report, package_dir / "VALUATION_SIMULATION_REPORT.md")
            print(f"  > Valuation Report: VALUATION_SIMULATION_REPORT.md")
            
        # 3. Reproduction Script
        # Create 'scripts' subdir in package to keep it clean
        pkg_scripts_dir = package_dir / "scripts"
        pkg_scripts_dir.mkdir(exist_ok=True)
        
        repro_script = Path("scripts/run_valuation_simulation.py")
        if repro_script.exists():
            shutil.copy2(repro_script, pkg_scripts_dir / "run_valuation_simulation.py")
            print(f"  > Reproduction Script: scripts/run_valuation_simulation.py")
            
    except Exception as e:
        logger.warning(f"Could not copy docs: {e}")
        
    print("\n" + "=" * 60)
    print("M&A PACKAGE GENERATION COMPLETE")
    print("=" * 60)
    print(f"Location: {package_dir.absolute()}")
    print("Contents:")
    for f in package_dir.iterdir():
        print(f"  - {f.name}")
    print("=" * 60)

if __name__ == "__main__":
    main()
