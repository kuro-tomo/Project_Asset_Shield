#!/usr/bin/env python3
"""
Silence Protocol Deployment Script V9.0

This script deploys and manages the enhanced Silence Protocol system.
It provides CLI commands for:
- Initializing the dead man's switch
- Sending heartbeats
- Triggering emergency protocols
- Scanning for PII
- Testing secure wipe functionality

WARNING: This script contains destructive operations.
Use with extreme caution in production environments.
"""

import os
import sys
import argparse
import hashlib
import getpass
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.shield.silence import (
    DeadlySilence,
    SilenceConfig,
    SecureWiper,
    PIISanitizer,
    DecoyGenerator,
    WipeMethod,
    TriggerType,
    initialize_silence_protocol
)


def cmd_init(args):
    """Initialize the silence protocol"""
    print("[*] Initializing Silence Protocol V9.0...")
    
    config = SilenceConfig(
        heartbeat_interval_seconds=args.interval,
        wipe_method=WipeMethod.DOD_3PASS if args.secure else WipeMethod.SIMPLE
    )
    
    # Create heartbeat file
    heartbeat_path = Path(config.heartbeat_file)
    heartbeat_path.write_text(datetime.now().isoformat(), encoding='utf-8')
    print(f"[+] Heartbeat file created: {heartbeat_path}")
    
    # Create canary files
    for canary in config.canary_files:
        canary_path = Path(canary)
        canary_hash = hashlib.sha256(
            f"canary_{datetime.now().isoformat()}".encode()
        ).hexdigest()
        canary_path.write_text(canary_hash, encoding='utf-8')
        print(f"[+] Canary file created: {canary_path}")
    
    print("[+] Silence Protocol initialized successfully.")
    print(f"[!] Heartbeat required every {args.interval} seconds.")
    print("[!] Keep the heartbeat file updated to prevent automatic trigger.")


def cmd_heartbeat(args):
    """Send a heartbeat signal"""
    config = SilenceConfig()
    silence = DeadlySilence(config)
    
    code = args.code if args.code else None
    
    if silence.send_heartbeat(code):
        print(f"[+] Heartbeat sent at {datetime.now().isoformat()}")
    else:
        print("[-] Heartbeat failed")
        sys.exit(1)


def cmd_panic(args):
    """Trigger panic button"""
    print("[!] PANIC BUTTON ACTIVATION")
    print("[!] This will PERMANENTLY DESTROY critical files.")
    
    if not args.force:
        confirm = input("[?] Type 'CONFIRM DESTRUCTION' to proceed: ")
        if confirm != "CONFIRM DESTRUCTION":
            print("[-] Aborted.")
            sys.exit(1)
    
    # Get panic code
    if args.code:
        code = args.code
    else:
        code = getpass.getpass("[?] Enter panic sequence: ")
    
    config = SilenceConfig()
    silence = DeadlySilence(config)
    
    if silence.trigger_panic(code):
        print("[!] PANIC PROTOCOL EXECUTED")
        print("[!] Critical files have been destroyed.")
    else:
        print("[-] Invalid panic sequence.")
        sys.exit(1)


def cmd_scan_pii(args):
    """Scan for PII in project files"""
    print("[*] Scanning for Personal Identifiable Information...")
    
    config = SilenceConfig()
    sanitizer = PIISanitizer(config.pii_patterns)
    
    scan_path = Path(args.path) if args.path else Path('.')
    findings = sanitizer.scan_directory(scan_path)
    
    if not findings:
        print("[+] No PII detected.")
        return
    
    print(f"[!] Found {len(findings)} potential PII instances:")
    for finding in findings:
        print(f"    - {finding['file']}: {finding['count']} matches")
        if args.verbose:
            for match in finding['matches'][:3]:  # Show first 3
                print(f"      > {match[:50]}...")
    
    if args.sanitize:
        print("\n[*] Sanitizing files...")
        for finding in findings:
            file_path = Path(finding['file'])
            if sanitizer.sanitize_file(file_path):
                print(f"[+] Sanitized: {file_path}")
            else:
                print(f"[-] Failed to sanitize: {file_path}")


def cmd_test_wipe(args):
    """Test secure wipe on a dummy file"""
    print("[*] Testing secure wipe functionality...")
    
    # Create test file
    test_file = Path(args.file) if args.file else Path('.silence_test_file')
    test_content = "SENSITIVE DATA " * 1000  # ~15KB of test data
    
    test_file.write_text(test_content, encoding='utf-8')
    original_size = test_file.stat().st_size
    print(f"[+] Created test file: {test_file} ({original_size} bytes)")
    
    # Determine wipe method
    method_map = {
        'simple': WipeMethod.SIMPLE,
        'dod3': WipeMethod.DOD_3PASS,
        'dod7': WipeMethod.DOD_7PASS,
        'gutmann': WipeMethod.GUTMANN
    }
    method = method_map.get(args.method, WipeMethod.DOD_3PASS)
    
    print(f"[*] Wiping with method: {method.name} ({method.value} passes)")
    
    if SecureWiper.secure_delete(test_file, method):
        print("[+] Secure wipe completed successfully.")
        if test_file.exists():
            print("[-] WARNING: File still exists after wipe!")
        else:
            print("[+] File successfully destroyed.")
    else:
        print("[-] Secure wipe failed.")


def cmd_status(args):
    """Check silence protocol status"""
    config = SilenceConfig()
    
    print("[*] Silence Protocol Status")
    print("=" * 40)
    
    # Check heartbeat
    heartbeat_path = Path(config.heartbeat_file)
    if heartbeat_path.exists():
        try:
            timestamp = heartbeat_path.read_text(encoding='utf-8').strip()
            last_beat = datetime.fromisoformat(timestamp)
            age = datetime.now() - last_beat
            print(f"[+] Heartbeat: ACTIVE")
            print(f"    Last beat: {timestamp}")
            print(f"    Age: {age.total_seconds():.0f} seconds")
            
            timeout = config.heartbeat_interval_seconds + config.grace_period_seconds
            remaining = timeout - age.total_seconds()
            if remaining > 0:
                print(f"    Time until trigger: {remaining:.0f} seconds")
            else:
                print(f"    [!] OVERDUE by {-remaining:.0f} seconds!")
        except Exception as e:
            print(f"[-] Heartbeat: ERROR ({e})")
    else:
        print("[-] Heartbeat: NOT FOUND")
    
    # Check canary files
    print("\n[*] Canary Files:")
    for canary in config.canary_files:
        canary_path = Path(canary)
        if canary_path.exists():
            print(f"    [+] {canary}: OK")
        else:
            print(f"    [-] {canary}: MISSING")
    
    # Check critical targets
    print("\n[*] Critical Targets:")
    for target in config.critical_targets[:5]:  # Show first 5
        target_path = Path(target)
        if target_path.exists():
            if target_path.is_file():
                size = target_path.stat().st_size
                print(f"    [+] {target}: {size} bytes")
            else:
                print(f"    [+] {target}: (directory)")
        else:
            print(f"    [-] {target}: NOT FOUND")


def cmd_daemon(args):
    """Run silence protocol as a daemon"""
    print("[*] Starting Silence Protocol Daemon...")
    print(f"[*] Heartbeat interval: {args.interval} seconds")
    print("[*] Press Ctrl+C to stop (requires valid heartbeat)")
    
    config = SilenceConfig(
        heartbeat_interval_seconds=args.interval
    )
    
    silence = DeadlySilence(config)
    
    # Register callback for logging
    def on_trigger(trigger_type: TriggerType):
        print(f"\n[!] SILENCE PROTOCOL TRIGGERED: {trigger_type.name}")
        print("[!] Executing destruction sequence...")
    
    silence.register_callback(on_trigger)
    
    try:
        silence.start_monitoring()
        
        # Keep main thread alive
        while silence.is_active:
            import time
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[*] Stopping daemon...")
        silence.stop_monitoring()
        print("[+] Daemon stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Silence Protocol V9.0 - Dead Man's Switch Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Initialize protocol:     python silence_protocol.py init --interval 3600
  Send heartbeat:          python silence_protocol.py heartbeat
  Check status:            python silence_protocol.py status
  Scan for PII:            python silence_protocol.py scan --verbose
  Test secure wipe:        python silence_protocol.py test-wipe --method dod3
  Run as daemon:           python silence_protocol.py daemon --interval 1800
  
WARNING: The 'panic' command will PERMANENTLY DESTROY critical files.
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize silence protocol')
    init_parser.add_argument('--interval', type=int, default=3600,
                            help='Heartbeat interval in seconds (default: 3600)')
    init_parser.add_argument('--secure', action='store_true',
                            help='Use DoD 3-pass wipe method')
    
    # Heartbeat command
    hb_parser = subparsers.add_parser('heartbeat', help='Send heartbeat signal')
    hb_parser.add_argument('--code', type=str, help='Optional verification code')
    
    # Panic command
    panic_parser = subparsers.add_parser('panic', help='Trigger panic button')
    panic_parser.add_argument('--code', type=str, help='Panic sequence code')
    panic_parser.add_argument('--force', action='store_true',
                             help='Skip confirmation prompt')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan for PII')
    scan_parser.add_argument('--path', type=str, help='Path to scan')
    scan_parser.add_argument('--verbose', '-v', action='store_true',
                            help='Show detailed findings')
    scan_parser.add_argument('--sanitize', action='store_true',
                            help='Automatically sanitize found PII')
    
    # Test wipe command
    test_parser = subparsers.add_parser('test-wipe', help='Test secure wipe')
    test_parser.add_argument('--file', type=str, help='File to test wipe on')
    test_parser.add_argument('--method', choices=['simple', 'dod3', 'dod7', 'gutmann'],
                            default='dod3', help='Wipe method to use')
    
    # Status command
    subparsers.add_parser('status', help='Check protocol status')
    
    # Daemon command
    daemon_parser = subparsers.add_parser('daemon', help='Run as daemon')
    daemon_parser.add_argument('--interval', type=int, default=3600,
                              help='Heartbeat interval in seconds')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate command
    commands = {
        'init': cmd_init,
        'heartbeat': cmd_heartbeat,
        'panic': cmd_panic,
        'scan': cmd_scan_pii,
        'test-wipe': cmd_test_wipe,
        'status': cmd_status,
        'daemon': cmd_daemon
    }
    
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
