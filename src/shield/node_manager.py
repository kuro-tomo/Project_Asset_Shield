import socket
import os
import sys

def identify_node():
    """機体（Node）を識別する"""
    hostname = socket.gethostname()
    # 環境変数 JOR_NODE_TYPE があれば優先、なければホスト名で判定
    node_type = os.getenv('JOR_NODE_TYPE', '').upper()
    
    if node_type == 'MASTER' or "TIRnoMacBook-Pro" in hostname:
        return "MASTER (M4 - Home Base)"
    elif node_type == 'AGENT' or "Air" in hostname:
        return "AGENT (M2 Air - Mobile)"
    else:
        return f"UNKNOWN NODE ({hostname})"

def main():
    node = identify_node()
    print(f"--- Project JÖRMUNGANDR Genesis ---")
    print(f"Current Node: {node}")
    
    if "MASTER" in node:
        print("[Status] Initializing Master Services...")
        print("[Status] J-Quants V2 Stream Standby: OK")
        print("[Status] gRPC Relay Server: Initializing...")
        # ここにMaster専用の起動ロジックを追加していく
    else:
        print("[Status] Initializing Agent Services...")
        print("[Status] Secure Tunnel to Master: Checking...")
        print("[Status] Monitoring Dashboard: Standby")
        # ここにAgent専用の起動ロジックを追加していく

if __name__ == "__main__":
    main()
