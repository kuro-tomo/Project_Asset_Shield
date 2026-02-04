import os

def check_structure():
    print("=== JÖRMUNGANDR System Configuration Check ===")

    # Critical files to verify
    targets = {
        "Root Files": [
            "watcher.py",
            "jquants_mock_cluster.py",
            "silence_protocol.py",
            ".owner_pulse"
        ],
        "Modules (Intelligence Layer)": [
            "modules/brain.py",
            "modules/evolution.py",
            "modules/silence.py"
        ]
    }

    for category, files in targets.items():
        print(f"\n[{category}]")
        for f in files:
            status = "EXISTS" if os.path.exists(f) else "MISSING"
            print(f" - {f:<25} : {status}")

    print("\n" + "="*46)

if __name__ == "__main__":
    check_structure()