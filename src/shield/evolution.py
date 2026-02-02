import os
import json

class EvolutionEngine:
    def __init__(self, log_file="sovereign_audit.log"):
        self.log_file = log_file

    def analyze_performance(self):
        if not os.path.exists(self.log_file):
            return None
        try:
            with open(self.log_file, "r") as f:
                logs = f.readlines()
            
            failures = [line for line in logs if "STOP-LOSS" in line]
            if len(failures) > 3:
                return {"action": "TIGHTEN_RISK", "penalty_increase": 0.2}
            
            successes = [line for line in logs if "STRATEGIC" in line]
            if len(successes) > 5:
                return {"action": "OPTIMIZE_THRESHOLD", "threshold_decrease": 0.02}
        except:
            pass
        return None

    def evolve_brain(self, brain):
        insight = self.analyze_performance()
        if insight:
            if insight["action"] == "TIGHTEN_RISK":
                brain.risk_penalty += insight["penalty_increase"]
                brain.adaptive_threshold = min(0.90, brain.adaptive_threshold + 0.05)
            elif insight["action"] == "OPTIMIZE_THRESHOLD":
                brain.adaptive_threshold = max(0.40, brain.adaptive_threshold - insight["threshold_decrease"])
            
            brain._save_memory()
            return f"SUCCESS:{insight['action']}"
        return "STABLE"
