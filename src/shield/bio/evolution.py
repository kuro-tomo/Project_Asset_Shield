import json
import os
import random
import logging

logger = logging.getLogger("Bio.Evolution")

class EvolutionEngine:
    """
    Self-Evolution Module (Adaptation)
    Analyzes performance and mutates strategic parameters (Genes) to improve fitness.
    """
    def __init__(self, brain_memory_path="memory_default.json"):
        self.brain_memory_path = brain_memory_path
        self.history_path = "data/evolution_history.json"
        
        # Mutation constraints
        self.constraints = {
            "risk_penalty": (0.5, 3.0),
            "adaptive_threshold": (0.4, 0.95)
        }

    def _load_brain_state(self):
        if os.path.exists(self.brain_memory_path):
            with open(self.brain_memory_path, 'r') as f:
                return json.load(f)
        return {"risk_penalty": 1.5, "adaptive_threshold": 0.65}

    def _save_brain_state(self, state):
        with open(self.brain_memory_path, 'w') as f:
            json.dump(state, f, indent=4)

    def mutate(self, intensity=0.1):
        """
        Triggers a genetic mutation on the brain's parameters.
        intensity: Magnitude of change (0.0 to 1.0)
        """
        current_genes = self._load_brain_state()
        new_genes = current_genes.copy()
        
        # Mutate Risk Penalty
        delta_risk = (random.random() - 0.5) * intensity * 2 # -0.1 to +0.1
        new_genes["risk_penalty"] = max(
            self.constraints["risk_penalty"][0],
            min(self.constraints["risk_penalty"][1], current_genes.get("risk_penalty", 1.5) + delta_risk)
        )

        # Mutate Threshold
        delta_thresh = (random.random() - 0.5) * intensity # -0.05 to +0.05
        new_genes["adaptive_threshold"] = max(
            self.constraints["adaptive_threshold"][0],
            min(self.constraints["adaptive_threshold"][1], current_genes.get("adaptive_threshold", 0.65) + delta_thresh)
        )

        logger.info(f"🧬 MUTATION TRIGGERED: {current_genes} -> {new_genes}")
        self._save_brain_state(new_genes)
        self._log_evolution("MUTATION", current_genes, new_genes)
        return new_genes

    def adapt_to_stress(self):
        """Rapid adaptation for high-stress environments (Defensive Evolution)."""
        logger.warning("🛡️ DEFENSIVE EVOLUTION TRIGGERED")
        current_genes = self._load_brain_state()
        new_genes = current_genes.copy()
        
        # Increase defensiveness significantly
        new_genes["risk_penalty"] += 0.5
        new_genes["adaptive_threshold"] += 0.1
        
        # Clamp
        new_genes["risk_penalty"] = min(self.constraints["risk_penalty"][1], new_genes["risk_penalty"])
        new_genes["adaptive_threshold"] = min(self.constraints["adaptive_threshold"][1], new_genes["adaptive_threshold"])
        
        self._save_brain_state(new_genes)
        self._log_evolution("STRESS_ADAPTATION", current_genes, new_genes)

    def _log_evolution(self, event_type, old_genes, new_genes):
        entry = {
            "timestamp": "Now", # In real impl use datetime
            "event": event_type,
            "old": old_genes,
            "new": new_genes
        }
        # Append to history file (simplified)
        try:
            history = []
            if os.path.exists(self.history_path):
                with open(self.history_path, 'r') as f:
                    history = json.load(f)
            history.append(entry)
            with open(self.history_path, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to log evolution: {e}")
