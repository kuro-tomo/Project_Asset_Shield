import sys
import os
import unittest

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from shield.brain import ShieldBrain

class TestShieldBrain(unittest.TestCase):
    def setUp(self):
        # Use a test ID to avoid messing with real memory files
        self.brain = ShieldBrain(target_id="test_unit")
        # Reset memory for test
        if os.path.exists(self.brain.memory_file):
            os.remove(self.brain.memory_file)
        self.brain = ShieldBrain(target_id="test_unit")

    def tearDown(self):
        if os.path.exists(self.brain.memory_file):
            os.remove(self.brain.memory_file)

    def test_learning_mechanism(self):
        initial_threshold = self.brain.adaptive_threshold
        initial_risk = self.brain.risk_penalty
        
        print(f"Initial: Threshold={initial_threshold}, Risk={initial_risk}")

        # 1. Simulate Loss
        self.brain.record_trade_result(-1000)
        
        print(f"After Loss: Threshold={self.brain.adaptive_threshold}, Risk={self.brain.risk_penalty}")
        
        self.assertGreater(self.brain.adaptive_threshold, initial_threshold, "Threshold should increase on loss")
        self.assertGreater(self.brain.risk_penalty, initial_risk, "Risk penalty should increase on loss")

        # 2. Simulate Win
        current_risk = self.brain.risk_penalty
        self.brain.record_trade_result(1000)
        
        print(f"After Win: Threshold={self.brain.adaptive_threshold}, Risk={self.brain.risk_penalty}")
        
        self.assertLess(self.brain.adaptive_threshold, 0.75, "Threshold should decrease on win") # 0.65 + 0.10 - 0.01 = 0.74
        
        # This assertion verifies the FIX. Currently it will FAIL if risk_penalty doesn't decrease.
        self.assertLess(self.brain.risk_penalty, current_risk, "Risk penalty should decrease on win (recovery)")

if __name__ == "__main__":
    unittest.main()
