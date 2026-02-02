import unittest
import os
import shutil
import time
from src.shield.bio.core import BioCore
from src.shield.bio.repair import IntegrityMonitor
from src.shield.bio.evolution import EvolutionEngine

class TestBioSystem(unittest.TestCase):
    def setUp(self):
        self.test_manifest = "test_manifest.json"
        self.test_backup_dir = "test_backups"
        self.test_file = "test_vital_organ.txt"
        
        # Setup environment
        if not os.path.exists(self.test_backup_dir):
            os.makedirs(self.test_backup_dir)
        
        with open(self.test_file, "w") as f:
            f.write("ORIGINAL_DNA")

    def tearDown(self):
        if os.path.exists(self.test_manifest):
            os.remove(self.test_manifest)
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.test_backup_dir):
            shutil.rmtree(self.test_backup_dir)

    def test_repair_mechanism(self):
        monitor = IntegrityMonitor(manifest_path=self.test_manifest, backup_dir=self.test_backup_dir)
        monitor.critical_files = [self.test_file]
        
        # 1. Generate Manifest (Healthy State)
        monitor.generate_manifest()
        
        # 2. Corrupt the file
        with open(self.test_file, "w") as f:
            f.write("CORRUPTED_DNA")
            
        # 3. Check and Repair
        status = monitor.check_integrity()
        self.assertFalse(status) # Should detect corruption
        
        with open(self.test_file, "r") as f:
            content = f.read()
        self.assertEqual(content, "ORIGINAL_DNA") # Should be restored

    def test_evolution_mechanism(self):
        engine = EvolutionEngine(brain_memory_path="test_memory.json")
        initial_genes = engine._load_brain_state()
        
        # Mutate
        new_genes = engine.mutate(intensity=0.5)
        self.assertNotEqual(initial_genes["risk_penalty"], new_genes["risk_penalty"])
        
        if os.path.exists("test_memory.json"):
            os.remove("test_memory.json")

if __name__ == '__main__':
    unittest.main()
