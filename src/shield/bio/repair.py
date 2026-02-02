import os
import hashlib
import json
import shutil
import logging
from datetime import datetime

logger = logging.getLogger("Bio.Repair")

class IntegrityMonitor:
    """
    Self-Repair Module (Regeneration)
    Monitors critical system files and restores them if corruption is detected.
    """
    def __init__(self, manifest_path="data/dna_manifest.json", backup_dir="data/backups"):
        self.manifest_path = manifest_path
        self.backup_dir = backup_dir
        self.critical_files = [
            "src/shield/brain.py",
            "src/shield/nexus.py",
            "src/shield/bio/core.py"
        ]
        self._ensure_backup_dir()

    def _ensure_backup_dir(self):
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)

    def calculate_hash(self, filepath):
        """Calculates SHA256 hash of a file."""
        if not os.path.exists(filepath):
            return None
        sha256_hash = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {filepath}: {e}")
            return None

    def generate_manifest(self):
        """Generates a new manifest (DNA) from current state. Should be run on clean deploy."""
        manifest = {}
        for filepath in self.critical_files:
            file_hash = self.calculate_hash(filepath)
            if file_hash:
                manifest[filepath] = file_hash
                # Create a backup immediately
                self._backup_file(filepath)
        
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)
        logger.info(f"DNA Manifest generated at {self.manifest_path}")

    def _backup_file(self, filepath):
        """Backs up a valid file to the backup directory."""
        filename = os.path.basename(filepath)
        backup_path = os.path.join(self.backup_dir, filename)
        try:
            shutil.copy2(filepath, backup_path)
        except Exception as e:
            logger.error(f"Backup failed for {filepath}: {e}")

    def check_integrity(self):
        """Checks current files against manifest. Triggers repair if needed."""
        if not os.path.exists(self.manifest_path):
            logger.warning("No DNA Manifest found. Generating new one...")
            self.generate_manifest()
            return True

        with open(self.manifest_path, "r") as f:
            manifest = json.load(f)

        status = True
        for filepath, expected_hash in manifest.items():
            current_hash = self.calculate_hash(filepath)
            
            if current_hash is None:
                logger.error(f"MISSING ORGAN: {filepath}")
                self.repair_file(filepath)
                status = False
            elif current_hash != expected_hash:
                logger.error(f"DNA CORRUPTION DETECTED: {filepath}")
                self.repair_file(filepath)
                status = False
            else:
                pass # Healthy
        
        return status

    def repair_file(self, filepath):
        """Restores a file from backup."""
        filename = os.path.basename(filepath)
        backup_path = os.path.join(self.backup_dir, filename)
        
        if os.path.exists(backup_path):
            try:
                shutil.copy2(backup_path, filepath)
                logger.info(f"REGENERATION SUCCESSFUL: {filepath}")
            except Exception as e:
                logger.critical(f"REGENERATION FAILED for {filepath}: {e}")
        else:
            logger.critical(f"NO STEM CELL (BACKUP) FOUND for {filepath}. Cannot repair.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    monitor = IntegrityMonitor()
    monitor.check_integrity()
