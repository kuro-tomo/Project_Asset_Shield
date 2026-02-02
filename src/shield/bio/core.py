import time
import threading
import logging
from .repair import IntegrityMonitor
from .evolution import EvolutionEngine
from .replication import ReplicationManager

logger = logging.getLogger("Bio.Core")

class BioCore:
    """
    Biological Core (Life Support System)
    Orchestrates the three major biological functions:
    1. Self-Repair (Regeneration)
    2. Self-Evolution (Adaptation)
    3. Self-Replication (Propagation)
    """
    def __init__(self, check_interval=60):
        self.check_interval = check_interval
        self.running = False
        
        # Initialize Organ Systems
        self.repair_system = IntegrityMonitor()
        self.evolution_system = EvolutionEngine()
        self.replication_system = ReplicationManager()
        
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._life_cycle_loop, daemon=True)
        self.thread.start()
        logger.info("🟢 BIO-CORE ACTIVATED: Life support systems online.")

    def stop(self):
        self.running = False
        self.replication_system.kill_all()
        logger.info("🔴 BIO-CORE DEACTIVATED.")

    def _life_cycle_loop(self):
        """The heartbeat of the system."""
        while self.running:
            try:
                # 1. Regeneration Check
                is_healthy = self.repair_system.check_integrity()
                if not is_healthy:
                    logger.warning("🩹 System repaired itself during cycle.")

                # 2. Propagation Check
                self.replication_system.check_colony_health()

                # 3. Evolution Check (Placeholder for periodic mutation if needed)
                # Typically evolution is triggered by trade events, but we can do periodic drift
                # self.evolution_system.mutate(intensity=0.01) 

            except Exception as e:
                logger.error(f"⚠️ BIO-CORE EXCEPTION: {e}")
            
            time.sleep(self.check_interval)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    core = BioCore(check_interval=5)
    core.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        core.stop()
