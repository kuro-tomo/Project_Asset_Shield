import subprocess
import os
import psutil
import logging
import time

logger = logging.getLogger("Bio.Replication")

class ReplicationManager:
    """
    Self-Replication Module (Propagation)
    Manages the population of active agent nodes. Spawns new instances to maintain colony health.
    """
    def __init__(self, target_population=2, agent_script="scripts/nodes/tokyo_inference.py"):
        self.target_population = target_population
        self.agent_script = agent_script
        self.active_agents = []

    def check_colony_health(self):
        """Checks if the population is at target levels. Spawns/Kills as needed."""
        # Prune dead processes
        self.active_agents = [p for p in self.active_agents if p.poll() is None]
        
        current_pop = len(self.active_agents)
        logger.info(f"🦠 COLONY STATUS: {current_pop}/{self.target_population} active agents")

        if current_pop < self.target_population:
            deficit = self.target_population - current_pop
            logger.warning(f"⚠️ POPULATION CRITICAL: Spawning {deficit} new agents...")
            for _ in range(deficit):
                self.spawn_agent()
        
    def spawn_agent(self):
        """Spawns a new agent process (Mitosis)."""
        try:
            # Launch detached process
            process = subprocess.Popen(
                ["python3", self.agent_script],
                cwd=os.getcwd(),
                stdout=subprocess.DEVNULL, # Silence output for background workers
                stderr=subprocess.DEVNULL
            )
            self.active_agents.append(process)
            logger.info(f"✅ SPAWNED NEW AGENT (PID: {process.pid})")
            return True
        except Exception as e:
            logger.error(f"❌ MITOSIS FAILED: {e}")
            return False

    def kill_all(self):
        """Apoptosis: Terminates all spawned agents."""
        logger.warning("☠️ TRIGGERING APOPTOSIS (Kill All)")
        for p in self.active_agents:
            if p.poll() is None:
                p.terminate()
        self.active_agents = []

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = ReplicationManager(target_population=1)
    manager.check_colony_health()
    time.sleep(5)
    manager.kill_all()
