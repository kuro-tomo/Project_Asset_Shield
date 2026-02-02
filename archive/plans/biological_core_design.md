# Biological Core Architecture Design (Shield BioCore)

## Overview
This document outlines the architecture for incorporating the "Three Major Functions" (Self-Repair, Self-Evolution, Self-Replication) into Project Asset Shield. These functions will be encapsulated in a new module `src/shield/bio/`.

## 1. Self-Repair (自己再生 - Regeneration)
**Goal:** Ensure system integrity by automatically detecting and fixing corrupted or missing components.

### Mechanism: `IntegrityMonitor`
- **Manifest:** A JSON file (`dna_manifest.json`) storing SHA256 hashes of critical source files (e.g., `brain.py`, `nexus.py`).
- **Surveillance:** A background thread checks current file hashes against the manifest every N minutes.
- **Action:**
    - If a file is missing or hash mismatch:
        1. Log the "Injury" (corruption).
        2. Retrieve the "Stem Cell" (backup content) from a secure archive or the Manifest itself (if small enough) or a backup directory.
        3. Overwrite/Restore the file.
        4. Restart the affected service if necessary.

### Structure
- `src/shield/bio/repair.py`
    - `class IntegrityMonitor`
    - `generate_manifest()`
    - `check_integrity()`
    - `repair_file()`

## 2. Self-Evolution (自己進化 - Evolution)
**Goal:** Improve performance over time without human intervention.

### Mechanism: `EvolutionEngine` (Enhanced)
- **Feedback Loop:**
    - Monitors `audit_global.db` or `brain_memory.json` for performance metrics (PnL, Win Rate, Drawdown).
- **Triggers:**
    - **Stagnation:** No profit for X ticks -> Trigger Mutation.
    - **Crisis:** Sharp drawdown -> Trigger Adaptation (Defensive Mode).
- **Mutation:**
    - Randomly adjusts `risk_penalty`, `adaptive_threshold`, or `lookback_period` within safe bounds.
    - Creates a "generation" version (e.g., Brain v1.1).
- **Selection:**
    - If the new generation performs better, it becomes the standard. If worse, it rolls back.

### Structure
- `src/shield/bio/evolution.py` (Refactor of existing `evolution.py`)
    - `class EvolutionEngine`
    - `assess_fitness()`
    - `mutate_dna()`

## 3. Self-Replication (自己増殖 - Propagation)
**Goal:** Expand processing power and survivability by spawning new instances.

### Mechanism: `ReplicationManager`
- **Node Discovery:**
    - Master node maintains a registry of active Agents.
- **Spawning:**
    - If CPU load is high OR active agents < Target (e.g., 3):
        - **Local:** Spawn a new Python process (`multiprocessing` or `subprocess`) running `scripts/nodes/tokyo_predator.py` or similar.
        - **Remote (Future):** SSH into another server and deploy a Docker container.
- **Resurrection:**
    - If an Agent stops sending heartbeats, the Manager declares it "Dead" and spawns a replacement.

### Structure
- `src/shield/bio/replication.py`
    - `class ReplicationManager`
    - `spawn_agent()`
    - `check_colony_health()`

## Integration
These modules will be orchestrated by `src/shield/bio/core.py` which initializes them on system startup.

```python
# src/shield/bio/core.py

class BioCore:
    def __init__(self):
        self.repair = IntegrityMonitor()
        self.evolution = EvolutionEngine()
        self.replication = ReplicationManager()

    def start_life_cycle(self):
        # Start background threads for each biological function
        pass
```
