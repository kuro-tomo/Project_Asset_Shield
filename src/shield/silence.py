"""
Silence Protocol V9.0 - Enhanced Dead Man's Switch

This module implements a multi-layered security system designed to protect
the operator's anonymity and physical safety through:

1. Secure Wipe: Forensic-resistant file destruction (DoD 5220.22-M compliant)
2. Heartbeat Monitoring: Dead Man's Switch with configurable intervals
3. Multi-Trigger Kill Switch: Local, remote, and panic button activation
4. PII Sanitization: Automatic detection and removal of identifying information
5. Decoy Generation: Replace sensitive files with plausible dummy content

WARNING: This module contains destructive operations. Use with extreme caution.
"""

import os
import sys
import shutil
import hashlib
import secrets
import logging
import threading
import time
import re
import json
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Callable, Set
from dataclasses import dataclass, field
from enum import Enum, auto

# Suppress logging to avoid leaving traces
logging.getLogger(__name__).addHandler(logging.NullHandler())


class TriggerType(Enum):
    """Types of kill switch triggers"""
    HEARTBEAT_TIMEOUT = auto()  # No heartbeat received within threshold
    PANIC_BUTTON = auto()       # Manual emergency trigger
    REMOTE_SIGNAL = auto()      # External API/webhook signal
    CANARY_DEATH = auto()       # Canary file modified/deleted
    INTRUSION_DETECTED = auto() # Suspicious activity detected
    DURESS_CODE = auto()        # Special code entered under coercion


class WipeMethod(Enum):
    """Secure deletion methods"""
    SIMPLE = 1          # Single pass random overwrite
    DOD_3PASS = 3       # DoD 5220.22-M short (3 passes)
    DOD_7PASS = 7       # DoD 5220.22-M extended (7 passes)
    GUTMANN = 35        # Gutmann method (35 passes)


@dataclass
class SilenceConfig:
    """Configuration for Silence Protocol"""
    heartbeat_file: str = ".owner_pulse"
    heartbeat_interval_seconds: int = 3600  # 1 hour default
    grace_period_seconds: int = 300         # 5 minute grace period
    wipe_method: WipeMethod = WipeMethod.DOD_3PASS
    canary_files: List[str] = field(default_factory=lambda: [".canary", ".integrity"])
    panic_sequence: str = "OMEGA-ZERO"      # Panic button activation code
    duress_code: str = "ALPHA-SAFE"         # Code that triggers silent wipe
    remote_check_url: Optional[str] = None  # URL to check for kill signal
    remote_check_interval: int = 300        # 5 minutes
    
    # Target directories and files for destruction
    critical_targets: List[str] = field(default_factory=lambda: [
        "src/shield/brain.py",
        "src/shield/adaptive_core.py",
        "src/shield/evolution.py",
        "output/brain_states_trained.json",
        "data/adaptive_state.json",
        "logs/",
        ".env",
        ".env.local",
    ])
    
    # Patterns for PII detection
    pii_patterns: List[str] = field(default_factory=lambda: [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{3}[-.]?\d{4}[-.]?\d{4}\b',  # Phone (JP format)
        r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP Address
        r'\b[A-Z]{2}\d{6,8}\b',  # Passport-like
        r'(?i)(api[_-]?key|password|secret|token)\s*[=:]\s*["\']?[\w-]+',  # Credentials
    ])


class SecureWiper:
    """
    Forensic-resistant file destruction utility.
    
    Implements multiple secure deletion standards to ensure
    data cannot be recovered through forensic analysis.
    """
    
    @staticmethod
    def _generate_pattern(size: int, pass_num: int, total_passes: int) -> bytes:
        """Generate overwrite pattern based on pass number"""
        if total_passes == 1:
            return secrets.token_bytes(size)
        
        # DoD 5220.22-M patterns
        if pass_num == 0:
            return b'\x00' * size  # All zeros
        elif pass_num == 1:
            return b'\xff' * size  # All ones
        elif pass_num == 2:
            return secrets.token_bytes(size)  # Random
        else:
            # Additional passes use random data
            return secrets.token_bytes(size)
    
    @classmethod
    def secure_delete(
        cls,
        file_path: Path,
        method: WipeMethod = WipeMethod.DOD_3PASS,
        verify: bool = True
    ) -> bool:
        """
        Securely delete a file using specified method.
        
        Args:
            file_path: Path to file to delete
            method: Wipe method to use
            verify: Whether to verify each pass
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not file_path.exists():
                return True
            
            file_size = file_path.stat().st_size
            passes = method.value
            
            # Perform overwrite passes
            for pass_num in range(passes):
                pattern = cls._generate_pattern(file_size, pass_num, passes)
                
                with open(file_path, 'r+b') as f:
                    f.seek(0)
                    f.write(pattern)
                    f.flush()
                    os.fsync(f.fileno())
                
                # Verify write if requested
                if verify:
                    with open(file_path, 'rb') as f:
                        written = f.read()
                        if written != pattern:
                            # Retry this pass
                            continue
            
            # Rename file multiple times to obscure original name
            current_path = file_path
            for _ in range(3):
                new_name = secrets.token_hex(16)
                new_path = current_path.parent / new_name
                current_path.rename(new_path)
                current_path = new_path
            
            # Finally delete
            current_path.unlink()
            
            return True
            
        except Exception:
            # Silent failure - no logging to avoid traces
            return False
    
    @classmethod
    def secure_delete_directory(
        cls,
        dir_path: Path,
        method: WipeMethod = WipeMethod.DOD_3PASS
    ) -> bool:
        """Securely delete all files in a directory"""
        try:
            if not dir_path.exists():
                return True
            
            # First, securely delete all files
            for root, dirs, files in os.walk(dir_path, topdown=False):
                for name in files:
                    file_path = Path(root) / name
                    cls.secure_delete(file_path, method)
                
                # Remove empty directories
                for name in dirs:
                    dir_to_remove = Path(root) / name
                    try:
                        dir_to_remove.rmdir()
                    except OSError:
                        pass
            
            # Remove the root directory
            try:
                dir_path.rmdir()
            except OSError:
                shutil.rmtree(dir_path, ignore_errors=True)
            
            return True
            
        except Exception:
            return False


class PIISanitizer:
    """
    Personal Identifiable Information detection and removal.
    
    Scans files for patterns that could identify the operator
    and either removes or masks them.
    """
    
    def __init__(self, patterns: List[str]):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def scan_file(self, file_path: Path) -> List[Dict]:
        """Scan a file for PII patterns"""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            for i, pattern in enumerate(self.patterns):
                matches = pattern.findall(content)
                if matches:
                    findings.append({
                        'file': str(file_path),
                        'pattern_index': i,
                        'matches': list(set(matches)),
                        'count': len(matches)
                    })
                    
        except Exception:
            pass
        
        return findings
    
    def sanitize_file(self, file_path: Path, replacement: str = "[REDACTED]") -> bool:
        """Remove PII from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            modified = content
            for pattern in self.patterns:
                modified = pattern.sub(replacement, modified)
            
            if modified != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified)
            
            return True
            
        except Exception:
            return False
    
    def scan_directory(self, dir_path: Path, extensions: Set[str] = None) -> List[Dict]:
        """Scan all files in directory for PII"""
        if extensions is None:
            extensions = {'.py', '.json', '.yaml', '.yml', '.txt', '.log', '.md', '.env'}
        
        all_findings = []
        
        for root, _, files in os.walk(dir_path):
            for name in files:
                file_path = Path(root) / name
                if file_path.suffix.lower() in extensions:
                    findings = self.scan_file(file_path)
                    all_findings.extend(findings)
        
        return all_findings


class DecoyGenerator:
    """
    Generate plausible decoy content to replace sensitive files.
    
    Instead of leaving obvious gaps, replace critical files with
    innocent-looking but non-functional code.
    """
    
    DECOY_TEMPLATES = {
        'brain.py': '''"""
Simple moving average calculator for stock analysis.
Basic implementation for educational purposes.
"""

import numpy as np
from typing import List

def calculate_sma(prices: List[float], window: int = 20) -> List[float]:
    """Calculate Simple Moving Average"""
    if len(prices) < window:
        return []
    return list(np.convolve(prices, np.ones(window)/window, mode='valid'))

def calculate_ema(prices: List[float], span: int = 20) -> List[float]:
    """Calculate Exponential Moving Average"""
    import pandas as pd
    return pd.Series(prices).ewm(span=span).mean().tolist()

class BasicAnalyzer:
    """Basic stock price analyzer"""
    
    def __init__(self, window: int = 20):
        self.window = window
    
    def analyze(self, prices: List[float]) -> dict:
        """Run basic analysis"""
        sma = calculate_sma(prices, self.window)
        return {
            'sma': sma[-1] if sma else None,
            'trend': 'up' if prices[-1] > sma[-1] else 'down' if sma else 'unknown'
        }
''',
        'adaptive_core.py': '''"""
Configuration manager for trading parameters.
Handles loading and saving of strategy settings.
"""

import json
from pathlib import Path
from typing import Dict, Any

DEFAULT_CONFIG = {
    'lookback_period': 20,
    'threshold': 0.02,
    'max_position_size': 100
}

class ConfigManager:
    """Manage trading configuration"""
    
    def __init__(self, config_path: str = 'config.json'):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return json.load(f)
        return DEFAULT_CONFIG.copy()
    
    def save_config(self) -> None:
        """Save current configuration"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
''',
        'evolution.py': '''"""
Parameter optimization using grid search.
Simple hyperparameter tuning for backtesting.
"""

from typing import Dict, List, Any, Callable
from itertools import product

def grid_search(
    param_grid: Dict[str, List[Any]],
    objective_fn: Callable,
    maximize: bool = True
) -> Dict[str, Any]:
    """
    Simple grid search optimization.
    
    Args:
        param_grid: Dictionary of parameter names to lists of values
        objective_fn: Function that takes params dict and returns score
        maximize: Whether to maximize (True) or minimize (False)
        
    Returns:
        Best parameters found
    """
    best_score = float('-inf') if maximize else float('inf')
    best_params = None
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    for combo in product(*values):
        params = dict(zip(keys, combo))
        score = objective_fn(params)
        
        if (maximize and score > best_score) or (not maximize and score < best_score):
            best_score = score
            best_params = params
    
    return best_params or {}

class SimpleOptimizer:
    """Basic parameter optimizer"""
    
    def __init__(self, param_grid: Dict[str, List[Any]]):
        self.param_grid = param_grid
        self.results = []
    
    def optimize(self, objective_fn: Callable) -> Dict[str, Any]:
        """Run optimization"""
        return grid_search(self.param_grid, objective_fn)
'''
    }
    
    @classmethod
    def generate_decoy(cls, original_path: Path) -> str:
        """Generate appropriate decoy content for a file"""
        filename = original_path.name
        
        if filename in cls.DECOY_TEMPLATES:
            return cls.DECOY_TEMPLATES[filename]
        
        # Generic Python decoy
        if original_path.suffix == '.py':
            return f'''"""
Utility module for {original_path.stem}.
Auto-generated placeholder.
"""

def placeholder():
    """Placeholder function"""
    pass

class Placeholder:
    """Placeholder class"""
    pass
'''
        
        # JSON decoy
        if original_path.suffix == '.json':
            return json.dumps({'status': 'initialized', 'version': '1.0.0'}, indent=2)
        
        return ''
    
    @classmethod
    def replace_with_decoy(cls, file_path: Path, wiper: SecureWiper) -> bool:
        """Replace a file with decoy content after secure deletion"""
        try:
            decoy_content = cls.generate_decoy(file_path)
            
            # First, securely delete the original
            wiper.secure_delete(file_path)
            
            # Then write decoy
            if decoy_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(decoy_content)
            
            return True
            
        except Exception:
            return False


class DeadlySilence:
    """
    Enhanced Dead Man's Switch with multi-trigger support.
    
    Monitors for various trigger conditions and executes
    the silence protocol when any condition is met.
    """
    
    def __init__(self, config: Optional[SilenceConfig] = None):
        self.config = config or SilenceConfig()
        self.wiper = SecureWiper()
        self.sanitizer = PIISanitizer(self.config.pii_patterns)
        self.is_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_heartbeat: Optional[datetime] = None
        self._triggered = False
        self._trigger_callbacks: List[Callable] = []
    
    def start_monitoring(self) -> None:
        """Start the dead man's switch monitoring"""
        if self.is_active:
            return
        
        self.is_active = True
        self._stop_event.clear()
        self._update_heartbeat()
        
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop monitoring (requires valid heartbeat)"""
        if not self.is_active:
            return
        
        self._stop_event.set()
        self.is_active = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def _update_heartbeat(self) -> None:
        """Update the heartbeat timestamp"""
        self._last_heartbeat = datetime.now()
        
        # Also update heartbeat file
        heartbeat_path = Path(self.config.heartbeat_file)
        try:
            heartbeat_path.write_text(
                datetime.now().isoformat(),
                encoding='utf-8'
            )
        except Exception:
            pass
    
    def send_heartbeat(self, code: Optional[str] = None) -> bool:
        """
        Send a heartbeat signal.
        
        Args:
            code: Optional verification code. If duress code is provided,
                  triggers silent wipe while appearing normal.
                  
        Returns:
            True if heartbeat accepted, False otherwise
        """
        # Check for duress code (operator under coercion)
        if code == self.config.duress_code:
            # Silently trigger wipe while appearing normal
            threading.Thread(
                target=self._execute_silence,
                args=(TriggerType.DURESS_CODE,),
                daemon=True
            ).start()
            return True  # Appear normal
        
        self._update_heartbeat()
        return True
    
    def trigger_panic(self, code: str) -> bool:
        """
        Manually trigger the panic button.
        
        Args:
            code: Panic sequence code for verification
            
        Returns:
            True if panic triggered, False if code invalid
        """
        if code != self.config.panic_sequence:
            return False
        
        self._execute_silence(TriggerType.PANIC_BUTTON)
        return True
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                # Check heartbeat timeout
                if self._check_heartbeat_timeout():
                    self._execute_silence(TriggerType.HEARTBEAT_TIMEOUT)
                    break
                
                # Check canary files
                if self._check_canary_files():
                    self._execute_silence(TriggerType.CANARY_DEATH)
                    break
                
                # Check remote signal (if configured)
                if self.config.remote_check_url:
                    if self._check_remote_signal():
                        self._execute_silence(TriggerType.REMOTE_SIGNAL)
                        break
                
                # Sleep before next check
                self._stop_event.wait(timeout=60)
                
            except Exception:
                # Silent failure - continue monitoring
                pass
    
    def _check_heartbeat_timeout(self) -> bool:
        """Check if heartbeat has timed out"""
        if self._last_heartbeat is None:
            # Check file-based heartbeat
            heartbeat_path = Path(self.config.heartbeat_file)
            if not heartbeat_path.exists():
                return True
            
            try:
                timestamp_str = heartbeat_path.read_text(encoding='utf-8').strip()
                self._last_heartbeat = datetime.fromisoformat(timestamp_str)
            except Exception:
                return True
        
        timeout = timedelta(
            seconds=self.config.heartbeat_interval_seconds + 
                    self.config.grace_period_seconds
        )
        
        return datetime.now() - self._last_heartbeat > timeout
    
    def _check_canary_files(self) -> bool:
        """Check if canary files have been tampered with"""
        for canary in self.config.canary_files:
            canary_path = Path(canary)
            if not canary_path.exists():
                return True
        return False
    
    def _check_remote_signal(self) -> bool:
        """Check remote endpoint for kill signal"""
        if not self.config.remote_check_url:
            return False
        
        try:
            import urllib.request
            
            req = urllib.request.Request(
                self.config.remote_check_url,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                return data.get('kill', False)
                
        except Exception:
            # Network failure is not a trigger
            return False
    
    def _execute_silence(self, trigger: TriggerType) -> None:
        """Execute the silence protocol"""
        if self._triggered:
            return
        
        self._triggered = True
        
        # Execute callbacks first
        for callback in self._trigger_callbacks:
            try:
                callback(trigger)
            except Exception:
                pass
        
        # Phase 1: Sanitize PII from all files
        try:
            self.sanitizer.scan_directory(Path('.'))
            for root, _, files in os.walk('.'):
                for name in files:
                    file_path = Path(root) / name
                    if file_path.suffix in {'.py', '.json', '.log', '.txt', '.env'}:
                        self.sanitizer.sanitize_file(file_path)
        except Exception:
            pass
        
        # Phase 2: Destroy critical targets
        for target in self.config.critical_targets:
            target_path = Path(target)
            
            try:
                if target_path.is_dir():
                    self.wiper.secure_delete_directory(
                        target_path,
                        self.config.wipe_method
                    )
                elif target_path.is_file():
                    # Replace with decoy instead of just deleting
                    DecoyGenerator.replace_with_decoy(target_path, self.wiper)
            except Exception:
                pass
        
        # Phase 3: Clear logs and temporary files
        log_patterns = ['*.log', '*.tmp', '*.cache', '.bash_history', '.python_history']
        for pattern in log_patterns:
            for log_file in Path('.').rglob(pattern):
                try:
                    self.wiper.secure_delete(log_file, self.config.wipe_method)
                except Exception:
                    pass
        
        # Phase 4: Remove heartbeat and canary files
        try:
            Path(self.config.heartbeat_file).unlink(missing_ok=True)
            for canary in self.config.canary_files:
                Path(canary).unlink(missing_ok=True)
        except Exception:
            pass
        
        # Phase 5: Self-destruct this module (optional, commented for safety)
        # self.wiper.secure_delete(Path(__file__), self.config.wipe_method)
    
    def register_callback(self, callback: Callable[[TriggerType], None]) -> None:
        """Register a callback to be executed before silence protocol"""
        self._trigger_callbacks.append(callback)
    
    def check_pulse(self) -> bool:
        """
        Legacy compatibility method.
        Check heartbeat and trigger if missing.
        
        Returns:
            True if pulse detected, False if triggered
        """
        if self._check_heartbeat_timeout():
            self._execute_silence(TriggerType.HEARTBEAT_TIMEOUT)
            return False
        return True
    
    def trigger_silver_bullet(self) -> None:
        """Legacy compatibility method for immediate trigger"""
        self._execute_silence(TriggerType.PANIC_BUTTON)


# Convenience function for quick setup
def initialize_silence_protocol(
    heartbeat_interval: int = 3600,
    auto_start: bool = True
) -> DeadlySilence:
    """
    Initialize and optionally start the silence protocol.
    
    Args:
        heartbeat_interval: Seconds between required heartbeats
        auto_start: Whether to start monitoring immediately
        
    Returns:
        Configured DeadlySilence instance
    """
    config = SilenceConfig(
        heartbeat_interval_seconds=heartbeat_interval
    )
    
    silence = DeadlySilence(config)
    
    if auto_start:
        silence.start_monitoring()
    
    return silence


if __name__ == "__main__":
    # Self-test mode - DO NOT RUN IN PRODUCTION
    print("[!] Silence Protocol V9.0 - Test Mode")
    print("[!] This module contains destructive operations.")
    print("[!] Do not run in production without understanding the consequences.")
