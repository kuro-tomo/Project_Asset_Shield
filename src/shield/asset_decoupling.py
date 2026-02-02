"""
Asset Decoupling Protocol for Asset Shield V2
Version-J (Transferable) / Version-Q (IP Reserved) Separation

Implements secure separation of:
- Version-J: Stable production model for white-label transfer
- Version-Q: Next-gen core logic and experimental weights (encrypted, retained)
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import base64

# Optional cryptography support
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logging.warning("cryptography package not available. Encryption features disabled.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AssetVersion(Enum):
    """Asset version classification"""
    VERSION_J = "version_j"  # Transferable (譲渡対象)
    VERSION_Q = "version_q"  # IP Reserved (非譲渡)


class AssetType(Enum):
    """Asset type classification"""
    SOURCE_CODE = "source_code"
    MODEL_WEIGHTS = "model_weights"
    CONFIGURATION = "configuration"
    DOCUMENTATION = "documentation"
    DATA = "data"
    INFRASTRUCTURE = "infrastructure"


@dataclass
class AssetManifestEntry:
    """Single asset entry in the manifest"""
    asset_id: str
    name: str
    path: str
    asset_type: AssetType
    version: AssetVersion
    sha256_hash: str
    size_bytes: int
    encrypted: bool = False
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            "asset_type": self.asset_type.value,
            "version": self.version.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AssetManifestEntry':
        data["asset_type"] = AssetType(data["asset_type"])
        data["version"] = AssetVersion(data["version"])
        return cls(**data)


@dataclass
class AssetManifest:
    """Complete asset manifest for audit"""
    manifest_id: str
    project_name: str = "Asset Shield V2"
    version: str = "2.1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    entries: List[AssetManifestEntry] = field(default_factory=list)
    integrity_hash: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "manifest_id": self.manifest_id,
            "project_name": self.project_name,
            "version": self.version,
            "created_at": self.created_at,
            "entries": [e.to_dict() for e in self.entries],
            "integrity_hash": self.integrity_hash
        }
    
    def calculate_integrity_hash(self) -> str:
        """Calculate SHA-256 hash of entire manifest"""
        content = json.dumps(
            [e.to_dict() for e in self.entries],
            sort_keys=True
        )
        return hashlib.sha256(content.encode()).hexdigest()


class EncryptionManager:
    """
    Encryption manager for Version-Q assets.
    Uses Fernet symmetric encryption with PBKDF2 key derivation.
    """
    
    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize encryption manager.
        
        Args:
            master_key: Master encryption key (or from ASSET_SHIELD_KEY env var)
        """
        self.master_key = master_key or os.environ.get("ASSET_SHIELD_KEY", "")
        self._fernet: Optional[Any] = None
        
        if CRYPTO_AVAILABLE and self.master_key:
            self._initialize_fernet()
    
    def _initialize_fernet(self) -> None:
        """Initialize Fernet cipher with derived key"""
        salt = b'asset_shield_v2_salt'  # In production, use random salt stored securely
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
        self._fernet = Fernet(key)
        
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data"""
        if not self._fernet:
            raise RuntimeError("Encryption not available. Set ASSET_SHIELD_KEY.")
        return self._fernet.encrypt(data)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data"""
        if not self._fernet:
            raise RuntimeError("Decryption not available. Set ASSET_SHIELD_KEY.")
        return self._fernet.decrypt(encrypted_data)
    
    def encrypt_file(self, input_path: str, output_path: str) -> bool:
        """Encrypt a file"""
        try:
            with open(input_path, 'rb') as f:
                data = f.read()
            encrypted = self.encrypt(data)
            with open(output_path, 'wb') as f:
                f.write(encrypted)
            return True
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return False
    
    def decrypt_file(self, input_path: str, output_path: str) -> bool:
        """Decrypt a file"""
        try:
            with open(input_path, 'rb') as f:
                encrypted = f.read()
            data = self.decrypt(encrypted)
            with open(output_path, 'wb') as f:
                f.write(data)
            return True
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return False


class AssetDecouplingManager:
    """
    Asset Decoupling Manager
    
    Manages separation of Version-J (transferable) and Version-Q (retained) assets.
    """
    
    # Default classification rules
    VERSION_J_PATTERNS = [
        # Transferable production code
        "modules/jquants_client.py",
        "modules/itayose_analyzer.py",
        "modules/execution_core.py",
        "modules/screener.py",
        "modules/money_management.py",
        "modules/tracker.py",
        "main_execution.py",
        "requirements.txt",
        "config/*",
        "dashboard/*",
        "*.md",
    ]
    
    VERSION_Q_PATTERNS = [
        # Retained IP
        "modules/adaptive_core.py",  # Core adaptive logic
        "modules/brain.py",          # Learning core
        "modules/evolution.py",      # Evolution engine
        "modules/nexus*.py",         # Nexus processor
        "*.weights",                 # Model weights
        "*.pkl",                     # Pickled models
        "memory_*.json",             # Learned parameters
        "adaptive_state.json",       # Adaptive state
    ]
    
    def __init__(
        self,
        project_root: str = ".",
        manifest_path: str = "asset_manifest.json",
        encryption_manager: Optional[EncryptionManager] = None
    ):
        """
        Initialize Asset Decoupling Manager.
        
        Args:
            project_root: Root directory of the project
            manifest_path: Path to save/load manifest
            encryption_manager: Optional encryption manager for Version-Q
        """
        self.project_root = Path(project_root)
        self.manifest_path = manifest_path
        self.encryption = encryption_manager or EncryptionManager()
        self.manifest: Optional[AssetManifest] = None
        
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _classify_asset(self, file_path: str) -> AssetVersion:
        """Classify asset as Version-J or Version-Q"""
        from fnmatch import fnmatch
        
        rel_path = str(Path(file_path).relative_to(self.project_root))
        
        # Check Version-Q patterns first (higher priority)
        for pattern in self.VERSION_Q_PATTERNS:
            if fnmatch(rel_path, pattern):
                return AssetVersion.VERSION_Q
        
        # Check Version-J patterns
        for pattern in self.VERSION_J_PATTERNS:
            if fnmatch(rel_path, pattern):
                return AssetVersion.VERSION_J
        
        # Default to Version-Q (conservative approach)
        return AssetVersion.VERSION_Q
    
    def _determine_asset_type(self, file_path: str) -> AssetType:
        """Determine asset type from file extension"""
        ext = Path(file_path).suffix.lower()
        
        if ext in ['.py', '.rs', '.js', '.ts']:
            return AssetType.SOURCE_CODE
        elif ext in ['.weights', '.pkl', '.pt', '.onnx', '.h5']:
            return AssetType.MODEL_WEIGHTS
        elif ext in ['.json', '.yaml', '.yml', '.toml', '.ini']:
            return AssetType.CONFIGURATION
        elif ext in ['.md', '.txt', '.rst', '.pdf']:
            return AssetType.DOCUMENTATION
        elif ext in ['.csv', '.parquet', '.db', '.sqlite']:
            return AssetType.DATA
        elif ext in ['.tf', '.dockerfile', '.sh']:
            return AssetType.INFRASTRUCTURE
        else:
            return AssetType.SOURCE_CODE
    
    def scan_project(self, exclude_patterns: Optional[List[str]] = None) -> AssetManifest:
        """
        Scan project and create asset manifest.
        
        Args:
            exclude_patterns: Patterns to exclude from scan
            
        Returns:
            AssetManifest with all classified assets
        """
        exclude = exclude_patterns or [
            '__pycache__',
            '.git',
            '.venv',
            'venv',
            'node_modules',
            '*.pyc',
            '.DS_Store'
        ]
        
        manifest_id = f"MANIFEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        manifest = AssetManifest(manifest_id=manifest_id)
        
        from fnmatch import fnmatch
        
        for root, dirs, files in os.walk(self.project_root):
            # Filter excluded directories
            dirs[:] = [d for d in dirs if not any(fnmatch(d, p) for p in exclude)]
            
            for file in files:
                # Skip excluded files
                if any(fnmatch(file, p) for p in exclude):
                    continue
                    
                file_path = Path(root) / file
                
                try:
                    rel_path = str(file_path.relative_to(self.project_root))
                    version = self._classify_asset(str(file_path))
                    asset_type = self._determine_asset_type(str(file_path))
                    file_hash = self._calculate_file_hash(str(file_path))
                    file_size = file_path.stat().st_size
                    
                    entry = AssetManifestEntry(
                        asset_id=hashlib.md5(rel_path.encode()).hexdigest()[:12],
                        name=file,
                        path=rel_path,
                        asset_type=asset_type,
                        version=version,
                        sha256_hash=file_hash,
                        size_bytes=file_size
                    )
                    manifest.entries.append(entry)
                    
                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")
        
        manifest.integrity_hash = manifest.calculate_integrity_hash()
        self.manifest = manifest
        
        logger.info(f"Scanned {len(manifest.entries)} assets")
        return manifest
    
    def export_version_j(self, output_dir: str) -> Dict[str, Any]:
        """
        Export Version-J assets for transfer.
        
        Args:
            output_dir: Directory to export assets
            
        Returns:
            Export summary
        """
        if not self.manifest:
            self.scan_project()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported = []
        skipped = []
        
        for entry in self.manifest.entries:
            if entry.version == AssetVersion.VERSION_J:
                src = self.project_root / entry.path
                dst = output_path / entry.path
                
                try:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file
                    import shutil
                    shutil.copy2(src, dst)
                    exported.append(entry.path)
                    
                except Exception as e:
                    logger.error(f"Failed to export {entry.path}: {e}")
                    skipped.append(entry.path)
            else:
                skipped.append(entry.path)
        
        # Create export manifest
        export_manifest = {
            "export_id": f"EXPORT_J_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "source_manifest": self.manifest.manifest_id,
            "exported_count": len(exported),
            "exported_files": exported,
            "skipped_count": len(skipped),
            "timestamp": datetime.now().isoformat()
        }
        
        manifest_file = output_path / "EXPORT_MANIFEST.json"
        with open(manifest_file, 'w') as f:
            json.dump(export_manifest, f, indent=2)
        
        logger.info(f"Exported {len(exported)} Version-J assets to {output_dir}")
        return export_manifest
    
    def secure_version_q(self, output_dir: str) -> Dict[str, Any]:
        """
        Secure Version-Q assets with encryption.
        
        Args:
            output_dir: Directory to store encrypted assets
            
        Returns:
            Security summary
        """
        if not self.manifest:
            self.scan_project()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        secured = []
        failed = []
        
        for entry in self.manifest.entries:
            if entry.version == AssetVersion.VERSION_Q:
                src = self.project_root / entry.path
                
                if CRYPTO_AVAILABLE and self.encryption.master_key:
                    # Encrypt the file
                    dst = output_path / f"{entry.path}.encrypted"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    
                    if self.encryption.encrypt_file(str(src), str(dst)):
                        entry.encrypted = True
                        secured.append(entry.path)
                    else:
                        failed.append(entry.path)
                else:
                    # Just track without encryption
                    secured.append(entry.path)
                    logger.warning(f"Encryption unavailable for {entry.path}")
        
        # Create security manifest
        security_manifest = {
            "security_id": f"SECURE_Q_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "source_manifest": self.manifest.manifest_id,
            "secured_count": len(secured),
            "secured_files": secured,
            "encrypted": CRYPTO_AVAILABLE and bool(self.encryption.master_key),
            "failed_count": len(failed),
            "timestamp": datetime.now().isoformat()
        }
        
        manifest_file = output_path / "SECURITY_MANIFEST.json"
        with open(manifest_file, 'w') as f:
            json.dump(security_manifest, f, indent=2)
        
        logger.info(f"Secured {len(secured)} Version-Q assets")
        return security_manifest
    
    def generate_audit_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive audit report for due diligence.
        
        Returns:
            Audit report dictionary
        """
        if not self.manifest:
            self.scan_project()
        
        version_j_assets = [e for e in self.manifest.entries if e.version == AssetVersion.VERSION_J]
        version_q_assets = [e for e in self.manifest.entries if e.version == AssetVersion.VERSION_Q]
        
        # Calculate totals by type
        type_breakdown = {}
        for entry in self.manifest.entries:
            type_name = entry.asset_type.value
            if type_name not in type_breakdown:
                type_breakdown[type_name] = {"version_j": 0, "version_q": 0, "total_bytes": 0}
            
            if entry.version == AssetVersion.VERSION_J:
                type_breakdown[type_name]["version_j"] += 1
            else:
                type_breakdown[type_name]["version_q"] += 1
            type_breakdown[type_name]["total_bytes"] += entry.size_bytes
        
        report = {
            "report_id": f"AUDIT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "project_name": self.manifest.project_name,
            "project_version": self.manifest.version,
            "manifest_integrity_hash": self.manifest.integrity_hash,
            "summary": {
                "total_assets": len(self.manifest.entries),
                "version_j_count": len(version_j_assets),
                "version_q_count": len(version_q_assets),
                "version_j_bytes": sum(e.size_bytes for e in version_j_assets),
                "version_q_bytes": sum(e.size_bytes for e in version_q_assets),
            },
            "type_breakdown": type_breakdown,
            "version_j_assets": [e.to_dict() for e in version_j_assets],
            "version_q_assets": [
                {
                    "asset_id": e.asset_id,
                    "name": e.name,
                    "asset_type": e.asset_type.value,
                    "encrypted": e.encrypted,
                    # Redact path and hash for security
                    "path": "[REDACTED]",
                    "sha256_hash": "[REDACTED]"
                }
                for e in version_q_assets
            ],
            "generated_at": datetime.now().isoformat(),
            "disclaimer": "Version-Q assets are intellectual property reserved by the originator."
        }
        
        return report
    
    def save_manifest(self, path: Optional[str] = None) -> None:
        """Save manifest to file"""
        if not self.manifest:
            raise ValueError("No manifest to save. Run scan_project() first.")
        
        save_path = path or self.manifest_path
        with open(save_path, 'w') as f:
            json.dump(self.manifest.to_dict(), f, indent=2)
        logger.info(f"Manifest saved to {save_path}")
    
    def load_manifest(self, path: Optional[str] = None) -> AssetManifest:
        """Load manifest from file"""
        load_path = path or self.manifest_path
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        entries = [AssetManifestEntry.from_dict(e) for e in data.get("entries", [])]
        self.manifest = AssetManifest(
            manifest_id=data["manifest_id"],
            project_name=data.get("project_name", "Asset Shield V2"),
            version=data.get("version", "2.1.0"),
            created_at=data.get("created_at", ""),
            entries=entries,
            integrity_hash=data.get("integrity_hash", "")
        )
        
        return self.manifest
    
    def verify_integrity(self) -> Tuple[bool, List[str]]:
        """
        Verify integrity of all assets against manifest.
        
        Returns:
            Tuple of (all_valid, list_of_mismatches)
        """
        if not self.manifest:
            raise ValueError("No manifest loaded. Run scan_project() or load_manifest() first.")
        
        mismatches = []
        
        for entry in self.manifest.entries:
            file_path = self.project_root / entry.path
            
            if not file_path.exists():
                mismatches.append(f"MISSING: {entry.path}")
                continue
            
            current_hash = self._calculate_file_hash(str(file_path))
            if current_hash != entry.sha256_hash:
                mismatches.append(f"MODIFIED: {entry.path}")
        
        return len(mismatches) == 0, mismatches


if __name__ == "__main__":
    # Test asset decoupling
    manager = AssetDecouplingManager(project_root=".")
    
    # Scan project
    print("=== Scanning Project ===")
    manifest = manager.scan_project()
    
    print(f"Total Assets: {len(manifest.entries)}")
    print(f"Manifest ID: {manifest.manifest_id}")
    print(f"Integrity Hash: {manifest.integrity_hash[:16]}...")
    
    # Generate audit report
    print("\n=== Audit Report ===")
    report = manager.generate_audit_report()
    
    print(f"Version-J Assets: {report['summary']['version_j_count']}")
    print(f"Version-Q Assets: {report['summary']['version_q_count']}")
    
    print("\nType Breakdown:")
    for type_name, counts in report['type_breakdown'].items():
        print(f"  {type_name}: J={counts['version_j']}, Q={counts['version_q']}")
    
    # Save manifest
    manager.save_manifest("asset_manifest.json")
    print("\nManifest saved to asset_manifest.json")
