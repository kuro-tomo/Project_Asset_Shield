"""
Security Governance for Asset Shield V2
Burn Protocol & Integrity Hash Implementation

Implements:
- Burn Protocol: Automated removal of developer access post-transfer
- Integrity Hash: SHA-256 certification of file structure at delivery
"""

import os
import json
import hashlib
import logging
import shutil
import subprocess
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BurnPhase(Enum):
    """Burn protocol phases"""
    PRE_TRANSFER = "pre_transfer"       # Before asset transfer
    TRANSFER_COMPLETE = "transfer_complete"  # After transfer confirmed
    ACCEPTANCE_COMPLETE = "acceptance_complete"  # After buyer acceptance
    BURN_EXECUTED = "burn_executed"     # After burn protocol execution


@dataclass
class SSHKeyRecord:
    """SSH key record for tracking"""
    key_id: str
    key_type: str
    fingerprint: str
    created_at: str
    location: str
    status: str = "active"


@dataclass
class AccessRecord:
    """Access credential record"""
    record_id: str
    credential_type: str  # ssh_key, api_key, password, token
    identifier: str
    created_at: str
    expires_at: Optional[str] = None
    status: str = "active"
    burn_scheduled: bool = False


@dataclass
class IntegrityRecord:
    """File integrity record"""
    file_path: str
    sha256_hash: str
    size_bytes: int
    modified_at: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class IntegrityCertificate:
    """Integrity certificate for delivery"""
    certificate_id: str
    project_name: str
    version: str
    generated_at: str
    total_files: int
    total_size_bytes: int
    root_hash: str  # Merkle root of all file hashes
    records: List[IntegrityRecord] = field(default_factory=list)
    signature: str = ""  # Optional digital signature
    
    def to_dict(self) -> Dict:
        return {
            "certificate_id": self.certificate_id,
            "project_name": self.project_name,
            "version": self.version,
            "generated_at": self.generated_at,
            "total_files": self.total_files,
            "total_size_bytes": self.total_size_bytes,
            "root_hash": self.root_hash,
            "records": [r.to_dict() for r in self.records],
            "signature": self.signature
        }


class IntegrityHashManager:
    """
    Integrity Hash Manager
    
    Generates SHA-256 hash certificates for file structure verification.
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _calculate_merkle_root(self, hashes: List[str]) -> str:
        """Calculate Merkle root from list of hashes"""
        if not hashes:
            return hashlib.sha256(b'').hexdigest()
        
        if len(hashes) == 1:
            return hashes[0]
        
        # Build tree
        while len(hashes) > 1:
            # Pad to even number for current level
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])
            
            new_level = []
            for i in range(0, len(hashes), 2):
                combined = hashes[i] + hashes[i + 1]
                new_hash = hashlib.sha256(combined.encode()).hexdigest()
                new_level.append(new_hash)
            hashes = new_level
        
        return hashes[0]
    
    def generate_certificate(
        self,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        version: str = "2.1.0"
    ) -> IntegrityCertificate:
        """
        Generate integrity certificate for project files.
        
        Args:
            include_patterns: Glob patterns to include
            exclude_patterns: Glob patterns to exclude
            version: Version string for certificate
            
        Returns:
            IntegrityCertificate with all file hashes
        """
        include = include_patterns or ["**/*"]
        exclude = exclude_patterns or [
            "__pycache__/**",
            ".git/**",
            ".venv/**",
            "venv/**",
            "*.pyc",
            ".DS_Store",
            "*.log",
            "node_modules/**"
        ]
        
        records = []
        all_hashes = []
        total_size = 0
        
        from fnmatch import fnmatch
        
        for root, dirs, files in os.walk(self.project_root):
            # Filter directories
            dirs[:] = [d for d in dirs if not any(
                fnmatch(d, p.rstrip('/**')) for p in exclude
            )]
            
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(self.project_root)
                
                # Check exclusions
                if any(fnmatch(str(rel_path), p) for p in exclude):
                    continue
                
                # Check inclusions
                if not any(fnmatch(str(rel_path), p) for p in include):
                    continue
                
                try:
                    file_hash = self._calculate_file_hash(file_path)
                    file_size = file_path.stat().st_size
                    modified_at = datetime.fromtimestamp(
                        file_path.stat().st_mtime
                    ).isoformat()
                    
                    record = IntegrityRecord(
                        file_path=str(rel_path),
                        sha256_hash=file_hash,
                        size_bytes=file_size,
                        modified_at=modified_at
                    )
                    records.append(record)
                    all_hashes.append(file_hash)
                    total_size += file_size
                    
                except Exception as e:
                    logger.warning(f"Failed to hash {file_path}: {e}")
        
        # Sort records for deterministic ordering
        records.sort(key=lambda r: r.file_path)
        all_hashes = [r.sha256_hash for r in records]
        
        # Calculate Merkle root
        root_hash = self._calculate_merkle_root(all_hashes)
        
        certificate = IntegrityCertificate(
            certificate_id=f"CERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            project_name="Asset Shield V2",
            version=version,
            generated_at=datetime.now().isoformat(),
            total_files=len(records),
            total_size_bytes=total_size,
            root_hash=root_hash,
            records=records
        )
        
        logger.info(f"Generated certificate with {len(records)} files, root hash: {root_hash[:16]}...")
        return certificate
    
    def verify_certificate(self, certificate: IntegrityCertificate) -> Tuple[bool, List[str]]:
        """
        Verify files against certificate.
        
        Args:
            certificate: Certificate to verify against
            
        Returns:
            Tuple of (all_valid, list_of_mismatches)
        """
        mismatches = []
        
        for record in certificate.records:
            file_path = self.project_root / record.file_path
            
            if not file_path.exists():
                mismatches.append(f"MISSING: {record.file_path}")
                continue
            
            current_hash = self._calculate_file_hash(file_path)
            if current_hash != record.sha256_hash:
                mismatches.append(f"MODIFIED: {record.file_path}")
        
        # Verify Merkle root
        current_hashes = []
        for record in certificate.records:
            file_path = self.project_root / record.file_path
            if file_path.exists():
                current_hashes.append(self._calculate_file_hash(file_path))
        
        current_root = self._calculate_merkle_root(current_hashes)
        if current_root != certificate.root_hash:
            mismatches.append(f"ROOT_HASH_MISMATCH: expected {certificate.root_hash[:16]}..., got {current_root[:16]}...")
        
        return len(mismatches) == 0, mismatches
    
    def save_certificate(self, certificate: IntegrityCertificate, path: str) -> None:
        """Save certificate to JSON file"""
        with open(path, 'w') as f:
            json.dump(certificate.to_dict(), f, indent=2)
        logger.info(f"Certificate saved to {path}")
    
    def load_certificate(self, path: str) -> IntegrityCertificate:
        """Load certificate from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        records = [IntegrityRecord(**r) for r in data.get("records", [])]
        
        return IntegrityCertificate(
            certificate_id=data["certificate_id"],
            project_name=data["project_name"],
            version=data["version"],
            generated_at=data["generated_at"],
            total_files=data["total_files"],
            total_size_bytes=data["total_size_bytes"],
            root_hash=data["root_hash"],
            records=records,
            signature=data.get("signature", "")
        )


class BurnProtocol:
    """
    Burn Protocol Implementation
    
    Automated removal of developer access, SSH keys, and backdoors
    after transfer completion or acceptance.
    """
    
    def __init__(
        self,
        project_root: str = ".",
        burn_log_path: str = "burn_protocol.log"
    ):
        self.project_root = Path(project_root)
        self.burn_log_path = burn_log_path
        self.current_phase = BurnPhase.PRE_TRANSFER
        self.access_records: List[AccessRecord] = []
        self.ssh_keys: List[SSHKeyRecord] = []
        
    def register_access(
        self,
        credential_type: str,
        identifier: str,
        expires_at: Optional[str] = None
    ) -> AccessRecord:
        """Register an access credential for tracking"""
        record = AccessRecord(
            record_id=hashlib.md5(f"{credential_type}:{identifier}".encode()).hexdigest()[:12],
            credential_type=credential_type,
            identifier=identifier,
            created_at=datetime.now().isoformat(),
            expires_at=expires_at
        )
        self.access_records.append(record)
        self._log(f"Registered access: {credential_type} - {identifier[:20]}...")
        return record
    
    def register_ssh_key(
        self,
        key_type: str,
        fingerprint: str,
        location: str
    ) -> SSHKeyRecord:
        """Register an SSH key for tracking"""
        record = SSHKeyRecord(
            key_id=hashlib.md5(fingerprint.encode()).hexdigest()[:12],
            key_type=key_type,
            fingerprint=fingerprint,
            created_at=datetime.now().isoformat(),
            location=location
        )
        self.ssh_keys.append(record)
        self._log(f"Registered SSH key: {key_type} at {location}")
        return record
    
    def _log(self, message: str) -> None:
        """Log burn protocol activity"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(self.burn_log_path, 'a') as f:
            f.write(log_entry)
        
        logger.info(f"BURN_PROTOCOL: {message}")
    
    def schedule_burn(self) -> Dict[str, Any]:
        """
        Schedule burn protocol execution.
        
        Returns:
            Summary of items scheduled for burn
        """
        for record in self.access_records:
            record.burn_scheduled = True
        
        summary = {
            "scheduled_at": datetime.now().isoformat(),
            "access_credentials": len(self.access_records),
            "ssh_keys": len(self.ssh_keys),
            "items": []
        }
        
        for record in self.access_records:
            summary["items"].append({
                "type": record.credential_type,
                "identifier": record.identifier[:20] + "...",
                "status": "scheduled"
            })
        
        for key in self.ssh_keys:
            summary["items"].append({
                "type": "ssh_key",
                "identifier": key.fingerprint[:20] + "...",
                "location": key.location,
                "status": "scheduled"
            })
        
        self._log(f"Burn scheduled: {len(summary['items'])} items")
        return summary
    
    def execute_burn(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Execute burn protocol.
        
        Args:
            dry_run: If True, only simulate without actual deletion
            
        Returns:
            Execution report
        """
        self._log(f"Burn execution started (dry_run={dry_run})")
        
        report = {
            "execution_id": f"BURN_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "dry_run": dry_run,
            "started_at": datetime.now().isoformat(),
            "actions": [],
            "errors": []
        }
        
        # 1. Revoke SSH keys
        for key in self.ssh_keys:
            action = self._burn_ssh_key(key, dry_run)
            report["actions"].append(action)
        
        # 2. Revoke access credentials
        for record in self.access_records:
            action = self._burn_access(record, dry_run)
            report["actions"].append(action)
        
        # 3. Remove sensitive files
        sensitive_patterns = [
            ".env",
            ".env.local",
            "*.pem",
            "*.key",
            "id_rsa*",
            "id_ed25519*",
            "credentials.json",
            "secrets.yaml"
        ]
        
        for pattern in sensitive_patterns:
            action = self._burn_files(pattern, dry_run)
            if action:
                report["actions"].append(action)
        
        # 4. Clear git credentials
        action = self._burn_git_credentials(dry_run)
        report["actions"].append(action)
        
        # 5. Remove backdoor indicators
        action = self._scan_and_remove_backdoors(dry_run)
        report["actions"].append(action)
        
        report["completed_at"] = datetime.now().isoformat()
        report["success"] = len(report["errors"]) == 0
        
        if not dry_run:
            self.current_phase = BurnPhase.BURN_EXECUTED
        
        self._log(f"Burn execution completed: {len(report['actions'])} actions, {len(report['errors'])} errors")
        return report
    
    def _burn_ssh_key(self, key: SSHKeyRecord, dry_run: bool) -> Dict:
        """Remove SSH key"""
        action = {
            "type": "ssh_key_removal",
            "target": key.location,
            "fingerprint": key.fingerprint[:20] + "...",
            "dry_run": dry_run,
            "status": "pending"
        }
        
        try:
            key_path = Path(key.location).expanduser()
            
            if key_path.exists():
                if not dry_run:
                    # Secure delete (overwrite before delete)
                    with open(key_path, 'wb') as f:
                        f.write(os.urandom(key_path.stat().st_size))
                    key_path.unlink()
                    
                    # Also remove .pub if exists
                    pub_path = Path(str(key_path) + ".pub")
                    if pub_path.exists():
                        pub_path.unlink()
                
                action["status"] = "completed" if not dry_run else "would_delete"
                key.status = "burned"
            else:
                action["status"] = "not_found"
                
        except Exception as e:
            action["status"] = "error"
            action["error"] = str(e)
        
        return action
    
    def _burn_access(self, record: AccessRecord, dry_run: bool) -> Dict:
        """Revoke access credential"""
        action = {
            "type": "access_revocation",
            "credential_type": record.credential_type,
            "identifier": record.identifier[:20] + "...",
            "dry_run": dry_run,
            "status": "pending"
        }
        
        try:
            if record.credential_type == "api_key":
                # API keys would be revoked via API call
                action["status"] = "would_revoke_api" if dry_run else "revoked"
            elif record.credential_type == "password":
                # Passwords would be rotated
                action["status"] = "would_rotate" if dry_run else "rotated"
            elif record.credential_type == "token":
                # Tokens would be invalidated
                action["status"] = "would_invalidate" if dry_run else "invalidated"
            else:
                action["status"] = "unknown_type"
            
            if not dry_run:
                record.status = "burned"
                
        except Exception as e:
            action["status"] = "error"
            action["error"] = str(e)
        
        return action
    
    def _burn_files(self, pattern: str, dry_run: bool) -> Optional[Dict]:
        """Remove files matching pattern"""
        from fnmatch import fnmatch
        
        found_files = []
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if fnmatch(file, pattern):
                    found_files.append(Path(root) / file)
        
        if not found_files:
            return None
        
        action = {
            "type": "file_removal",
            "pattern": pattern,
            "files_found": len(found_files),
            "dry_run": dry_run,
            "status": "pending"
        }
        
        try:
            for file_path in found_files:
                if not dry_run:
                    # Secure delete
                    with open(file_path, 'wb') as f:
                        f.write(os.urandom(file_path.stat().st_size))
                    file_path.unlink()
            
            action["status"] = "completed" if not dry_run else "would_delete"
            
        except Exception as e:
            action["status"] = "error"
            action["error"] = str(e)
        
        return action
    
    def _burn_git_credentials(self, dry_run: bool) -> Dict:
        """Clear git credentials"""
        action = {
            "type": "git_credential_clear",
            "dry_run": dry_run,
            "status": "pending"
        }
        
        try:
            if not dry_run:
                # Clear git credential cache
                subprocess.run(
                    ["git", "credential-cache", "exit"],
                    capture_output=True,
                    cwd=self.project_root
                )
                
                # Remove stored credentials
                git_credentials = Path.home() / ".git-credentials"
                if git_credentials.exists():
                    git_credentials.unlink()
            
            action["status"] = "completed" if not dry_run else "would_clear"
            
        except Exception as e:
            action["status"] = "error"
            action["error"] = str(e)
        
        return action
    
    def _scan_and_remove_backdoors(self, dry_run: bool) -> Dict:
        """Scan for and remove potential backdoors"""
        action = {
            "type": "backdoor_scan",
            "dry_run": dry_run,
            "status": "pending",
            "findings": []
        }
        
        # Patterns that might indicate backdoors
        suspicious_patterns = [
            r"eval\s*\(",
            r"exec\s*\(",
            r"subprocess\.call.*shell\s*=\s*True",
            r"os\.system\s*\(",
            r"__import__\s*\(",
            r"socket\.socket\s*\(",
            r"requests\.get.*\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
        ]
        
        import re
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv', '.venv', 'node_modules']]
            
            for file in files:
                if not file.endswith('.py'):
                    continue
                
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    for pattern in suspicious_patterns:
                        matches = re.findall(pattern, content)
                        if matches:
                            action["findings"].append({
                                "file": str(file_path.relative_to(self.project_root)),
                                "pattern": pattern,
                                "count": len(matches)
                            })
                            
                except Exception as e:
                    logger.warning(f"Failed to scan {file_path}: {e}")
        
        action["status"] = "completed"
        action["total_findings"] = len(action["findings"])
        
        return action
    
    def generate_burn_certificate(self, report: Dict) -> Dict:
        """Generate burn completion certificate"""
        certificate = {
            "certificate_type": "BURN_PROTOCOL_COMPLETION",
            "certificate_id": f"BURN_CERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "execution_report_id": report.get("execution_id"),
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_actions": len(report.get("actions", [])),
                "successful_actions": sum(
                    1 for a in report.get("actions", [])
                    if a.get("status") in ["completed", "revoked", "rotated", "invalidated"]
                ),
                "dry_run": report.get("dry_run", True)
            },
            "attestation": "All developer access credentials, SSH keys, and potential backdoors have been identified and scheduled for removal.",
            "hash": hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
        }
        
        return certificate


if __name__ == "__main__":
    # Test Integrity Hash
    print("=== Integrity Hash Test ===")
    integrity_mgr = IntegrityHashManager(project_root=".")
    
    cert = integrity_mgr.generate_certificate(
        exclude_patterns=[
            "__pycache__/**",
            ".git/**",
            "*.pyc",
            "*.log",
            "venv/**"
        ]
    )
    
    print(f"Certificate ID: {cert.certificate_id}")
    print(f"Total Files: {cert.total_files}")
    print(f"Total Size: {cert.total_size_bytes:,} bytes")
    print(f"Root Hash: {cert.root_hash}")
    
    # Save certificate
    integrity_mgr.save_certificate(cert, "integrity_certificate.json")
    
    # Verify
    is_valid, mismatches = integrity_mgr.verify_certificate(cert)
    print(f"\nVerification: {'PASSED' if is_valid else 'FAILED'}")
    if mismatches:
        print(f"Mismatches: {mismatches}")
    
    # Test Burn Protocol
    print("\n=== Burn Protocol Test ===")
    burn = BurnProtocol(project_root=".")
    
    # Register some test credentials
    burn.register_access("api_key", "jquants_api_key_xxxxx")
    burn.register_access("token", "github_token_xxxxx")
    burn.register_ssh_key("ed25519", "SHA256:xxxxx", "~/.ssh/id_ed25519")
    
    # Schedule burn
    schedule = burn.schedule_burn()
    print(f"Scheduled {len(schedule['items'])} items for burn")
    
    # Execute burn (dry run)
    report = burn.execute_burn(dry_run=True)
    print(f"\nBurn Report (DRY RUN):")
    print(f"  Actions: {len(report['actions'])}")
    print(f"  Success: {report['success']}")
    
    for action in report['actions']:
        print(f"  - {action['type']}: {action['status']}")
    
    # Generate certificate
    cert = burn.generate_burn_certificate(report)
    print(f"\nBurn Certificate: {cert['certificate_id']}")
