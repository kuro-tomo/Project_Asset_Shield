#!/usr/bin/env python3
"""
Asset Shield - AWS Migration Helper (M&A Transfer Tool)
-------------------------------------------------------
This tool facilitates the migration of the Asset Shield trading system
from local environment (MacBook Pro) to AWS infrastructure.

It orchestrates:
1. Terraform Infrastructure Provisioning
2. Docker Container Build & Push
3. Environment Configuration

Usage:
    python aws_migration_helper.py [init|plan|deploy]
"""

import os
import sys
import subprocess
import argparse
import shutil
import json
from typing import Optional

# Colors for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'

def log(message: str, level: str = "INFO"):
    if level == "INFO":
        print(f"{GREEN}[INFO] {message}{RESET}")
    elif level == "WARN":
        print(f"{YELLOW}[WARN] {message}{RESET}")
    elif level == "ERROR":
        print(f"{RED}[ERROR] {message}{RESET}")

def check_prerequisites():
    """Check if required tools are installed."""
    tools = ["terraform", "docker", "aws"]
    missing = []
    
    for tool in tools:
        if not shutil.which(tool):
            missing.append(tool)
    
    if missing:
        log(f"Missing required tools: {', '.join(missing)}", "ERROR")
        log("Please install them before proceeding.", "ERROR")
        sys.exit(1)
    
    log("All prerequisites checked.", "INFO")

def run_command(command: str, cwd: Optional[str] = None) -> bool:
    """Run a shell command."""
    try:
        log(f"Executing: {command}", "INFO")
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd, 
            check=True, 
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        log(f"Command failed: {e}", "ERROR")
        return False

def setup_terraform(action: str):
    """Handle Terraform operations."""
    tf_dir = "infrastructure/terraform"
    
    if not os.path.exists(tf_dir):
        log(f"Terraform directory not found: {tf_dir}", "ERROR")
        sys.exit(1)

    # Check for API Key
    api_key = os.environ.get("JQUANTS_API_KEY")
    if not api_key:
        log("JQUANTS_API_KEY environment variable not set.", "WARN")
        log("You will be prompted for it during deployment.", "WARN")

    if action == "init":
        run_command("terraform init", cwd=tf_dir)
        
    elif action == "plan":
        # Create a tfvars file if it doesn't exist (basic template)
        tfvars_path = os.path.join(tf_dir, "production.tfvars")
        if not os.path.exists(tfvars_path):
            with open(tfvars_path, "w") as f:
                f.write('environment = "production"\n')
                f.write('project_name = "asset-shield"\n')
                if api_key:
                    f.write(f'jquants_api_key = "{api_key}"\n')
            log(f"Created template {tfvars_path}", "INFO")
            
        cmd = "terraform plan -var-file=production.tfvars"
        run_command(cmd, cwd=tf_dir)
        
    elif action == "deploy":
        cmd = "terraform apply -var-file=production.tfvars -auto-approve"
        if run_command(cmd, cwd=tf_dir):
            log("Infrastructure deployment successful.", "INFO")
            
            # Get ECR URL
            output = subprocess.check_output(
                "terraform output -raw ecr_repository_url", 
                shell=True, 
                cwd=tf_dir, 
                text=True
            ).strip()
            
            if output:
                build_and_push_docker(output)

def build_and_push_docker(ecr_url: str):
    """Build and push Docker image to ECR."""
    log("Starting Docker build and push sequence...", "INFO")
    
    # Login to ECR
    region = "ap-northeast-1" # Default to Tokyo
    login_cmd = f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {ecr_url}"
    run_command(login_cmd)
    
    # Build
    build_cmd = f"docker build -t asset-shield:latest -f infrastructure/Dockerfile ."
    run_command(build_cmd)
    
    # Tag
    tag_cmd = f"docker tag asset-shield:latest {ecr_url}:latest"
    run_command(tag_cmd)
    
    # Push
    push_cmd = f"docker push {ecr_url}:latest"
    run_command(push_cmd)
    
    log("Docker image deployed successfully!", "INFO")

def main():
    parser = argparse.ArgumentParser(description="Asset Shield AWS Migration Helper")
    parser.add_argument("action", choices=["init", "plan", "deploy"], help="Action to perform")
    
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print(" 🛡️  ASSET SHIELD - M&A TRANSFER TOOL")
    print("="*50 + "\n")
    
    check_prerequisites()
    setup_terraform(args.action)

if __name__ == "__main__":
    main()
