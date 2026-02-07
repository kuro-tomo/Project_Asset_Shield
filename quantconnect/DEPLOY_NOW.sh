#!/bin/bash
# ============================================================
# Asset Shield V3.2.0 - QuantConnect Deployment Script
# ============================================================
# Target: 11:30 JST
# ============================================================

set -e

export PATH="$PATH:/Users/MBP/Library/Python/3.9/bin"

echo "============================================================"
echo "Asset Shield V3.2.0 - QuantConnect Deployment"
echo "============================================================"
echo "Current time: $(date '+%H:%M:%S JST')"
echo "Target: 11:30 JST"
echo "============================================================"

cd /Users/MBP/Desktop/Project_Asset_Shield/quantconnect

# Check if logged in
echo ""
echo "[Step 1] Checking QuantConnect login..."
if ! lean whoami 2>/dev/null; then
    echo "Not logged in. Please run:"
    echo "  lean login"
    echo ""
    echo "Then re-run this script."
    exit 1
fi

echo "✓ Logged in"

# Method 1: Manual Upload Instructions
echo ""
echo "============================================================"
echo "MANUAL UPLOAD INSTRUCTIONS (if CLI fails)"
echo "============================================================"
echo ""
echo "1. Open QuantConnect: https://www.quantconnect.com/terminal"
echo ""
echo "2. Create new algorithm: 'AssetShield_Chunk1'"
echo "   Copy content from:"
echo "   cat AssetShieldJP/chunks/uploader_chunk1.py | pbcopy"
echo ""
echo "3. Run backtest (upload data)"
echo ""
echo "4. Repeat for chunks 2-4"
echo ""
echo "5. Create 'AssetShieldJP_Final' algorithm"
echo "   Copy content from:"
echo "   cat AssetShieldJP/main_objectstore_final.py | pbcopy"
echo ""
echo "6. Run final backtest"
echo ""
echo "============================================================"

# Copy first chunk to clipboard for quick start
echo ""
echo "[Quick Start] Copying Chunk 1 to clipboard..."
cat AssetShieldJP/chunks/uploader_chunk1.py | pbcopy
echo "✓ Chunk 1 copied to clipboard"
echo ""
echo "Paste into QuantConnect and run backtest!"
echo ""
echo "============================================================"
