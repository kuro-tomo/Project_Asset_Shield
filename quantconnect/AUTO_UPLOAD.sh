#!/bin/bash
# Asset Shield - Automated QuantConnect Upload
set -e

export PATH="$PATH:/Users/MBP/Library/Python/3.9/bin"
cd /Users/MBP/Desktop/Project_Asset_Shield/quantconnect

echo "=== Asset Shield Auto Upload ==="

# Check login
if ! lean whoami 2>/dev/null | grep -q "@"; then
    echo "Not logged in. Running lean login..."
    lean login
fi

echo "Logged in!"

# Create uploaders directory structure
PROJ_DIR="uploaders_qc"
mkdir -p $PROJ_DIR

# Function to create project, push, and backtest
upload_and_run() {
    local name=$1
    local file=$2
    local proj_path="$PROJ_DIR/$name"

    echo "=== Processing: $name ==="

    # Create project directory
    mkdir -p "$proj_path"
    cp "$file" "$proj_path/main.py"

    # Create config
    cat > "$proj_path/config.json" << EOF
{
    "algorithm-language": "Python",
    "parameters": {},
    "description": "$name",
    "cloud-id": 0
}
EOF

    # Push to cloud
    echo "Pushing $name to cloud..."
    cd "$proj_path"
    lean cloud push --project . 2>/dev/null || lean cloud push 2>/dev/null || echo "Push may have issues"

    # Run backtest
    echo "Running backtest for $name..."
    lean cloud backtest . --push 2>/dev/null || echo "Backtest queued"

    cd /Users/MBP/Desktop/Project_Asset_Shield/quantconnect
    echo "Done: $name"
    echo ""
    sleep 2
}

# Upload all parts
echo ""
echo "=== Uploading 6 data chunks ==="
upload_and_run "Upload72030A" "uploaders/upload_72030_A.py"
upload_and_run "Upload72030B" "uploaders/upload_72030_B.py"
upload_and_run "Upload67580A" "uploaders/upload_67580_A.py"
upload_and_run "Upload67580B" "uploaders/upload_67580_B.py"
upload_and_run "Upload83060A" "uploaders/upload_83060_A.py"
upload_and_run "Upload83060B" "uploaders/upload_83060_B.py"

echo ""
echo "=== Data upload complete! ==="
echo ""
echo "=== Now running main backtest ==="
upload_and_run "AssetShield3Stocks" "BACKTEST_3STOCKS.py"

echo ""
echo "=== ALL DONE ==="
echo "Check QuantConnect for results: https://www.quantconnect.com/terminal"
