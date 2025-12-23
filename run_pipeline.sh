#!/bin/bash
# Simple wrapper script to run the end-to-end pipeline

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Transformer UE Localization Pipeline${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
else
    echo -e "${RED}Warning: No virtual environment found at venv/${NC}"
    echo -e "${YELLOW}Consider creating one with: python -m venv venv${NC}"
fi

# Check if Python dependencies are installed
echo -e "${YELLOW}Checking dependencies...${NC}"
python -c "import torch, pytorch_lightning, zarr" 2>/dev/null || {
    echo -e "${RED}Missing dependencies!${NC}"
    echo -e "${YELLOW}Run: pip install -r requirements.txt${NC}"
    exit 1
}

# Run the pipeline
echo -e "${GREEN}Starting pipeline...${NC}"
echo ""

python run_pipeline.py "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Pipeline completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Pipeline failed with exit code $EXIT_CODE${NC}"
    echo -e "${RED}========================================${NC}"
fi

exit $EXIT_CODE
