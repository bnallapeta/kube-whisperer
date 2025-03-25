#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Ensure we're in the project root
cd "$(dirname "$0")"

# Create build directories if they don't exist
mkdir -p .build/{coverage,egg-info,pytest_cache}

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Run tests based on arguments
if [ "$1" == "--unit" ]; then
    echo -e "${GREEN}Running unit tests...${NC}"
    pytest tests -v -m "unit"
elif [ "$1" == "--integration" ]; then
    echo -e "${GREEN}Running integration tests...${NC}"
    pytest tests -v -m "integration"
elif [ "$1" == "--config" ]; then
    echo -e "${GREEN}Running configuration tests...${NC}"
    pytest tests -v -m "config" --cov=src --cov-report=term-missing
elif [ "$1" == "--all" ]; then
    echo -e "${GREEN}Running all tests...${NC}"
    pytest tests -v --cov=src --cov-report=term-missing --cov-report=html
else
    echo -e "${YELLOW}Usage: $0 [--unit|--integration|--config|--all]${NC}"
    echo "  --unit         Run unit tests only"
    echo "  --integration  Run integration tests only"
    echo "  --config       Run configuration tests only"
    echo "  --all         Run all tests and generate coverage report"
    exit 1
fi

# Check if tests passed
if [ $? -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo -e "${RED}Tests failed!${NC}"
    exit 1
fi 