#!/bin/bash
# Script to clean up build artifacts and temporary files

echo "Cleaning up build artifacts and temporary files..."

# Remove Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete

# Remove test artifacts
rm -rf .pytest_cache
rm -rf htmlcov
rm -f .coverage

# Remove build artifacts
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
rm -rf whisper_service.egg-info/

# Remove temporary files
rm -rf /tmp/whisper_audio/*

# Create clean build directories
mkdir -p .build/{coverage,egg-info,pytest_cache}

echo "Cleanup complete!" 