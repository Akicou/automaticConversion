#!/bin/bash
# GGUF Forge Update Script (Linux/macOS)
# Fetches latest changes and overwrites local files

echo "=== GGUF Forge Updater ==="
echo ""

# Get current branch
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
if [ -z "$branch" ]; then
    echo "Error: Not a git repository"
    exit 1
fi

echo "Current branch: $branch"
echo "Fetching latest from origin..."

# Fetch all updates from origin
git fetch --all

if [ $? -ne 0 ]; then
    echo "Error: Failed to fetch from origin"
    exit 1
fi

echo "Resetting to origin/$branch (overwriting local changes)..."

# Hard reset to origin branch (overwrites all local changes)
git reset --hard "origin/$branch"

if [ $? -ne 0 ]; then
    echo "Error: Failed to reset to origin"
    exit 1
fi

echo ""
echo "Update complete!"
echo "Latest commit:"
git log -1 --oneline
