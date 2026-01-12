# GGUF Forge Update Script (Windows PowerShell)
# Fetches latest changes and overwrites local files

Write-Host "=== GGUF Forge Updater ===" -ForegroundColor Cyan
Write-Host ""

# Get current branch
$branch = git rev-parse --abbrev-ref HEAD 2>$null
if (-not $branch) {
    Write-Host "Error: Not a git repository" -ForegroundColor Red
    exit 1
}

Write-Host "Current branch: $branch" -ForegroundColor Yellow
Write-Host "Fetching latest from origin..." -ForegroundColor Yellow

# Fetch all updates from origin
git fetch --all

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to fetch from origin" -ForegroundColor Red
    exit 1
}

Write-Host "Resetting to origin/$branch (overwriting local changes)..." -ForegroundColor Yellow

# Hard reset to origin branch (overwrites all local changes)
git reset --hard "origin/$branch"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to reset to origin" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Update complete!" -ForegroundColor Green
Write-Host "Latest commit:" -ForegroundColor Cyan
git log -1 --oneline
