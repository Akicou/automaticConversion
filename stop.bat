@echo off
echo Stopping GGUF Forge server...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr ABH') do (
    echo Killing process %%a on port 8000...
    taskkill /F /PID %%a >nul 2>&1
)
echo Done.
