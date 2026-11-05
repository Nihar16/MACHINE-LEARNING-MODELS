@echo off
REM Double-click launcher for Windows. Runs setup.ps1 without changing system policy.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
pause
