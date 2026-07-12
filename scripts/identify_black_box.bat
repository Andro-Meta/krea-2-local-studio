@echo off
:: Run this WHILE the black box is on screen. It lists every visible window
:: with its owning process, command line, and parent chain, and flags
:: borderless untitled windows (black-box candidates) with a blackness score.
cd /d "%~dp0.."
venv\Scripts\python.exe scripts\window_watchdog.py --snapshot
pause
