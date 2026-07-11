@echo off
:: Starts the window forensics watchdog in the background (no console window).
:: It logs every window that appears to logs\window_watchdog.jsonl and saves a
:: screenshot to logs\watchdog_shots\ whenever a large black borderless window
:: shows up - definitive evidence of what process created the mystery box.
cd /d "%~dp0.."
powershell -NoProfile -Command "Start-Process -WindowStyle Hidden -FilePath 'venv\Scripts\pythonw.exe' -ArgumentList 'scripts\window_watchdog.py'"
echo Watchdog started. Evidence will appear in logs\window_watchdog.jsonl
