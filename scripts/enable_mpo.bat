@echo off
:: Reverts disable_mpo.bat (re-enables Multi-Plane Overlay).
:: Right-click this file and choose "Run as administrator".
reg delete "HKLM\SOFTWARE\Microsoft\Windows\Dwm" /v OverlayTestMode /f
if %errorlevel%==0 (echo MPO re-enabled. Reboot to apply.) else (echo FAILED - run as administrator.)
pause
