@echo off
:: Disables Multi-Plane Overlay (MPO) - NVIDIA's documented fix for black
:: box / flicker artifacts on the physical display that don't appear in
:: screenshots. Reversible with enable_mpo.bat. Reboot to take effect.
:: Right-click this file and choose "Run as administrator".
reg add "HKLM\SOFTWARE\Microsoft\Windows\Dwm" /v OverlayTestMode /t REG_DWORD /d 5 /f
if %errorlevel%==0 (echo MPO disabled. Reboot to apply.) else (echo FAILED - run as administrator.)
pause
