@echo off
:: Installs the window watchdog into the current user's Startup folder so it
:: arms itself at every logon (no admin rights needed).
:: Remove with: del "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\krea_window_watchdog.vbs"
copy /Y "%~dp0watchdog_autostart.vbs" "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\krea_window_watchdog.vbs" >nul
if %errorlevel%==0 (echo Watchdog autostart installed.) else (echo FAILED to copy to Startup folder.)
