@echo off
:: Closes the Windows 11 "black ghost window" (Windows Input Experience /
:: TextInputHost.exe rendering an unclosable borderless black rectangle).
:: Safe: Windows respawns TextInputHost automatically on demand.
taskkill /IM TextInputHost.exe /F >nul 2>&1
echo Ghost window host restarted. If a black window remains, it belongs to
echo another app - press Alt+Tab to identify it.
