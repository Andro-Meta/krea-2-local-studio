@echo off
setlocal
set "KREA_WATCHDOG_INSTALL_ROOT=%~dp0.."
set "STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "KREA_WATCHDOG_VBS=%STARTUP_DIR%\krea_window_watchdog.vbs"

if not exist "%STARTUP_DIR%" mkdir "%STARTUP_DIR%"
powershell -NoProfile -Command "$ErrorActionPreference='Stop'; $r=[IO.Path]::GetFullPath($env:KREA_WATCHDOG_INSTALL_ROOT); $q=[char]34; $lines=@(''' Launches the Krea window watchdog silently at logon.','On Error Resume Next','Dim shell, fso, exePath, scriptPath, attempts','Set shell = CreateObject('+$q+'WScript.Shell'+$q+')','Set fso = CreateObject('+$q+'Scripting.FileSystemObject'+$q+')','exePath = '+$q+($r+'\venv\Scripts\pythonw.exe')+$q,'scriptPath = '+$q+($r+'\scripts\window_watchdog.py')+$q,'For attempts = 1 To 24','    If fso.FileExists(exePath) Then Exit For','    WScript.Sleep 5000','Next','If fso.FileExists(exePath) Then','    shell.CurrentDirectory = '+$q+$r+$q,'    shell.Run Chr(34) ^& exePath ^& Chr(34) ^& '+$q+' '+$q+' ^& Chr(34) ^& scriptPath ^& Chr(34), 0, False','End If'); [IO.File]::WriteAllLines($env:KREA_WATCHDOG_VBS,$lines,[Text.UTF8Encoding]::new($false))"
if errorlevel 1 (
    echo FAILED to install watchdog autostart.
    exit /b 1
)
echo Watchdog autostart installed.
