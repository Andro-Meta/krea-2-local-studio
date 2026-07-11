' Launches the black-box window forensics watchdog silently at logon.
' Installed into shell:startup by install_watchdog_autostart.bat.
' Waits for the E: drive to be ready (secondary drives can mount after the
' Startup folder runs) and never shows an error dialog.
On Error Resume Next
Dim shell, fso, exePath, scriptPath, attempts
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
exePath = "E:\Krea 2\venv\Scripts\pythonw.exe"
scriptPath = "E:\Krea 2\scripts\window_watchdog.py"
For attempts = 1 To 24 ' up to ~2 minutes
    If fso.FileExists(exePath) Then Exit For
    WScript.Sleep 5000
Next
If fso.FileExists(exePath) Then
    shell.CurrentDirectory = "E:\Krea 2"
    shell.Run """" & exePath & """ """ & scriptPath & """", 0, False
End If
