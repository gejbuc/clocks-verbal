@echo off
echo Stopping Clocks Verbal system...

:: Stop the Python orchestrator
powershell -NoProfile -Command "Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -match 'orchestrator.py' } | ForEach-Object { Stop-Process $_.ProcessId -Force }"

:: Stop the VLC idle video player
powershell -NoProfile -Command "Get-WmiObject Win32_Process | Where-Object { $_.Name -match 'vlc.exe' -and $_.CommandLine -match 'mediapipe_idle.mp4' } | ForEach-Object { Stop-Process $_.ProcessId -Force }"

echo Stopped.
timeout /t 3 >nul
