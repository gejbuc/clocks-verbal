@echo off
:: Relaunch in hidden mode to prevent console windows from staying open
if "%~1"=="" (
    powershell -WindowStyle Hidden -Command "Start-Process -FilePath cmd.exe -ArgumentList '/c \"%~f0\" hidden' -WindowStyle Hidden"
    exit /b
)

:: Change to the project directory
cd /d "%~dp0"

:: Start the idle video in VLC
start "" "C:\Program Files\VideoLAN\VLC\vlc.exe" --fullscreen --loop --no-video-title-show --no-qt-fs-controller "projects\idle_video\mediapipe_idle.mp4"

:: Wait 10 seconds
timeout /t 10 /nobreak >nul

:: Start the Python orchestrator quietly using pythonw.exe
start "" "%~dp0.conda\pythonw.exe" "%~dp0scripts\orchestrator.py"