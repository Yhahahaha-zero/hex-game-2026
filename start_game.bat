@echo off
rem Hex game launcher: auto-detect Python, install pygame if needed, then start.
cd /d "%~dp0"

rem ---- locate Python (py launcher first, then python command) ----
set PYCMD=py
where py >nul 2>nul
if errorlevel 1 (
    where python >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python not found. Please install Python 3.11+ from https://www.python.org/
        echo         and check "Add python.exe to PATH" during installation.
        pause
        exit /b 1
    )
    set PYCMD=python
)

rem ---- show which Python will be used (helps diagnosing problems) ----
%PYCMD% -c "import sys; print('[INFO] Python', sys.version.split()[0], 'at', sys.executable)"

rem ---- pygame already installed? then start directly ----
%PYCMD% -c "import pygame" >nul 2>nul
if not errorlevel 1 goto run

echo [INFO] pygame not installed, installing now...

rem ---- Python 3.14+ has no classic pygame wheels; pygame-ce is a drop-in
rem      replacement (same "import pygame", no code changes needed) ----
set PKG=pygame
%PYCMD% -c "import sys; sys.exit(0 if sys.version_info >= (3, 14) else 1)"
if not errorlevel 1 set PKG=pygame-ce

%PYCMD% -m pip install --upgrade pip >nul 2>nul
echo [INFO] Installing %PKG% from your default pip source ...
%PYCMD% -m pip install %PKG%
if errorlevel 1 (
    echo [WARN] Default source failed, retrying with Aliyun mirror ...
    %PYCMD% -m pip install %PKG% -i https://mirrors.aliyun.com/pypi/simple
)
if errorlevel 1 (
    echo [WARN] Retrying with official PyPI ...
    %PYCMD% -m pip install %PKG% -i https://pypi.org/simple
)
if errorlevel 1 (
    echo [ERROR] Automatic install failed. Possible causes:
    echo         1. No network access to pip mirrors.
    echo         2. Python is a restricted edition - install the full version from python.org.
    echo.
    echo         Manual install, then rerun this file:
    echo         %PYCMD% -m pip install %PKG%
    pause
    exit /b 1
)

:run
%PYCMD% hex_game.py
if errorlevel 1 pause
