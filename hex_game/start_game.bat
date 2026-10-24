@echo off
rem Double-click launcher for Hex game (pygame version)
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python launcher "py" not found. Please install Python 3 from python.org
    echo         and make sure "py launcher" is checked during installation.
    pause
    exit /b 1
)

py -c "import pygame" >nul 2>nul
if errorlevel 1 (
    echo [INFO] pygame not installed, installing now...
    py -m pip install pygame
    if errorlevel 1 (
        echo [ERROR] Failed to install pygame. Please run: py -m pip install pygame
        pause
        exit /b 1
    )
)

py hex_game.py
if errorlevel 1 pause
