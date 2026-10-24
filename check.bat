@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHON_EXE="
set "PYTHON_ARGS="

for /f "delims=" %%P in ('where py 2^>nul') do (
    "%%~P" -3 -m pip --version >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_EXE=%%~P"
        set "PYTHON_ARGS=-3"
        goto :python_found
    )
)

for /f "delims=" %%P in ('where python 2^>nul') do (
    "%%~P" -m pip --version >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_EXE=%%~P"
        set "PYTHON_ARGS="
        goto :python_found
    )
)

for /f "delims=" %%P in ('where python3 2^>nul') do (
    "%%~P" -m pip --version >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_EXE=%%~P"
        set "PYTHON_ARGS="
        goto :python_found
    )
)

echo [ERROR] No usable Python with pip was found.
echo         Run start_game.bat --check for environment diagnostics.
if not defined CI pause
exit /b 1

:python_found
"%PYTHON_EXE%" %PYTHON_ARGS% -c "import pygame" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] pygame is not installed.
    echo         Run start_game.bat once to install it, then rerun check.bat.
    if not defined CI pause
    exit /b 1
)

echo [1/2] Running game self-test...
"%PYTHON_EXE%" %PYTHON_ARGS% -B hex_game_test.py
if errorlevel 1 goto :fail

echo [2/2] Running unit tests...
"%PYTHON_EXE%" %PYTHON_ARGS% -B -m unittest -v test_hex_board.py test_hex_engine.py test_hex_match.py test_benchmark_ai.py test_ai.py test_mcts.py test_ai_worker.py test_ui.py test_p1.py
if errorlevel 1 goto :fail

echo [OK] All checks passed.
exit /b 0

:fail
echo [ERROR] One or more checks failed.
if not defined CI pause
exit /b 1
