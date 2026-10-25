@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "WITH_AI=0"
set "CHECK_ONLY=0"
set "VENV_DIR=.venv"

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--with-ai" (
    set "WITH_AI=1"
) else if /i "%~1"=="--check-only" (
    set "CHECK_ONLY=1"
) else if /i "%~1"=="--help" (
    goto usage
) else (
    echo [ERROR] Unknown option: %~1
    goto usage_error
)
shift
goto parse_args

:usage
echo Usage:
echo   setup_environment.bat [--with-ai] [--check-only]
echo.
echo   --with-ai      Also install PyTorch and NumPy for NeuralMCTSAI.
echo   --check-only   Check the current environment without installing.
exit /b 0

:usage_error
echo.
echo Run setup_environment.bat --help for usage.
exit /b 2

:args_done
set "VENV_PY=%CD%\%VENV_DIR%\Scripts\python.exe"

if "!CHECK_ONLY!"=="1" (
    if exist "%VENV_PY%" (
        set "CHECK_PY=%VENV_PY%"
    ) else (
        set "CHECK_PY="
        call :find_base_python
        if not defined CHECK_PY set "CHECK_PY=!BASE_PY!"
    )
    "!CHECK_PY!" -c "import sys, pygame; print('[OK] Python', sys.version.split()[0]); print('[OK] pygame', pygame.version.ver)"
    if errorlevel 1 (
        echo [ERROR] pygame is not available. Run setup_environment.bat.
        exit /b 1
    )
    if "!WITH_AI!"=="1" (
        "!CHECK_PY!" -c "import torch, numpy; print('[OK] torch', torch.__version__); print('[OK] numpy', numpy.__version__)"
        if errorlevel 1 (
            echo [ERROR] AI dependencies are not available.
            exit /b 1
        )
    )
    echo [OK] Environment check completed.
    exit /b 0
)

call :find_base_python
if not defined BASE_PY exit /b 1

"%BASE_PY%" %BASE_ARGS% -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)"
if errorlevel 1 (
    echo [ERROR] Python 3.11 or newer is required.
    exit /b 1
)

if not exist "%VENV_PY%" (
    echo [INFO] Creating virtual environment in %VENV_DIR% ...
    "%BASE_PY%" %BASE_ARGS% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
)

echo [INFO] Upgrading pip in virtual environment ...
"%VENV_PY%" -m pip install --disable-pip-version-check --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    exit /b 1
)

echo [INFO] Installing pygame runtime ...
"%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 14) else 1)"
if errorlevel 1 (
    set "PYGAME_PACKAGE=pygame"
) else (
    set "PYGAME_PACKAGE=pygame-ce"
)
"%VENV_PY%" -m pip install --disable-pip-version-check "!PYGAME_PACKAGE!"
if errorlevel 1 (
    echo [ERROR] Failed to install !PYGAME_PACKAGE!.
    exit /b 1
)

if "%WITH_AI%"=="1" (
    echo [INFO] Installing optional neural AI dependencies ...
    "%VENV_PY%" -m pip install --disable-pip-version-check -r requirements-ai.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install AI dependencies.
        exit /b 1
    )
)

echo [INFO] Running environment self-check ...
"%VENV_PY%" -c "import sys, pygame; print('[OK] Python', sys.version.split()[0]); print('[OK] pygame', pygame.version.ver)"
if errorlevel 1 (
    echo [ERROR] pygame self-check failed.
    exit /b 1
)
if "%WITH_AI%"=="1" (
    "%VENV_PY%" -c "import torch, numpy; print('[OK] torch', torch.__version__); print('[OK] numpy', numpy.__version__)"
    if errorlevel 1 (
        echo [ERROR] AI dependency self-check failed.
        exit /b 1
    )
)

echo [INFO] Running project checks with the virtual environment ...
set "HEX_PYTHON=%VENV_PY%"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
call check.bat
set "CHECK_EXIT=%ERRORLEVEL%"
if not "%CHECK_EXIT%"=="0" (
    echo [ERROR] Project checks failed.
    exit /b %CHECK_EXIT%
)

echo [OK] Environment setup completed.
echo [INFO] Start with: %VENV_DIR%\Scripts\python.exe hex_game.py
exit /b 0

:find_base_python
set "BASE_PY="
set "BASE_ARGS="
for /f "delims=" %%P in ('where py 2^>nul') do (
    "%%~P" -3 -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "BASE_PY=%%~P"
        set "BASE_ARGS=-3"
        goto :eof
    )
)
for /f "delims=" %%P in ('where python 2^>nul') do (
    "%%~P" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "BASE_PY=%%~P"
        goto :eof
    )
)
for /f "delims=" %%P in ('where python3 2^>nul') do (
    "%%~P" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "BASE_PY=%%~P"
        goto :eof
    )
)
echo [ERROR] No Python 3.11+ interpreter was found.
exit /b 1
