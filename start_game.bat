@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist "hex_game.py" (
    echo [ERROR] hex_game.py was not found next to this launcher.
    pause
    exit /b 1
)

rem Find a Python interpreter that can actually run pip.
rem A py.exe launcher may exist without any registered Python installation.
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
echo         Install Python 3 from https://www.python.org/
echo         and enable "Add python.exe to PATH" plus pip during installation.
echo         Then run this command to verify:
echo             python -m pip --version
pause
exit /b 1

:python_found
"%PYTHON_EXE%" %PYTHON_ARGS% -c "import sys; print('[INFO] Python', sys.version.split()[0], 'at', sys.executable)"

if /i "%~1"=="--check" (
    "%PYTHON_EXE%" %PYTHON_ARGS% -c "import pygame; print('[OK] pygame', pygame.version.ver)" 2>nul
    if errorlevel 1 (
        echo [INFO] pygame is not installed. A normal launch will install it.
        exit /b 1
    )
    echo [OK] Launcher environment check completed.
    exit /b 0
)

"%PYTHON_EXE%" %PYTHON_ARGS% -c "import pygame" >nul 2>nul
if not errorlevel 1 goto :run

echo [INFO] pygame is not installed. Installing it now...
set "PKG=pygame"
"%PYTHON_EXE%" %PYTHON_ARGS% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 14) else 1)"
if not errorlevel 1 set "PKG=pygame-ce"

echo [INFO] Installing %PKG% from the configured pip source...
"%PYTHON_EXE%" %PYTHON_ARGS% -m pip install --disable-pip-version-check %PKG%
if errorlevel 1 (
    echo [WARN] Default source failed. Retrying with the Aliyun mirror...
    "%PYTHON_EXE%" %PYTHON_ARGS% -m pip install --disable-pip-version-check %PKG% -i https://mirrors.aliyun.com/pypi/simple
)
if errorlevel 1 (
    echo [WARN] Mirror failed. Retrying with official PyPI...
    "%PYTHON_EXE%" %PYTHON_ARGS% -m pip install --disable-pip-version-check %PKG% -i https://pypi.org/simple
)
if errorlevel 1 (
    echo [ERROR] Automatic pygame installation failed.
    echo         Check the network connection, then run manually:
    echo             "%PYTHON_EXE%" %PYTHON_ARGS% -m pip install %PKG%
    pause
    exit /b 1
)

"%PYTHON_EXE%" %PYTHON_ARGS% -c "import pygame" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] pygame was installed but still cannot be imported.
    pause
    exit /b 1
)

:run
"%PYTHON_EXE%" %PYTHON_ARGS% "hex_game.py"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
