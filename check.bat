@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHON_EXE="
set "PYTHON_ARGS="
if defined HEX_PYTHON if exist "%HEX_PYTHON%" (
    set "PYTHON_EXE=%HEX_PYTHON%"
    set "PYTHON_ARGS="
    goto :python_found
)
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
    set "PYTHON_ARGS="
    goto :python_found
)

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
"%PYTHON_EXE%" %PYTHON_ARGS% -B -m unittest -v test_hex_board.py test_hex_engine.py test_hex_match.py test_hex_notation.py test_game_records_validation.py test_llm_config.py test_board_observation.py test_llm_client.py test_llm_advisor.py test_analysis_worker.py test_benchmark_ai.py test_ai.py test_mcts.py test_ai_worker.py test_ui.py test_vision.py test_memory_contracts.py test_memory_canonical.py test_memory_store.py test_memory_collector.py test_memory_integration.py test_memory_validator.py test_memory_features.py test_memory_retrieval.py test_memory_ai.py test_memory_tools.py test_memory_dataset.py test_memory_training.py test_memory_promotion.py test_memory_v2_tools.py test_memory_v3_pipeline.py test_memory_v3_selfplay.py test_memory_v3_registry_policy.py test_memory_v3_tools.py test_memory_v4.py test_memory_v4_tools.py test_p1.py test_p2.py test_evaluation.py test_tactics.py test_hybrid_ai.py test_ai_coordinator.py test_llm_candidate_tool.py test_strategic_ai.py test_tournament.py test_package_release.py test_e2e.py
if errorlevel 1 goto :fail

echo [OK] All checks passed.
exit /b 0

:fail
echo [ERROR] One or more checks failed.
if not defined CI pause
exit /b 1
