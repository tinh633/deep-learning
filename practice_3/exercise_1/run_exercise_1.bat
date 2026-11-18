@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
title Exercise 1 - Sentiment Analysis

echo === Exercise 1: setup ===
if not exist .venv (
    python -m venv .venv || goto :fail
)
call .venv\Scripts\activate.bat || goto :fail
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt || goto :fail

echo.
echo === Unit tests ===
python -m pytest tests -q || goto :fail

echo.
echo === Run sentiment analysis (downloads the model on first run) ===
python src\sentiment.py || goto :fail

echo.
echo DONE. Result: %~dp0outputs\sentiment_result.json
goto :end

:fail
echo.
echo *** FAILED - see the message above. ***
set EXITCODE=1

:end
if /i not "%~1"=="nopause" pause
exit /b %EXITCODE%
