@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set TOKENIZERS_PARALLELISM=false
title Exercise 2 - Fine-tune DistilBERT on IMDb

echo === Exercise 2: setup ===
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
echo === 1/5 Prepare data (downloads IMDb on first run) ===
python src\2_preparedata.py || goto :fail

echo.
echo === 2/5 Fine-tune (creates a NEW outputs\runs\lan_N; CPU takes several minutes) ===
python src\3_train.py || goto :fail

rem newest run folder (numeric sort, so lan_10 > lan_9)
for /f %%R in ('python -c "import sys; sys.path.insert(0,'src'); from config import RUNS; print(sorted(RUNS.glob('lan_*'), key=lambda p: int(p.name[4:]))[-1].name)"') do set EVAL_RUN=%%R
echo Using run: %EVAL_RUN%

rem validation stage: the test set is NOT touched
set RUN_TEST=0
echo.
echo === 3/5 Baselines on validation ===
python src\7_baseline.py || goto :fail
echo.
echo === 4/5 Evaluate on validation ===
python src\4_evaluate.py || goto :fail
echo.
echo === 5/5 Report and figures ===
python src\5_report.py || goto :fail

echo.
echo Validation stage finished for %EVAL_RUN%. Review visualization\%EVAL_RUN%\ before the final test.
echo The held-out test set must be used ONCE, after you accept this model.
set ANSWER=n
set /p ANSWER=Run the final test on %EVAL_RUN% now? [y/N]: 
if /i not "!ANSWER!"=="y" goto :done

set RUN_TEST=1
echo.
echo === Final test ===
if not exist outputs\baseline\baseline_test.json (
    python src\7_baseline.py --confirm-test || goto :fail
) else (
    echo baseline_test.json already exists - keeping it.
)
if exist outputs\runs\%EVAL_RUN%\test_metrics.json (
    echo test_metrics.json already exists for %EVAL_RUN% - refusing to overwrite.
) else (
    python src\4_evaluate.py --confirm-test || goto :fail
)
python src\5_report.py || goto :fail

:done
echo.
echo DONE. Report: %~dp0visualization\%EVAL_RUN%\REPORT.md
goto :end

:fail
echo.
echo *** FAILED - see the message above. ***
set EXITCODE=1

:end
if /i not "%~1"=="nopause" pause
exit /b %EXITCODE%
