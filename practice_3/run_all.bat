@echo off
cd /d "%~dp0"
call exercise_1\run_exercise_1.bat nopause
if errorlevel 1 goto :end
call exercise_2\run_exercise_2.bat nopause
:end
pause
