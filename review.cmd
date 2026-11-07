@echo off
cd /d "%~dp0"
python scriptsserve_review.py --pdf-root "%~dp0.." %*
pause
