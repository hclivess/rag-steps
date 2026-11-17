@echo off
rem Double-click this file: it installs what is needed on the first start (into the
rem .venv folder next to it), starts the web interface and opens it in the browser.
rem Close this window to stop it.
setlocal
cd /d "%~dp0"
title rag-sqlite
set "PY="
python -c "import sys" >nul 2>nul && set "PY=python"
if not defined PY py -3 -c "import sys" >nul 2>nul && set "PY=py -3"
if not defined PY (
  echo Python was not found. Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH", then start this again.
  pause
  exit /b 1
)
if not exist ".venv\installed.txt" (
  echo First start: installing into the .venv folder here. This takes a minute or two.
  %PY% -m venv .venv && ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt && echo ok> ".venv\installed.txt"
)
if not exist ".venv\installed.txt" (
  echo.
  echo The installation failed. The messages above say why.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m rag serve %*
if errorlevel 1 pause
