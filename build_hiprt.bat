@echo off
setlocal
where python >nul 2>nul
if errorlevel 1 goto try_py
python "%~dp0tools\build_hiprt.py" %*
exit /b %errorlevel%
:try_py
where py >nul 2>nul
if errorlevel 1 goto missing_python
py -3 "%~dp0tools\build_hiprt.py" %*
exit /b %errorlevel%
:missing_python
echo ERROR: Install Python 3.8 or newer and add it to PATH. 1>&2
exit /b 1
