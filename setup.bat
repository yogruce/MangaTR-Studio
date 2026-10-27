@echo off
title MangaTR-Studio - Otomatik Kurulum
cd /d "%~dp0"
echo ====================================================================
echo   MangaTR-Studio - Kurulum Baslatiliyor...
echo ====================================================================
set "PY_CMD=python"
py -3.10 --version >nul 2>&1 && set "PY_CMD=py -3.10"
py -3.11 --version >nul 2>&1 && set "PY_CMD=py -3.11"
py -3.12 --version >nul 2>&1 && set "PY_CMD=py -3.12"
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Python sanal ortami .venv olusturuluyor...
    %PY_CMD% -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
    echo [HATA] .venv olusturulamadi! Lutfen Python 3.10, 3.11 veya 3.12 kurun.
    pause
    exit /b 1
)
echo [2/3] Gerekli Python paketleri yukleniyor...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
echo [3/3] Yapay zeka modeli ve fontlar hazirlaniyor...
".venv\Scripts\python.exe" setup_project.py
echo.
echo ====================================================================
echo Kurulum tamamlandi! MangaTR-Studio.bat dosyasina cift tiklayabilirsiniz.
echo ====================================================================
pause
