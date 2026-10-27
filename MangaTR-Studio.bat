@echo off
title MangaTR-Studio - AI Manga Ceviri ve Typesetting Studyosu
cd /d "%~dp0"
echo ======================================================================
echo   MangaTR Studio - AI Manga Ceviri ve Typesetting Studyosu Baslatiliyor
echo ======================================================================
if not exist ".venv\Scripts\python.exe" (
    echo [BILGI] Sanal ortam bulunamadi. Otomatik kurulum baslatiliyor...
    call setup.bat
)
if not exist "vendor\comic-text-detector\models\comictextdetector.pt" (
    echo [BILGI] Model dosyasi indiriliyor...
    ".venv\Scripts\python.exe" setup_project.py
)
".venv\Scripts\python.exe" -m app.main %*
if errorlevel 1 (
    echo.
    echo [HATA] Program beklenmedik sekilde sonlandi.
    pause
)
