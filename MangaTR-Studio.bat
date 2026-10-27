@echo off
chcp 65001 >nul
title MangaTR-Studio — AI Manga Ceviri ve Typesetting Studyosu
cd /d "%~dp0"

echo ======================================================================
echo   MangaTR Studio — AI Manga Ceviri ve Typesetting Studyosu Baslatiliyor
echo ======================================================================

set "VENV_PY="
if exist "%~dp0.venv\Scripts\python.exe" (
    set "VENV_PY=%~dp0.venv\Scripts\python.exe"
) else if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "VENV_PY=%~dp0..\.venv\Scripts\python.exe"
)

if "%VENV_PY%"=="" (
    echo [BILGI] Sanal ortam ^(.venv^) bulunamadi. Otomatik kurulum ^(setup.bat^) baslatiliyor...
    call "%~dp0setup.bat"
    if exist "%~dp0.venv\Scripts\python.exe" (
        set "VENV_PY=%~dp0.venv\Scripts\python.exe"
    ) else (
        echo [HATA] Sanal ortam bulunamadi. Lutfen once setup.bat dosyasini calistirin.
        pause
        exit /b 1
    )
)

if not exist "%~dp0vendor\comic-text-detector\models\comictextdetector.pt" (
    echo [BILGI] Model dosyasi kontrol ediliyor / indiriliyor...
    "%VENV_PY%" "%~dp0setup_project.py"
)

"%VENV_PY%" -m app.main %*
if errorlevel 1 (
    echo.
    echo [HATA] Program beklenmedik sekilde sonlandi.
    pause
)
