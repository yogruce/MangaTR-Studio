@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ======================================================================
echo   MangaTR Studio — AI Manga Ceviri ve Typesetting Studyosu Baslatiliyor
echo ======================================================================
".\.venv\Scripts\python.exe" -m app.main %*
if errorlevel 1 (
    echo.
    echo [HATA] Program beklenmedik sekilde sonlandi.
    pause
)
