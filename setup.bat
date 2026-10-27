@echo off
chcp 65001 >nul
title MangaTR-Studio - Otomatik Kurulum (Venv + Paketler + Model)
cd /d "%~dp0"

echo ====================================================================
echo   MangaTR-Studio - Kurulum Baslatiliyor...
echo ====================================================================

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Python sanal ortami (.venv) olusturuluyor...
    python -m venv .venv
)

echo [2/3] Gerekli Python paketleri yukleniyor...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo [3/3] Yapay zeka modeli (comictextdetector.pt) ve fontlar hazirlaniyor...
".venv\Scripts\python.exe" setup_project.py

echo.
echo Kurulum basariyla tamamlandi! Uygulamayi baslatmak icin MangaTR-Studio.bat dosyasina cift tiklayabilirsiniz.
pause
