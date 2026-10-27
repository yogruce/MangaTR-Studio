"""
MangaTR-Studio - Standalone Windows .EXE Builder (build_exe.py)

Usage:
    .\.venv\Scripts\python.exe build_exe.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


def ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller kuruluyor...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])


def build() -> None:
    ensure_pyinstaller()
    print("=" * 70)
    print("MangaTR-Studio Standalone .EXE derlemesi başlatılıyor...")
    print("=" * 70)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--windowed",
        "--name",
        "MangaTR-Studio",
        "--add-data",
        f"{ROOT_DIR / 'fonts'};fonts",
        "--add-data",
        f"{ROOT_DIR / 'projects'};projects",
        "--add-data",
        f"{ROOT_DIR / 'vendor' / 'comic-text-detector'};vendor/comic-text-detector",
        "--collect-all",
        "easyocr",
        str(ROOT_DIR / "app" / "main.py"),
    ]

    subprocess.check_call(cmd, cwd=str(ROOT_DIR))
    print("\nDerleme tamamlandı!")
    print(f"Çalıştırılabilir dosya: {ROOT_DIR / 'dist' / 'MangaTR-Studio' / 'MangaTR-Studio.exe'}")


if __name__ == "__main__":
    build()
