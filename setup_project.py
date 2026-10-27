"""
MangaTR-Studio - Otomatik Kurulum & Model İndirme Aracı (setup_project.py)

Bu betik GitHub'dan projeyi klonlayan kullanıcılar için:
1. Gerekli klasör yapısını (`input`, `output`, `models`, `fonts`, `temp`) oluşturur.
2. `comictextdetector.pt` (~80 MB) model ağırlık dosyasını otomatik indirir.
3. Windows sistem fontlarından Türkçe destekli Comic Sans Bold (`comicbd.ttf`) varsa `fonts/` klasörüne kopyalar.
4. `.env` dosyası yoksa `.env.example` şablonundan bilgi mesajı gösterir.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DETECTOR_MODEL_PATH = ROOT_DIR / "vendor" / "comic-text-detector" / "models" / "comictextdetector.pt"
DETECTOR_MODEL_URL = (
    "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt"
)
FONTS_DIR = ROOT_DIR / "fonts"


def ensure_directories() -> None:
    for d in ("input", "output", "models", "fonts", "projects/default", "temp", "vendor/comic-text-detector/models"):
        (ROOT_DIR / d).mkdir(parents=True, exist_ok=True)


def ensure_detector_model() -> None:
    if DETECTOR_MODEL_PATH.exists() and DETECTOR_MODEL_PATH.stat().st_size > 50_000_000:
        print(f"[OK] Comic Text Detector modeli mevcut: {DETECTOR_MODEL_PATH}")
        return

    print(f"[İNDİRİLİYOR] Comic Text Detector modeli indiriliyor (~80 MB)...")
    print(f"Kaynak: {DETECTOR_MODEL_URL}")
    tmp_path = DETECTOR_MODEL_PATH.with_suffix(".pt.part")

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            downloaded = min(total_size, block_num * block_size)
            pct = (downloaded / total_size) * 100.0
            mb_done = downloaded / (1024 * 1024)
            mb_tot = total_size / (1024 * 1024)
            print(f"\r  -> {mb_done:.1f} / {mb_tot:.1f} MB (%{pct:.1f})", end="", flush=True)

    urllib.request.urlretrieve(DETECTOR_MODEL_URL, str(tmp_path), reporthook=_progress)
    print()
    tmp_path.replace(DETECTOR_MODEL_PATH)
    print(f"[OK] Model başarıyla kaydedildi: {DETECTOR_MODEL_PATH}")


def setup_local_windows_fonts() -> None:
    win_fonts = Path("C:/Windows/Fonts")
    mapping = {
        "comicbd.ttf": "ComicSansMS-Bold.ttf",
        "comic.ttf": "ComicSansMS-Regular.ttf",
    }
    for src_name, dst_name in mapping.items():
        src_path = win_fonts / src_name
        dst_path = FONTS_DIR / dst_name
        if src_path.exists() and not dst_path.exists():
            try:
                shutil.copy2(src_path, dst_path)
                print(f"[OK] Sistem fontu kopyalandı: {dst_name}")
            except Exception:
                pass


def main() -> None:
    print("=" * 68)
    print("  MangaTR-Studio — Proje Hazırlama & Model Kurulum Sihirbazı")
    print("=" * 68)
    ensure_directories()
    setup_local_windows_fonts()
    ensure_detector_model()
    print("=" * 68)
    print("Kurulum tamamlandı!")
    print("- Masaüstü Arayüzünü başlatmak için : python app/main.py  (veya MangaTR-Studio.bat)")
    print("- CLI ile sayfa çevirmek için       : python app/main.py --cli -i input/test.jpg -o output/")
    print("- Opsiyonel API ayarları için       : .env.example dosyasını .env olarak kopyalayın")
    print("=" * 68)


if __name__ == "__main__":
    main()
