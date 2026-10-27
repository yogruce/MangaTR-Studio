"""
MangaTR-Studio - Main Application Entry Point (app/main.py)

Usage:
  1. Launch Desktop GUI (default):
       python -m app.main

  2. Translate a single page via CLI:
       python -m app.main --cli --input input/test.jpg --output output

  3. Batch translate a folder / 400 chapters via CLI:
       python -m app.main --cli --batch-dir input --output output/batch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import INPUT_DIR, OUTPUT_DIR, StudioConfig  # noqa: E402
from app.core.batch_processor import BatchProcessor  # noqa: E402
from app.core.pipeline import MangaPipeline  # noqa: E402


def run_cli(args: argparse.Namespace) -> None:
    config = StudioConfig()
    config.translator.backend = args.backend
    config.detector.reading_order = args.reading_order
    pipeline = MangaPipeline(config)

    if args.batch_dir:
        batch = BatchProcessor(pipeline)
        in_root = Path(args.batch_dir)
        out_root = Path(args.output)
        print("=" * 74)
        print(f"MangaTR-Studio — Toplu Bölüm Çeviri (Batch): {in_root} -> {out_root}")
        print("=" * 74)

        def on_prog(idx: int, total: int, path: Path, status: str) -> None:
            print(f"[{idx:04d}/{total:04d}] {path.name:<28} | {status}")

        summary = batch.run_batch(
            input_root=in_root,
            output_root=out_root,
            save_debug_artifacts=args.debug,
            skip_existing=not args.overwrite,
            on_page_progress=on_prog,
        )
        print("-" * 74)
        print(
            f"Bitti! Toplam: {summary.total_pages}, Başarılı: {summary.succeeded_pages}, "
            f"Hatalı: {summary.failed_pages}, Süre: {summary.elapsed_seconds} sn"
        )
        return

    input_path = Path(args.input)
    output_dir = Path(args.output)
    print("=" * 74)
    print("MangaTR-Studio — Tam Sayfa Lokalizasyon Pipeline")
    print("=" * 74)
    print(f"Görsel : {input_path}")
    print(f"Çıkış  : {output_dir}")
    print(f"Cihaz  : {pipeline.device}")

    def on_step(msg: str, pct: float) -> None:
        print(f"[{int(pct * 100):3d}%] {msg}")

    res = pipeline.process_page(
        image_path=input_path,
        output_dir=output_dir,
        save_debug_artifacts=True,
        progress_callback=on_step,
    )

    print("\n" + "-" * 74)
    for b in res.blocks:
        print(f"[#{b['id']:02d}] EN: {b['text']}")
        print(f"      TR: {b['translation']}  (font={b.get('rendered_font_size')}pt, mod={b.get('clean_mode')})")
    print("-" * 74)
    print(f"Türkçe Final JPG : {res.output_image_path}")
    print(f"Proje JSON       : {res.output_json_path}")


def run_gui() -> None:
    import tkinter as tk
    from app.ui.main_window import MangaTRMainWindow

    root = tk.Tk()
    _app = MangaTRMainWindow(root)
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="MangaTR Studio — AI Manga Localization System")
    parser.add_argument("--cli", action="store_true", help="Arayüz yerine komut satırı modunda çalıştır")
    parser.add_argument("--input", type=str, default=str(INPUT_DIR / "test.jpg"), help="Tekil manga sayfası yolu")
    parser.add_argument("--batch-dir", type=str, default="", help="Toplu çevrilecek klasör yolu")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Çıktı klasörü")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "free_web", "local_llm", "gemini_api"])
    parser.add_argument("--reading-order", type=str, default="manga_rtl", choices=["manga_rtl", "comic_ltr"])
    parser.add_argument("--overwrite", action="store_true", help="Batch modunda mevcut dosyaların üzerine yaz")
    parser.add_argument("--debug", action="store_true", help="Temizlenmiş ve tespit görsellerini de kaydet")

    args = parser.parse_args()
    if args.cli or args.batch_dir:
        run_cli(args)
    else:
        run_gui()


if __name__ == "__main__":
    main()
