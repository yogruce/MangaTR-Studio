"""
MangaTR-Studio - Multi-Chapter Batch Processor (app/core/batch_processor.py)

Designed for batch translating 1 to 400+ manga chapters while preserving directory structure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from app.core.pipeline import MangaPipeline


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class BatchJobSummary:
    input_root: Path
    output_root: Path
    total_pages: int = 0
    succeeded_pages: int = 0
    failed_pages: int = 0
    elapsed_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)


class BatchProcessor:
    """
    Processes an entire folder of manga images or a multi-chapter directory tree
    (e.g., 400 chapters) into localized Turkish pages.
    """

    def __init__(self, pipeline: MangaPipeline) -> None:
        self.pipeline = pipeline
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def discover_images(self, input_root: Path) -> List[Path]:
        if input_root.is_file() and input_root.suffix.lower() in IMAGE_EXTENSIONS:
            return [input_root]
        images: List[Path] = []
        for p in sorted(input_root.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                # Skip already-generated debug/output files if user points to a mixed folder
                if any(
                    p.stem.endswith(suf)
                    for suf in ("_tr", "_cleaned", "_detected", "_mask", "_mask_refined")
                ):
                    continue
                images.append(p)
        return images

    def run_batch(
        self,
        input_root: Path | str,
        output_root: Path | str,
        save_debug_artifacts: bool = False,
        skip_existing: bool = True,
        on_page_progress: Optional[Callable[[int, int, Path, str], None]] = None,
    ) -> BatchJobSummary:
        self._cancel_requested = False
        in_root = Path(input_root).resolve()
        out_root = Path(output_root).resolve()
        out_root.mkdir(parents=True, exist_ok=True)

        start_t = time.time()
        all_images = self.discover_images(in_root)
        summary = BatchJobSummary(
            input_root=in_root,
            output_root=out_root,
            total_pages=len(all_images),
        )

        for idx, img_path in enumerate(all_images, start=1):
            if self._cancel_requested:
                break

            rel_parent = (
                img_path.parent.relative_to(in_root)
                if in_root.is_dir() and img_path.parent != in_root
                else Path(".")
            )
            target_dir = out_root / rel_parent
            expected_out = target_dir / f"{img_path.stem}_tr.jpg"

            if skip_existing and expected_out.exists():
                summary.succeeded_pages += 1
                if on_page_progress:
                    on_page_progress(idx, summary.total_pages, img_path, "Atlandı (Zaten mevcut)")
                continue

            if on_page_progress:
                on_page_progress(idx, summary.total_pages, img_path, "İşleniyor...")

            try:
                self.pipeline.process_page(
                    image_path=img_path,
                    output_dir=target_dir,
                    save_debug_artifacts=save_debug_artifacts,
                )
                summary.succeeded_pages += 1
                if on_page_progress:
                    on_page_progress(idx, summary.total_pages, img_path, "Tamamlandı")
            except Exception as exc:
                summary.failed_pages += 1
                err_msg = f"{img_path.name}: {exc}"
                summary.errors.append(err_msg)
                if on_page_progress:
                    on_page_progress(idx, summary.total_pages, img_path, f"HATA: {exc}")

        summary.elapsed_seconds = round(time.time() - start_t, 2)
        return summary
