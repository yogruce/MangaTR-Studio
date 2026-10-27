"""
MangaTR-Studio - End-to-End & Interactive Page Processing Pipeline (app/core/pipeline.py)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import StudioConfig
from app.core.project import ProjectManager
from app.engines.cleaner import BlockCleanInfo, MangaCleaner
from app.engines.detector import DetectedBlock, MangaTextDetector
from app.engines.ocr_engine import MangaOCREngine, OCRBlockResult
from app.engines.translator import MangaTranslator
from app.engines.typesetter import MangaTypesetter
from app.utils.json_utils import save_json


@dataclass
class PagePipelineResult:
    input_path: Path
    original_bgr: np.ndarray
    detected_vis_bgr: np.ndarray
    raw_mask: np.ndarray
    refined_mask: np.ndarray
    cleaned_bgr: np.ndarray
    final_tr_bgr: np.ndarray
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    clean_infos: List[BlockCleanInfo] = field(default_factory=list)
    output_image_path: Optional[Path] = None
    output_json_path: Optional[Path] = None


class MangaPipeline:
    """
    Coordinates the 5 stages of MangaTR-Studio:
    1. Detection & Duplicate Block Cleanup (+ Japanese SFX Rejection)
    2. Multi-Candidate Smart OCR
    3. Context-Aware & Vision-Assisted Page Translation
    4. Hybrid Bubble Cleaning & Inpainting (only on valid English dialogue blocks!)
    5. Turkish Typesetting & Instant Re-Typesetting
    """

    def __init__(
        self,
        config: Optional[StudioConfig] = None,
        project: Optional[ProjectManager] = None,
        device: Optional[str] = None,
    ) -> None:
        import torch

        self.config = config or StudioConfig()
        self.config.ensure_dirs()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.project = project or ProjectManager("default")

        self._detector: Optional[MangaTextDetector] = None
        self._ocr: Optional[MangaOCREngine] = None
        self._translator: Optional[MangaTranslator] = None
        self._cleaner: Optional[MangaCleaner] = None
        self._typesetter: Optional[MangaTypesetter] = None

    @property
    def detector(self) -> MangaTextDetector:
        if self._detector is None:
            self._detector = MangaTextDetector(self.config.detector, device=self.device)
        return self._detector

    @property
    def ocr(self) -> MangaOCREngine:
        if self._ocr is None:
            self._ocr = MangaOCREngine(self.config.ocr, device=self.device)
        return self._ocr

    @property
    def translator(self) -> MangaTranslator:
        if self._translator is None:
            self._translator = MangaTranslator(self.config.translator)
        return self._translator

    @property
    def cleaner(self) -> MangaCleaner:
        if self._cleaner is None:
            self._cleaner = MangaCleaner(self.config.cleaner)
        return self._cleaner

    @property
    def typesetter(self) -> MangaTypesetter:
        if self._typesetter is None:
            self._typesetter = MangaTypesetter(self.config.typesetter)
        return self._typesetter

    def process_page(
        self,
        image_path: Path | str,
        output_dir: Optional[Path | str] = None,
        save_debug_artifacts: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> PagePipelineResult:
        in_path = Path(image_path).resolve()
        if not in_path.exists():
            raise FileNotFoundError(f"Görüntü bulunamadı: {in_path}")

        def notify(msg: str, pct: float) -> None:
            if progress_callback:
                progress_callback(msg, pct)

        notify("Görsel okunuyor...", 0.05)
        img_bytes = np.fromfile(str(in_path), dtype=np.uint8)
        image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Görsel OpenCV tarafından okunamadı: {in_path}")

        # 1. Text Detection & Duplicate Cleanup
        notify("Metin bölgeleri ve satırlar tespit ediliyor (Comic Text Detector)...", 0.18)
        raw_mask, refined_mask, detected_blocks = self.detector.detect(image)

        # 2. Multi-Candidate Smart OCR
        notify(f"EasyOCR ile {len(detected_blocks)} konuşma bloğu okunuyor...", 0.40)
        ocr_results = self.ocr.recognize_blocks(image, detected_blocks, mask_refined=refined_mask)

        # Filter out blocks that are clearly non-English SFX before translation (unless Gemini Vision is enabled)
        use_gemini_vision = (
            self.config.translator.backend.lower() in ("auto", "gemini_api")
            and bool(self.config.translator.gemini_api_key.strip())
        )
        if not use_gemini_vision:
            filtered_pairs: List[Tuple[DetectedBlock, OCRBlockResult]] = [
                (blk, ocr_res)
                for blk, ocr_res in zip(detected_blocks, ocr_results)
                if ocr_res.is_valid_english and ocr_res.text.strip()
            ]
            detected_blocks = [p[0] for p in filtered_pairs]
            ocr_results = [p[1] for p in filtered_pairs]

        # 3. Context-Aware & Vision-Assisted Page Translation
        notify("Diyaloglar bağlama uygun Türkçeye çevriliyor...", 0.62)
        trans_inputs = [
            {"id": blk.id, "text": ocr_res.text, "crop": ocr_res.crop_image}
            for blk, ocr_res in zip(detected_blocks, ocr_results)
        ]
        translations = self.translator.translate_page(
            trans_inputs,
            custom_glossary=self.project.glossary,
        )
        trans_map = {t.block_id: t for t in translations}

        # Filter out any block flagged as non-English SFX (is_sfx=True) or empty translation
        # BEFORE cleaning/inpainting so Japanese SFX are never blurred or erased!
        active_blocks: List[DetectedBlock] = []
        active_ocr_results: List[OCRBlockResult] = []
        for blk, ocr_res in zip(detected_blocks, ocr_results):
            t_item = trans_map.get(blk.id)
            if t_item is not None and t_item.is_sfx:
                continue
            if t_item is not None and t_item.source_text.strip():
                ocr_res.text = t_item.source_text.strip()
            if not ocr_res.text.strip() and (t_item is None or not t_item.translated_text.strip()):
                continue
            active_blocks.append(blk)
            active_ocr_results.append(ocr_res)

        detected_blocks = active_blocks
        ocr_results = active_ocr_results

        # Re-number active blocks sequentially (1..N)
        old_to_new_id: Dict[int, int] = {}
        for new_idx, (blk, ocr_res) in enumerate(zip(detected_blocks, ocr_results), start=1):
            old_to_new_id[blk.id] = new_idx
            blk.id = new_idx
            ocr_res.block_id = new_idx

        trans_map = {
            old_to_new_id[old_id]: item
            for old_id, item in trans_map.items()
            if old_id in old_to_new_id
        }

        # 4. Hybrid Text Cleaning & Background Inpainting (only on valid active dialogue blocks!)
        notify("İngilizce metinler doğal arka plan korunarak temizleniyor...", 0.78)
        cleaned_bgr, clean_infos = self.cleaner.clean_page(image, refined_mask, detected_blocks)
        clean_map = {ci.block_id: ci for ci in clean_infos}

        # Build unified block records
        block_records: List[Dict[str, Any]] = []
        for blk, ocr_res in zip(detected_blocks, ocr_results):
            t_item = trans_map.get(blk.id)
            c_info = clean_map.get(blk.id)
            tr_text = t_item.translated_text if t_item else ""
            backend_used = t_item.backend_used if t_item else "none"

            line_dicts = [
                {
                    "line_index": lr.index,
                    "bbox": lr.bbox,
                    "text": lr.text,
                    "confidence": lr.confidence,
                }
                for lr in ocr_res.lines
            ]

            block_records.append(
                {
                    "id": blk.id,
                    "bbox": blk.bbox,
                    "safe_box": c_info.safe_text_box if c_info else blk.bbox,
                    "language": blk.language,
                    "vertical": blk.vertical,
                    "text": ocr_res.text,
                    "translation": tr_text,
                    "translation_backend": backend_used,
                    "confidence": ocr_res.ocr_confidence,
                    "ocr_confidence": ocr_res.ocr_confidence,
                    "mask_confidence": blk.mask_confidence,
                    "font_size": blk.font_size,
                    "render_font_size": 0,
                    "font_name": "",
                    "clean_mode": c_info.mode if c_info else "smart_bubble_fill",
                    "recovered_from_mask": blk.recovered_from_mask,
                    "merged_duplicates": blk.merged_duplicates,
                    "lines": line_dicts,
                    "line_bboxes": blk.line_bboxes,
                }
            )

        # 5. Turkish Typesetting
        notify("Türkçe metinler balon geometrisine göre yerleştiriliyor...", 0.90)
        final_tr_bgr, typeset_specs = self.typesetter.typeset_page(
            cleaned_bgr, block_records, clean_infos=clean_infos
        )
        spec_map = {s.block_id: s for s in typeset_specs}
        for rec in block_records:
            sp = spec_map.get(rec["id"])
            if sp:
                rec["rendered_font_size"] = sp.font_size
                rec["rendered_font_name"] = sp.font_name
                rec["rendered_lines"] = sp.lines

        vis_bgr = self._draw_detection_visualization(image, block_records)

        # 6. Save outputs if output_dir provided
        out_img_path: Optional[Path] = None
        out_json_path: Optional[Path] = None
        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = in_path.stem
            out_img_path = out_dir / f"{stem}_tr.jpg"
            out_json_path = out_dir / f"{stem}_tr.json"

            self._imwrite_unicode(out_img_path, final_tr_bgr, quality=self.config.jpeg_quality)

            if save_debug_artifacts:
                self._imwrite_unicode(out_dir / f"{stem}_cleaned.jpg", cleaned_bgr, quality=self.config.jpeg_quality)
                self._imwrite_unicode(out_dir / f"{stem}_detected.jpg", vis_bgr, quality=self.config.jpeg_quality)

            payload = {
                "source_image": in_path.name,
                "output_image": out_img_path.name,
                "image_width": int(image.shape[1]),
                "image_height": int(image.shape[0]),
                "device": self.device,
                "reading_order": self.config.detector.reading_order,
                "block_count": len(block_records),
                "blocks": block_records,
            }
            save_json(out_json_path, payload)

        notify("Tamamlandı!", 1.0)
        return PagePipelineResult(
            input_path=in_path,
            original_bgr=image,
            detected_vis_bgr=vis_bgr,
            raw_mask=raw_mask,
            refined_mask=refined_mask,
            cleaned_bgr=cleaned_bgr,
            final_tr_bgr=final_tr_bgr,
            blocks=block_records,
            clean_infos=clean_infos,
            output_image_path=out_img_path,
            output_json_path=out_json_path,
        )

    def retypeset_from_edits(self, page_result: PagePipelineResult) -> np.ndarray:
        """
        Instantly re-render the Turkish page after the user edits block bboxes,
        translations or font sizes in the UI, without re-running detection or OCR.
        """
        final_tr_bgr, specs = self.typesetter.typeset_page(
            page_result.cleaned_bgr,
            page_result.blocks,
            clean_infos=page_result.clean_infos,
        )
        spec_map = {s.block_id: s for s in specs}
        for rec in page_result.blocks:
            sp = spec_map.get(rec["id"])
            if sp:
                rec["rendered_font_size"] = sp.font_size
                rec["rendered_font_name"] = sp.font_name
                rec["rendered_lines"] = sp.lines

        page_result.final_tr_bgr = final_tr_bgr
        page_result.detected_vis_bgr = self._draw_detection_visualization(
            page_result.original_bgr, page_result.blocks
        )
        if page_result.output_image_path:
            self._imwrite_unicode(
                page_result.output_image_path, final_tr_bgr, quality=self.config.jpeg_quality
            )
        if page_result.output_json_path:
            save_json(
                page_result.output_json_path,
                {
                    "source_image": page_result.input_path.name,
                    "output_image": page_result.output_image_path.name if page_result.output_image_path else "",
                    "image_width": int(page_result.original_bgr.shape[1]),
                    "image_height": int(page_result.original_bgr.shape[0]),
                    "device": self.device,
                    "block_count": len(page_result.blocks),
                    "blocks": page_result.blocks,
                },
            )
        return final_tr_bgr

    def _draw_detection_visualization(
        self,
        image: np.ndarray,
        block_records: List[Dict[str, Any]],
    ) -> np.ndarray:
        vis = image.copy()
        for rec in block_records:
            idx = int(rec["id"])
            x1, y1, x2, y2 = rec["bbox"]
            for lb in rec.get("line_bboxes", []):
                lx1, ly1, lx2, ly2 = lb
                cv2.rectangle(vis, (lx1, ly1), (lx2, ly2), (0, 165, 255), 1)

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 210, 0), 2)
            label = f"#{idx:02d} ({rec.get('confidence', 0.0):.2f})"
            cv2.putText(
                vis,
                label,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 60, 0),
                2,
                cv2.LINE_AA,
            )
        return vis

    @staticmethod
    def _imwrite_unicode(path: Path, bgr_img: np.ndarray, quality: int = 95) -> None:
        ext = path.suffix.lower() or ".jpg"
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)] if ext in (".jpg", ".jpeg") else []
        ok, encoded = cv2.imencode(ext, bgr_img, params)
        if ok:
            encoded.tofile(str(path))
