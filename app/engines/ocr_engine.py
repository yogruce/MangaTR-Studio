"""
MangaTR-Studio - Line-Level & Block-Level EasyOCR Engine (app/engines/ocr_engine.py)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import easyocr
import numpy as np
import torch

from app.config import OCRConfig
from app.engines.detector import DetectedBlock, clamp_box
from app.utils.text_normalizer import clean_ocr_line, merge_ocr_lines, normalize_manga_english


@dataclass
class OCRLineResult:
    index: int
    bbox: List[int]
    text: str
    confidence: float


@dataclass
class OCRBlockResult:
    block_id: int
    bbox: List[int]
    text: str
    raw_text: str
    ocr_confidence: float
    mask_confidence: float
    lines: List[OCRLineResult] = field(default_factory=list)
    crop_image: Optional[np.ndarray] = None


class MangaOCREngine:
    """
    Manga-optimized EasyOCR engine:
    - Suppresses curved speech-bubble border lines outside the dilated text mask
    - Extracts structured line-by-line OCR results within each speech bubble
    - Stitches hyphenated line-wraps and normalizes manga typography
    """

    def __init__(self, config: Optional[OCRConfig] = None, device: Optional[str] = None) -> None:
        self.config = config or OCRConfig()
        use_cuda = (device == "cuda") if device else torch.cuda.is_available()
        self.gpu = bool(self.config.gpu and use_cuda)
        self.reader = easyocr.Reader(
            self.config.languages,
            gpu=self.gpu,
            verbose=False,
        )

    def recognize_blocks(
        self,
        image: np.ndarray,
        blocks: List[DetectedBlock],
        mask_refined: Optional[np.ndarray] = None,
    ) -> List[OCRBlockResult]:
        return [self.recognize_block(image, blk, mask_refined=mask_refined) for blk in blocks]

    def recognize_block(
        self,
        image: np.ndarray,
        block: DetectedBlock,
        mask_refined: Optional[np.ndarray] = None,
    ) -> OCRBlockResult:
        im_h, im_w = image.shape[:2]
        pad = self.config.block_padding
        bx1, by1, bx2, by2 = clamp_box(block.bbox, im_w, im_h)
        px1 = max(0, bx1 - pad)
        py1 = max(0, by1 - pad)
        px2 = min(im_w, bx2 + pad)
        py2 = min(im_h, by2 + pad)

        raw_crop = image[py1:py2, px1:px2].copy()

        # Prepare border-cleaned crop (removes curved bubble outlines at crop corners)
        clean_crop = self._prepare_clean_crop(
            image, [px1, py1, px2, py2], mask_refined, dilate_px=15
        )

        # Primary: structured line extraction on the border-cleaned block
        line_results, avg_conf = self._ocr_block_lines(clean_crop, [px1, py1, px2, py2])

        # Also run on unmasked padded raw_crop in case mask_refined clipped punctuation
        raw_padded = cv2.copyMakeBorder(
            raw_crop, 10, 10, 10, 10, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        raw_line_results, raw_avg_conf = self._ocr_block_lines(raw_padded, [px1, py1, px2, py2])

        clean_merged = merge_ocr_lines([lr.text for lr in line_results])
        raw_merged = merge_ocr_lines([lr.text for lr in raw_line_results])

        if not clean_merged and raw_merged:
            chosen_lines = raw_line_results
            chosen_text = raw_merged
            chosen_conf = raw_avg_conf
        else:
            chosen_lines = line_results if line_results else raw_line_results
            chosen_text = clean_merged if clean_merged else raw_merged
            chosen_conf = max(avg_conf, raw_avg_conf)

        raw_joined = " ".join(lr.text for lr in chosen_lines).strip()
        final_text = (
            normalize_manga_english(chosen_text)
            if self.config.normalize_manga_text
            else chosen_text
        )

        return OCRBlockResult(
            block_id=block.id,
            bbox=block.bbox,
            text=final_text,
            raw_text=raw_joined,
            ocr_confidence=round(float(chosen_conf), 4),
            mask_confidence=block.mask_confidence,
            lines=chosen_lines,
            crop_image=raw_crop,
        )

    def _prepare_clean_crop(
        self,
        image: np.ndarray,
        box: List[int],
        mask_refined: Optional[np.ndarray],
        dilate_px: int = 15,
    ) -> np.ndarray:
        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return crop

        if self.config.clean_bubble_borders and mask_refined is not None:
            m_crop = mask_refined[y1:y2, x1:x2]
            if m_crop.size > 0 and np.any(m_crop > 30):
                _, bin_m = cv2.threshold(m_crop, 30, 255, cv2.THRESH_BINARY)
                ksize = max(5, int(dilate_px) * 2 + 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                safe_zone = cv2.dilate(bin_m, kernel, iterations=1)

                ring = (safe_zone > 0) & (bin_m == 0)
                if np.any(ring):
                    bg_color = np.median(crop[ring], axis=0).astype(np.uint8)
                else:
                    bg_color = np.array([255, 255, 255], dtype=np.uint8)

                if float(np.mean(bg_color)) > 175:
                    crop[safe_zone == 0] = bg_color

        margin = 10
        return cv2.copyMakeBorder(
            crop,
            margin,
            margin,
            margin,
            margin,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    def _ocr_block_lines(
        self,
        padded_crop: np.ndarray,
        crop_origin_box: List[int],
    ) -> Tuple[List[OCRLineResult], float]:
        if padded_crop.size == 0:
            return [], 0.0

        scale = self.config.scale
        scaled = (
            cv2.resize(padded_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            if scale != 1.0
            else padded_crop
        )

        results = self.reader.readtext(
            scaled,
            detail=1,
            paragraph=False,
            width_ths=0.7,
            height_ths=0.5,
            text_threshold=0.35,
            low_text=0.22,
            link_threshold=0.20,
            mag_ratio=1.0,
        )
        if not results:
            return [], 0.0

        snippets = []
        for poly, txt, conf in results:
            t = clean_ocr_line(str(txt))
            if not t:
                continue
            arr = np.asarray(poly, dtype=np.float32) / scale - 10.0
            x1 = int(round(crop_origin_box[0] + float(np.min(arr[:, 0]))))
            y1 = int(round(crop_origin_box[1] + float(np.min(arr[:, 1]))))
            x2 = int(round(crop_origin_box[0] + float(np.max(arr[:, 0]))))
            y2 = int(round(crop_origin_box[1] + float(np.max(arr[:, 1]))))
            snippets.append(([x1, y1, x2, y2], t, float(conf)))

        if not snippets:
            return [], 0.0

        snippets.sort(key=lambda s: ((s[0][1] + s[0][3]) * 0.5, s[0][0]))

        # Group horizontal fragments on the same line into a single OCRLineResult
        rows: List[List[Tuple[List[int], str, float]]] = [[snippets[0]]]
        for snip in snippets[1:]:
            prev_row = rows[-1]
            prev_cy = float(np.mean([(b[1] + b[3]) * 0.5 for b, _, _ in prev_row]))
            prev_h = float(np.mean([max(10, b[3] - b[1]) for b, _, _ in prev_row]))
            curr_cy = (snip[0][1] + snip[0][3]) * 0.5
            if abs(curr_cy - prev_cy) <= 0.55 * prev_h:
                prev_row.append(snip)
            else:
                rows.append([snip])

        line_objs: List[OCRLineResult] = []
        all_confs: List[float] = []
        for idx, row in enumerate(rows, start=1):
            row.sort(key=lambda s: s[0][0])
            rx1 = min(s[0][0] for s in row)
            ry1 = min(s[0][1] for s in row)
            rx2 = max(s[0][2] for s in row)
            ry2 = max(s[0][3] for s in row)
            r_text = " ".join(s[1] for s in row).strip()
            r_conf = float(np.mean([s[2] for s in row]))
            all_confs.append(r_conf)
            line_objs.append(
                OCRLineResult(
                    index=idx,
                    bbox=[rx1, ry1, rx2, ry2],
                    text=normalize_manga_english(r_text)
                    if self.config.normalize_manga_text
                    else r_text,
                    confidence=round(r_conf, 4),
                )
            )

        return line_objs, (float(np.mean(all_confs)) if all_confs else 0.0)
