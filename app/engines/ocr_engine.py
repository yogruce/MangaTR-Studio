"""
MangaTR-Studio - Multi-Candidate Smart Manga OCR Engine (app/engines/ocr_engine.py)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import cv2
import easyocr
import numpy as np
import torch

from app.config import OCRConfig
from app.engines.detector import DetectedBlock, clamp_box
from app.utils.text_normalizer import clean_ocr_line, merge_ocr_lines, normalize_manga_english


_COMMON_ENGLISH_VOCAB: Set[str] = {
    "I", "A", "TO", "OF", "IN", "IT", "IS", "MY", "ME", "WE", "HE", "SO", "ON", "AT", "BY", "DO", "GO", "NO", "OR", "UP",
    "IF", "BE", "AS", "AN", "US", "AM", "GET", "OUT", "FOR", "NOT", "YOU", "ALL", "CAN", "HER", "WAS", "ONE", "OUR", "DAY",
    "HAD", "HAS", "HIM", "HIS", "HOW", "MAN", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY", "WHO", "BOY", "DID", "ITS", "LET",
    "PUT", "SAY", "SHE", "TOO", "USE", "DAD", "MOM", "KID", "THE", "AND", "ARE", "BUT", "NEVER", "WANT", "REMEMBER",
    "THOSE", "DAYS", "AGAIN", "HEAD", "PAST", "SAINT", "BEGONE", "STAY", "SEALED", "WERE", "HELL", "ALRIGHT", "DONE",
    "KEEP", "TOGETHER", "MENTAL", "DEFENSES", "DROP", "HUFF", "AAH", "AAA", "AAAA", "YEAH", "RIGHT", "WHALE", "ISLAND",
    "HUNTER", "STORM", "SPEED", "HOURS", "SHIP", "STEER", "WOW", "HEY", "WHAT", "WHEN", "WHERE", "WHY", "COME", "STOP",
    "WAIT", "LOOK", "HERE", "THERE", "THIS", "THAT", "WITH", "FROM", "HAVE", "WILL", "YOUR", "KNOW", "TIME", "BACK",
}


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
    is_valid_english: bool = True


class MangaOCREngine:
    """
    Manga-optimized Multi-Candidate EasyOCR engine:
    - Evaluates Standard, Dark-Border Flood-Fill (for white-outlined text on black panels),
      and Mask-Guided (for speed-line / screentone bubbles) crops
    - Selects the candidate with the highest English vocabulary & confidence score
    - Identifies and rejects unreadable Japanese SFX false-positives
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
        m_crop = mask_refined[py1:py2, px1:px2].copy() if mask_refined is not None else None

        candidates: List[np.ndarray] = [raw_crop]

        # Variant 2: Border-cleaned crop (removes curved bubble walls at corners)
        if m_crop is not None:
            clean_crop = self._prepare_clean_crop(raw_crop, m_crop, dilate_px=14)
            candidates.append(clean_crop)

            # Variant 3: Tight mask-guided crop (erases speed-lines passing through the bubble)
            mask_guided = self._prepare_mask_guided_crop(raw_crop, m_crop)
            candidates.append(mask_guided)

        # Variant 4: Dark-border flood-fill crop (converts white-outlined black text on dark panels into black-on-white)
        dark_ff = self._prepare_dark_border_floodfill(raw_crop)
        if dark_ff is not None:
            candidates.append(dark_ff)

        best_lines: List[OCRLineResult] = []
        best_text = ""
        best_conf = 0.0
        best_score = -1.0

        for cand_img in candidates:
            padded = cv2.copyMakeBorder(
                cand_img, 12, 12, 12, 12, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            line_results, avg_conf = self._ocr_block_lines(padded, [px1, py1, px2, py2])
            merged = merge_ocr_lines([lr.text for lr in line_results])
            score = self._english_quality_score(merged, avg_conf)
            if score > best_score:
                best_score = score
                best_lines = line_results
                best_text = merged
                best_conf = avg_conf

        raw_joined = " ".join(lr.text for lr in best_lines).strip()
        final_text = (
            normalize_manga_english(best_text)
            if self.config.normalize_manga_text
            else best_text
        )

        # Determine if this block is valid English text or a Japanese SFX false positive
        alpha_words = re.findall(r"[A-Za-z]{2,}", final_text)
        is_valid = bool(alpha_words) and (
            block.language == "eng" or best_score >= 1.8 or best_conf >= 0.40
        )

        return OCRBlockResult(
            block_id=block.id,
            bbox=block.bbox,
            text=final_text if is_valid else "",
            raw_text=raw_joined,
            ocr_confidence=round(float(best_conf), 4),
            mask_confidence=block.mask_confidence,
            lines=best_lines if is_valid else [],
            crop_image=raw_crop,
            is_valid_english=is_valid,
        )

    def _english_quality_score(self, text: str, conf: float) -> float:
        if not text:
            return 0.0
        words = re.findall(r"[A-Za-z']+", text.upper())
        if not words:
            return 0.0
        clean_words = [w.strip("'") for w in words if w.strip("'")]
        if not clean_words:
            return 0.0
        known_exact = sum(1 for w in clean_words if w in _COMMON_ENGLISH_VOCAB)
        multi_char = sum(1 for w in clean_words if len(w) >= 3)
        alpha_chars = sum(len(w) for w in clean_words)
        return (known_exact * 2.6 + multi_char * 0.9 + alpha_chars * 0.15) * (0.45 + 0.55 * max(0.0, conf))

    def _prepare_dark_border_floodfill(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        For white-outlined black text on a dark/black panel (e.g., flashback panels),
        flood-fill the outer dark background connected to the crop edges with white (255)
        so the interior black letters stand out cleanly on a pure white background.
        """
        if crop_bgr.size == 0:
            return None
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h < 8 or w < 8:
            return None

        border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        if float(np.median(border_pixels)) >= 115.0:
            return None

        _, bw = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        ff = bw.copy()
        mask = np.zeros((h + 2, w + 2), np.uint8)
        for x in range(w):
            if ff[0, x] == 0:
                cv2.floodFill(ff, mask, (x, 0), 255)
            if ff[h - 1, x] == 0:
                cv2.floodFill(ff, mask, (x, h - 1), 255)
        for y in range(h):
            if ff[y, 0] == 0:
                cv2.floodFill(ff, mask, (0, y), 255)
            if ff[y, w - 1] == 0:
                cv2.floodFill(ff, mask, (w - 1, y), 255)
        return cv2.cvtColor(ff, cv2.COLOR_GRAY2BGR)

    def _prepare_mask_guided_crop(self, crop_bgr: np.ndarray, m_crop: np.ndarray) -> np.ndarray:
        """Keep original pixels only inside the dilated text mask, erasing speed-lines outside letters."""
        if crop_bgr.size == 0 or m_crop.size == 0:
            return crop_bgr
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        zone = cv2.dilate(m_crop, k, iterations=1) > 25
        if not np.any(zone):
            return crop_bgr
        out = np.full_like(crop_bgr, 255)
        out[zone] = crop_bgr[zone]
        return out

    def _prepare_clean_crop(
        self,
        crop: np.ndarray,
        m_crop: np.ndarray,
        dilate_px: int = 14,
    ) -> np.ndarray:
        out = crop.copy()
        if out.size == 0 or m_crop.size == 0:
            return out

        if self.config.clean_bubble_borders and np.any(m_crop > 30):
            _, bin_m = cv2.threshold(m_crop, 30, 255, cv2.THRESH_BINARY)
            ksize = max(5, int(dilate_px) * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            safe_zone = cv2.dilate(bin_m, kernel, iterations=1)

            ring = (safe_zone > 0) & (bin_m == 0)
            if np.any(ring):
                bg_color = np.median(out[ring], axis=0).astype(np.uint8)
            else:
                bg_color = np.array([255, 255, 255], dtype=np.uint8)

            if float(np.mean(bg_color)) > 175:
                out[safe_zone == 0] = bg_color

        return out

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

        # Use balanced CRAFT thresholds so adjacent vertical text lines do not fuse into a single square blob
        results = self.reader.readtext(
            scaled,
            detail=1,
            paragraph=False,
            width_ths=0.7,
            height_ths=0.5,
            text_threshold=0.60,
            low_text=0.35,
            link_threshold=0.35,
            mag_ratio=1.0,
        )
        if not results:
            return [], 0.0

        snippets = []
        for poly, txt, conf in results:
            t = clean_ocr_line(str(txt))
            if not t:
                continue
            arr = np.asarray(poly, dtype=np.float32) / scale - 12.0
            x1 = int(round(crop_origin_box[0] + float(np.min(arr[:, 0]))))
            y1 = int(round(crop_origin_box[1] + float(np.min(arr[:, 1]))))
            x2 = int(round(crop_origin_box[0] + float(np.max(arr[:, 0]))))
            y2 = int(round(crop_origin_box[1] + float(np.max(arr[:, 1]))))
            snippets.append(([x1, y1, x2, y2], t, float(conf)))

        if not snippets:
            return [], 0.0

        snippets.sort(key=lambda s: ((s[0][1] + s[0][3]) * 0.5, s[0][0]))

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
