"""
MangaTR-Studio - Text Detection, Duplicate Cleanup & Line Polygon Engine (app/engines/detector.py)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from app.config import DetectorConfig, VENDOR_DIR

if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from inference import TextDetector  # noqa: E402
from utils.textmask import REFINEMASK_ANNOTATION  # noqa: E402


def clamp_box(box: List[int | float], width: int, height: int) -> List[int]:
    """Clamp [x1, y1, x2, y2] to image boundaries."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return [x1, y1, x2, y2]


def box_area(box: List[int]) -> float:
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))


def box_intersection(a: List[int], b: List[int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((x2 - x1) * (y2 - y1))


def box_iou(a: List[int], b: List[int]) -> float:
    inter = box_intersection(a, b)
    if inter <= 0.0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def box_containment(a: List[int], b: List[int]) -> float:
    """Return max(inter / area(a), inter / area(b))."""
    inter = box_intersection(a, b)
    if inter <= 0.0:
        return 0.0
    min_area = min(box_area(a), box_area(b))
    return inter / min_area if min_area > 0.0 else 0.0


def poly_to_bbox(poly: List[List[int]] | np.ndarray, width: int, height: int) -> List[int]:
    arr = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
    x1 = int(np.min(arr[:, 0]))
    y1 = int(np.min(arr[:, 1]))
    x2 = int(np.max(arr[:, 0]))
    y2 = int(np.max(arr[:, 1]))
    return clamp_box([x1, y1, x2, y2], width, height)


def bbox_to_poly(box: List[int]) -> List[List[int]]:
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


@dataclass
class DetectedBlock:
    id: int
    bbox: List[int]
    lines: List[List[List[int]]]  # List of 4-point polygons [[x,y], [x,y], [x,y], [x,y]]
    line_bboxes: List[List[int]]
    language: str = "eng"
    vertical: bool = False
    font_size: float = 24.0
    angle: int = 0
    mask_confidence: float = 0.0
    reading_order_weight: Optional[float] = None
    recovered_from_mask: bool = False
    merged_duplicates: int = 0
    raw_detector: Dict[str, Any] = field(default_factory=dict)


class MangaTextDetector:
    """
    High-precision manga text block & line detector built on Comic Text Detector,
    with duplicate block suppression, Japanese SFX filtering, smart vertical line
    completion, and manga reading-order sorting.
    """

    def __init__(self, config: Optional[DetectorConfig] = None, device: Optional[str] = None) -> None:
        self.config = config or DetectorConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not self.config.model_path.exists():
            if getattr(self.config, "auto_download_model", True):
                self._download_model_weights()
            else:
                raise FileNotFoundError(f"Comic Text Detector model not found: {self.config.model_path}")

        self.detector = TextDetector(
            model_path=str(self.config.model_path),
            input_size=self.config.input_size,
            device=self.device,
            half=False,
            nms_thresh=self.config.nms_thresh,
            conf_thresh=self.config.conf_thresh,
            act="leaky",
        )

    def _download_model_weights(self) -> None:
        """Automatically download comictextdetector.pt on first run if missing."""
        import urllib.request

        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.config.model_path.with_suffix(".pt.part")
        print(f"[MangaTR-Studio] Model indiriliyor: {self.config.model_url} -> {self.config.model_path}")
        try:
            urllib.request.urlretrieve(self.config.model_url, str(tmp_path))
            tmp_path.replace(self.config.model_path)
            print("[MangaTR-Studio] Model indirme tamamlandı.")
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Comic Text Detector modeli indirilemedi ({self.config.model_url}): {exc}\n"
                "Lütfen `python setup_project.py` komutunu çalıştırın."
            ) from exc

    def detect(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[DetectedBlock]]:
        """
        Run detection on a BGR image.
        Returns:
            raw_mask, refined_mask, detected_blocks
        """
        im_h, im_w = image.shape[:2]
        gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        raw_mask, mask_refined, raw_blocks = self.detector(
            image,
            refine_mode=REFINEMASK_ANNOTATION,
            keep_undetected_mask=True,
        )

        # Convert raw TextBlocks into internal candidates
        candidates: List[DetectedBlock] = []
        for idx, blk in enumerate(raw_blocks, start=1):
            bbox = clamp_box(blk.xyxy, im_w, im_h)
            raw_lang = str(getattr(blk, "language", "eng") or "eng")
            is_vert = bool(getattr(blk, "vertical", False))

            # Smart same-bubble vertical completion:
            # If YOLO cut off the bottom text line inside the same white speech bubble (e.g. "DAYS WERE HELL!!"),
            # expand y2 to include the connected horizontal text mask without crossing any dark panel border.
            bbox = self._expand_cut_bottom_line(bbox, gray_full, mask_refined, im_w, im_h)

            raw_lines = np.asarray(getattr(blk, "lines", []), dtype=np.int32)
            lines_list: List[List[List[int]]] = []
            line_bboxes: List[List[int]] = []

            if raw_lines.size > 0:
                for poly in raw_lines.reshape(-1, 4, 2):
                    lb = poly_to_bbox(poly, im_w, im_h)
                    line_bboxes.append(lb)
                    lines_list.append(poly.astype(int).tolist())
            else:
                line_bboxes.append(bbox)
                lines_list.append(bbox_to_poly(bbox))

            # If bbox expanded below the last line_bbox, re-extract complete line_bboxes from mask_refined
            if line_bboxes and (bbox[3] - max(lb[3] for lb in line_bboxes)) >= 14:
                split_polys, split_boxes = self._extract_lines_from_mask(mask_refined, bbox, im_w, im_h)
                if len(split_boxes) >= len(line_bboxes):
                    lines_list, line_bboxes = split_polys, split_boxes

            # Sort lines top-to-bottom, left-to-right
            sorted_pairs = sorted(
                zip(lines_list, line_bboxes),
                key=lambda item: ((item[1][1] + item[1][3]) * 0.5, item[1][0]),
            )
            lines_list = [p[0] for p in sorted_pairs]
            line_bboxes = [p[1] for p in sorted_pairs]

            if len(lines_list) == 1 and (bbox[3] - bbox[1]) > 42:
                split_polys, split_boxes = self._extract_lines_from_mask(mask_refined, bbox, im_w, im_h)
                if len(split_boxes) > 1:
                    lines_list, line_bboxes = split_polys, split_boxes

            weight_val = getattr(blk, "weight", None)
            try:
                weight_float = None if weight_val is None else round(float(weight_val), 2)
            except (TypeError, ValueError):
                weight_float = None

            try:
                native_dict = blk.to_dict()
            except Exception:
                native_dict = {}

            line_heights = [max(12, lb[3] - lb[1]) for lb in line_bboxes]
            est_font_size = float(np.median(line_heights)) if line_heights else float(getattr(blk, "font_size", 24.0))

            candidates.append(
                DetectedBlock(
                    id=idx,
                    bbox=bbox,
                    lines=lines_list,
                    line_bboxes=line_bboxes,
                    language=raw_lang,
                    vertical=is_vert,
                    font_size=round(est_font_size, 1),
                    angle=int(getattr(blk, "angle", 0)),
                    mask_confidence=self._compute_mask_confidence(raw_mask, mask_refined, bbox, line_bboxes),
                    reading_order_weight=weight_float,
                    recovered_from_mask=False,
                    merged_duplicates=0,
                    raw_detector=native_dict,
                )
            )

        # Step 1: Deduplicate overlapping blocks (e.g. when YOLO detects both 'eng' and 'ja' on the same bubble)
        unique_blocks = self._deduplicate_blocks(candidates, raw_mask, mask_refined, im_w, im_h)

        # Step 1b: Drop pure vertical Japanese SFX blocks ('ja' that did not merge with any 'eng' block)
        unique_blocks = [
            b for b in unique_blocks if not (b.language == "ja" and b.vertical)
        ]

        # Step 2: Recover missed enclosed WHITE speech bubbles from mask_refined (e.g. "'BOUT RIGHT...")
        # while strictly ignoring background Japanese SFX, faces, eyebrows, and screentone art!
        if self.config.recover_undetected_mask:
            unique_blocks, mask_refined = self._recover_or_clean_undetected_mask(
                image, unique_blocks, raw_mask, mask_refined, im_w, im_h
            )

        # Step 3: Sort in natural manga reading order and re-assign sequential IDs (1..N)
        sorted_blocks = self._sort_reading_order(unique_blocks, im_w, im_h)
        for new_id, block in enumerate(sorted_blocks, start=1):
            block.id = new_id

        return raw_mask, mask_refined, sorted_blocks

    def _expand_cut_bottom_line(
        self,
        bbox: List[int],
        gray_full: np.ndarray,
        mask_refined: np.ndarray,
        im_w: int,
        im_h: int,
    ) -> List[int]:
        """
        If a speech bubble bbox stopped above its final text line (e.g. 'DAYS WERE HELL!!'),
        expand y2 downward ONLY if the transition strip is bright (inside the same white bubble)
        and mask_refined has connected text strokes immediately below y2.
        """
        x1, y1, x2, y2 = bbox
        probe_y2 = min(im_h, y2 + 36)
        if probe_y2 <= y2 + 4:
            return bbox

        # Check that we are not crossing a black panel border at y2
        transition_strip = gray_full[max(0, y2 - 2) : min(im_h, y2 + 4), x1:x2]
        if transition_strip.size == 0 or float(np.mean(transition_strip)) < 155.0:
            return bbox

        below_mask = mask_refined[y2:probe_y2, max(0, x1 - 45) : min(im_w, x2 + 45)]
        ys, xs = np.where(below_mask > 40)
        if ys.size > 25:
            ox = max(0, x1 - 45)
            new_x1 = min(x1, max(0, ox + int(xs.min()) - 4))
            new_x2 = max(x2, min(im_w, ox + int(xs.max()) + 4))
            new_y2 = min(im_h, y2 + int(ys.max()) + 4)
            return [new_x1, y1, new_x2, new_y2]
        return bbox

    def _compute_mask_confidence(
        self,
        raw_mask: np.ndarray,
        mask_refined: np.ndarray,
        bbox: List[int],
        line_bboxes: List[List[int]],
    ) -> float:
        """Compute a real [0, 1] confidence score from refined text segmentation mask quality."""
        x1, y1, x2, y2 = bbox
        crop_ref = mask_refined[y1:y2, x1:x2]
        if crop_ref.size == 0:
            return 0.0

        fg_pixels = np.count_nonzero(crop_ref > 127)
        if fg_pixels == 0:
            return 0.0

        box_a = max(1.0, float((x2 - x1) * (y2 - y1)))
        density = fg_pixels / box_a
        density_score = min(1.0, density / 0.12) if density < 0.12 else min(1.0, (0.85 - density) / 0.40)
        mean_strength = float(np.mean(crop_ref[crop_ref > 60])) / 255.0
        conf = max(0.75, min(0.995, 0.35 * density_score + 0.65 * mean_strength))
        return round(conf, 4)

    def _deduplicate_blocks(
        self,
        blocks: List[DetectedBlock],
        raw_mask: np.ndarray,
        mask_refined: np.ndarray,
        im_w: int,
        im_h: int,
    ) -> List[DetectedBlock]:
        """
        Merge duplicate/overlapping blocks produced when YOLO detects both 'eng' and 'ja'
        classes on the same speech bubble. Prefers the candidate with richer multi-line polygons.
        """
        if len(blocks) <= 1:
            return blocks

        used = [False] * len(blocks)
        merged_blocks: List[DetectedBlock] = []

        for i, blk_a in enumerate(blocks):
            if used[i]:
                continue

            cluster = [blk_a]
            used[i] = True

            for j in range(i + 1, len(blocks)):
                if used[j]:
                    continue
                blk_b = blocks[j]
                iou = box_iou(blk_a.bbox, blk_b.bbox)
                cont = box_containment(blk_a.bbox, blk_b.bbox)
                if iou >= self.config.iou_dup_thresh or cont >= self.config.contain_dup_thresh:
                    cluster.append(blk_b)
                    used[j] = True

            if len(cluster) == 1:
                merged_blocks.append(blk_a)
                continue

            # If any candidate in the cluster was detected as 'eng', mark the merged block as 'eng'
            has_eng = any(b.language == "eng" for b in cluster)

            cluster.sort(
                key=lambda b: (len(b.lines), b.mask_confidence, box_area(b.bbox)),
                reverse=True,
            )
            primary = cluster[0]
            if has_eng:
                primary.language = "eng"

            ux1 = min(b.bbox[0] for b in cluster)
            uy1 = min(b.bbox[1] for b in cluster)
            ux2 = max(b.bbox[2] for b in cluster)
            uy2 = max(b.bbox[3] for b in cluster)
            union_bbox = clamp_box([ux1, uy1, ux2, uy2], im_w, im_h)

            primary.bbox = union_bbox
            primary.merged_duplicates = len(cluster) - 1
            primary.mask_confidence = self._compute_mask_confidence(
                raw_mask, mask_refined, primary.bbox, primary.line_bboxes
            )
            merged_blocks.append(primary)

        return merged_blocks

    def _extract_lines_from_mask(
        self,
        mask_refined: np.ndarray,
        bbox: List[int],
        im_w: int,
        im_h: int,
    ) -> Tuple[List[List[List[int]]], List[List[int]]]:
        """Split a text block region into horizontal line boxes using morphological horizontal grouping."""
        x1, y1, x2, y2 = bbox
        sub = mask_refined[y1:y2, x1:x2]
        if sub.size == 0:
            return [bbox_to_poly(bbox)], [bbox]

        _, bw = cv2.threshold(sub, 30, 255, cv2.THRESH_BINARY)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 2))
        connected = cv2.dilate(bw, horiz_kernel, iterations=1)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(connected)
        raw_boxes: List[List[int]] = []
        for idx in range(1, num_labels):
            lx, ly, lw, lh, area = stats[idx]
            if lw < 8 or lh < 6 or area < 35:
                continue
            raw_boxes.append(clamp_box([x1 + lx, y1 + ly, x1 + lx + lw, y1 + ly + lh], im_w, im_h))

        if not raw_boxes:
            return [bbox_to_poly(bbox)], [bbox]

        raw_boxes.sort(key=lambda b: (b[1], b[0]))
        merged_lines: List[List[int]] = [raw_boxes[0]]
        for box in raw_boxes[1:]:
            prev = merged_lines[-1]
            prev_cy = 0.5 * (prev[1] + prev[3])
            curr_cy = 0.5 * (box[1] + box[3])
            prev_h = max(1, prev[3] - prev[1])
            curr_h = max(1, box[3] - box[1])
            if abs(curr_cy - prev_cy) < 0.55 * max(prev_h, curr_h):
                merged_lines[-1] = [
                    min(prev[0], box[0]),
                    min(prev[1], box[1]),
                    max(prev[2], box[2]),
                    max(prev[3], box[3]),
                ]
            else:
                merged_lines.append(box)

        polys = [bbox_to_poly(lb) for lb in merged_lines]
        return polys, merged_lines

    def _recover_or_clean_undetected_mask(
        self,
        image: np.ndarray,
        blocks: List[DetectedBlock],
        raw_mask: np.ndarray,
        mask_refined: np.ndarray,
        im_w: int,
        im_h: int,
    ) -> Tuple[List[DetectedBlock], np.ndarray]:
        """
        Inspect connected components in mask_refined that are not covered by any DetectedBlock.
        - Small isolated components in the page margin (like page numbers '46') are removed from mask_refined.
        - ONLY recover an unboxed component if it sits inside a clean, uniform white speech bubble
          (e.g. "'BOUT RIGHT..."), strictly ignoring Japanese SFX (ゴゴゴ, ぐぐぐ, ヴァ) and artwork!
        """
        cleaned_mask = mask_refined.copy()
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(cleaned_mask, 30, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 18))
        grouped = cv2.dilate(thresh, kernel, iterations=1)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(grouped)

        next_id = len(blocks) + 1
        for i in range(1, num_labels):
            x, y, w, h, _ = stats[i]
            coarse_box = [int(x), int(y), int(x + w), int(y + h)]
            box_a = max(1.0, box_area(coarse_box))

            max_ov = max(
                (box_intersection(b.bbox, coarse_box) / box_a for b in blocks),
                default=0.0,
            )
            if max_ov >= 0.20:
                continue

            ys, xs = np.where(thresh[y : y + h, x : x + w] > 0)
            if len(xs) == 0:
                continue

            tight_box = clamp_box(
                [int(x + xs.min()), int(y + ys.min()), int(x + xs.max() + 1), int(y + ys.max() + 1)],
                im_w,
                im_h,
            )
            tw = tight_box[2] - tight_box[0]
            th = tight_box[3] - tight_box[1]

            in_bottom_margin = tight_box[1] > int(im_h * 0.90)
            in_top_margin = tight_box[3] < int(im_h * 0.04)
            is_tiny_noise = tw < 34 and th < 26

            if (in_bottom_margin and tw < 55 and th < 35) or in_top_margin or is_tiny_noise:
                pad = 4
                cleaned_mask[
                    max(0, tight_box[1] - pad) : min(im_h, tight_box[3] + pad),
                    max(0, tight_box[0] - pad) : min(im_w, tight_box[2] + pad),
                ] = 0
                continue

            # Strictly verify whether this unboxed mask region is inside a clean white speech bubble!
            # Background Japanese SFX (ゴゴゴ, ヴァ, ぐぐぐ) sit on screentone/artwork with high std or large size.
            if 36 <= tw <= 180 and 16 <= th <= 95:
                sub_gray = gray_img[tight_box[1] : tight_box[3], tight_box[0] : tight_box[2]]
                sub_mask = thresh[tight_box[1] : tight_box[3], tight_box[0] : tight_box[2]]
                bg_pixels = sub_gray[sub_mask == 0]
                if bg_pixels.size > 20:
                    bg_mean = float(np.mean(bg_pixels))
                    bg_std = float(np.std(bg_pixels))
                else:
                    bg_mean = 0.0
                    bg_std = 999.0

                # Only promote to a block if background is uniform white/light (> 212 mean, < 36 std)
                if bg_mean >= 212.0 and bg_std <= 36.0:
                    polys, line_boxes = self._extract_lines_from_mask(cleaned_mask, tight_box, im_w, im_h)
                    line_heights = [max(12, lb[3] - lb[1]) for lb in line_boxes]
                    est_font_size = float(np.median(line_heights)) if line_heights else float(th)

                    blocks.append(
                        DetectedBlock(
                            id=next_id,
                            bbox=tight_box,
                            lines=polys,
                            line_bboxes=line_boxes,
                            language="eng",
                            vertical=False,
                            font_size=round(est_font_size, 1),
                            angle=0,
                            mask_confidence=self._compute_mask_confidence(
                                raw_mask, cleaned_mask, tight_box, line_boxes
                            ),
                            reading_order_weight=None,
                            recovered_from_mask=True,
                            merged_duplicates=0,
                            raw_detector={"recovered_from_mask": True, "xyxy": tight_box},
                        )
                    )
                    next_id += 1
                    continue

            # Otherwise, it is an unboxed Japanese SFX or artwork stroke — remove from cleaned_mask
            # so cleaner never blurs or inpaints it!
            pad = 2
            cleaned_mask[
                max(0, tight_box[1] - pad) : min(im_h, tight_box[3] + pad),
                max(0, tight_box[0] - pad) : min(im_w, tight_box[2] + pad),
            ] = 0

        return blocks, cleaned_mask

    def _sort_reading_order(
        self,
        blocks: List[DetectedBlock],
        im_w: int,
        im_h: int,
    ) -> List[DetectedBlock]:
        """
        Sort blocks into natural reading bands (top-to-bottom rows, then right-to-left for manga
        or left-to-right for western comics).
        """
        if not blocks:
            return []

        rtl = self.config.reading_order != "comic_ltr"

        band_tol = max(75.0, im_h * 0.075)
        by_y = sorted(blocks, key=lambda b: 0.5 * (b.bbox[1] + b.bbox[3]))

        bands: List[List[DetectedBlock]] = []
        for blk in by_y:
            cy = 0.5 * (blk.bbox[1] + blk.bbox[3])
            if not bands:
                bands.append([blk])
                continue
            band_cy = float(np.mean([0.5 * (x.bbox[1] + x.bbox[3]) for x in bands[-1]]))
            if abs(cy - band_cy) <= band_tol:
                bands[-1].append(blk)
            else:
                bands.append([blk])

        ordered: List[DetectedBlock] = []
        for band in bands:
            if rtl:
                band.sort(key=lambda b: (-0.5 * (b.bbox[0] + b.bbox[2]), b.bbox[1]))
            else:
                band.sort(key=lambda b: (0.5 * (b.bbox[0] + b.bbox[2]), b.bbox[1]))
            ordered.extend(band)

        return ordered
