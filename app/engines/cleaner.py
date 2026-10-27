"""
MangaTR-Studio - Hybrid Speech Bubble & Background Text Cleaner (app/engines/cleaner.py)

Implements the 2-Tier cleaning strategy:
1. Uniform Speech Bubbles -> Exact background tone sampling + border-safe mask fill
2. Textured / Floating Text -> Localized Telea inpainting via MangaInpainter
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.config import CleanerConfig
from app.engines.detector import DetectedBlock, clamp_box
from app.engines.inpainter import MangaInpainter


@dataclass
class BlockCleanInfo:
    block_id: int
    bbox: List[int]
    mode: str  # "smart_bubble_fill" or "complex_inpaint"
    bg_color_bgr: Tuple[int, int, int]
    fg_color_bgr: Tuple[int, int, int]
    is_dark_bubble: bool
    safe_text_box: List[int]  # Usable interior box [x1, y1, x2, y2] for typesetting


class MangaCleaner:
    """
    Cleans original English text from manga pages while strictly protecting:
    - Speech bubble borders
    - Artwork outside the text bounding box (e.g., pipes, hair, panel borders)
    """

    def __init__(self, config: Optional[CleanerConfig] = None) -> None:
        self.config = config or CleanerConfig()
        self.inpainter = MangaInpainter(radius=self.config.inpaint_radius)

    def clean_page(
        self,
        image: np.ndarray,
        mask_refined: np.ndarray,
        blocks: List[DetectedBlock],
    ) -> Tuple[np.ndarray, List[BlockCleanInfo]]:
        cleaned = image.copy()
        im_h, im_w = cleaned.shape[:2]
        clean_infos: List[BlockCleanInfo] = []

        for blk in blocks:
            info = self._clean_single_block(cleaned, mask_refined, blk, im_w, im_h)
            clean_infos.append(info)

        return cleaned, clean_infos

    def _clean_single_block(
        self,
        canvas: np.ndarray,
        mask_refined: np.ndarray,
        block: DetectedBlock,
        im_w: int,
        im_h: int,
    ) -> BlockCleanInfo:
        # Strictly constrain cleaning window to block.bbox (+3px) so outside artwork is never touched
        pad = 3
        bx1, by1, bx2, by2 = clamp_box(block.bbox, im_w, im_h)
        wx1 = max(0, bx1 - pad)
        wy1 = max(0, by1 - pad)
        wx2 = min(im_w, bx2 + pad)
        wy2 = min(im_h, by2 + pad)

        patch = canvas[wy1:wy2, wx1:wx2]
        raw_m = mask_refined[wy1:wy2, wx1:wx2].copy()

        # Also build a polygon mask from block.line_bboxes so any stray mask noise outside lines is ignored
        if len(block.line_bboxes) > 0:
            line_gate = np.zeros_like(raw_m)
            for lb in block.line_bboxes:
                lx1 = max(0, lb[0] - wx1 - 6)
                ly1 = max(0, lb[1] - wy1 - 6)
                lx2 = min(raw_m.shape[1], lb[2] - wx1 + 6)
                ly2 = min(raw_m.shape[0], lb[3] - wy1 + 6)
                line_gate[ly1:ly2, lx1:lx2] = 255
            raw_m = cv2.bitwise_and(raw_m, line_gate)

        _, core_mask = cv2.threshold(raw_m, 25, 255, cv2.THRESH_BINARY)

        # Fallback if mask_refined was empty inside this block: Otsu threshold dark letters inside bbox
        if np.count_nonzero(core_mask) < 15:
            gray_p = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            _, otsu_inv = cv2.threshold(gray_p, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            core_mask = otsu_inv

        # Remove connected components in core_mask that touch the outer boundary of the patch
        # (these are speech bubble outlines or artwork lines entering the box, not text glyphs)
        core_mask = self._strip_border_touching_components(core_mask)

        # Dilate core_mask to cover anti-aliased letter edges
        ksize = max(3, int(self.config.mask_dilate_kernel))
        if ksize % 2 == 0:
            ksize += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        dilated_mask = cv2.dilate(core_mask, kernel, iterations=self.config.mask_dilate_iters)

        # Protect dark pixels touching the patch boundary from being overwritten by dilation
        if self.config.preserve_bubble_borders:
            gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            dark_pixels = (gray_patch < self.config.dark_border_thresh).astype(np.uint8) * 255
            border_lines = self._keep_only_border_touching_components(dark_pixels)
            dilated_mask[border_lines > 0] = 0

        # Sample background pixels around dilated_mask
        outer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        outer_ring = (cv2.dilate(dilated_mask, outer_kernel, iterations=1) > 0) & (dilated_mask == 0)

        if np.any(outer_ring):
            bg_samples = patch[outer_ring]
        else:
            bg_samples = patch[dilated_mask == 0] if np.any(dilated_mask == 0) else np.array([[255, 255, 255]])

        bg_median = np.median(bg_samples, axis=0).astype(np.uint8)
        bg_std = float(np.std(cv2.cvtColor(bg_samples.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2GRAY)))

        # Sample foreground (ink) color from core_mask
        if np.any(core_mask > 0):
            fg_samples = patch[core_mask > 0]
            fg_median = np.percentile(fg_samples, 15, axis=0).astype(np.uint8)
        else:
            fg_median = np.array([0, 0, 0], dtype=np.uint8)

        is_dark_bubble = float(np.mean(bg_median)) < 110.0

        # Decide between Tier 1 (Smart Speech Bubble Fill) and Tier 2 (Complex Background Inpainting)
        if bg_std <= self.config.uniform_std_thresh and not block.recovered_from_mask:
            mode = "smart_bubble_fill"
            filled = patch.copy()
            filled[dilated_mask > 0] = bg_median
            # Smooth 1px edge transition
            alpha = cv2.GaussianBlur(dilated_mask.astype(np.float32) / 255.0, (3, 3), 0)[..., None]
            result_patch = np.clip(
                filled.astype(np.float32) * alpha + patch.astype(np.float32) * (1.0 - alpha),
                0,
                255,
            ).astype(np.uint8)
            canvas[wy1:wy2, wx1:wx2] = result_patch
        else:
            mode = "complex_inpaint"
            canvas[wy1:wy2, wx1:wx2] = self.inpainter.inpaint_region(patch, dilated_mask)

        # Estimate usable safe bubble box around block.bbox for Turkish typesetting
        safe_box = self._estimate_safe_bubble_box(canvas, block.bbox, bg_median, im_w, im_h)

        return BlockCleanInfo(
            block_id=block.id,
            bbox=block.bbox,
            mode=mode,
            bg_color_bgr=(int(bg_median[0]), int(bg_median[1]), int(bg_median[2])),
            fg_color_bgr=(int(fg_median[0]), int(fg_median[1]), int(fg_median[2])),
            is_dark_bubble=is_dark_bubble,
            safe_text_box=safe_box,
        )

    def _strip_border_touching_components(self, bin_mask: np.ndarray) -> np.ndarray:
        """Remove connected components in bin_mask that touch the outer 1px frame AND are large/elongated."""
        h, w = bin_mask.shape[:2]
        if h < 6 or w < 6:
            return bin_mask
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        out = bin_mask.copy()
        for i in range(1, num_labels):
            x, y, cw, ch, area = stats[i]
            touches_edge = (x == 0) or (y == 0) or (x + cw >= w) or (y + ch >= h)
            if touches_edge and (cw > w * 0.65 or ch > h * 0.65):
                out[labels == i] = 0
        return out

    def _keep_only_border_touching_components(self, dark_mask: np.ndarray) -> np.ndarray:
        """Return mask of dark components that touch the patch boundary (bubble outlines)."""
        h, w = dark_mask.shape[:2]
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
        border_only = np.zeros_like(dark_mask)
        for i in range(1, num_labels):
            x, y, cw, ch, _ = stats[i]
            touches_edge = (x == 0) or (y == 0) or (x + cw >= w) or (y + ch >= h)
            if touches_edge:
                border_only[labels == i] = 255
        return border_only

    def _estimate_safe_bubble_box(
        self,
        cleaned_canvas: np.ndarray,
        bbox: List[int],
        bg_median: np.ndarray,
        im_w: int,
        im_h: int,
    ) -> List[int]:
        """
        Slightly expand the English text bbox into the surrounding white speech bubble interior
        (without crossing any black bubble outline) so longer Turkish translations fit comfortably.
        """
        x1, y1, x2, y2 = clamp_box(bbox, im_w, im_h)
        bw = x2 - x1
        bh = y2 - y1

        # Maximum allowed expansion in each direction
        max_dx = min(22, max(6, int(bw * 0.16)))
        max_dy = min(18, max(4, int(bh * 0.14)))

        gray = cv2.cvtColor(cleaned_canvas, cv2.COLOR_BGR2GRAY)
        bg_gray = float(np.mean(bg_median))

        if bg_gray < 170:
            # Not a standard light bubble; keep tight bbox
            return [x1, y1, x2, y2]

        # Probe left, right, up, down until dark pixels (< 160) appear
        nx1, ny1, nx2, ny2 = x1, y1, x2, y2
        y_mid1 = y1 + int(bh * 0.2)
        y_mid2 = y2 - int(bh * 0.2)
        x_mid1 = x1 + int(bw * 0.2)
        x_mid2 = x2 - int(bw * 0.2)

        for dx in range(1, max_dx + 1):
            if x1 - dx <= 2:
                break
            col = gray[y_mid1:y_mid2, x1 - dx]
            if col.size == 0 or np.min(col) < 175:
                break
            nx1 = x1 - dx

        for dx in range(1, max_dx + 1):
            if x2 + dx >= im_w - 2:
                break
            col = gray[y_mid1:y_mid2, x2 + dx]
            if col.size == 0 or np.min(col) < 175:
                break
            nx2 = x2 + dx

        for dy in range(1, max_dy + 1):
            if y1 - dy <= 2:
                break
            row = gray[y1 - dy, x_mid1:x_mid2]
            if row.size == 0 or np.min(row) < 175:
                break
            ny1 = y1 - dy

        for dy in range(1, max_dy + 1):
            if y2 + dy >= im_h - 2:
                break
            row = gray[y2 + dy, x_mid1:x_mid2]
            if row.size == 0 or np.min(row) < 175:
                break
            ny2 = y2 + dy

        # Keep an inner safety margin of 5px away from the bubble wall
        nx1 = min(x1, nx1 + 5)
        ny1 = min(y1, ny1 + 4)
        nx2 = max(x2, nx2 - 5)
        ny2 = max(y2, ny2 - 4)

        return clamp_box([nx1, ny1, nx2, ny2], im_w, im_h)
