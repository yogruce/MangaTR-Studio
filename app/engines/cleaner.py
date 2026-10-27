"""
MangaTR-Studio - Hybrid Speech Bubble & Background Text Cleaner (app/engines/cleaner.py)

Implements a 3-Tier cleaning strategy:
1. Dark/Black Flashback Panels -> Erases both interior letters and white outline halos into the dark panel
2. Uniform Light Speech Bubbles -> Exact background tone sampling + speedline-safe interior text fill
3. Textured / Floating Text -> Localized Telea inpainting via MangaInpainter
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
    - Artwork outside the text bounding box
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
        pad = 4
        bx1, by1, bx2, by2 = clamp_box(block.bbox, im_w, im_h)
        wx1 = max(0, bx1 - pad)
        wy1 = max(0, by1 - pad)
        wx2 = min(im_w, bx2 + pad)
        wy2 = min(im_h, by2 + pad)

        patch = canvas[wy1:wy2, wx1:wx2]
        raw_m = mask_refined[wy1:wy2, wx1:wx2].copy()
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        ph, pw = gray_patch.shape[:2]

        # Build a gate covering the union of block.bbox and line_bboxes
        line_gate = np.zeros_like(raw_m)
        gx1 = max(0, bx1 - wx1)
        gy1 = max(0, by1 - wy1)
        gx2 = min(pw, bx2 - wx1)
        gy2 = min(ph, by2 - wy1)
        line_gate[gy1:gy2, gx1:gx2] = 255
        for lb in block.line_bboxes:
            lx1 = max(0, lb[0] - wx1 - 6)
            ly1 = max(0, lb[1] - wy1 - 6)
            lx2 = min(pw, lb[2] - wx1 + 6)
            ly2 = min(ph, lb[3] - wy1 + 6)
            line_gate[ly1:ly2, lx1:lx2] = 255

        raw_m = cv2.bitwise_and(raw_m, line_gate)
        _, core_mask = cv2.threshold(raw_m, 22, 255, cv2.THRESH_BINARY)

        # Fallback if mask_refined was empty inside this block
        if np.count_nonzero(core_mask) < 15:
            _, otsu_inv = cv2.threshold(gray_patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            core_mask = cv2.bitwise_and(otsu_inv, line_gate)

        # Check outer border of patch to detect dark/black panels (e.g., flashback panels with white-outlined text)
        border_gray = np.concatenate(
            [gray_patch[0, :], gray_patch[-1, :], gray_patch[:, 0], gray_patch[:, -1]]
        )
        border_median_gray = float(np.median(border_gray)) if border_gray.size > 0 else 255.0
        border_std_gray = float(np.std(border_gray)) if border_gray.size > 0 else 0.0

        # Case A: Dark / Black Panel (like "I NEVER WANT TO REMEMBER..." or "GET OUT OF MY HEAD...!!!")
        if border_median_gray < 85.0 and border_std_gray < 45.0:
            border_bgr_pixels = np.concatenate(
                [patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]], axis=0
            )
            bg_median = np.median(border_bgr_pixels, axis=0).astype(np.uint8)
            fg_median = np.array([255, 255, 255], dtype=np.uint8)

            # Inside a dark panel, both the core text mask AND any white outline halo (> border_median + 25)
            # inside line_gate belong to the text and must be erased into the dark panel!
            halo_mask = ((gray_patch > (border_median_gray + 25.0)) & (line_gate > 0)).astype(np.uint8) * 255
            combined = cv2.bitwise_or(core_mask, halo_mask)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            dilated_mask = cv2.dilate(combined, k, iterations=2)

            filled = patch.copy()
            filled[dilated_mask > 0] = bg_median
            alpha = cv2.GaussianBlur(dilated_mask.astype(np.float32) / 255.0, (3, 3), 0)[..., None]
            canvas[wy1:wy2, wx1:wx2] = np.clip(
                filled.astype(np.float32) * alpha + patch.astype(np.float32) * (1.0 - alpha),
                0,
                255,
            ).astype(np.uint8)

            return BlockCleanInfo(
                block_id=block.id,
                bbox=block.bbox,
                mode="smart_bubble_fill",
                bg_color_bgr=(int(bg_median[0]), int(bg_median[1]), int(bg_median[2])),
                fg_color_bgr=(255, 255, 255),
                is_dark_bubble=True,
                safe_text_box=block.bbox,
            )

        # Case B: Standard Light Bubble or Complex/Speedline Background
        ksize = max(3, int(self.config.mask_dilate_kernel))
        if ksize % 2 == 0:
            ksize += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        dilated_mask = cv2.dilate(core_mask, kernel, iterations=self.config.mask_dilate_iters)

        # Also include any white halo pixels immediately adjacent to core_mask (for white-stroked floating text)
        halo_ring = cv2.dilate(core_mask, kernel, iterations=2)

        # Protect dark bubble walls ONLY in the outer 4px frame of the patch — NEVER zero out core_mask!
        if self.config.preserve_bubble_borders and ph > 12 and pw > 12:
            dark_pixels = (gray_patch < self.config.dark_border_thresh).astype(np.uint8) * 255
            border_lines = self._keep_only_border_touching_components(dark_pixels)
            # Only protect border_lines in the outer 4px rim and never where core_mask is active
            inner_safe = np.zeros_like(border_lines)
            inner_safe[4 : ph - 4, 4 : pw - 4] = 255
            rim_border = border_lines & (~inner_safe) & (~core_mask)
            dilated_mask[rim_border > 0] = 0

        # Sample background pixels around dilated_mask
        outer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        outer_ring = (cv2.dilate(dilated_mask, outer_kernel, iterations=1) > 0) & (dilated_mask == 0)

        if np.any(outer_ring):
            bg_samples = patch[outer_ring]
        else:
            bg_samples = patch[dilated_mask == 0] if np.any(dilated_mask == 0) else np.array([[255, 255, 255]])

        # For light speech bubbles that have thin black speed-lines passing through them,
        # use the 75th percentile if mean brightness is > 160 so black speed-lines don't turn the fill gray
        bg_gray_vals = cv2.cvtColor(bg_samples.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        if border_median_gray >= 165.0 and float(np.percentile(bg_gray_vals, 70)) > 210.0:
            bg_median = np.percentile(bg_samples, 75, axis=0).astype(np.uint8)
        else:
            bg_median = np.median(bg_samples, axis=0).astype(np.uint8)

        bg_std = float(np.std(bg_gray_vals))

        if np.any(core_mask > 0):
            fg_samples = patch[core_mask > 0]
            fg_median = np.percentile(fg_samples, 15, axis=0).astype(np.uint8)
        else:
            fg_median = np.array([0, 0, 0], dtype=np.uint8)

        is_dark_bubble = float(np.mean(bg_median)) < 105.0

        # If this is a light speech bubble (even with speedlines passing at the edges), fill text cleanly with white/bg_median
        if (
            border_median_gray >= 165.0
            and (bg_std <= self.config.uniform_std_thresh or float(np.mean(bg_median)) >= 215.0)
            and not block.recovered_from_mask
        ):
            mode = "smart_bubble_fill"
            filled = patch.copy()
            filled[dilated_mask > 0] = bg_median
            alpha = cv2.GaussianBlur(dilated_mask.astype(np.float32) / 255.0, (3, 3), 0)[..., None]
            result_patch = np.clip(
                filled.astype(np.float32) * alpha + patch.astype(np.float32) * (1.0 - alpha),
                0,
                255,
            ).astype(np.uint8)
            canvas[wy1:wy2, wx1:wx2] = result_patch
        else:
            mode = "complex_inpaint"
            # For floating text with a white halo over dark/complex art, include the white halo in inpainting
            inpaint_mask = cv2.bitwise_or(dilated_mask, halo_ring & ((gray_patch > 200).astype(np.uint8) * 255))
            canvas[wy1:wy2, wx1:wx2] = self.inpainter.inpaint_region(patch, inpaint_mask)

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
        x1, y1, x2, y2 = clamp_box(bbox, im_w, im_h)
        bw = x2 - x1
        bh = y2 - y1

        max_dx = min(22, max(6, int(bw * 0.16)))
        max_dy = min(18, max(4, int(bh * 0.14)))

        gray = cv2.cvtColor(cleaned_canvas, cv2.COLOR_BGR2GRAY)
        bg_gray = float(np.mean(bg_median))

        if bg_gray < 170:
            return [x1, y1, x2, y2]

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

        nx1 = min(x1, nx1 + 5)
        ny1 = min(y1, ny1 + 4)
        nx2 = max(x2, nx2 - 5)
        ny2 = max(y2, ny2 - 4)

        return clamp_box([nx1, ny1, nx2, ny2], im_w, im_h)
