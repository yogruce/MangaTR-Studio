"""
MangaTR-Studio - Complex Background Inpainting Engine (app/engines/inpainter.py)
"""

from __future__ import annotations

import cv2
import numpy as np


class MangaInpainter:
    """
    Reconstructs complex/textured manga backgrounds (screentones, hatching lines, SFX areas)
    underneath removed text masks using Telea / Navier-Stokes inpainting with edge feathering.
    """

    def __init__(self, radius: int = 4) -> None:
        self.radius = max(2, int(radius))

    def inpaint_region(
        self,
        patch_bgr: np.ndarray,
        patch_mask: np.ndarray,
    ) -> np.ndarray:
        if patch_bgr.size == 0 or patch_mask.size == 0:
            return patch_bgr

        _, bin_mask = cv2.threshold(patch_mask, 25, 255, cv2.THRESH_BINARY)
        if not np.any(bin_mask > 0):
            return patch_bgr

        # Slightly dilate mask so anti-aliased letter fringes are not propagated by Telea
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        inpaint_mask = cv2.dilate(bin_mask, kernel, iterations=1)

        reconstructed = cv2.inpaint(
            patch_bgr,
            inpaint_mask,
            inpaintRadius=self.radius,
            flags=cv2.INPAINT_TELEA,
        )

        # Feather transition at mask boundary
        alpha = cv2.GaussianBlur(inpaint_mask.astype(np.float32) / 255.0, (3, 3), 0)[..., None]
        blended = (reconstructed.astype(np.float32) * alpha + patch_bgr.astype(np.float32) * (1.0 - alpha))
        return np.clip(blended, 0, 255).astype(np.uint8)
