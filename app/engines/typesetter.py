"""
MangaTR-Studio - Professional Turkish Manga Typesetting Engine (app/engines/typesetter.py)

Features:
- Turkish-aware uppercase & glyph rendering (Ç, Ğ, İ, Ö, Ş, Ü)
- Automatic font style selection (Normal dialogue vs. Shout/Exclamation)
- Balanced speech-bubble line wrapping & dynamic font size fitting
- Automatic white/black stroke halo on floating or complex inpainted backgrounds
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.config import FONTS_DIR, TypesetterConfig
from app.engines.cleaner import BlockCleanInfo
from app.utils.text_normalizer import tr_upper


@dataclass
class TypesetBlockSpec:
    block_id: int
    bbox: List[int]
    safe_box: List[int]
    text_tr: str
    font_name: str = ""
    font_size: int = 0
    lines: List[str] = None
    is_floating_or_complex: bool = False
    is_dark_bubble: bool = False


class MangaTypesetter:
    """
    Fits and renders localized Turkish dialogue into cleaned manga speech bubbles.
    """

    def __init__(self, config: Optional[TypesetterConfig] = None) -> None:
        self.config = config or TypesetterConfig()
        self._font_cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}

    def typeset_page(
        self,
        cleaned_image_bgr: np.ndarray,
        blocks: List[Dict[str, object]],
        clean_infos: Optional[List[BlockCleanInfo]] = None,
    ) -> Tuple[np.ndarray, List[TypesetBlockSpec]]:
        """
        Render Turkish translations onto `cleaned_image_bgr`.
        `blocks` is a list of dicts containing at least:
          - id: int
          - bbox: [x1, y1, x2, y2]
          - translation / text_tr: str
        """
        im_h, im_w = cleaned_image_bgr.shape[:2]
        rgb = cv2.cvtColor(cleaned_image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)

        clean_map: Dict[int, BlockCleanInfo] = {}
        if clean_infos:
            for ci in clean_infos:
                clean_map[ci.block_id] = ci

        specs: List[TypesetBlockSpec] = []

        for b in blocks:
            bid = int(b.get("id", 0))
            raw_tr = str(b.get("translation") or b.get("text_tr") or "").strip()
            if not raw_tr:
                continue

            bbox = [int(v) for v in b.get("bbox", [0, 0, 100, 100])]
            ci = clean_map.get(bid)
            safe_box = ci.safe_text_box if ci else [int(v) for v in b.get("safe_box", bbox)]
            sub_patch = cleaned_image_bgr[safe_box[1] : safe_box[3], safe_box[0] : safe_box[2]]
            if ci is not None:
                is_dark = ci.is_dark_bubble
            else:
                is_dark = bool(sub_patch.size > 0 and float(np.mean(sub_patch)) < 105.0)

            # Enable stroke halo automatically if background under safe_box has speed-lines, borders, or texture
            has_bg_texture = bool(
                sub_patch.size > 0
                and (float(np.std(sub_patch)) > 12.0 or (not is_dark and float(np.percentile(sub_patch, 5)) < 200.0))
            )
            is_complex = (
                (ci.mode == "complex_inpaint") if ci else (b.get("clean_mode") == "complex_inpaint")
            ) or has_bg_texture or is_dark

            display_text = tr_upper(raw_tr) if self.config.force_uppercase else raw_tr
            font_file = self._choose_font_file(display_text, str(b.get("font_name", "")))

            override_size = int(b.get("render_font_size", 0) or 0)
            spec = self._fit_and_draw_block(
                draw=draw,
                block_id=bid,
                bbox=bbox,
                safe_box=safe_box,
                text=display_text,
                font_file=font_file,
                is_complex=is_complex,
                is_dark=is_dark,
                im_w=im_w,
                im_h=im_h,
                override_font_size=override_size,
            )
            specs.append(spec)

        out_rgb = np.asarray(pil_img, dtype=np.uint8)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        return out_bgr, specs

    def _choose_font_file(self, text: str, explicit_font: str = "") -> str:
        if explicit_font and (FONTS_DIR / explicit_font).exists():
            return explicit_font
        if text.endswith("!!") or text.endswith("?!") or text.endswith("!?"):
            if (FONTS_DIR / self.config.shout_font).exists():
                return self.config.shout_font
        return self.config.default_font

    def _get_font(self, font_file: str, size: int) -> ImageFont.FreeTypeFont:
        key = (font_file, size)
        if key in self._font_cache:
            return self._font_cache[key]

        candidates = [
            FONTS_DIR / font_file,
            FONTS_DIR / self.config.default_font,
            Path("C:/Windows/Fonts/comicbd.ttf"),
            FONTS_DIR / "ChakraPetch-Bold.ttf",
            Path("C:/Windows/Fonts/arialbd.ttf"),
            Path("C:/Windows/Fonts/segoeui.ttf"),
        ]
        for path in candidates:
            if path.exists():
                try:
                    font = ImageFont.truetype(str(path), size=size)
                    self._font_cache[key] = font
                    return font
                except Exception:
                    continue

        fallback = ImageFont.load_default()
        self._font_cache[key] = fallback
        return fallback

    def _fit_and_draw_block(
        self,
        draw: ImageDraw.ImageDraw,
        block_id: int,
        bbox: List[int],
        safe_box: List[int],
        text: str,
        font_file: str,
        is_complex: bool,
        is_dark: bool,
        im_w: int,
        im_h: int,
        override_font_size: int = 0,
    ) -> TypesetBlockSpec:
        sx1, sy1, sx2, sy2 = safe_box
        box_w = max(48, sx2 - sx1)
        box_h = max(32, sy2 - sy1)

        pad_x = max(1, int(box_w * self.config.box_padding_ratio))
        pad_y = max(1, int(box_h * self.config.box_padding_ratio))
        usable_w = max(44, box_w - 2 * pad_x)
        usable_h = max(28, box_h - 2 * pad_y)

        # Cap maximum font size relative to bubble dimensions so tiny bubbles don't get huge letters
        area_cap = int(min(self.config.max_font_size, max(18, min(usable_w * 0.36, usable_h * 0.58))))
        min_sz = 10 if (sx2 - sx1) < 72 else self.config.min_font_size

        if override_font_size > 0:
            best_size = max(10, min(self.config.max_font_size, override_font_size))
            font = self._get_font(font_file, best_size)
            best_lines = self._wrap_text_balanced(draw, text, font, usable_w)
        else:
            best_size = min_sz
            best_lines = [text]

            for candidate_size in range(area_cap, min_sz - 1, -1):
                font = self._get_font(font_file, candidate_size)
                lines = self._wrap_text_balanced(draw, text, font, usable_w)
                if self._lines_fit(draw, lines, font, usable_w, usable_h):
                    best_size = candidate_size
                    best_lines = lines
                    break
            else:
                font = self._get_font(font_file, min_sz)
                best_lines = self._wrap_text_balanced(draw, text, font, usable_w)

        font = self._get_font(font_file, best_size)

        # Measure total block height and line heights
        line_metrics = []
        for line in best_lines:
            l_bbox = draw.textbbox((0, 0), line, font=font)
            lw = l_bbox[2] - l_bbox[0]
            lh = max(int(best_size * 0.88), l_bbox[3] - l_bbox[1])
            line_metrics.append((line, lw, lh))

        line_gap = max(2, int(round(best_size * (self.config.line_spacing - 1.0))))
        total_h = sum(m[2] for m in line_metrics) + max(0, len(line_metrics) - 1) * line_gap

        # Center vertically and horizontally around the safe_box center
        cx = 0.5 * (sx1 + sx2)
        cy = 0.5 * (sy1 + sy2)
        curr_y = int(round(cy - total_h * 0.5))

        if is_dark:
            fill_color = (255, 255, 255)
            stroke_color = (0, 0, 0)
        else:
            fill_color = (18, 18, 18)
            stroke_color = (255, 255, 255)

        stroke_w = self.config.stroke_width if (is_complex and self.config.auto_stroke_on_complex_bg) else 0

        for line_str, lw, lh in line_metrics:
            lx = int(round(cx - lw * 0.5))
            lx = max(2, min(im_w - lw - 2, lx))
            curr_y = max(2, min(im_h - lh - 2, curr_y))

            if stroke_w > 0:
                draw.text(
                    (lx, curr_y),
                    line_str,
                    font=font,
                    fill=fill_color,
                    stroke_width=stroke_w,
                    stroke_fill=stroke_color,
                )
            else:
                draw.text(
                    (lx, curr_y),
                    line_str,
                    font=font,
                    fill=fill_color,
                )
            curr_y += lh + line_gap

        return TypesetBlockSpec(
            block_id=block_id,
            bbox=bbox,
            safe_box=safe_box,
            text_tr=text,
            font_name=font_file,
            font_size=best_size,
            lines=best_lines,
            is_floating_or_complex=is_complex,
            is_dark_bubble=is_dark,
        )

    def _wrap_text_balanced(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_w: int,
    ) -> List[str]:
        """
        Wrap text into lines that fit within `max_w` while keeping line widths balanced
        (avoiding single short orphan words on the last line).
        """
        words = text.split()
        if not words:
            return [""]
        if len(words) == 1:
            return [words[0]]

        # First do greedy wrap to find minimum required line count
        greedy_lines = self._greedy_wrap(draw, words, font, max_w)
        if len(greedy_lines) <= 1:
            return greedy_lines

        # Try balancing line widths for the same number of lines (or +1 if bubble is tall/narrow)
        target_n = len(greedy_lines)
        best_candidate = greedy_lines
        best_score = self._layout_penalty(draw, greedy_lines, font, max_w)

        for num_lines in (target_n, target_n + 1):
            if num_lines > len(words):
                continue
            balanced = self._partition_words_even(words, num_lines)
            score = self._layout_penalty(draw, balanced, font, max_w)
            if score < best_score:
                best_score = score
                best_candidate = balanced

        return best_candidate

    def _greedy_wrap(
        self,
        draw: ImageDraw.ImageDraw,
        words: List[str],
        font: ImageFont.FreeTypeFont,
        max_w: int,
    ) -> List[str]:
        lines: List[str] = []
        current = words[0]
        for w in words[1:]:
            test = f"{current} {w}"
            tb = draw.textbbox((0, 0), test, font=font)
            if (tb[2] - tb[0]) <= max_w:
                current = test
            else:
                lines.append(current)
                current = w
        lines.append(current)
        return lines

    def _partition_words_even(self, words: List[str], k: int) -> List[str]:
        """Partition `words` into `k` contiguous lines with approximately equal character length."""
        if k <= 1:
            return [" ".join(words)]
        if k >= len(words):
            return list(words)

        total_chars = sum(len(w) for w in words) + (len(words) - k)
        target_per_line = total_chars / float(k)

        lines: List[str] = []
        curr_words: List[str] = []
        curr_len = 0
        remaining_lines = k

        for idx, w in enumerate(words):
            rem_words = len(words) - idx
            if rem_words == remaining_lines:
                if curr_words:
                    lines.append(" ".join(curr_words))
                    curr_words = []
                    remaining_lines -= 1
                lines.append(w)
                remaining_lines -= 1
                continue

            new_len = curr_len + (1 if curr_words else 0) + len(w)
            if curr_words and new_len > target_per_line * 1.18 and remaining_lines > 1:
                lines.append(" ".join(curr_words))
                curr_words = [w]
                curr_len = len(w)
                remaining_lines -= 1
            else:
                curr_words.append(w)
                curr_len = new_len

        if curr_words:
            lines.append(" ".join(curr_words))

        return lines

    def _layout_penalty(
        self,
        draw: ImageDraw.ImageDraw,
        lines: List[str],
        font: ImageFont.FreeTypeFont,
        max_w: int,
    ) -> float:
        widths = []
        for line in lines:
            tb = draw.textbbox((0, 0), line, font=font)
            w = tb[2] - tb[0]
            if w > max_w:
                return 1e9 + (w - max_w) * 1000.0
            widths.append(float(w))
        if not widths:
            return 0.0
        return float(np.std(widths)) + 8.0 * len(lines)

    def _lines_fit(
        self,
        draw: ImageDraw.ImageDraw,
        lines: List[str],
        font: ImageFont.FreeTypeFont,
        max_w: int,
        max_h: int,
    ) -> bool:
        total_h = 0
        font_sz = getattr(font, "size", 16)
        line_gap = max(2, int(round(font_sz * (self.config.line_spacing - 1.0))))

        for line in lines:
            tb = draw.textbbox((0, 0), line, font=font)
            lw = tb[2] - tb[0]
            lh = max(int(font_sz * 0.88), tb[3] - tb[1])
            if lw > max_w:
                return False
            total_h += lh

        total_h += max(0, len(lines) - 1) * line_gap
        return total_h <= max_h
