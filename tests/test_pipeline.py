"""
MangaTR-Studio - Automated Pipeline Verification Suite (tests/test_pipeline.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.engines.detector import box_containment, box_iou  # noqa: E402
from app.utils.text_normalizer import merge_ocr_lines, normalize_manga_english, tr_upper  # noqa: E402


def test_duplicate_metrics() -> None:
    box_a = [658, 126, 766, 241]
    box_b = [660, 126, 764, 242]
    assert box_iou(box_a, box_b) > 0.90
    assert box_containment(box_a, box_b) > 0.95


def test_ocr_normalization_and_hyphenation() -> None:
    merged = merge_ocr_lines(["HE'S A", "HUNT-", "ER!!"])
    assert merged == "HE'S A HUNTER!!"

    norm1 = normalize_manga_english(":You EMBARKED AT WHALE ISLAND, RiGht?")
    assert norm1 == "...YOU EMBARKED AT WHALE ISLAND, RIGHT?"

    norm2 = normalize_manga_english("AT OUR ClpRENT SPEeD, We'll BE IN It IN SAY...THRee HOUPS")
    assert norm2 == "AT OUR CURRENT SPEED, WE'LL BE IN IT IN SAY...THREE HOURS."

    norm3 = normalize_manga_english("V 'BOUT RIGHT...")
    assert norm3 == "'BOUT RIGHT..."


def test_turkish_uppercase() -> None:
    assert tr_upper("İngilizce ve Türkçe çeviri") == "İNGİLİZCE VE TÜRKÇE ÇEVİRİ"
    assert tr_upper("babanın mesleği ne?") == "BABANIN MESLEĞİ NE?"


if __name__ == "__main__":
    test_duplicate_metrics()
    test_ocr_normalization_and_hyphenation()
    test_turkish_uppercase()
    print("All MangaTR-Studio unit tests passed!")
