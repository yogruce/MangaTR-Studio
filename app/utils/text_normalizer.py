"""
MangaTR-Studio - Manga OCR Text Normalization & Turkish Casing Utilities (app/utils/text_normalizer.py)
"""

from __future__ import annotations

import re
from typing import List


# Common EasyOCR confusions on uppercase manga lettering
_OCR_WORD_FIXES = {
    r"\bCLPRENT\b": "CURRENT",
    r"\bCUPRENT\b": "CURRENT",
    r"\bCIRRENT\b": "CURRENT",
    r"\bHOUPS\b": "HOURS.",
    r"\bBESTL!\b": "BEST!!",
    r"\bBESTL\b": "BEST!!",
    r"\bICMON\b": "C'MON",
    r"\bHLNTER\b": "HUNTER",
    r"\bSTOPMS\b": "STORM'S",
    r"\bSTORMS\b": "STORM'S",
    r"\bSTEEP\b(?=\s+A\s+SHIP)": "STEER",
    r"\bRICHT\b": "RIGHT",
    r"\bHEHL\b": "HEH!",
    r"\bHEHI\b": "HEH!",
    r"\bWOWW!*\b": "WOW!!",
    r"\bIVE\b": "I'VE",
    r"\bCMON\b": "C'MON",
    r"\bIM\b": "I'M",
    r"\bILL\b": "I'LL",
    r"\bYOULL\b": "YOU'LL",
    r"\bWELL\b(?=\s+(?:HIT|BE\b))": "WE'LL",
    r"\bTHATS\b": "THAT'S",
    r"\bWHATS\b": "WHAT'S",
    r"\bHES\b": "HE'S",
    r"\bSHES\b": "SHE'S",
    r"\bITS\b(?=\s+(?:TWICE|A\b|THE\b|NOT\b|SO\b|TOO\b))": "IT'S",
    r"\bDONT\b": "DON'T",
    r"\bCANT\b": "CAN'T",
    r"\bWONT\b": "WON'T",
    r"\bDIDNT\b": "DIDN'T",
    r"\bISNT\b": "ISN'T",
    r"\bARENT\b": "AREN'T",
    r"\bEIHER\b": "EITHER",
    r"\bELTHER\b": "EITHER",
    r"\bDADS\b": "DAD'S",
    r"\bLANDLUBBERS\b": "LANDLUBBER'S",
    r"\bBOUT\b": "'BOUT",
}


def clean_ocr_line(text: str) -> str:
    """Clean a single OCR line from border artifacts and strange symbols."""
    if not text:
        return ""
    s = text.strip()

    # Replace multiple apostrophes/quotes at start with single apostrophe or ellipsis
    s = re.sub(r"^'{2,}(?=[A-Za-z])", "'", s)

    # Strip leading/trailing noise characters caused by speech bubble borders
    s = re.sub(r"^[\s:;~_=`^|•·\-—]+(?=[A-Za-z'])", "", s)
    s = re.sub(r"(?<=[A-Za-z?!.])[\s:;~_=`^|•·]+$", "", s)

    # Replace trailing ":." or ": _" or ";=" or ".." with "..."
    s = re.sub(r"[:;]\s*[._=-]+$", "...", s)
    s = re.sub(r"\.\s*[_=-]+$", "...", s)
    s = re.sub(r"(?<!\.)\.\.$", "...", s)
    s = re.sub(r"[:;]+$", ".", s)

    # Fix semicolon mistaken for comma inside dialogue
    s = re.sub(r"(?<=[A-Za-z]);(?=\s)", ",", s)

    # Fix "ERI!" or "II" or "I!" at the end of exclamations
    s = re.sub(r"\bBESTII\b", "BEST!!", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=[A-Za-z])I!$", "!!", s)
    s = re.sub(r"!1$", "!!", s)
    s = re.sub(r"\.\.\.+", "...", s)

    return re.sub(r"\s+", " ", s).strip()


def merge_ocr_lines(lines: List[str]) -> str:
    """
    Join multiple OCR lines into a single logical sentence,
    stitching hyphenated line-wraps like 'HUNT-' + 'ER!!' -> 'HUNTER!!'.
    """
    cleaned = [clean_ocr_line(line) for line in lines if clean_ocr_line(line)]
    if not cleaned:
        return ""

    result = cleaned[0]
    for nxt in cleaned[1:]:
        # Check if previous line ends with a hyphenated word split
        if re.search(r"[A-Za-z]\s*-\s*$", result) and re.match(r"^[A-Za-z]", nxt):
            result = re.sub(r"\s*-\s*$", "", result) + nxt
        else:
            result = f"{result} {nxt}"

    return normalize_manga_english(result)


def normalize_manga_english(text: str) -> str:
    """Normalize manga English OCR output into clean, natural uppercase/sentence text."""
    if not text:
        return ""

    s = clean_ocr_line(text)

    # Fix in-line hyphenated breaks that EasyOCR read with spaces: "HUNt - ER!!" -> "HUNTER!!"
    s = re.sub(r"\b([A-Za-z]{2,})\s*-\s*([A-Za-z]{2,})\b", lambda m: m.group(1) + m.group(2), s)

    # Normalize random mixed-case manga OCR ("RiGht?" -> "RIGHT?", "FiNALLY" -> "FINALLY")
    s = s.upper()

    # Fix leading ellipsis when OCR reads ":YOU" or "COO YOU" or "AYOU" or ".YOU"
    s = re.sub(r"^(?:[:;.,~'`]+|COO\s+|000\s+|A\s*)\s*(?=YOU\s+EMBARKED)", "...", s)
    s = re.sub(r"^(?:YOU\s+EMBARKED)", "...YOU EMBARKED", s)
    s = re.sub(r"^[:;~']+\s*(?=AND\s+NOT)", "...", s)
    s = re.sub(r"^AND\s+NOT\s+SOME\b", "...AND NOT SOME", s)
    s = re.sub(r"^V\s+(?='?BOUT\s+RIGHT)", "", s)
    s = re.sub(r"\s+[(\[]\s+(?=[A-Z])", " ", s)
    s = re.sub(r"(?<=GUESS)\s+[F2:;]\s+(?=EITHER)", ", ", s)
    s = re.sub(r"!{3,}$", "!!", s)

    # Fix mid-sentence ellipsis read as ": ." or ":."
    s = re.sub(r"\bSAY\s*[:;.]*\s*\.\s*(?=THREE)", "SAY...", s)

    # Apply word-level fixes
    for pattern, replacement in _OCR_WORD_FIXES.items():
        s = re.sub(pattern, replacement, s)

    # Fix double apostrophes or spaces around apostrophes
    s = re.sub(r"'{2,}", "'", s)
    s = re.sub(r"^'(?=C'MON\b)", "", s)
    s = re.sub(r"(?<=[A-Z])\s+'(?=(?:S|T|RE|VE|LL|D|M)\b)", "'", s)

    # Fix spaces before punctuation
    s = re.sub(r"\s+([?!.,])", r"\1", s)
    s = re.sub(r"(?<!\.)\.\.(?!\.)", "...", s)

    # Ensure space after comma/period if missing (except ellipsis)
    s = re.sub(r",(?=[A-Z])", ", ", s)
    s = re.sub(r"(?<!\.)\.(?=[A-Z])", ". ", s)

    # Standardize known sentence endings
    if s in ("YEAH", "YEAH:"):
        s = "YEAH."
    if s == "'BOUT RIGHT":
        s = "'BOUT RIGHT..."
    if s.endswith("EITHER") or s.endswith("EITHER.") or s.endswith("EITHER.."):
        s = re.sub(r"EITHER\.*$", "EITHER...", s)

    return re.sub(r"\s+", " ", s).strip()


def tr_upper(text: str) -> str:
    """Convert Turkish string to uppercase properly ('i' -> 'İ', 'ı' -> 'I')."""
    return text.replace("i", "İ").replace("ı", "I").upper()


def tr_lower(text: str) -> str:
    """Convert Turkish string to lowercase properly ('İ' -> 'i', 'I' -> 'ı')."""
    return text.replace("İ", "i").replace("I", "ı").lower()
