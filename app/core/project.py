"""
MangaTR-Studio - Project Memory & Glossary Manager (app/core/project.py)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import DEFAULT_GLOSSARY, PROJECTS_DIR
from app.utils.json_utils import load_json, save_json


@dataclass
class DialogueBlockRecord:
    id: int
    bbox: List[int]
    safe_box: List[int]
    text_en: str
    text_tr: str
    ocr_confidence: float = 0.0
    mask_confidence: float = 0.0
    font_size: int = 0
    font_name: str = ""
    clean_mode: str = "smart_bubble_fill"
    is_dark_bubble: bool = False
    recovered_from_mask: bool = False
    lines: List[Dict[str, Any]] = field(default_factory=list)


class ProjectManager:
    """
    Manages series glossary/terminology memory and page translation project files
    across 400+ chapters.
    """

    def __init__(self, project_name: str = "default") -> None:
        self.project_name = project_name
        self.project_dir = PROJECTS_DIR / project_name
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.glossary_path = self.project_dir / "glossary.json"
        self.glossary: Dict[str, str] = self._load_or_init_glossary()

    def _load_or_init_glossary(self) -> Dict[str, str]:
        if self.glossary_path.exists():
            try:
                data = load_json(self.glossary_path)
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
            except Exception:
                pass
        glossary = dict(DEFAULT_GLOSSARY)
        save_json(self.glossary_path, glossary)
        return glossary

    def save_glossary(self, glossary: Optional[Dict[str, str]] = None) -> None:
        if glossary is not None:
            self.glossary = dict(glossary)
        save_json(self.glossary_path, self.glossary)

    def add_term(self, english_term: str, turkish_term: str) -> None:
        en = english_term.strip()
        tr = turkish_term.strip()
        if en and tr:
            self.glossary[en] = tr
            self.save_glossary()

    def remove_term(self, english_term: str) -> None:
        en = english_term.strip()
        if en in self.glossary:
            del self.glossary[en]
            self.save_glossary()
