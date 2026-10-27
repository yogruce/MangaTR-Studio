"""
MangaTR-Studio - Central Configuration & Environment Loader (app/config.py)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = ROOT_DIR / "app"
VENDOR_DIR = ROOT_DIR / "vendor" / "comic-text-detector"
DETECTOR_MODEL_PATH = VENDOR_DIR / "models" / "comictextdetector.pt"
DETECTOR_MODEL_URL = (
    "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt"
)

FONTS_DIR = ROOT_DIR / "fonts"
MODELS_DIR = ROOT_DIR / "models"
INPUT_DIR = ROOT_DIR / "input"
OUTPUT_DIR = ROOT_DIR / "output"
PROJECTS_DIR = ROOT_DIR / "projects"
TEMP_DIR = ROOT_DIR / "temp"
ENV_FILE_PATH = ROOT_DIR / ".env"


def load_dotenv_file(env_path: Path = ENV_FILE_PATH) -> None:
    """Load key=value pairs from `.env` into `os.environ` without external dependencies."""
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass


load_dotenv_file()


DEFAULT_GLOSSARY: Dict[str, str] = {
    "Whale Island": "Balina Adası",
    "Hunter Association": "Avcılar Birliği",
    "Hunter Exam": "Avcılık Sınavı",
    "Hunter": "Avcı",
}


@dataclass
class DetectorConfig:
    model_path: Path = DETECTOR_MODEL_PATH
    model_url: str = DETECTOR_MODEL_URL
    auto_download_model: bool = True
    input_size: int = 1024
    conf_thresh: float = 0.38
    nms_thresh: float = 0.35
    iou_dup_thresh: float = 0.42
    contain_dup_thresh: float = 0.68
    recover_undetected_mask: bool = True
    reading_order: str = "manga_rtl"  # "manga_rtl" (right-to-left) or "comic_ltr" (left-to-right)


@dataclass
class OCRConfig:
    languages: List[str] = field(default_factory=lambda: ["en"])
    gpu: bool = True
    scale: float = 2.2
    block_padding: int = 6
    line_padding: int = 5
    use_line_level_ocr: bool = True
    clean_bubble_borders: bool = True
    normalize_manga_text: bool = True


@dataclass
class TranslatorConfig:
    # Supported backends: "auto", "free_web", "local_llm", "gemini_api", "openai_api"
    backend: str = field(default_factory=lambda: os.getenv("MANGATR_BACKEND", "auto"))
    source_lang: str = "en"
    target_lang: str = "tr"
    ollama_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    )
    openai_compat_url: str = field(
        default_factory=lambda: os.getenv(
            "OPENAI_COMPAT_URL", "https://api.openai.com/v1/chat/completions"
        )
    )
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    glossary: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_GLOSSARY))


@dataclass
class CleanerConfig:
    mask_dilate_kernel: int = 5
    mask_dilate_iters: int = 2
    inpaint_radius: int = 4
    uniform_std_thresh: float = 22.0
    preserve_bubble_borders: bool = True
    dark_border_thresh: int = 90


@dataclass
class TypesetterConfig:
    default_font: str = "ComicSansMS-Bold.ttf"
    shout_font: str = "ComicSansMS-Bold.ttf"
    sfx_font: str = "Bangers-Regular.ttf"
    min_font_size: int = 14
    max_font_size: int = 32
    line_spacing: float = 1.12
    box_padding_ratio: float = 0.04
    force_uppercase: bool = True
    auto_stroke_on_complex_bg: bool = True
    stroke_width: int = 3


@dataclass
class StudioConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    translator: TranslatorConfig = field(default_factory=TranslatorConfig)
    cleaner: CleanerConfig = field(default_factory=CleanerConfig)
    typesetter: TypesetterConfig = field(default_factory=TypesetterConfig)
    jpeg_quality: int = 95

    def ensure_dirs(self) -> None:
        for folder in (FONTS_DIR, MODELS_DIR, INPUT_DIR, OUTPUT_DIR, PROJECTS_DIR, TEMP_DIR):
            folder.mkdir(parents=True, exist_ok=True)
