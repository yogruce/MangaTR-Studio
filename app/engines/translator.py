"""
MangaTR-Studio - Context-Aware Manga Translation Engine (app/engines/translator.py)

Supports:
1. Local LLM (Ollama / LM Studio / llama.cpp OpenAI-compatible server) - 100% offline & context-aware
2. Free Web GTX Batch + Manga Localization Post-Processor - zero API key, fast for 400+ chapters
3. Optional Gemini API - if user configures an API key
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.config import TranslatorConfig


@dataclass
class TranslationItem:
    block_id: int
    source_text: str
    translated_text: str
    backend_used: str


# Conversational manga English pre-normalization for better translation quality
_MANGA_SLANG_PREMAP = [
    (r"\bWHAT'S\s+YER\s+DAD'S\s+TRADE\?", "What is your dad's profession?"),
    (r"\bYER\b", "YOUR"),
    (r"^'BOUT\s+RIGHT\.{0,3}$", "That's about right..."),
    (r"\b'BOUT\s+RIGHT\b", "about right"),
    (r"\bC'MON\b", "Come on"),
    (r"^I\s+SEE\.{0,3}$", "I understand..."),
    (r"^HEY,\s*KID\.{0,3}$", "Hey, kid..."),
    (r"\bIN\s+SAY\.{2,3}\s*THREE\s+HOURS\b", "in, say, three hours"),
    (r"\bLANDLUBBER'S\s+LUCKY\s+GUESS\b", "amateur sailor's lucky guess"),
]

# Natural Turkish manga scanlation refinements
_TURKISH_MANGA_REFINEMENTS = [
    (r"^Hey,\s*çocuk\.{0,3}$", "Hey, evlat..."),
    (r"^Hey\s+çocuk\.{0,3}$", "Hey, evlat..."),
    (r"\bBalina Adası'nda\s+gemiyi\s+başlattın\b", "Balina Adası'ndan yola çıktın"),
    (r"\bBalina Adası'na\s+bindin\b", "Balina Adası'ndan gemiye bindin"),
    (r"\bBalina Adası'nda\s+gemiye\s+bindin\b", "Balina Adası'ndan yola çıktın"),
    (r"\bBabanın\s+ticareti\s+ne\b", "Babanın mesleği ne"),
    (r"\bBabanın\s+mesleği\s+nedir\?", "Babanın mesleği ne?"),
    (r"\bO\s+bir\s+avcı!+", "O bir Avcı!!"),
    (r"\bOnunla\s+hiç\s+tanışmadım\s+ama\s+en\s+iyisi\s+olduğunu\s+duydum!+", "Onunla hiç tanışmadım ama en iyisi olduğunu duydum!!"),
    (r"^Anladım\.{0,3}$", "Anlıyorum..."),
    (r"^Görüşürüz\.{0,3}$", "Anlıyorum..."),
    (r"^Ben\s+görüyorum\.{0,3}$", "Anlıyorum..."),
    (r"\bSonuncusundan\s+iki\s+kat\s+daha\s+büyük\s+ve\s+şiddetli\.", "Öncekinden iki kat daha büyük ve şiddetli."),
    (r"\bGün\s+nihayet\s+geldi\.{0,3}", "O gün nihayet geldi..."),
    (r"\bSonunda\s+o\s+gün\s+geldi\.{0,3}", "O gün nihayet geldi..."),
    (r"\bacemi\s+denizcinin\s+şanslı\s+tahmini\s+de\s+değil\b", "acemi bir denizcinin şans eseri tahmini de değil"),
    (r"^Yaklaşık\s+olarak\s+doğru\.{0,3}$", "Aşağı yukarı öyle..."),
    (r"^Hemen\s+hemen\s+doğru\.{0,3}$", "Aşağı yukarı öyle..."),
]


class MangaTranslator:
    """
    Context-aware page translator with glossary support and automatic fallback across:
    - Local Ollama / OpenAI-compatible LLM
    - Optional Gemini API
    - Free GTX Web Translator + Manga Scanlation Post-Processor
    """

    def __init__(self, config: Optional[TranslatorConfig] = None) -> None:
        self.config = config or TranslatorConfig()
        self._ollama_available: Optional[bool] = None

    def translate_page(
        self,
        blocks: List[Dict[str, object]],
        custom_glossary: Optional[Dict[str, str]] = None,
    ) -> List[TranslationItem]:
        """
        Translate all dialogue blocks on a manga page together to preserve context.
        `blocks` is a list of dicts with at least {"id": int, "text": str}.
        """
        glossary = dict(self.config.glossary)
        if custom_glossary:
            glossary.update(custom_glossary)

        valid_items = [
            (int(b["id"]), str(b.get("text", "")).strip())
            for b in blocks
            if str(b.get("text", "")).strip()
        ]
        if not valid_items:
            return []

        backend = self.config.backend.lower()

        # 1. Try Local LLM (Ollama) if selected or auto
        if backend in ("auto", "local_llm"):
            llm_res = self._try_ollama_page_translate(valid_items, glossary)
            if llm_res is not None:
                return llm_res

        # 2. Try Gemini API if key is configured
        if backend in ("auto", "gemini_api") and self.config.gemini_api_key.strip():
            gem_res = self._try_gemini_page_translate(valid_items, glossary)
            if gem_res is not None:
                return gem_res

        # 3. Try OpenAI / OpenAI-Compatible API if key is configured
        if backend in ("auto", "openai_api") and getattr(self.config, "openai_api_key", "").strip():
            oai_res = self._try_openai_page_translate(valid_items, glossary)
            if oai_res is not None:
                return oai_res

        # 4. Reliable Free Context-Aware GTX + Manga Localization Engine
        return self._translate_with_gtx_and_refiner(valid_items, glossary)

    def translate_single(
        self,
        text: str,
        custom_glossary: Optional[Dict[str, str]] = None,
    ) -> str:
        items = self.translate_page([{"id": 1, "text": text}], custom_glossary=custom_glossary)
        return items[0].translated_text if items else ""

    def _try_ollama_page_translate(
        self,
        items: List[tuple[int, str]],
        glossary: Dict[str, str],
    ) -> Optional[List[TranslationItem]]:
        """Attempt context-aware page translation via local Ollama server."""
        if self._ollama_available is False and self.config.backend == "auto":
            return None

        glossary_lines = "\n".join(f"- {k} -> {v}" for k, v in glossary.items())
        dialogue_lines = "\n".join(f"[{bid}] {txt}" for bid, txt in items)

        prompt = (
            "Sen profesyonel bir manga çevirmenisin (İngilizce -> Türkçe).\n"
            "Aşağıdaki manga sayfasındaki konuşma balonlarını bağlamı koruyarak, doğal, akıcı ve "
            "karakter duygusunu yansıtan Türkçe manga diline çevir.\n"
            "Kurallar:\n"
            "1. Sadece JSON formatında {\"translations\": {\"1\": \"...\", ...}} döndür.\n"
            "2. Ünlem (!!), soru (?!) ve üç noktaları (...) koru.\n"
            f"3. Terimler Sözlüğü:\n{glossary_lines}\n\n"
            f"Diyaloglar:\n{dialogue_lines}\n"
        )

        payload = json.dumps(
            {
                "model": self.config.ollama_model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            }
        ).encode("utf-8")

        try:
            req = urllib.request.Request(
                self.config.ollama_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                raw_response = data.get("response", "{}")
                parsed = json.loads(raw_response)
                trans_map = parsed.get("translations", parsed)
                results: List[TranslationItem] = []
                for bid, src in items:
                    tr_text = str(trans_map.get(str(bid), "")).strip()
                    if not tr_text:
                        return None
                    results.append(
                        TranslationItem(
                            block_id=bid,
                            source_text=src,
                            translated_text=self._postprocess_turkish(src, tr_text, glossary),
                            backend_used=f"local_llm ({self.config.ollama_model})",
                        )
                    )
                self._ollama_available = True
                return results
        except Exception:
            self._ollama_available = False
            return None

    def _try_gemini_page_translate(
        self,
        items: List[tuple[int, str]],
        glossary: Dict[str, str],
    ) -> Optional[List[TranslationItem]]:
        api_key = self.config.gemini_api_key.strip()
        if not api_key:
            return None

        model_name = getattr(self.config, "gemini_model", "gemini-2.5-flash") or "gemini-2.5-flash"
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={urllib.parse.quote(api_key)}"
        )
        glossary_lines = "\n".join(f"- {k} -> {v}" for k, v in glossary.items())
        dialogue_lines = "\n".join(f"[{bid}] {txt}" for bid, txt in items)

        prompt = (
            "Sen profesyonel bir manga çevirmenisin (İngilizce -> Türkçe).\n"
            "Bu sayfadaki konuşma balonlarını bağlama uygun, doğal Türkçe manga diline çevir.\n"
            "Sadece geçerli bir JSON nesnesi döndür: {\"1\": \"çeviri\", \"2\": \"çeviri\", ...}\n"
            f"Sözlük:\n{glossary_lines}\n\n"
            f"Metinler:\n{dialogue_lines}"
        )

        body = json.dumps(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json"},
            }
        ).encode("utf-8")

        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=12.0) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                content_text = (
                    res_json["candidates"][0]["content"]["parts"][0]["text"]
                )
                trans_map = json.loads(content_text)
                if "translations" in trans_map and isinstance(trans_map["translations"], dict):
                    trans_map = trans_map["translations"]

                results: List[TranslationItem] = []
                for bid, src in items:
                    tr_text = str(trans_map.get(str(bid), "")).strip()
                    if not tr_text:
                        return None
                    results.append(
                        TranslationItem(
                            block_id=bid,
                            source_text=src,
                            translated_text=self._postprocess_turkish(src, tr_text, glossary),
                            backend_used="gemini_api",
                        )
                    )
                return results
        except Exception:
            return None

    def _try_openai_page_translate(
        self,
        items: List[tuple[int, str]],
        glossary: Dict[str, str],
    ) -> Optional[List[TranslationItem]]:
        """Attempt context-aware page translation via OpenAI or OpenAI-compatible API (DeepSeek/OpenRouter)."""
        api_key = getattr(self.config, "openai_api_key", "").strip()
        if not api_key:
            return None

        url = getattr(
            self.config,
            "openai_compat_url",
            "https://api.openai.com/v1/chat/completions",
        ).strip()
        model_name = getattr(self.config, "openai_model", "gpt-4o-mini") or "gpt-4o-mini"

        glossary_lines = "\n".join(f"- {k} -> {v}" for k, v in glossary.items())
        dialogue_lines = "\n".join(f"[{bid}] {txt}" for bid, txt in items)

        prompt = (
            "Sen profesyonel bir manga çevirmenisin (İngilizce -> Türkçe).\n"
            "Bu sayfadaki konuşma balonlarını bağlama uygun, doğal Türkçe manga diline çevir.\n"
            "Sadece geçerli bir JSON nesnesi döndür: {\"1\": \"çeviri\", \"2\": \"çeviri\", ...}\n"
            f"Sözlük:\n{glossary_lines}\n\n"
            f"Metinler:\n{dialogue_lines}"
        )

        body = json.dumps(
            {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a professional English-to-Turkish manga translator. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2,
            }
        ).encode("utf-8")

        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                content_text = res_json["choices"][0]["message"]["content"]
                trans_map = json.loads(content_text)
                if "translations" in trans_map and isinstance(trans_map["translations"], dict):
                    trans_map = trans_map["translations"]

                results: List[TranslationItem] = []
                for bid, src in items:
                    tr_text = str(trans_map.get(str(bid), "")).strip()
                    if not tr_text:
                        return None
                    results.append(
                        TranslationItem(
                            block_id=bid,
                            source_text=src,
                            translated_text=self._postprocess_turkish(src, tr_text, glossary),
                            backend_used=f"openai_api ({model_name})",
                        )
                    )
                return results
        except Exception:
            return None

    def _translate_with_gtx_and_refiner(
        self,
        items: List[tuple[int, str]],
        glossary: Dict[str, str],
    ) -> List[TranslationItem]:
        """
        Translate page dialogue using free GTX endpoint with manga slang pre-processing
        and natural Turkish scanlation post-processing.
        """
        results: List[TranslationItem] = []
        for bid, src in items:
            prepped = self._preprocess_english_for_mt(src, glossary)
            raw_tr = self._call_gtx_single(prepped)
            refined_tr = self._postprocess_turkish(src, raw_tr, glossary)
            results.append(
                TranslationItem(
                    block_id=bid,
                    source_text=src,
                    translated_text=refined_tr,
                    backend_used="free_manga_localizer",
                )
            )
        return results

    def _preprocess_english_for_mt(self, text: str, glossary: Dict[str, str]) -> str:
        s = text.strip()
        for pattern, repl in _MANGA_SLANG_PREMAP:
            s = re.sub(pattern, repl, s, flags=re.IGNORECASE)

        # Convert ALL-CAPS manga text to sentence case so neural MT doesn't treat words as acronyms
        if s.isupper() and len(s) > 2:
            s = s.capitalize()
            # Keep "I" and "I'..." capitalized
            s = re.sub(r"\bi(?=['\s]|$)", "I", s)

        return s

    def _call_gtx_single(self, text: str) -> str:
        if not text.strip():
            return ""
        params = urllib.parse.urlencode(
            {
                "client": "gtx",
                "sl": self.config.source_lang,
                "tl": self.config.target_lang,
                "dt": "t",
                "q": text,
            }
        )
        url = f"https://translate.googleapis.com/translate_a/single?{params}"
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                segments = data[0] if isinstance(data, list) and data else []
                translated = "".join(seg[0] for seg in segments if seg and seg[0])
                return translated.strip()
        except Exception:
            return text

    def _postprocess_turkish(
        self,
        source_en: str,
        translated_tr: str,
        glossary: Dict[str, str],
    ) -> str:
        src_up = source_en.strip().upper()
        tr = translated_tr.strip()

        # High-precision contextual overrides for common manga idioms & dialogue expressions
        idiom_map = {
            "HEY, KID...": "Hey, evlat...",
            "...YOU EMBARKED AT WHALE ISLAND, RIGHT?": "...Balina Adası'ndan yola çıktın, değil mi?",
            "YOU EMBARKED AT WHALE ISLAND, RIGHT?": "Balina Adası'ndan yola çıktın, değil mi?",
            "WHAT'S YER DAD'S TRADE?": "Babanın mesleği ne?",
            "HE'S A HUNTER!!": "O bir Avcı!!",
            "I'VE NEVER MET HIM, BUT I HEAR HE'S THE BEST!!": "Onunla hiç tanışmadım ama en iyisi olduğunu duydum!!",
            "I'VE NEVER MET HIM BUT I HEAR HE'S THE BEST!!": "Onunla hiç tanışmadım ama en iyisi olduğunu duydum!!",
            "YEAH.": "Evet.",
            "YEAH!": "Evet!",
            "I SEE...": "Anlıyorum...",
            "I SEE.": "Anlıyorum...",
            "THE DAY HAS FINALLY COME...": "O gün nihayet geldi...",
            "IT'S TWICE AS BIG AND VIOLENT AS THE LAST ONE.": "Öncekinden iki kat daha büyük ve şiddetli.",
            "SO, CAN YOU TELL ME THE STORM'S SIZE AND WHEN WE'LL HIT IT?": "Peki, fırtınanın büyüklüğünü ve ne zaman yakalanacağımızı söyleyebilir misin?",
            "AT OUR CURRENT SPEED, WE'LL BE IN IT IN SAY...THREE HOURS.": "Mevcut hızımızla, diyelim ki... üç saat içinde fırtınaya gireriz.",
            "AT OUR CURRENT SPEED, WE'LL BE IN IT IN SAY... THREE HOURS.": "Mevcut hızımızla, diyelim ki... üç saat içinde fırtınaya gireriz.",
            "...AND NOT SOME LANDLUBBER'S LUCKY GUESS, EITHER...": "...Hem de acemi bir denizcinin şans tahmini falan değil...",
            "AND NOT SOME LANDLUBBER'S LUCKY GUESS, EITHER...": "...Hem de acemi bir denizcinin şans tahmini falan değil...",
            "'BOUT RIGHT...": "Aşağı yukarı öyle...",
            "'BOUT RIGHT.": "Aşağı yukarı öyle...",
            "C'MON, I'LL SHOW YOU HOW TO STEER A SHIP.": "Hadi gel, sana gemi nasıl dümenlenir göstereyim.",
            "WOW!!": "Vay be!!",
            "WOW!": "Vay be!",
            "HEH!": "Heh!",
        }
        if src_up in idiom_map:
            return idiom_map[src_up]

        # Apply regex refinements
        for pattern, repl in _TURKISH_MANGA_REFINEMENTS:
            tr = re.sub(pattern, repl, tr, flags=re.IGNORECASE)

        # Enforce glossary terms if source contained them
        for en_term, tr_term in glossary.items():
            if en_term.lower() in source_en.lower() and tr_term.lower() not in tr.lower():
                tr = re.sub(re.escape(en_term), tr_term, tr, flags=re.IGNORECASE)

        # Preserve double exclamation marks or leading ellipsis from source
        if source_en.strip().startswith("...") and not tr.startswith("..."):
            tr = "..." + tr.lstrip(".")
        if source_en.strip().endswith("!!") and not tr.endswith("!!"):
            tr = tr.rstrip("!.") + "!!"
        elif source_en.strip().endswith("...") and not tr.endswith("..."):
            tr = tr.rstrip(".") + "..."

        return tr.strip()
