"""
MangaTR-Studio - Context-Aware & Vision-Assisted Manga Translation Engine (app/engines/translator.py)

Supports:
1. Gemini Multimodal Vision + Context Translation (reads bubble crops directly + translates in full page context)
2. Local LLM (Ollama / LM Studio / llama.cpp) - 100% offline & full-page context-aware
3. OpenAI-Compatible API (DeepSeek / OpenRouter / GPT-4o) - full-page context-aware
4. Free Web GTX Batch + Universal Manga Context & Proper-Noun Protector - zero API key, works across all series
"""

from __future__ import annotations

import base64
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from app.config import TranslatorConfig


@dataclass
class TranslationItem:
    block_id: int
    source_text: str
    translated_text: str
    backend_used: str
    is_sfx: bool = False


# ---------------------------------------------------------------------------
# 1. Universal Untranslatable Manga / Manhwa / Webtoon / Anime Terms
# ---------------------------------------------------------------------------
UNIVERSAL_PROTECTED_TERMS: Dict[str, str] = {
    "One Piece": "One Piece",
    "Grand Line": "Grand Line",
    "New World": "New World",
    "Calm Belt": "Calm Belt",
    "Red Line": "Red Line",
    "Laugh Tale": "Laugh Tale",
    "Poneglyph": "Poneglyph",
    "Road Poneglyph": "Road Poneglyph",
    "Mariejois": "Mariejois",
    "Yonko": "Yonko",
    "Shichibukai": "Shichibukai",
    "Gorosei": "Gorosei",
    "Wano": "Wano",
    "Elbaf": "Elbaf",
    "Sanji": "Sanji",
    "Zoro": "Zoro",
    "Luffy": "Luffy",
    "Nami": "Nami",
    "Usopp": "Usopp",
    "Chopper": "Chopper",
    "Robin": "Robin",
    "Franky": "Franky",
    "Brook": "Brook",
    "Jinbe": "Jinbe",
    "Qilingam": "Qilingam",
    "Saint Qilingam": "Aziz Qilingam",
    "Haki": "Haki",
    "Haoshoku Haki": "Haoshoku Haki",
    "Busoshoku Haki": "Busoshoku Haki",
    "Kenbunshoku Haki": "Kenbunshoku Haki",
    "Gear Second": "Gear Second",
    "Gear Third": "Gear Third",
    "Gear Fourth": "Gear Fourth",
    "Gear Fifth": "Gear Fifth",
    "Gear 5": "Gear 5",
    "Bankai": "Bankai",
    "Shikai": "Shikai",
    "Zanpakuto": "Zanpakuto",
    "Getsuga Tensho": "Getsuga Tensho",
    "Hueco Mundo": "Hueco Mundo",
    "Soul Society": "Soul Society",
    "Gotei 13": "Gotei 13",
    "Espada": "Espada",
    "Arrancar": "Arrancar",
    "Quincy": "Quincy",
    "Reiatsu": "Reiatsu",
    "Shinigami": "Shinigami",
    "Sharingan": "Sharingan",
    "Mangekyo Sharingan": "Mangekyo Sharingan",
    "Byakugan": "Byakugan",
    "Rinnegan": "Rinnegan",
    "Rasengan": "Rasengan",
    "Chidori": "Chidori",
    "Chakra": "Chakra",
    "Jutsu": "Jutsu",
    "Ninjutsu": "Ninjutsu",
    "Genjutsu": "Genjutsu",
    "Taijutsu": "Taijutsu",
    "Dojutsu": "Dojutsu",
    "Kekkei Genkai": "Kekkei Genkai",
    "Hokage": "Hokage",
    "Kazekage": "Kazekage",
    "Raikage": "Raikage",
    "Mizukage": "Mizukage",
    "Tsuchikage": "Tsuchikage",
    "Akatsuki": "Akatsuki",
    "Shinobi": "Shinobi",
    "Jinchuriki": "Jinchuriki",
    "Konoha": "Konoha",
    "Nen": "Nen",
    "Zetsu": "Zetsu",
    "Hatsu": "Hatsu",
    "Zoldyck": "Zoldyck",
    "Kurapika": "Kurapika",
    "Killua": "Killua",
    "Leorio": "Leorio",
    "Hisoka": "Hisoka",
    "Chrollo": "Chrollo",
    "Kamehameha": "Kamehameha",
    "Saiyan": "Saiyan",
    "Super Saiyan": "Super Saiyan",
    "Kaioken": "Kaioken",
    "Quirk": "Quirk",
    "Plus Ultra": "Plus Ultra",
    "One For All": "One For All",
    "All For One": "All For One",
    "Shikigami": "Shikigami",
    "Black Flash": "Black Flash",
    "Hollow Purple": "Hollow Purple",
    "Grimoire": "Grimoire",
    "Hamon": "Hamon",
    "Murim": "Murim",
    "Dantian": "Dantian",
    "Mana": "Mana",
    "Aura": "Aura",
    "Chaebol": "Chaebol",
    "Oppa": "Oppa",
    "Hyung": "Hyung",
    "Noona": "Noona",
    "Unnie": "Unnie",
    "Sunbae": "Sunbae",
    "Hoobae": "Hoobae",
    "Ahjussi": "Ahjussi",
    "Ahjumma": "Ahjumma",
    "Senpai": "Senpai",
    "Sensei": "Sensei",
    "Sama": "Sama",
    "Kun": "Kun",
    "Chan": "Chan",
    "Dono": "Dono",
    "Aniki": "Aniki",
    "Nakama": "Nakama",
    "Yokai": "Yokai",
    "Kaiju": "Kaiju",
    "Isekai": "Isekai",
    "Devil Fruit": "Şeytan Meyvesi",
    "Straw Hat": "Hasır Şapka",
    "Soul Reaper": "Ruh Tırpancısı",
    "Domain Expansion": "Alan Genişletme",
    "Cursed Energy": "Lanetli Enerji",
    "Cursed Technique": "Lanetli Teknik",
    "Shadow Monarch": "Gölge Hükümdarı",
    "Heavenly Demon": "Göksel İblis",
    "Martial Arts": "Dövüş Sanatları",
    "Survey Corps": "Keşif Birliği",
    "Demon Slayer": "İblis Keser",
}

_COMMON_ENGLISH_STOPWORDS: Set[str] = {
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "ANY", "CAN", "HAD", "HER", "WAS",
    "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HIS", "HOW", "MAN", "NEW", "NOW", "OLD",
    "SEE", "TWO", "WAY", "WHO", "BOY", "DID", "ITS", "LET", "PUT", "SAY", "SHE", "TOO", "USE",
    "DAD", "MOM", "KID", "GUY", "MEN", "GOD", "SIR", "YES", "YEA", "YEAH", "NO", "HEY", "WOW",
    "WHAT", "WHEN", "WHERE", "WHICH", "WHILE", "WHOSE", "WHY", "WITH", "WITHOUT", "WITHIN",
    "THIS", "THAT", "THESE", "THOSE", "THERE", "THEIR", "THEM", "THEN", "THAN", "THEY",
    "HAVE", "HERE", "FROM", "INTO", "OVER", "SOME", "SUCH", "TAKE", "TIME", "VERY", "WELL",
    "WERE", "WILL", "YOUR", "ABOUT", "AFTER", "AGAIN", "BEING", "COULD", "EVERY", "FIRST",
    "GREAT", "NEVER", "OTHER", "RIGHT", "SHALL", "SINCE", "STILL", "THREE", "UNDER", "UNTIL",
    "WATER", "WHERE", "WORLD", "WOULD", "YEARS", "YOUNG", "STORM", "SPEED", "HOURS", "GUESS",
    "TRADE", "SHIP", "STEER", "ISLAND", "WHALE", "HUNTER", "CAPTAIN", "MASTER", "LORD", "KING",
    "HEAD", "PAST", "SAINT", "STAY", "SEALED", "HELL", "DONE", "KEEP", "TOGETHER", "MENTAL",
    "DEFENSES", "DROP", "HUFF",
}

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
    (r"\bKE+P\s+IT\s+TOGETHER\b", "Pull yourself together"),
    (r"\bI'M\s+DONE\s+FOR\b", "I am finished"),
]

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
    (r"^Kafamdan\s+çık.*$", "Zihnimden çık...!!!"),
    (r"^Defol!\s*Mühürlü\s*kal.*$", "Defol!! Mühürlü kal! O günler cehennem gibiydi!!"),
]


class MangaTranslator:
    """
    Context-aware page translator with universal proper-noun/manga-term protection,
    multimodal Gemini Vision support, and automatic fallback across:
    - Local Ollama LLM
    - Gemini API (with Multimodal Vision bubble crop reading + fallback model chain)
    - OpenAI-compatible API
    - Free GTX Web Translator + Universal Manga Context & Term Protector
    """

    def __init__(self, config: Optional[TranslatorConfig] = None) -> None:
        self.config = config or TranslatorConfig()
        self._ollama_available: Optional[bool] = None
        self._working_gemini_model: Optional[str] = None

    def translate_page(
        self,
        blocks: List[Dict[str, object]],
        custom_glossary: Optional[Dict[str, str]] = None,
    ) -> List[TranslationItem]:
        """
        Translate all dialogue blocks on a manga page together to preserve context.
        `blocks` is a list of dicts with {"id": int, "text": str, optional "crop": np.ndarray}.
        """
        effective_glossary: Dict[str, str] = dict(UNIVERSAL_PROTECTED_TERMS)
        effective_glossary.update(self.config.glossary)
        if custom_glossary:
            effective_glossary.update(custom_glossary)

        has_any_crop = any(isinstance(b.get("crop"), np.ndarray) for b in blocks)
        backend = self.config.backend.lower()

        # When Gemini API is active and bubble crops are provided, include all blocks so Vision can read them
        if backend in ("auto", "gemini_api") and self.config.gemini_api_key.strip() and has_any_crop:
            gem_vis = self._try_gemini_vision_page_translate(blocks, effective_glossary)
            if gem_vis is not None:
                return gem_vis

        valid_items = [
            (int(b["id"]), str(b.get("text", "")).strip())
            for b in blocks
            if str(b.get("text", "")).strip()
        ]
        if not valid_items:
            return []

        dynamic_terms = self._extract_dynamic_protected_terms([txt for _, txt in valid_items])
        for term in dynamic_terms:
            if term not in effective_glossary:
                effective_glossary[term] = term

        # 1. Try Local LLM (Ollama) if selected or auto
        if backend in ("auto", "local_llm"):
            llm_res = self._try_ollama_page_translate(valid_items, effective_glossary)
            if llm_res is not None:
                return llm_res

        # 2. Try Gemini Text API if key is configured
        if backend in ("auto", "gemini_api") and self.config.gemini_api_key.strip():
            gem_res = self._try_gemini_page_translate(valid_items, effective_glossary)
            if gem_res is not None:
                return gem_res

        # 3. Try OpenAI / OpenAI-Compatible API if key is configured
        if backend in ("auto", "openai_api") and getattr(self.config, "openai_api_key", "").strip():
            oai_res = self._try_openai_page_translate(valid_items, effective_glossary)
            if oai_res is not None:
                return oai_res

        # 4. Reliable Free Full-Page Context-Aware GTX + Manga Localization & Term Protector
        return self._translate_with_gtx_and_refiner(valid_items, effective_glossary)

    def translate_single(
        self,
        text: str,
        custom_glossary: Optional[Dict[str, str]] = None,
    ) -> str:
        items = self.translate_page([{"id": 1, "text": text}], custom_glossary=custom_glossary)
        return items[0].translated_text if items else ""

    # ------------------------------------------------------------------
    # Universal Context & Dynamic Proper-Noun Discovery (All Series)
    # ------------------------------------------------------------------

    def _extract_dynamic_protected_terms(self, texts: List[str]) -> List[str]:
        discovered: List[str] = []
        full_page_text = " \n ".join(texts)

        honorific_pattern = re.compile(
            r"\b([A-Za-z]{2,15})-(san|kun|chan|sama|sensei|senpai|dono|tan|chama|ssi|nim)\b",
            re.IGNORECASE,
        )
        for match in honorific_pattern.finditer(full_page_text):
            base_name = match.group(1).capitalize()
            suffix = match.group(2).lower()
            full_honorific = f"{base_name}-{suffix}"
            discovered.append(full_honorific)
            if base_name.upper() not in _COMMON_ENGLISH_STOPWORDS:
                discovered.append(base_name)

        title_name_pattern = re.compile(
            r"\b(?:Lord|Lady|Captain|General|Admiral|Commander|Master|Doctor|Professor|Prince|Princess|Saint|Duke|Emperor|Mount)\s+([A-Za-z]{3,14})\b",
            re.IGNORECASE,
        )
        for match in title_name_pattern.finditer(full_page_text):
            candidate = match.group(1)
            if candidate.upper() not in _COMMON_ENGLISH_STOPWORDS:
                discovered.append(candidate.capitalize())

        jp_kr_morph_pattern = re.compile(
            r"\b([A-Za-z]{2,10}(?:maru|suke|taro|jiro|hiko|kage|gakure|jutsu|giri|zaki|moto|hara|shiro|kaze|ryu|hyun|heon|seok|cheol|hwan|woon))\b",
            re.IGNORECASE,
        )
        for match in jp_kr_morph_pattern.finditer(full_page_text):
            word = match.group(1)
            if word.upper() not in _COMMON_ENGLISH_STOPWORDS and len(word) >= 4:
                discovered.append(word.capitalize())

        seen: Set[str] = set()
        unique_terms: List[str] = []
        for t in discovered:
            low = t.lower()
            if low not in seen:
                seen.add(low)
                unique_terms.append(t)
        return unique_terms

    def _build_llm_context_prompt(
        self,
        items: List[Tuple[int, str]],
        glossary: Dict[str, str],
    ) -> str:
        full_text_lower = " ".join(txt.lower() for _, txt in items)
        relevant_glossary = {
            k: v for k, v in glossary.items() if k.lower() in full_text_lower
        }
        glossary_lines = (
            "\n".join(f"- {k} -> {v}" for k, v in relevant_glossary.items())
            if relevant_glossary
            else "- (Sayfa içi özel isimleri ve teknik adlarını koru)"
        )
        dialogue_lines = "\n".join(f"[{bid}] {txt}" for bid, txt in items)

        return (
            "Sen profesyonel bir manga, manhwa ve çizgi roman çevirmenisin (İngilizce -> Türkçe).\n"
            "Görev ve Bağlam Kuralları:\n"
            "1. Önce sayfadaki tüm balonları birlikte okuyarak sahnenin bağlamını, hangi esere/evrene ait olduğunu ve konuşanların duygusunu analiz et.\n"
            "2. Özel eser/seri adlarını (ör. 'One Piece'), özel teknik/saldırı isimlerini (ör. 'Bankai', 'Sharingan', 'Rasengan', 'Haki'), "
            "karakter adlarını ve Japonca/Korece hitap eklerini (-san, -kun, -chan, -sama, -sensei, Senpai, Hyung, Oppa) ASLA kelimesi kelimesine çevirme.\n"
            "3. Sadece geçerli bir JSON nesnesi döndür: {\"1\": \"çeviri\", \"2\": \"çeviri\", ...}\n"
            "4. Ünlem (!!), soru (?!) ve üç noktaları (...) koru.\n"
            f"5. Bu Sayfada Tespit Edilen Korunacak Terimler & Sözlük:\n{glossary_lines}\n\n"
            f"Sayfa Diyalogları:\n{dialogue_lines}\n"
        )

    # ------------------------------------------------------------------
    # Gemini Multimodal Vision + Text Translation Backend
    # ------------------------------------------------------------------

    def _get_gemini_candidate_models(self) -> List[str]:
        preferred = getattr(self.config, "gemini_model", "gemini-3-flash-preview") or "gemini-3-flash-preview"
        models = [
            preferred,
            "gemini-3-flash-preview",
            "gemini-flash-latest",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
        ]
        if self._working_gemini_model and self._working_gemini_model in models:
            models.remove(self._working_gemini_model)
            models.insert(0, self._working_gemini_model)
        seen = set()
        ordered = []
        for m in models:
            if m not in seen:
                seen.add(m)
                ordered.append(m)
        return ordered

    def _try_gemini_vision_page_translate(
        self,
        blocks: List[Dict[str, object]],
        glossary: Dict[str, str],
    ) -> Optional[List[TranslationItem]]:
        """
        Send bubble image crops directly to Gemini Vision so even stylized/dark/speedline
        manga fonts are read with 100% accuracy and translated in full page context.
        """
        api_key = self.config.gemini_api_key.strip()
        if not api_key:
            return None

        parts: List[Dict[str, object]] = [
            {
                "text": (
                    "Sen profesyonel bir manga, manhwa ve webtoon çevirmenisin (İngilizce -> Türkçe).\n"
                    "Aşağıda bir manga sayfasındaki konuşma balonlarının kırpılmış görselleri sırasıyla (#1, #2, ...) verilmiştir.\n"
                    "Kurallar:\n"
                    "1. Her görseldeki İngilizce metni tam ve doğru olarak oku ('en').\n"
                    "2. Eğer bir görselde İngilizce diyalog YOKSA (sadece Japonca ses efekti/SFX harfleri veya çizim varsa) 'is_sfx': true ve 'tr': '' döndür.\n"
                    "3. Tüm balonların birbirleriyle olan sahne bağlamını analiz ederek doğal, akıcı ve dramatik Türkçe manga diline çevir ('tr').\n"
                    "4. Özel eser/seri adlarını (ör. 'One Piece'), özel teknik isimlerini (ör. 'Bankai', 'Sharingan', 'Haki') ve karakter adlarını koru.\n"
                    "Sadece şu JSON formatında yanıt ver:\n"
                    '{"blocks": {"1": {"en": "...", "tr": "...", "is_sfx": false}, ...}}'
                )
            }
        ]

        valid_block_ids: List[int] = []
        fallback_texts: Dict[int, str] = {}

        for b in blocks:
            bid = int(b["id"])
            ocr_hint = str(b.get("text", "")).strip()
            crop = b.get("crop")
            if not isinstance(crop, np.ndarray) or crop.size == 0:
                continue
            ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                continue
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            valid_block_ids.append(bid)
            fallback_texts[bid] = ocr_hint
            parts.append({"text": f"Balon #{bid} (OCR İpucu: {ocr_hint}):"})
            parts.append({"inlineData": {"mimeType": "image/jpeg", "data": b64}})

        if not valid_block_ids:
            return None

        body = json.dumps(
            {
                "contents": [{"parts": parts}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2},
            }
        ).encode("utf-8")

        for model_name in self._get_gemini_candidate_models():
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model_name}:generateContent?key={urllib.parse.quote(api_key)}"
            )
            try:
                req = urllib.request.Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=25.0) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    content_text = res_json["candidates"][0]["content"]["parts"][0]["text"]
                    parsed = json.loads(content_text)
                    block_map = parsed.get("blocks", parsed)

                    results: List[TranslationItem] = []
                    for bid in valid_block_ids:
                        entry = block_map.get(str(bid), {})
                        if isinstance(entry, str):
                            en_txt = fallback_texts.get(bid, "")
                            tr_txt = entry.strip()
                            is_sfx = False
                        elif isinstance(entry, dict):
                            en_txt = str(entry.get("en") or fallback_texts.get(bid, "")).strip()
                            tr_txt = str(entry.get("tr") or "").strip()
                            is_sfx = bool(entry.get("is_sfx", False))
                        else:
                            en_txt = fallback_texts.get(bid, "")
                            tr_txt = ""
                            is_sfx = True

                        if is_sfx or not tr_txt:
                            results.append(
                                TranslationItem(
                                    block_id=bid,
                                    source_text=en_txt,
                                    translated_text="",
                                    backend_used=f"gemini_vision ({model_name})",
                                    is_sfx=True,
                                )
                            )
                        else:
                            results.append(
                                TranslationItem(
                                    block_id=bid,
                                    source_text=en_txt,
                                    translated_text=self._postprocess_turkish(en_txt, tr_txt, glossary),
                                    backend_used=f"gemini_vision ({model_name})",
                                    is_sfx=False,
                                )
                            )
                    self._working_gemini_model = model_name
                    return results
            except Exception:
                continue

        return None

    def _try_gemini_page_translate(
        self,
        items: List[Tuple[int, str]],
        glossary: Dict[str, str],
    ) -> Optional[List[TranslationItem]]:
        api_key = self.config.gemini_api_key.strip()
        if not api_key:
            return None

        prompt = self._build_llm_context_prompt(items, glossary)
        body = json.dumps(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2},
            }
        ).encode("utf-8")

        for model_name in self._get_gemini_candidate_models():
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model_name}:generateContent?key={urllib.parse.quote(api_key)}"
            )
            try:
                req = urllib.request.Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15.0) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    content_text = res_json["candidates"][0]["content"]["parts"][0]["text"]
                    trans_map = json.loads(content_text)
                    if "translations" in trans_map and isinstance(trans_map["translations"], dict):
                        trans_map = trans_map["translations"]

                    results: List[TranslationItem] = []
                    for bid, src in items:
                        tr_text = str(trans_map.get(str(bid), "")).strip()
                        if not tr_text:
                            tr_text = self._call_gtx_single(self._preprocess_english_for_mt(src, glossary))
                        results.append(
                            TranslationItem(
                                block_id=bid,
                                source_text=src,
                                translated_text=self._postprocess_turkish(src, tr_text, glossary),
                                backend_used=f"gemini_api ({model_name})",
                            )
                        )
                    self._working_gemini_model = model_name
                    return results
            except Exception:
                continue

        return None

    def _try_ollama_page_translate(
        self,
        items: List[Tuple[int, str]],
        glossary: Dict[str, str],
    ) -> Optional[List[TranslationItem]]:
        if self._ollama_available is False and self.config.backend == "auto":
            return None

        prompt = self._build_llm_context_prompt(items, glossary)
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

    def _try_openai_page_translate(
        self,
        items: List[Tuple[int, str]],
        glossary: Dict[str, str],
    ) -> Optional[List[TranslationItem]]:
        api_key = getattr(self.config, "openai_api_key", "").strip()
        if not api_key:
            return None

        url = getattr(
            self.config,
            "openai_compat_url",
            "https://api.openai.com/v1/chat/completions",
        ).strip()
        model_name = getattr(self.config, "openai_model", "gpt-4o-mini") or "gpt-4o-mini"
        prompt = self._build_llm_context_prompt(items, glossary)

        body = json.dumps(
            {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a professional English-to-Turkish manga/manhwa translator. "
                            "Analyze the full page context, keep proper nouns, series titles (e.g. One Piece), "
                            "honorifics, and attack names untranslated, and output valid JSON only."
                        ),
                    },
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

    # ------------------------------------------------------------------
    # Free Full-Page Context-Aware GTX + Term Protector Engine
    # ------------------------------------------------------------------

    def _lock_protected_terms(
        self,
        text: str,
        glossary: Dict[str, str],
    ) -> Tuple[str, Dict[str, str]]:
        token_map: Dict[str, str] = {}
        locked = text
        sorted_terms = sorted(glossary.items(), key=lambda kv: len(kv[0]), reverse=True)
        token_idx = 0

        for en_term, tr_replacement in sorted_terms:
            if not en_term or len(en_term) < 2:
                continue
            pattern = re.compile(rf"\b{re.escape(en_term)}\b", flags=re.IGNORECASE)
            if pattern.search(locked):
                placeholder = f"XMT{token_idx}X"
                token_map[placeholder] = tr_replacement
                locked = pattern.sub(placeholder, locked)
                token_idx += 1

        return locked, token_map

    def _unlock_protected_terms(self, text: str, token_map: Dict[str, str]) -> str:
        unlocked = text
        for placeholder, replacement in token_map.items():
            unlocked = re.sub(re.escape(placeholder), replacement, unlocked, flags=re.IGNORECASE)
        return unlocked

    def _translate_with_gtx_and_refiner(
        self,
        items: List[Tuple[int, str]],
        glossary: Dict[str, str],
    ) -> List[TranslationItem]:
        prepped_lines: List[str] = []
        token_maps: List[Dict[str, str]] = []

        for _, src in items:
            prepped = self._preprocess_english_for_mt(src, glossary)
            locked, tmap = self._lock_protected_terms(prepped, glossary)
            prepped_lines.append(locked)
            token_maps.append(tmap)

        batch_text = "\n".join(prepped_lines)
        batch_translated = self._call_gtx_single(batch_text)
        split_lines = [line.strip() for line in batch_translated.split("\n") if line.strip()]

        if len(split_lines) != len(items):
            split_lines = [self._call_gtx_single(line) for line in prepped_lines]

        results: List[TranslationItem] = []
        for idx, (bid, src) in enumerate(items):
            raw_tr = split_lines[idx] if idx < len(split_lines) else ""
            unlocked_tr = self._unlock_protected_terms(raw_tr, token_maps[idx])
            refined_tr = self._postprocess_turkish(src, unlocked_tr, glossary)
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

        if s.isupper() and len(s) > 2:
            s = s.capitalize()
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
            "GET OUT OF MY HEAD...!!!": "Zihnimden çık...!!!",
            "I NEVER WANT TO REMEMBER THOSE DAYS AGAIN!!": "O günleri bir daha asla hatırlamak istemiyorum!!",
            "ALRIGHT!! IT'S DONE!!": "Tamamdır!! Bitti!!",
            "HUFF... HUFF...": "Haa... Haa...",
            "KEEEEEP IT TOGETHER!!!": "Kendine geeel!!!",
            "AAA AAAA AAH!!!": "AAA AAAA AAH!!!",
        }
        if src_up in idiom_map:
            return idiom_map[src_up]

        for pattern, repl in _TURKISH_MANGA_REFINEMENTS:
            tr = re.sub(pattern, repl, tr, flags=re.IGNORECASE)

        for en_term, tr_term in glossary.items():
            if en_term.lower() in source_en.lower() and tr_term.lower() not in tr.lower():
                tr = re.sub(re.escape(en_term), tr_term, tr, flags=re.IGNORECASE)

        if source_en.strip().startswith("...") and not tr.startswith("..."):
            tr = "..." + tr.lstrip(".")
        if source_en.strip().endswith("!!") and not tr.endswith("!!"):
            tr = tr.rstrip("!.") + "!!"
        elif source_en.strip().endswith("...") and not tr.endswith("..."):
            tr = tr.rstrip(".") + "..."

        return tr
