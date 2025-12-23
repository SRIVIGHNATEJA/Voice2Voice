from difflib import get_close_matches


# ─────────────────────────────────────────────
# Canonical Indian language definitions
# ─────────────────────────────────────────────

# Code → human-readable name (used for ASR routing & logs)
INDIAN_LANG_MAP = {
    "as": "Assamese",
    "bn": "Bengali",
    "brx": "Bodo",
    "doi": "Dogri",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "kok": "Konkani",
    "ks": "Kashmiri",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mni": "Manipuri",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "sat": "Santali",
    "sd": "Sindhi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}

# Code-only set (used for fast membership checks)
INDIAN_LANGS = set(INDIAN_LANG_MAP.keys())


# ─────────────────────────────────────────────
# Supported spoken language aliases
# ─────────────────────────────────────────────

SUPPORTED_LANGUAGES = {
    "english": "en",
    "hindi": "hi",
    "telugu": "te",
    "tamil": "ta",
    "german": "de",
    "french": "fr",
    "bengali": "bn",
    "marathi": "mr",
    "kannada": "kn",
    "malayalam": "ml",
    "japanese": "ja",
    "spanish": "es",
    "gujarati": "gu",
    "punjabi": "pa",
}


# ─────────────────────────────────────────────
# Language normalization utilities
# ─────────────────────────────────────────────

def normalize_language_input(text: str) -> str:
    """
    Normalizes spoken or typed language input.
    """
    return (
        text.lower()
        .replace("language", "")
        .replace("lang", "")
        .strip()
    )


def get_language_code(spoken_input: str):
    """
    Matches spoken/typed input to a supported language.
    Returns (lang_code, canonical_name) or (None, None).
    """
    normalized = normalize_language_input(spoken_input)

    matches = get_close_matches(
        normalized,
        SUPPORTED_LANGUAGES.keys(),
        n=1,
        cutoff=0.5 
    )

    if matches:
        lang_name = matches[0]
        print(f"✅ Interpreted as: {lang_name.capitalize()}")
        return SUPPORTED_LANGUAGES[lang_name], lang_name

    return None, None


def detect_target_language_manually():
    """
    Manual fallback for target language selection.
    """
    print("🔡 Please type the target language name (e.g., English, Hindi):")
    typed_input = input("Your input: ").strip()

    code, name = get_language_code(typed_input)

    if code:
        return code, name

    print("⚠️ Not recognized. Suggestions:")
    suggestions = get_close_matches(
        normalize_language_input(typed_input),
        SUPPORTED_LANGUAGES.keys(),
        n=3
    )
    print("🔎 Close matches:", ", ".join(suggestions))
    return None, None


# ─────────────────────────────────────────────
# Verified NLLB language codes
# ─────────────────────────────────────────────

NLLB_LANG_CODE_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "te": "tel_Telu",
    "ta": "tam_Taml",
    "bn": "ben_Beng",
    "ml": "mal_Mlym",
    "kn": "kan_Knda",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
    "ne": "npi_Deva",
    "or": "ory_Orya",
    "as": "asm_Beng",
    "sd": "snd_Arab",
    "si": "sin_Sinh",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab",
    "pt": "por_Latn",
    "it": "ita_Latn",
}
