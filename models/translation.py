from transformers import pipeline
import torch
from utils.system import device
from utils.language import NLLB_LANG_CODE_MAP, INDIAN_LANGS

# ─────────────────────────────────────────────
# 🔒 Global cached pipeline
# ─────────────────────────────────────────────
_nllb_pipeline = None


# ─────────────────────────────────────────────
# 🌐 NLLB Translation Pipeline (lazy-loaded)
# ─────────────────────────────────────────────
def get_nllb_pipeline():
    global _nllb_pipeline
    if _nllb_pipeline is None:
        print("🔁 Loading NLLB-200 translation model...")
        _nllb_pipeline = pipeline(
            "translation",
            model="facebook/nllb-200-distilled-600M",
            device=0 if device == "cuda" else -1
        )
        
        if device == "cpu":
            print("⚡ Applying dynamic quantization to NLLB (Int8)...")
            # 🔧 Force QNNPACK for macOS ARM64
            torch.backends.quantized.engine = "qnnpack"
            try:
                _nllb_pipeline.model = torch.quantization.quantize_dynamic(
                    _nllb_pipeline.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            except Exception as e:
                print(f"⚠️ Quantization failed: {e}. Proceeding with float32.")
    return _nllb_pipeline


# ─────────────────────────────────────────────
# 🔁 Core Translation Function
# ─────────────────────────────────────────────
def translate_with_nllb(text, src_lang_code, tgt_lang_code):
    """
    Translates text using NLLB-200.
    """
    if not text or not text.strip():
        print("⚠️ Empty input text. Skipping translation.")
        return None

    try:
        nllb = get_nllb_pipeline()

        src_nllb = NLLB_LANG_CODE_MAP.get(src_lang_code, "eng_Latn")
        tgt_nllb = NLLB_LANG_CODE_MAP.get(tgt_lang_code, "eng_Latn")

        translated = nllb(
            text,
            src_lang=src_nllb,
            tgt_lang=tgt_nllb,
            max_length=512
        )

        return translated[0]["translation_text"]

    except Exception as e:
        print(f"[ERROR] NLLB Translation failed: {e}")
        return None


# ─────────────────────────────────────────────
# 🔀 Translation Router (logging helper)
# ─────────────────────────────────────────────
def route_translation_pipeline(
    transcribed_text,
    detected_lang_code,
    target_lang_code="en"
):
    print(f"\n[INFO] Source Language Code : {detected_lang_code}")
    print(f"[INFO] Target Language Code : {target_lang_code}")

    translated_text = translate_with_nllb(
        transcribed_text,
        detected_lang_code,
        target_lang_code
    )

    if translated_text:
        print(f"[SUCCESS] Translation Output:\n{translated_text}")
    else:
        print("[FAILURE] Translation produced no output.")

    return translated_text


# ─────────────────────────────────────────────
# 🏷️ Translation + Language Classification
# ─────────────────────────────────────────────
def nllb_translate_and_classify(text, src_lang_code, tgt_lang_code):
    try:
        translated_text = translate_with_nllb(
            text, src_lang_code, tgt_lang_code
        )

        if not translated_text:
            return None, None, None

        lang_type = (
            "indian"
            if tgt_lang_code in INDIAN_LANGS
            else "international"
        )

        return translated_text, tgt_lang_code, lang_type

    except Exception as e:
        print(f"[ERROR] NLLB translate+classify failed: {e}")
        return None, None, None
