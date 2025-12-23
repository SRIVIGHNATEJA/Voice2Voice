import torch
import soundfile as sf
import numpy as np
import os
import pyttsx3
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from utils.system import device
from utils.audio import play_wav

import contextlib

# ─────────────────────────────────────────────
# 🔒 Global caches (CRITICAL)
# ─────────────────────────────────────────────
_parler_model = None
_tokenizer = None
_desc_tokenizer = None
_engine = None


# ─────────────────────────────────────────────
# 🧠 Lazy-load Indic Parler TTS
# ─────────────────────────────────────────────
def load_parler_tts():
    global _parler_model, _tokenizer, _desc_tokenizer

    if _parler_model is None:
        print("🔁 Loading Indic-Parler TTS model...")
        
        # 🤫 Suppress noisy config dumps from transformers/parler
        with open(os.devnull, "w") as fnull:
            with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                _parler_model = ParlerTTSForConditionalGeneration.from_pretrained(
                    "ai4bharat/indic-parler-tts"
                ).to(device)
                _parler_model.eval()

                _tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
                _desc_tokenizer = AutoTokenizer.from_pretrained(
                    _parler_model.config.text_encoder._name_or_path
                )

        # Ensure padding
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        if _desc_tokenizer.pad_token is None:
            _desc_tokenizer.pad_token = _desc_tokenizer.eos_token

    return _parler_model, _tokenizer, _desc_tokenizer


# ─────────────────────────────────────────────
# 🔊 Lazy-load pyttsx3 (fallback)
# ─────────────────────────────────────────────
def get_tts_engine():
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 180)
        _engine.setProperty("volume", 1.0)
    return _engine


# International voices (macOS)
LANG_VOICE_MAP = {
    "en": "com.apple.voice.compact.en-US.Samantha",
    "es": "com.apple.voice.compact.es-ES.Monica",
    "fr": "com.apple.voice.compact.fr-FR.Thomas",
    "zh": "com.apple.voice.compact.zh-CN.Tingting",
    "ar": "com.apple.voice.compact.ar-001.Maged",
    "pt": "com.apple.voice.compact.pt-BR.Luciana",
    "ru": "com.apple.voice.compact.ru-RU.Milena",
    "ja": "com.apple.voice.compact.ja-JP.Kyoko",
    "de": "com.apple.voice.compact.de-DE.Anna"
}


# ─────────────────────────────────────────────
# 🎙️ Indic-Parler TTS
# ─────────────────────────────────────────────
def generate_speech(prompt, description, output_file="output.wav"):
    if not prompt or not prompt.strip():
        print("⚠️ Empty prompt. Skipping TTS.")
        return

    try:
        model, tokenizer, desc_tokenizer = load_parler_tts()

        desc = desc_tokenizer(description, return_tensors="pt").to(device)
        prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            audio = model.generate(
                input_ids=desc.input_ids,
                attention_mask=desc.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask
            )

        audio_arr = audio.cpu().numpy().squeeze()

        if audio_arr.size == 0 or np.all(audio_arr == 0) or np.isnan(audio_arr).any():
            print("❌ Invalid waveform generated.")
            return

        sf.write(output_file, audio_arr, model.config.sampling_rate)
        play_wav(output_file)

    except Exception as e:
        print(f"🔥 Indic-Parler TTS error: {e}")


# ─────────────────────────────────────────────
# 🔀 Unified Playback Router
# ─────────────────────────────────────────────
def play_tts_output(text, lang_type, tgt_lang_code):
    if not text or not text.strip():
        print("⚠️ Empty text. Skipping playback.")
        return

    # ── Indian languages → Indic-Parler
    if lang_type == "indian":
        description = "A calm voice with natural pace, clarity and stability."
        generate_speech(text, description, output_file="output_indic.wav")
        return

    # ── International → pyttsx3
    engine = get_tts_engine()
    voice_id = LANG_VOICE_MAP.get(tgt_lang_code, LANG_VOICE_MAP["en"])
    engine.setProperty("voice", voice_id)

    print(f"🔊 Playing via pyttsx3 [{tgt_lang_code}]")
    engine.say(text)
    engine.runAndWait()
