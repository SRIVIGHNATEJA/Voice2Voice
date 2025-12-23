import whisper
import torch
import torchaudio
from transformers import AutoModel

from utils.system import device
from utils.language import (
    INDIAN_LANG_MAP,
    get_language_code,
    detect_target_language_manually
)
from utils.audio import record_audio


# ─────────────────────────────────────────────
# 🔒 Global cached models (lazy-loaded)
# ─────────────────────────────────────────────
_whisper_model = None
_indic_model = None


# ─────────────────────────────────────────────
# 🎤 Whisper ASR (Primary / Authoritative)
# ─────────────────────────────────────────────
def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print("🔁 Loading Whisper model...")
        _whisper_model = whisper.load_model("small").to(device)
        _whisper_model.eval()
    return _whisper_model


# ─────────────────────────────────────────────
# 🪷 IndicConformer (Detection / Experimental)
# ─────────────────────────────────────────────
def get_indic_model():
    global _indic_model
    if _indic_model is None:
        print("🔁 Loading IndicConformer model...")
        _indic_model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            trust_remote_code=True
        ).to(device)
        _indic_model.eval()
    return _indic_model


# ─────────────────────────────────────────────
# 🌐 Language Detection (Whisper-based)
# ─────────────────────────────────────────────
def detect_input_language_whisper(audio_path: str, model):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    detected_code = max(probs, key=probs.get)

    is_indian = detected_code in INDIAN_LANG_MAP
    lang_name = INDIAN_LANG_MAP.get(detected_code, "International")

    print(f"🌐 Detected Language Code: {detected_code}")
    print(f"🌐 Interpreted as: {lang_name}")

    return detected_code, lang_name, is_indian


# ─────────────────────────────────────────────
# 📝 Whisper Transcription
# ─────────────────────────────────────────────
def whisper_transcribe(audio_path: str, model):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    return result.text


# ─────────────────────────────────────────────
# 🪷 IndicConformer Transcription
# ─────────────────────────────────────────────
def indic_transcribe(audio_path: str, lang_code: str = "hi"):
    """
    Transcribe using IndicConformer for Indian languages.
    """
    indic_model = get_indic_model()
    
    # Load audio at 16kHz (IndicConformer requirement)
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Move to device
    waveform = waveform.to(device)
    
    # Call IndicConformer model (returns list of transcriptions)
    with torch.no_grad():
        transcription = indic_model(waveform, lang_code, "ctc")
    
    return transcription if isinstance(transcription, str) else transcription[0]


# ─────────────────────────────────────────────
# 🔀 Unified Transcription Router
# ─────────────────────────────────────────────
def transcribe_audio(audio_path: str):
    """
    Final ASR behavior:
      • Whisper detects language
      • Indian languages → IndicConformer transcription
      • International languages → Whisper transcription
    """
    whisper_model = get_whisper_model()

    lang_code, lang_name, is_indic = detect_input_language_whisper(
        audio_path, whisper_model
    )

    if is_indic:
        print(f"🛤️ Indian language detected ({lang_name}).")
        print("🪷 Using IndicConformer for transcription...")
        transcription = indic_transcribe(audio_path, lang_code)
    else:
        print("🛤️ International language detected.")
        print("🌍 Using Whisper for transcription...")
        transcription = whisper_transcribe(audio_path, whisper_model)

    print(f"📝 Final Transcription: {transcription}")
    return transcription


# ─────────────────────────────────────────────
# 🎯 Target Language Detection (Voice-based)
# ─────────────────────────────────────────────
def detect_target_language_by_voice(attempts=2):
    whisper_model = get_whisper_model()

    for attempt in range(attempts):
        print(
            f"🗣️ Attempt {attempt + 1}/{attempts}: "
            "Speak target language name (e.g., Tamil, Hindi, German)"
        )

        file_path = record_audio(
            duration=1.75,
            filename=f"target_attempt_{attempt + 1}.wav"
        )

        audio = whisper.load_audio(file_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)

        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(whisper_model, mel, options)
        spoken = result.text.strip()

        print(f"📝 You said: {spoken}")

        lang_code, lang_name = get_language_code(spoken)
        if lang_code:
            return lang_code, lang_name

        print("⚠️ Not confident. Try again...\n")

    return None, None


# ─────────────────────────────────────────────
# 🧭 Unified Target Language Selector
# ─────────────────────────────────────────────
def get_target_language():
    print("🎯 Target Language Selection Started")

    code, name = detect_target_language_by_voice()

    if not code:
        print("🔁 Switching to manual input...")
        code, name = detect_target_language_manually()

    if code:
        print(f"✅ Final Target Language: {name.capitalize()} ({code})")
    else:
        print("❌ Language not supported.")

    return code, name
