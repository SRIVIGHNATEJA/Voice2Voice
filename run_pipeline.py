import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import datetime
import logging
from time import perf_counter

# Global silence
logging.getLogger("transformers").setLevel(logging.ERROR)

from utils.system import device
from utils.language import INDIAN_LANG_MAP
from utils.audio import record_audio, play_beep

from models.asr import transcribe_audio, get_target_language
from models.translation import translate_with_nllb
from models.tts import play_tts_output

# 🆕 System resource tracking
import psutil


def run_voice2voice_evaluation(
    duration=2,
    output_log="evaluation_log.json"
):
    """
    Journal-grade execution + evaluation harness.
    Runs the full Voice2Voice pipeline once and logs metrics.
    """

    # ─── Deterministic runtime setup ──────────────────────
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    timings = {}
    outputs = {}

    start_total = perf_counter()

    # 🆕 Init CPU & RAM tracking
    process = psutil.Process()
    cpu_times_start = process.cpu_times()
    rss_start = process.memory_info().rss
    peak_rss = rss_start

    # ─── Phase 1: Audio Capture ───────────────────────────
    play_beep()
    t0 = perf_counter()
    audio_path = record_audio(duration=duration, filename="input.wav")
    timings["recording_time_sec"] = perf_counter() - t0

    # Track RAM growth
    peak_rss = max(peak_rss, process.memory_info().rss)

    # ─── Phase 2: ASR (Detection + Transcription) ─────────
    t0 = perf_counter()
    transcription = transcribe_audio(audio_path)
    timings["asr_time_sec"] = perf_counter() - t0
    outputs["transcription"] = transcription

    peak_rss = max(peak_rss, process.memory_info().rss)

    # ─── Phase 3: Target Language Selection ───────────────
    play_beep()
    t0 = perf_counter()
    tgt_lang_code, tgt_lang_name = get_target_language()
    timings["target_lang_selection_sec"] = perf_counter() - t0
    outputs["target_language"] = {
        "code": tgt_lang_code,
        "name": tgt_lang_name
    }

    peak_rss = max(peak_rss, process.memory_info().rss)

    # ─── Phase 4: Translation ─────────────────────────────
    t0 = perf_counter()
    translated_text = translate_with_nllb(
        transcription,
        src_lang_code=None,
        tgt_lang_code=tgt_lang_code
    )
    timings["translation_time_sec"] = perf_counter() - t0
    outputs["translation"] = translated_text

    peak_rss = max(peak_rss, process.memory_info().rss)

    if translated_text:
        print(f"🔄 Translated Text: {translated_text}")
    else:
        print("⚠️ Translation failed or returned empty.")

    # ─── Phase 5: TTS Synthesis ───────────────────────────
    t0 = perf_counter()
    lang_type = (
        "indian"
        if tgt_lang_code in INDIAN_LANG_MAP
        else "international"
    )
    play_tts_output(translated_text, lang_type, tgt_lang_code)
    timings["tts_time_sec"] = perf_counter() - t0

    peak_rss = max(peak_rss, process.memory_info().rss)

    # ─── Final Aggregation ───────────────────────────────
    total_runtime = perf_counter() - start_total
    timings["total_pipeline_time_sec"] = total_runtime

    # 🧮 CPU cost calculation (Q1-style)
    cpu_times_end = process.cpu_times()
    cpu_time_used = (
        (cpu_times_end.user + cpu_times_end.system)
        - (cpu_times_start.user + cpu_times_start.system)
    )

    avg_cores_used = cpu_time_used / total_runtime if total_runtime > 0 else 0.0
    num_cores = psutil.cpu_count(logical=True)
    normalized_percent = (avg_cores_used / num_cores) * 100 if num_cores else 0.0

    # 🧮 Peak RAM (MB)
    peak_ram_mb = peak_rss / (1024 * 1024)

    evaluation_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "device": device,
        "timings": timings,
        "outputs": outputs,
        "system_metrics": {
            "avg_cores_used": avg_cores_used,
            "normalized_cpu_percent": normalized_percent,
            "peak_ram_usage_mb": peak_ram_mb
        }
    }

    # 📊 Print Timing Report (BEFORE return!)
    print("\n" + "=" * 40)
    print("⏱️  EXECUTION TIMING REPORT (CPU)")
    print("=" * 40)
    print(f"🎤 Recording:      {timings['recording_time_sec']:.2f}s")
    print(f"📝 ASR + Detect:   {timings['asr_time_sec']:.2f}s")
    print(f"🎯 Lang Selection: {timings['target_lang_selection_sec']:.2f}s")
    print(f"🔄 Translation:    {timings['translation_time_sec']:.2f}s")
    print(f"🗣️ TTS Generation: {timings['tts_time_sec']:.2f}s")
    print("-" * 40)
    print(f"🚀 TOTAL TIME:     {total_runtime:.2f}s")
    print(f"🖥️ Avg CPU Used:  {avg_cores_used:.2f} cores")
    print(f"📊 CPU Capacity:  {normalized_percent:.2f}%")
    print(f"💾 Peak RAM:      {peak_ram_mb:.2f} MB")
    print("=" * 40 + "\n")

    # Append log (SAFE read-modify-write to maintain valid JSON array)
    try:
        if os.path.exists(output_log):
            with open(output_log, "r") as f:
                try:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = []
                except json.JSONDecodeError:
                    existing = []
        else:
            existing = []

        existing.append(evaluation_entry)

        with open(output_log, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
            f.write("\n")

    except Exception as e:
        print(f"⚠️ Failed to update log file: {e}")

    return evaluation_entry


if __name__ == "__main__":
    run_voice2voice_evaluation()
