import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Designs a stable Butterworth bandpass filter.
    """
    nyq = 0.5 * fs

    # Ensure cutoffs are strictly within Nyquist
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.999)

    return butter(order, [low, high], btype="band")


def bandpass_filter(data, lowcut=80.0, highcut=7900.0, fs=16000, order=4):
    """
    Applies bandpass filtering to audio data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)


def record_audio(duration=3, fs=16000, filename="user_input.wav", playback=False):
    """
    Records audio, applies bandpass filtering, and saves to disk.
    This preprocessing is deterministic and reproducible.
    """
    print(f"🎤 Recording ({duration}s)…")

    data = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype="float32"
    )
    sd.wait()

    raw = data.flatten()
    filtered = bandpass_filter(raw, fs=fs)

    if playback:
        print("🔊 Playing back filtered audio…")
        sd.play(filtered.astype(np.float32), fs)
        sd.wait()

    sf.write(filename, filtered, fs)
    print(f"✅ Filtered audio saved: {filename}")

    return filename


def play_beep(duration=0.07, freq=1000, fs=44100):
    """
    Plays a short beep sound to cue recording.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    beep = np.sin(2 * np.pi * freq * t).astype(np.float32)

    sd.play(beep, fs)
    sd.wait()


def play_wav(filename):
    """
    Plays a WAV file using sounddevice (blocking).
    """
    try:
        data, fs = sf.read(filename, dtype="float32")
        sd.play(data, fs)
        sd.wait()
    except Exception as e:
        print(f"⚠️ Audio playback failed: {e}")
