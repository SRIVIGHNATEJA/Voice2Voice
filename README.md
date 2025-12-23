# A Unified Neural Framework for Offline Multilingual Speech-to-Speech Translation with Adaptive ASR Routing and Cross-Lingual TTS


A modular, reproducible **voice-to-voice translation research system** supporting Indian and international languages.
This repository represents a refactored, file-based implementation derived from a controlled research notebook and is structured to ensure **deterministic execution, dependency isolation, and academic reproducibility**.

This codebase is intended to **support the experimental methodology described in the associated research paper** and is not positioned as a deployment-ready or end-user system.

---

## Overview

The system implements a **multi-stage voice-to-voice inference pipeline** comprising:

1. Audio capture and preprocessing
2. Automatic speech recognition (ASR)
3. Language identification and model routing
4. Neural machine translation (NMT)
5. Text-to-speech synthesis (TTS)

Model-specific routing decisions, architectural rationale, and evaluation methodology are **intentionally described in the research paper** and not exhaustively documented here.

---

## Project Structure

```
Voice2Voice/
│
├── run_pipeline.py
├── requirements.txt
│
├── models/
│   ├── asr.py
│   ├── translation.py
│   └── tts.py
│
├── utils/
│   ├── audio.py
│   ├── language.py
│   └── system.py
│
├── setup/
│   ├── setup_env.py
│   ├── warmup_models.py
│   └── README.md
│
└── notebooks/
    └── notebook_reference.ipynb
```

The directory layout mirrors the logical separation of system components used during experimentation. Internal module behavior is documented inline within the source code and discussed in the paper.

---

## Environment & Reproducibility

* **Execution mode**: CPU-only (as stated in the research paper)
* **GPU / MPS**: Explicitly disabled to avoid hardware-induced variance
* **Model loading**: Lazy-loaded and cached at runtime
* **Warmup**: Optional warmup utilities provided to stabilize first-run latency

> Platform-specific TTS backends (e.g., `pyttsx3`) are used strictly for usability in non-Indian language cases and are **excluded from latency, quality, and comparative evaluation**.

---

## Installation

Execution assumes familiarity with the system architecture and controlled experimental setups.
It is strongly recommended to use a **fresh virtual environment**.

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

All dependencies are pinned to ensure reproducibility across runs.

---

## Optional Setup & Warmup

```bash
python setup/setup_env.py
python setup/warmup_models.py
```

Warmup utilities preload model weights and tokenizers to reduce first-run variance and support consistent timing measurements.

---

## Running the System

```bash
python run_pipeline.py
```

The entry point executes the complete voice-to-voice inference cycle defined in the methodology section of the paper and logs:

* Phase-wise execution times
* Language routing decisions
* System resource usage

Generated logs are intended for **offline analysis and reporting**, not for real-time benchmarking claims.

---

## Model Attribution & Citations

This project makes exclusive use of **pretrained models**. No claim of authorship is made for any model component.

### Automatic Speech Recognition (ASR)

* **Whisper** – OpenAI
  Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision*, 2022

* **IndicConformer** – AI4Bharat (IIT Madras)

### Neural Machine Translation (NMT)

* **NLLB-200** – Meta AI
  Costa-jussà et al., *No Language Left Behind*, 2022

### Text-to-Speech (TTS)

* **Indic-Parler-TTS** – AI4Bharat

### Fallback TTS (Non-evaluated)

* **pyttsx3** – Platform-dependent speech synthesis (used solely for usability)

Formal citations and license details should be included in the paper or appendix as appropriate.

---

## License & Usage

This repository is intended **exclusively for academic and research use**.
It does **not constitute a benchmark, product, or deployment-ready system**.

Users are responsible for complying with individual model licenses provided by their respective authors.

---

## Acknowledgements

* OpenAI – Whisper
* AI4Bharat – IndicConformer, Indic-Parler-TTS
* Meta AI – NLLB-200

---

**Context**: Academic research project
