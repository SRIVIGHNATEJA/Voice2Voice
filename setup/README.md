# Setup Scripts

This directory contains **optional auxiliary utilities** intended for
environment validation and runtime consistency.

These scripts are **not part of the experimental methodology** described
in the research paper.

## Files

- `setup_env.py`  
  Performs optional environment checks and inspects local Hugging Face caches.
  This script does **not modify model logic, routing decisions, or inference behavior**.

- `warmup_models.py`  
  Executes preliminary inference passes for ASR, translation, and TTS models
  to preload model weights and tokenizers.

  This is intended solely to:
  - avoid first-run initialization overhead
  - ensure stable timing measurements

## Important Notes

- Running these scripts is **optional**
- They are **not required** to reproduce reported experimental results
- Warm-up execution does **not** affect accuracy, routing, or output quality
- All evaluation metrics are collected only during controlled pipeline execution
- The primary reproducible entry point is `run_pipeline.py`

These utilities are provided for runtime consistency and user convenience
and are **explicitly excluded from experimental analysis**.
