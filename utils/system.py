import torch
import platform
import psutil
import warnings
import logging
import transformers

# 🤫 Suppress noisy logs & warnings for clean terminal output
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

# ─────────────────────────────────────────────
# Device Configuration (CPU-ONLY for research)
# ─────────────────────────────────────────────

device = "cpu"

# ─────────────────────────────────────────────
# System Metadata (for reproducibility & logging)
# ─────────────────────────────────────────────

def get_system_info():
    return {
        "device": device,
        "platform": f"{platform.system()} {platform.release()}",
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
    }


def log_device():
    print(f"🔥 Using device: {device.upper()} (CPU-locked for reproducibility)")
