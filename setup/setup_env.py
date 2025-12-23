"""
Optional environment utility.

This script performs a Hugging Face cache inspection to help users
verify downloaded models and disk usage.

It does NOT install packages.
It does NOT modify experiment logic.
It does NOT affect evaluation or reported results.

Safe to run multiple times.
"""

import os
import shutil

def scan_huggingface_cache():
    print("🔍 Scanning Hugging Face cache...\n")

    # Ensure huggingface-cli exists
    if shutil.which("huggingface-cli") is None:
        print("⚠️ huggingface-cli not found. Skipping cache scan.")
        print("ℹ️ Install via: pip install huggingface-hub")
        return

    os.system("huggingface-cli scan-cache")

if __name__ == "__main__":
    scan_huggingface_cache()
