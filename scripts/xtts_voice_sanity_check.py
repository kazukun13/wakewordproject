# xtts_voice_sanity_check.py
# xtts_voice_sanity_check.py

"""
XTTS Voice Sanity Check Script

This script validates reference speaker voice clips used for XTTS voice cloning.
It checks for:
- Sample rate compliance (expected: 16kHz)
- Duration within bounds (default: 3.0s to 10.0s)
- Mono channel (required by XTTS)
- File readability

Usage:
    Place reference speaker `.wav` files in the `reference_voices/` directory.
    Then run the script to validate the audio files.

Author: [Your Name]
Date: [YYYY-MM-DD]
"""

import os
import soundfile as sf

#
# Configuration
#
REF_DIR = "reference_voices"
TARGET_SR = 16000
MIN_LEN = 3.0  # in seconds
MAX_LEN = 10.0  # in seconds

print(f"\n🔍 Validating XTTS reference voice clips in: {REF_DIR}\n")

for fname in sorted(os.listdir(REF_DIR)):
    if not fname.endswith(".wav"):
        continue

    path = os.path.join(REF_DIR, fname)

    try:
        audio, sr = sf.read(path)
        dur = len(audio) / sr  # duration in seconds

        problems = []

        # Check sample rate
        if sr != TARGET_SR:
            problems.append(f"sample rate is {sr}Hz")

        # Check duration
        if dur < MIN_LEN:
            problems.append(f"too short ({dur:.1f}s)")
        elif dur > MAX_LEN:
            problems.append(f"too long ({dur:.1f}s)")

        # Check mono channel
        if audio.ndim != 1:
            problems.append("not mono")

        # Print summary
        if problems:
            print(f"{fname:<25} ⚠️  {'; '.join(problems)}")
        else:
            print(f"{fname:<25} ✅ valid ({dur:.1f}s @ {sr}Hz)")

    except Exception as e:
        print(f"{fname:<25} ❌ error reading file: {e}")