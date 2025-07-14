"""whisper_testbench_xtts.py
Benchmarks how well Whisper transcribes wake words like "hey room" using XTTS samples.
This script evaluates the model's accuracy on positive and negative samples located in the data/samples_xtts directories.
"""

import os
import whisper
import difflib
from glob import glob

# Expecting these folders to contain .wav samples
POS_DIR = "data/samples_xtts/positive"
NEG_DIR = "data/samples_xtts/negative"
# Wake phrase specific to XTTS test; change as needed for other tests
WAKE_PHRASE = "hey room"

model = whisper.load_model("base")  # could use "tiny" or "medium" depending on speed/accuracy needs

def transcribe(path):
    try:
        result = model.transcribe(path)
        txt = result["text"].strip().lower()
        score = difflib.SequenceMatcher(None, txt, WAKE_PHRASE).ratio()
        return txt, score
    except Exception as e:
        return f"[ERR] {e}", 0.0

def eval_dir(folder, label):
    print(f"\n▶ {label.upper()} | {folder}")
    files = glob(os.path.join(folder, "*.wav"))
    total, correct = 0, 0

    for f in files:
        guess, score = transcribe(f)
        match = score >= 0.75
        is_correct = (match and label == "positive") or (not match and label == "negative")
        print(f"{os.path.basename(f):<30} → '{guess}' ({score:.2f}) {'✓' if is_correct else '✗'}")
        total += 1
        correct += int(is_correct)

    acc = 100 * correct / total if total else 0.0
    print(f"\n[{label}] Accuracy: {acc:.1f}% ({correct}/{total})\n")
    return acc

if __name__ == "__main__":
    acc_pos = eval_dir(POS_DIR, "positive")
    acc_neg = eval_dir(NEG_DIR, "negative")

    print("\n=== FINAL SUMMARY ===")
    print(f"✅ POSITIVE: {acc_pos:.1f}%")
    print(f"❌ NEGATIVE: {acc_neg:.1f}%")