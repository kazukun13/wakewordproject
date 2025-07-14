"""
yourtts.py

Generate positive (“Hi Room”) and negative wake‑word samples with YourTTS
and save them to the structured training folders used in this project.

- Uses 15 reference speaker WAVs from `reference_voices/`.
- Creates 100 positive + 100 negative clips per speaker.
- Ensures each output clip is 0.5–3 seconds long at 16 kHz.
- Outputs to:
    data/training_data/finalhiroom/
    data/training_data/finalnotwakeword/

"""

import os
from TTS.api import TTS
import librosa


# Configuration
POS_DIR = "data/training_data/finalhiroom"
NEG_DIR = "data/training_data/finalnotwakeword"
REF_DIR = "reference_voices"
N_SAMPLES = 100
MIN_DUR = 0.5
MAX_DUR = 3.0

# make output folders
os.makedirs(POS_DIR, exist_ok=True)
os.makedirs(NEG_DIR, exist_ok=True)

# basic YourTTS setup
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=True, gpu=False)

# helper
def synth(text, ref, out):
    tts.tts_to_file(
        text=text,
        speaker_wav=ref,
        language="en",
        file_path=out
    )


pos_lines = ["Hi Room!", "Hi Room", "Hii... Room"]
neg_lines = ["Hey broom", "Hey zoom", "They groom", "Play doom", "Hey soon", "Hi zoom", "Wake your room"]

refs = [f"{REF_DIR}/speaker{i+1}.wav" for i in range(15)]

print(f"Generating positive samples to {POS_DIR} ...")
# gen positives
for ref in refs:
    sid = os.path.splitext(os.path.basename(ref))[0]
    for line in pos_lines:
        for i in range(N_SAMPLES):
            fname = line.lower().replace(" ", "")
            out = f"{POS_DIR}/{fname}_{sid}_{i:03}.wav"
            synth(line, ref, out)
            try:
                dur = librosa.get_duration(path=out)
                if dur < MIN_DUR or dur > MAX_DUR:
                    os.remove(out)
            except:
                pass

print(f"Generating negative samples to {NEG_DIR} ...")
# gen negatives
for ref in refs:
    sid = os.path.splitext(os.path.basename(ref))[0]
    for i in range(N_SAMPLES):
        line = neg_lines[i % len(neg_lines)]
        out = f"{NEG_DIR}/neg_{sid}_{i:03}.wav"
        synth(line, ref, out)
        try:
            dur = librosa.get_duration(path=out)
            if dur < MIN_DUR or dur > MAX_DUR:
                os.remove(out)
        except:
            pass

print("Done.")
