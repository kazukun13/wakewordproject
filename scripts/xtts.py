import os
import random
from TTS.api import TTS as TTSModel
from pydub import AudioSegment

# ----------------------------------------------------------
# Script: xtts.py
# Purpose: Generate synthetic wakeword and negative audio samples using XTTS v2
#          and save them to structured training folders for model development.
# ----------------------------------------------------------

# Set FFmpeg path for pydub
AudioSegment.converter = "/opt/homebrew/bin/ffmpeg"

# Safe loading for custom config
import torch.serialization
from TTS.tts.configs.xtts_config import XttsConfig
torch.serialization.add_safe_globals([XttsConfig])

# Load XTTS model (CPU for now)
tts = TTSModel(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=False)

# Reference speaker voices (ensure these exist)
ref_voices = {
    f"speaker{i}": f"reference_voices/speaker{i}.wav"
    for i in range(1, 16)
}

# Phrases to generate (positive = wakeword; negative = similar but incorrect)
pos_texts = ["Hey Room", "Hey, Room!", "Heyy Rooom", "Hey... Room", "Hi Room", "Wake Up Room"]
neg_texts = ["Hey broom", "Hello room", "Hello roomie", "Hey Doom", "Hey drone", "Wake broom", "Hi groom"]

# Output folders
for d in ["data/training_data/heyroom", "data/training_data/notwakeword"]:
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        fp = os.path.join(d, f)
        if os.path.isfile(fp):
            os.remove(fp)

n = 100  # samples per speaker

# Helper function to process, augment and save audio clips
def augment_save(src, dst):
    audio = AudioSegment.from_wav(src)
    dur = len(audio)
    if dur < 500 or dur > 3000:
        print(f"Skipping {src} ({dur}ms)")
        return
    pitch = random.uniform(0.95, 1.05)
    speed = random.uniform(0.95, 1.05)
    audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * pitch)})
    audio = audio.set_frame_rate(16000)
    try:
        audio = audio.speedup(playback_speed=speed)
    except:
        pass
    audio.export(dst, format="wav")

# Generate positive samples
print("Generating positive samples...")
for spk, wav in ref_voices.items():
    if not os.path.exists(wav):
        print(f"[WARN] Skipping {spk}: reference voice file not found.")
        continue
    for i in range(n):
        txt = random.choice(pos_texts)
        tts.tts_to_file(text=txt, speaker_wav=wav, language="en", file_path="temp.wav")
        augment_save("temp.wav", f"data/training_data/heyroom/xtts_{spk}_{i:03}.wav")

# Generate negative samples
print("Generating negative samples...")
for spk, wav in ref_voices.items():
    if not os.path.exists(wav):
        continue
    for i in range(n * 2):
        txt = random.choice(neg_texts)
        tts.tts_to_file(text=txt, speaker_wav=wav, language="en", file_path="temp.wav")
        augment_save("temp.wav", f"data/training_data/notwakeword/xtts_{spk}_{i:03}.wav")

# Clean up
if os.path.exists("temp.wav"):
    os.remove("temp.wav")

print("Done.")