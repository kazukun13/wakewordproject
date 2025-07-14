import os
import random
from TTS.api import TTS

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

alts = [
    "Hey broom", "Hey doom", "Hey groom", "Hey zoom", "Hey Roam",
    "Hey droon", "Hey drone", "Hey doom", "Hey roon", "Hey rune"
]

out_dir = "data/training_data/finalnotwakeword"
os.makedirs(out_dir, exist_ok=True)

for spk in range(1, 16):
    ref = f"reference_voices/speaker{spk}.wav"
    for i in range(100):
        line = random.choice(alts)
        fname = f"final_s{spk}_{i:03}.wav"
        tts.tts_to_file(text=line, speaker_wav=ref, language="en", file_path=os.path.join(out_dir, fname))

print("done.")
