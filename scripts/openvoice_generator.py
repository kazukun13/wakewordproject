import os
import random
import torch
from melo.api import TTS
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from pydub import AudioSegment
import nltk

nltk.download('averaged_perceptron_tagger_eng')

base_dir = os.path.dirname(os.path.dirname(__file__))

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_converter = "openvoice/checkpoints_v2/converter"
se_dir = "openvoice/checkpoints_v2/base_speakers/ses"

tone_converter = ToneColorConverter(f"{ckpt_converter}/config.json", device=device)
tone_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")

reference_voices = {
    f"speaker{i}": os.path.join(base_dir, f"reference_voices/speaker{i}.wav")
    for i in range(1, 16)
}

neg_lines = ["Woke up room", "Wake Room", "Awake Room", "Hiya Room", "Yay Room!"]

os.makedirs("data/training_data/finalhiroom", exist_ok=True)
os.makedirs("data/training_data/finalnotwakeword", exist_ok=True)

n = 100  # samples per speaker

def augment_and_save(wav_path, out_path):
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
        print(f"[SKIP] {wav_path} missing or corrupt")
        return
    try:
        audio = AudioSegment.from_wav(wav_path)
    except Exception as e:
        print(f"[FAIL] Couldn't read {wav_path}: {e}")
        os.remove(wav_path)
        return

    pitch = random.uniform(0.95, 1.05)
    speed = random.uniform(0.95, 1.05)
    audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * pitch)})
    audio = audio.set_frame_rate(16000)

    try:
        audio = audio.speedup(playback_speed=speed)
    except:
        pass

    dur = len(audio)
    if dur < 500 or dur > 3000:
        print(f"[DROP] {wav_path} — {dur}ms")
        os.remove(wav_path)
        return

    audio.export(out_path, format="wav")

source_se = torch.load(os.path.join(se_dir, "en-india.pth")).to(device)

tts = TTS(language="EN", device=device)
spk_id = tts.hps.data.spk2id["EN_INDIA"]

for sid, wav in reference_voices.items():
    for i in range(n):
        line = random.choice(neg_lines)
        tmp = f"temp_{sid}_{i}.wav"
        raw = f"data/training_data/finalnotwakeword/raw_{sid}_{i:03}.wav"
        final = f"data/training_data/finalnotwakeword/final_{sid}_{i:03}.wav"

        try:
            tgt_se, _ = se_extractor.get_se(wav, tone_converter, vad=False)
        except AssertionError:
            print(f"[SKIP] {wav} too short")
            continue

        try:
            if os.path.exists(tmp):
                os.remove(tmp)
            tts.tts_to_file(line, spk_id, tmp, speed=random.uniform(0.95, 1.05))
            if not os.path.exists(tmp) or os.path.getsize(tmp) < 1000:
                print(f"[MISS] TTS failed on {sid} — '{line}'")
                continue
            os.rename(tmp, raw)
        except Exception as e:
            print(f"[ERR] TTS gen fail {sid}: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)
            continue

        tone_converter.convert(
            audio_src_path=raw,
            src_se=source_se,
            tgt_se=tgt_se,
            output_path=final,
            message="@MyShell"
        )

        augment_and_save(final, final)

import glob
for f in glob.glob("data/training_data/**/raw*.wav", recursive=True):
    os.remove(f)

print("Done.")
