import os
import csv
import random
from pydub import AudioSegment
from pydub.effects import speedup

skipped_log_path = "background_overlay_skipped.csv"

def overlay_background_noise(clean_path, noise_dir, output_path, snr_db, log_writer=None, fname=""):
    clean = AudioSegment.from_wav(clean_path)
    noise_files = [f for f in os.listdir(noise_dir) if f.endswith(".wav")]
    if not noise_files:
        print(f"[WARN] No noise in {noise_dir}, skipping")
        return

    noise_path = os.path.join(noise_dir, random.choice(noise_files))
    noise = AudioSegment.from_wav(noise_path)

    clean = clean.normalize()
    noise = noise.normalize()

    dur = len(clean)
    speed_factor = 1.0 if dur < 1000 else random.uniform(0.95, 1.05)
    clean = speedup(clean, playback_speed=speed_factor)

    if len(noise) < len(clean):
        noise *= ((len(clean) // len(noise)) + 1)
    noise = noise[:len(clean)]

    if len(noise) > len(clean):
        shift = random.randint(0, len(noise) - len(clean))
        noise = noise[shift:shift + len(clean)]

    noise += random.uniform(-5, 0)

    clean_rms = clean.rms
    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    if noise.rms > 0:
        adjust = 20 * (random.random() - 0.5)
        noise = noise - (noise.rms - target_noise_rms) + adjust

    combined = clean.overlay(noise)

    if len(combined) < 500 or len(combined) > 3000:
        print(f"[SKIP] {fname} - duration {len(combined)}ms")
        if log_writer:
            with open(skipped_log_path, mode='a', newline='') as skipped_file:
                writer = csv.writer(skipped_file)
                writer.writerow([fname, f"Duration {len(combined)}ms", dur, clean.rms])
        clean.export(output_path, format="wav")
        return

    combined.export(output_path, format="wav")
    if log_writer:
        log_writer.writerow([fname, os.path.basename(noise_path), round(snr_db, 2), clean.rms, noise.rms, len(clean)])

sample_dirs = {
    "data/training_data/heyroom": "data/training_data/finalheyroom",
    "data/training_data/archived_notwakeword": "data/training_data/finalnotwakeword"
}

noise_dir = "assets/background"

with open("background_overlay_log.csv", 'w', newline='') as log_file:
    log_writer = csv.writer(log_file)
    log_writer.writerow(["Filename", "Noise Used", "SNR (dB)", "Clean RMS", "Noise RMS", "Duration (ms)"])

with open(skipped_log_path, 'w', newline='') as skipped_file:
    skipped_writer = csv.writer(skipped_file)
    skipped_writer.writerow(["Filename", "Reason", "Clean Duration (ms)", "Clean RMS"])

with open("background_overlay_log.csv", 'a', newline='') as log_file:
    log_writer = csv.writer(log_file)

    for src_dir, out_dir in sample_dirs.items():
        os.makedirs(out_dir, exist_ok=True)
        files = [f for f in os.listdir(src_dir) if f.endswith(".wav")]
        to_process = random.sample(files, len(files) // 2)
        done = 0
        tries = 0
        max_tries = len(to_process) * 30

        while done < len(to_process) and tries < max_tries:
            fname = random.choice(files)
            in_path = os.path.join(src_dir, fname)
            out_path = os.path.join(out_dir, fname)
            snr = random.uniform(5, 15)
            try:
                overlay_background_noise(in_path, noise_dir, out_path, snr, log_writer, fname)
                print(f"[OK] {out_path}")
                done += 1
            except Exception as e:
                print(f"[FAIL] {fname}: {e}")
            tries += 1