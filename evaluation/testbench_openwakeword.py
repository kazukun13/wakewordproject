# testbench_openwakeword.py
# Testbench to evaluate a custom wakeword model with labeled samples

import os
import csv
import soundfile as sf
import numpy as np
from collections import Counter
from speechbrain.pretrained import EncoderClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import torch

# --- Config ---
model_path = "wakeword_classifier.joblib"
positive_dir = "training_data/heyroom"
negative_dir = "training_data/not_heyroom"
output_csv = "custom_model_test_results.csv"
threshold = 0.5

# --- Load model and scaler ---
print("Loading model and scaler...")
model = joblib.load(model_path)
# --- Load ECAPA embedder ---
embedder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cpu"},
    savedir="pretrained_models/ecapa"
)

# --- Embedding extractor ---
def extract_embedding(wav_path):
    signal, sr = sf.read(wav_path)
    if len(signal.shape) > 1:  # Stereo to mono
        signal = np.mean(signal, axis=1)
    emb = embedder.encode_batch(torch.tensor(signal).unsqueeze(0)).squeeze().detach().numpy()
    return emb

# --- Evaluate a directory ---
def evaluate_dir(directory, label):
    results = []
    for fname in os.listdir(directory):
        if fname.endswith(".wav"):
            path = os.path.join(directory, fname)
            try:
                emb = extract_embedding(path)
                prob = model.predict_proba([emb])[0][1]
                prediction = "positive" if prob > threshold else "negative"
                correct = prediction == label
                results.append((fname, prob, prediction, label, correct))
            except Exception as e:
                print(f"Failed on {fname}: {e}")
    return results

# --- Run tests ---
print("Evaluating positive samples...")
positive_results = evaluate_dir(positive_dir, "positive")

print("Evaluating negative samples...")
negative_results = evaluate_dir(negative_dir, "negative")

all_results = positive_results + negative_results

# --- Summary Metrics ---
total = len(all_results)
correct = sum(1 for r in all_results if r[-1])
accuracy = correct / total * 100 if total else 0

# Confusion matrix
conf_matrix = Counter()
for _, _, pred, truth, _ in all_results:
    conf_matrix[(truth, pred)] += 1

print("\n✅ Accuracy: {:.2f}% ({} out of {})".format(accuracy, correct, total))
print("\nConfusion Matrix:")
for (truth, pred), count in conf_matrix.items():
    print(f"{truth} predicted as {pred}: {count}")

# --- Write CSV ---
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Filename", "Score", "Prediction", "GroundTruth", "Correct"])
    writer.writerows(all_results)

print(f"\nResults written to {output_csv}")