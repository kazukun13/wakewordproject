import os
import torch
import torchaudio
import joblib
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from speechbrain.pretrained import EncoderClassifier
import matplotlib.pyplot as plt
import sounddevice as sd

# quick config — just edit here
POS_DIR = "training_data/heyroom"
NEG_DIR = "training_data/not_heyroom"
THRESH = 0.5

# load trained stuff
model = joblib.load("wakeword_classifier.joblib")
try:
    scaler = joblib.load("wakeword_scaler.joblib")
    use_scaler = True
except:
    scaler = None
    use_scaler = False

# speaker-agnostic embedding model
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

def embed(wav_path):
    x, fs = torchaudio.load(wav_path)
    if fs != 16000:
        x = torchaudio.transforms.Resample(fs, 16000)(x)
    with torch.no_grad():
        emb = classifier.encode_batch(x)
    return emb.squeeze().numpy()

def classify(embedding):
    if use_scaler:
        embedding = scaler.transform([embedding])
    prob = model.predict_proba([embedding])[0][1]
    pred = int(prob >= THRESH)
    return prob, pred

# main eval loop
def run_eval():
    probs = []
    y_pred = []
    y_true = []

    for label, folder in [("pos", POS_DIR), ("neg", NEG_DIR)]:
        y = 1 if label == "pos" else 0
        for f in os.listdir(folder):
            if not f.endswith(".wav"): continue
            try:
                p, pred = classify(embed(os.path.join(folder, f)))
                y_true.append(y)
                y_pred.append(pred)
                probs.append(p)
                print(f"{f} -> {p:.4f} | pred={pred} | true={y}")
            except Exception as e:
                print(f"[FAIL] {f}: {e}")

    print("\n== Metrics ==")
    print(classification_report(y_true, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_true, probs):.4f}")
    return y_true, probs

def mic_check(seconds=2):
    print(f"Listening for {seconds}s...")
    fs = 16000
    rec = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    audio = torch.tensor(rec.T)
    with torch.no_grad():
        emb = classifier.encode_batch(audio)
    emb_np = emb.squeeze().numpy()
    if use_scaler:
        emb_np = scaler.transform([emb_np])
    prob = model.predict_proba([emb_np])[0][1]
    pred = int(prob >= THRESH)
    print(f"[Mic] {prob:.4f} → {'Wakeword' if pred else 'Nope'}")
    return prob, pred

def plot_probs(probs, y_true):
    plt.figure(figsize=(8, 4))
    plt.hist(probs, bins=50, alpha=0.7)
    plt.axvline(THRESH, color='r', linestyle='--')
    plt.title("Confidence Scores")
    plt.xlabel("P(wakeword)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(y_true, probs)
    plt.figure(figsize=(8, 4))
    plt.plot(t, p[:-1], label="Precision")
    plt.plot(t, r[:-1], label="Recall")
    plt.axvline(THRESH, color='r', linestyle='--')
    plt.title("Threshold Tuning")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    y_true, probs = run_eval()
    plot_probs(probs, y_true)