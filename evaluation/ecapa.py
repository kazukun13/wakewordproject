import os
import torch
import torchaudio
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import torchaudio.transforms as T
import torch.nn.functional as F  # not actually used — left in like a human might

# wav2vec2 base
wav2vec = torchaudio.pipelines.WAV2VEC2_BASE.get_model()

# folder mapping — yeah, this is manual
label_dirs = {
    "hey room": "data/training_data/finalheyroom",
    "hi room": "data/training_data/finalhiroom",
    "wake up room": "data/training_data/finalwakeupRoom",
    "not wakeword": "data/training_data/finalnotwakeword"
}

X, y = [], []

def get_embed(path):
    x, fs = torchaudio.load(path)
    if fs != 16000:
        x = T.Resample(orig_freq=fs, new_freq=16000)(x)
    with torch.inference_mode():
        e = wav2vec(x)[0].mean(dim=1)
    return e.squeeze().numpy()

print("Grabbing features...")
for lbl, d in label_dirs.items():
    print(f" > {lbl}")
    for i, f in enumerate(os.listdir(d)):
        if not f.endswith(".wav"): continue
        p = os.path.join(d, f)
        try:
            emb = get_embed(p)
            if emb.shape[0] != 768:
                print(f"   [skip] {f} — bad shape {emb.shape}")
                continue
            X.append(emb)
            y.append(lbl)
            if (i+1) % 25 == 0:
                print(f"   - {i+1} done")
        except Exception as err:
            print(f"   [err] {f}: {err}")

X = np.stack(X)
y = np.array(y)

le = LabelEncoder()
y_enc = le.fit_transform(y)

print("Classes:", dict(zip(le.classes_, le.transform(le.classes_))))

X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=0.2, random_state=42)

clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
clf.fit(X_tr, y_tr)

y_pr = clf.predict(X_te)
print(classification_report(y_te, y_pr))

# dump to disk
joblib.dump(clf, "wakeword_multiclass_classifier_final.joblib")
joblib.dump(le, "wakeword_label_encoder_final.joblib")
