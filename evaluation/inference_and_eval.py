# HeyRoom Wake Word Detection

HeyRoom is a robust wake word detection system built using synthetic voice generation, background noise augmentation, and a machine learning pipeline powered by ECAPA embeddings and a multi-class classifier.

The goal? Let your system respond to custom phrases like "Hey Room", "Hi Room", and "Wake Up Room"—even in noisy environments.

---

## 🔍 What It Does

This project allows your system to recognize three different custom wake phrases:
- **Hey Room**
- **Hi Room**
- **Wake Up Room**

It also handles **non-wake phrases** effectively to avoid false activations.

---

## 🚀 Quick Start

1. **Clone the Repo**

```bash
git clone https://github.com/yourname/heyroom-wakeword.git
cd heyroom-wakeword
```

2. **Set up the Environment**

Use your preferred method:

**With venv**
```bash
python3 -m venv heyroom-env
source heyroom-env/bin/activate
pip install -r requirements.txt
```

**With Conda**
```bash
conda create -n heyroom python=3.10
conda activate heyroom
pip install -r requirements.txt
```

---

## 🧪 How It Works

1. **Sample Generation**: Synthetic samples were generated using OpenVoice, YourTTS, and XTTS.
2. **Background Noise**: Random noise (Indian streets, fans, etc.) was overlayed to simulate real-world environments.
3. **Embedding**: We use Wav2Vec 2.0 for extracting embeddings.
4. **Classification**: A multi-class model (RandomForest/MLPClassifier) maps these embeddings to phrases.
5. **Evaluation**: Accuracy, precision, recall, F1, and ROC-AUC are measured.

---

## 📁 Folder Structure

```
data/
  training_data/
    finalheyroom/
    finalhiroom/
    finalwakeupRoom/
    finalnotwakeword/
assets/
  background/       # Noise WAVs
  source/           # Raw source audio
scripts/
  openvoice_generator.py
evaluation/
  ecapa.py
  inference_and_eval.py
models/
  wakeword_multiclass_classifier.joblib
  wakeword_label_encoder.joblib
```

---

## 📊 Results

Our best model achieved:
- **Accuracy**: ~88%
- **Macro F1**: ~0.89
- **ROC-AUC**: ~0.91
- Balanced performance across all 4 classes

---

## 📦 Requirements

All dependencies are listed in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

---

## 📝 Notes

- Tested on Python 3.10
- All audio is 16kHz WAV mono
- You’ll need to manually download large TTS models if regenerating data

---

## 💡 Ideas for Improvement

- Real-user recordings
- Post-training model compression (e.g., TFLite export)
- On-device inference with real-time streaming
- Integrate VAD and noise suppression at inference

---

## 🧠 Credits

- [openWakeWord](https://github.com/dscripka/openWakeWord)
- [MyShell OpenVoice](https://github.com/myshell-ai/OpenVoice)
- [SpeechBrain](https://github.com/speechbrain/speechbrain)
- [YourTTS](https://github.com/Edresson/YourTTS)

---

## 📄 License

MIT License
