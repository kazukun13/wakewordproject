
### README.md

```
#  Wakeword Detection: "Hey Room", "Hi Room", "Wakeup Room"

A custom wakeword detection system optimized for Indian-accented English using TTS, voice cloning, and OpenWakeWord-compatible training.

---

##  Features
- Detects 3 wakephrases: `Hey Room`, `Hi Room`, `Wakeup Room`
- Indian-accent voice diversity via XTTS, OpenVoice, and YourTTS
- Background noise augmentation for realism
- Model trained with ECAPA embeddings + logistic regression
- Real-time mic inference and evaluation scripts

---

##  Project Structure
```

heyroom/
├── data/                # Cleaned and structured dataset
├── models/              # Trained models and config
├── scripts/             # Sample generation tools
├── evaluation/          # Accuracy testing, Whisper clarity checks
├── inference/           # Real-time mic detection
├── train/               # Training scripts
├── requirements.txt     # Python dependencies
├── .gitignore           # Ignore unnecessary files
└── README.md            # You're reading this

````

---

##  Setup
```bash
# 1. Create venv
python3 -m venv heyroom-env
source heyroom-env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
````

---

## 🧪 Evaluate Model

```bash
python evaluation/testbench_openwakeword.py
```

Outputs:

* Accuracy, confusion matrix
* Confidence scores per file
* `custom_model_test_results.csv`

---

##  Live Mic Wakeword Detection

```bash
python inference/infer_wakeword.py
```

---

##  Retrain Wakeword Model

```bash
python train/train_openwakeword.py
```

Ensure that `training_data/` has folders:

* `heyroom/`, `hiroom/`, `wakeuproom/`, `not_wakeword/`

---

##  Generate Samples

Use XTTS, OpenVoice, or YourTTS from `scripts/` folder to create `.wav` files for each phrase. See `xtts.py`, `openvoice_generator.py`, etc.

---

##  Optional: Whisper Evaluation

```bash
python scripts/whisper_testbench.py
```

Check how clearly TTS samples are transcribed by Whisper.

---

##  Acknowledgements

* [OpenWakeWord](https://github.com/david-berthelot/openWakeWord)
* [Coqui TTS](https://github.com/coqui-ai/TTS)
* [SpeechBrain ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)

```
```

*This README.md was partially generated with CHATGPT