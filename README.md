# 🧭 Project Structure & Setup Guide

## 📁 Folder Structure

```bash
mohit_project/
├── datasets/                            # Contains MELD.Raw dataset
│   └── MELD.Raw/
│       ├── train/
│       ├── dev/
│       └── test/
├── opensmile-3.0.2-windows-x86_64/      # openSMILE binaries and configs
├── pipe_data/                           # Intermediate and final outputs
│   ├── train/
│   ├── dev/
│   └── test/
├── convert_mp4_to_wav.py                # Converts mp4 to wav
├── extract_audio_features.py            # Extracts prosody features via openSMILE
├── extract_meld_text_audio.py           # Matches text with audio paths
├── main_pipeline.py                     # Orchestrates full pipeline
├── train_models.py                      # Trains ML models on extracted features
├── README.md
└── requirements.txt
```

---

## ⚙️ Instructions to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/emotion_detection.git
cd emotion_detection
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download MELD Dataset

* Visit: [https://affective-meld.github.io/](https://affective-meld.github.io/)
* Extract it to:

```
datasets/MELD.Raw/
```

### 4. Download openSMILE

* From: [https://audeering.github.io/opensmile/](https://audeering.github.io/opensmile/)
* Place the folder in root:

```
opensmile-3.0.2-windows-x86_64/
```

### 5. Run the Pipeline

```bash
python main_pipeline.py
```

This will:

* Extract text + audio info
* Convert `.mp4` to `.wav`
* Extract prosody features using openSMILE
* Merge results into CSVs (train/dev/test)

### 6. Train ML Models

```bash
python train_models.py
```

> This trains SVM & Random Forest on prosodic + textual features.

---

## 🧠 Output Files

* `pipe_data/train/meld_audio_prosody.csv`
* `pipe_data/dev/meld_audio_prosody.csv`
* `pipe_data/test/meld_audio_prosody.csv`

These files contain cleaned, engineered features ready for training & evaluation.

---

## 🧾 Python Requirements (requirements.txt)

```txt
pandas
numpy
scikit-learn
moviepy
soundfile
openpyxl
```

---

Feel free to add experiment logs, evaluation scripts, or visualization tools as you proceed!
