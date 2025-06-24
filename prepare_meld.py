# === SCRIPT 1: Extract Utterance + Audio Info from MELD ===

import pandas as pd
import os

# CONFIG
BASE_DIR = "C:\\Users\\Administrator\\Desktop\\mohit_project\\datasets\\MELD.Raw\\MELD.Raw\\train"
CSV_PATH = os.path.join(BASE_DIR, "train_sent_emo.csv")
AUDIO_DIR = os.path.join(BASE_DIR, "train_splits")

# Load CSV
df = pd.read_csv(CSV_PATH)

# Add audio paths
df["audio_path"] = df.apply(
    lambda row: os.path.join(AUDIO_DIR, f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"),
    axis=1
)
df["audio_exists"] = df["audio_path"].apply(os.path.exists)

# Filter valid entries
df = df[df["audio_exists"]].reset_index(drop=True)

# Save result
df[["Utterance", "Emotion", "audio_path"]].to_csv("C:\\Users\\Administrator\\Desktop\\mohit_project\\pipe_data\\meld_audio_text.csv", index=False)
