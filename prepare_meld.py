import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--split', required=True, choices=['train', 'dev', 'test'])
args = parser.parse_args()

SPLIT = args.split
DATA_DIR = f"C:/Users/Administrator/Desktop/mohit_project/datasets/MELD.Raw/MELD.Raw/{SPLIT}"
CSV_PATH = os.path.join(DATA_DIR, f"{SPLIT}_sent_emo.csv")
AUDIO_DIR = os.path.join(DATA_DIR, f"{SPLIT}_splits")
OUTPUT_DIR = f"C:/Users/Administrator/Desktop/mohit_project/pipe_data/{SPLIT}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "meld_audio_text.csv")

if os.path.exists(OUTPUT_CSV):
    print(f"⏩ Skipping {SPLIT} - Already exists: {OUTPUT_CSV}")
    exit()

df = pd.read_csv(CSV_PATH)
df["audio_path"] = df.apply(
    lambda row: os.path.join(AUDIO_DIR, f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"),
    axis=1
)
df["audio_exists"] = df["audio_path"].apply(os.path.exists)
df = df[df["audio_exists"]].reset_index(drop=True)

df[["Utterance", "Emotion", "audio_path"]].to_csv(OUTPUT_CSV, index=False)
print(f"✅ Created: {OUTPUT_CSV} ({len(df)} rows)")
