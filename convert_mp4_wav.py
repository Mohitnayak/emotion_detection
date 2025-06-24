import os
import pandas as pd
import argparse
from moviepy.editor import AudioFileClip

parser = argparse.ArgumentParser()
parser.add_argument('--split', required=True, choices=['train', 'dev', 'test'])
args = parser.parse_args()

SPLIT = args.split
BASE_DIR = f"C:/Users/Administrator/Desktop/mohit_project/pipe_data/{SPLIT}"
INPUT_CSV = os.path.join(BASE_DIR, "meld_audio_text.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "meld_with_wavs.csv")
WAV_DIR = os.path.join(BASE_DIR, "wav_outputs")
os.makedirs(WAV_DIR, exist_ok=True)

if os.path.exists(OUTPUT_CSV):
    print(f"⏩ Skipping {SPLIT} - Already exists: {OUTPUT_CSV}")
    exit()

df = pd.read_csv(INPUT_CSV)
wav_paths = []

for i, row in df.iterrows():
    mp4_path = row["audio_path"]
    wav_path = os.path.join(WAV_DIR, f"utt_{i}.wav")

    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
        print(f"⏩ Exists: utt_{i}.wav")
        wav_paths.append(wav_path)
        continue

    try:
        print(f"🎧 Converting: {mp4_path} → utt_{i}.wav")
        clip = AudioFileClip(mp4_path)
        clip.write_audiofile(wav_path, fps=16000, nbytes=2, codec='pcm_s16le')
        wav_paths.append(wav_path)
    except Exception as e:
        print(f"❌ Failed for: {mp4_path}\n{e}")
        wav_paths.append(None)

df["wav_path"] = wav_paths
df = df[df["wav_path"].notnull()]
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Created: {OUTPUT_CSV} ({len(df)} valid WAVs)")
