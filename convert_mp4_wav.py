import os
import pandas as pd
from moviepy.editor import AudioFileClip

# === CONFIGURATION ===
INPUT_CSV = "C:\\Users\\Administrator\\Desktop\\mohit_project\\pipe_data\\meld_audio_text.csv"
WAV_DIR = "C:\\Users\\Administrator\\Desktop\\mohit_project\\pipe_data\\wav_outputs"
os.makedirs(WAV_DIR, exist_ok=True)

# Load utterance + audio_path
df = pd.read_csv(INPUT_CSV)
wav_paths = []

for i, row in df.iterrows():
    mp4_path = row["audio_path"]
    wav_path = os.path.join(WAV_DIR, f"utt_{i}.wav")

    # ✅ Skip if already converted
    if os.path.exists(wav_path):
        print(f"⏩ Skipping existing file: utt_{i}.wav")
        wav_paths.append(wav_path)
        continue

    # 🔄 Convert if not found
    try:
        print(f"🎧 Converting: {mp4_path} → utt_{i}.wav")
        clip = AudioFileClip(mp4_path)
        clip.write_audiofile(wav_path, fps=16000, nbytes=2, codec='pcm_s16le')
        wav_paths.append(wav_path)
    except Exception as e:
        print(f"❌ Failed for: {mp4_path}\n{e}")
        wav_paths.append(None)

# Save updated list with .wav paths
df["wav_path"] = wav_paths
df = df[df["wav_path"].notnull()]
df.to_csv("C:\\Users\\Administrator\\Desktop\\mohit_project\\pipe_data\\meld_with_wavs.csv", index=False)
print(f"\n✅ Conversion done. Total usable wavs: {len(df)}")
