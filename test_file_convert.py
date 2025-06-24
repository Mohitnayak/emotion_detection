import os
import pandas as pd
from moviepy.editor import AudioFileClip

# === CONFIGURATION ===
FULL_CSV = "C:\\Users\\Administrator\\Desktop\\mohit_project\\pipe_data\\meld_audio_text.csv"
WAV_DIR = "C:\\Users\\Administrator\\Desktop\\mohit_project\\pipe_data\\test_wav_outputs"
os.makedirs(WAV_DIR, exist_ok=True)

# === Load CSV ===
if not os.path.exists(FULL_CSV):
    raise FileNotFoundError(f"CSV not found: {FULL_CSV}")

print(f"📄 Loaded CSV: {FULL_CSV}")
full_df = pd.read_csv(FULL_CSV)

# === Collect first 4 mp4s with audio ===
valid_rows = []
print(f"🔍 Searching for valid audio files...")
for idx, row in full_df.iterrows():
    mp4_path = row["audio_path"]

    if not os.path.exists(mp4_path):
        print(f"⚠️ Skipping missing: {mp4_path}")
        continue

    try:
        clip = AudioFileClip(mp4_path)
        duration = clip.duration
        if duration and duration > 0:
            print(f"✅ Audio found ({duration:.2f} sec): {mp4_path}")
            valid_rows.append(row)
        else:
            print(f"❌ Empty audio: {mp4_path}")
    except Exception as e:
        print(f"❌ Failed to load {mp4_path}: {e}")
        continue

    if len(valid_rows) >= 4:
        break

# === If no valid files, stop ===
if not valid_rows:
    print("❌ No valid audio files found.")
    exit()

# === Convert to .wav ===
test_df = pd.DataFrame(valid_rows).reset_index(drop=True)
wav_paths = []

print("\n🎧 Starting conversion to WAV...")
for i, row in test_df.iterrows():
    mp4_path = row["audio_path"]
    wav_path = os.path.join(WAV_DIR, f"test_utt_{i}.wav")

    try:
        clip = AudioFileClip(mp4_path)
        clip.write_audiofile(wav_path, fps=16000, nbytes=2, codec='pcm_s16le')
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
            print(f"✅ File written successfully: {wav_path}")
        else:
            print(f"❌ File write failed or empty: {wav_path}")

        wav_paths.append(wav_path)
        print(f"✅ Converted: {wav_path}")
    except Exception as e:
        print(f"❌ Conversion failed for {mp4_path}: {e}")
        wav_paths.append(None)

test_df["wav_path"] = wav_paths
test_df = test_df[test_df["wav_path"].notnull()].reset_index(drop=True)

# === Save output CSV ===
output_csv = "C:\\Users\\Administrator\\Desktop\\mohit_project\\pipe_data\\meld_test_4_wavs.csv"
test_df.to_csv(output_csv, index=False)

print(f"\n✅ Finished. CSV saved to: {output_csv}")
print(f"🟢 Converted {len(test_df)} .mp4 files to .wav successfully.")
