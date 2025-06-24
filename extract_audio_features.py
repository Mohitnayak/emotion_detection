import os
import pandas as pd
import subprocess

# === CONFIGURATION ===
INPUT_CSV = r"C:\Users\Administrator\Desktop\mohit_project\pipe_data\meld_with_wavs.csv"
SMILE_PATH = r"C:\Users\Administrator\Desktop\mohit_project\opensmile-3.0.2-windows-x86_64\opensmile-3.0.2-windows-x86_64\bin\SMILExtract.exe"
CONFIG_PATH = r"C:\Users\Administrator\Desktop\mohit_project\opensmile-3.0.2-windows-x86_64\opensmile-3.0.2-windows-x86_64\config\gemaps\v01a\GeMAPSv01a.conf"
OUTPUT_FOLDER = r"C:\Users\Administrator\Desktop\mohit_project\pipe_data\prosody_features"
FINAL_OUTPUT = r"C:\Users\Administrator\Desktop\mohit_project\pipe_data\meld_audio_prosody.csv"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Function to parse ARFF-style CSV from openSMILE ===
def read_opensmile_csv(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    attributes = [line.split()[1] for line in lines if line.startswith("@attribute")]
    data_line = next((line for line in lines if not line.startswith("@") and line.strip()), None)
    if not data_line:
        return pd.DataFrame()
    values = data_line.strip().split(',')
    return pd.DataFrame([values], columns=attributes)

# === Load input CSV ===
df = pd.read_csv(INPUT_CSV)
print(f"📥 Loaded {len(df)} rows")

summary_rows = []

# === Extract + Summarize ===
for idx, row in df.iterrows():
    wav_path = row["wav_path"]
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 2000:
        print(f"⚠️ Skipping invalid file: {wav_path}")
        continue

    out_csv = os.path.join(OUTPUT_FOLDER, f"utt_{idx}.csv")
    cmd = [
        SMILE_PATH,
        "-C", CONFIG_PATH,
        "-I", wav_path,
        "-O", out_csv
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        feats = read_opensmile_csv(out_csv)
        if feats.empty:
            print(f"⚠️ No features for: {wav_path}")
            continue

        feats = feats.apply(pd.to_numeric, errors='coerce')  # convert all to float
        feats["utt_id"] = f"utt_{idx}"
        feats["Emotion"] = row["Emotion"]
        feats["Utterance"] = row["Utterance"]
        summary_rows.append(feats)

    except subprocess.CalledProcessError as e:
        print(f"❌ openSMILE failed for {wav_path}: {e}")

# === Postprocess ===
if not summary_rows:
    print("⚠️ No data to summarize.")
    exit()

combined = pd.concat(summary_rows, ignore_index=True)

groups = {
    "Pitch": [c for c in combined.columns if "F0" in c],
    "Loudness": [c for c in combined.columns if "loudness" in c],
    "JitterShimmer": [c for c in combined.columns if "jitter" in c or "shimmer" in c],
    "HNR": [c for c in combined.columns if "HNR" in c],
    "Formants": [c for c in combined.columns if any(f in c for f in ["F1", "F2", "F3"])],
}

summary_data = []

for _, row in combined.iterrows():
    row_summary = {
        "utt_id": row["utt_id"],
        "Emotion": row["Emotion"],
        "Utterance": row["Utterance"]
    }
    for group, cols in groups.items():
        values = row[cols].dropna()
        if not values.empty:
            row_summary[f"{group}_mean"] = values.mean()
            row_summary[f"{group}_std"] = values.std()
            row_summary[f"{group}_min"] = values.min()
            row_summary[f"{group}_max"] = values.max()
    summary_data.append(row_summary)

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(FINAL_OUTPUT, index=False)
print(f"✅ Saved prosodic summary to:\n{FINAL_OUTPUT}")
