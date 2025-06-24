import pandas as pd

# === Load Extracted Features ===
csv_path = r"C:\Users\Administrator\Desktop\mohit_project\pipe_data\meld_audio_prosody.csv"
df = pd.read_csv(csv_path)

# === Define Feature Groups ===
pitch_feats = [col for col in df.columns if "F0semitone" in col]
loudness_feats = [col for col in df.columns if "loudness" in col]
jitter_feats = [col for col in df.columns if "jitter" in col or "shimmer" in col]
hnr_feats = [col for col in df.columns if "HNR" in col]
formant_feats = [col for col in df.columns if "F1" in col or "F2" in col or "F3" in col]

groups = {
    "Pitch": pitch_feats,
    "Loudness": loudness_feats,
    "JitterShimmer": jitter_feats,
    "HNR": hnr_feats,
    "Formants": formant_feats
}

# === Compute Summary Statistics ===
summary = {}

for group, cols in groups.items():
    subset = df[cols]
    summary[f"{group}_mean"] = subset.mean(axis=1)
    summary[f"{group}_std"] = subset.std(axis=1)
    summary[f"{group}_min"] = subset.min(axis=1)
    summary[f"{group}_max"] = subset.max(axis=1)

summary_df = pd.DataFrame(summary)
summary_df["Emotion"] = df["Emotion"]
summary_df["utt_id"] = df["utt_id"]
summary_df["Utterance"] = df["Utterance"]

# === Save Postprocessed Data ===
out_path = r"C:\Users\Administrator\Desktop\mohit_project\pipe_data\meld_prosody_summary.csv"
summary_df.to_csv(out_path, index=False)

print(f"✅ Postprocessed summary saved to:\n{out_path}")
print(f"📊 Columns: {list(summary_df.columns)}")
