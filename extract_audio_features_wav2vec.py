import os
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from tqdm import tqdm
import argparse

# ========== ARGUMENTS ==========
parser = argparse.ArgumentParser()
parser.add_argument('--split', required=True, choices=['train', 'dev', 'test'])
args = parser.parse_args()
split = args.split

BASE = f"C:/Users/Administrator/Desktop/mohit_project/pipe_data/{split}"
INPUT_CSV = os.path.join(BASE, "meld_with_wavs.csv")
OUTPUT_FOLDER = os.path.join(BASE, "wav2vec_embeddings")
FINAL_OUTPUT = os.path.join(BASE, "meld_audio_wav2vec.csv")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if os.path.exists(FINAL_OUTPUT):
    print(f"⏩ Skipping {split} - Already exists: {FINAL_OUTPUT}")
    exit()

# ========== LOAD MODEL ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
model.eval()

# ========== AUDIO LOADING ==========
def load_audio(path, target_sr=16000):
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform[0]

# ========== FEATURE EXTRACTION ==========
def extract_wav2vec_features(wav_path):
    waveform = load_audio(wav_path)
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs.to(device))
    # Use mean-pooled hidden states as utterance embedding
    embeddings = outputs.last_hidden_state.squeeze(0).mean(dim=0).cpu().numpy()
    return embeddings

# ========== PROCESS DATA ==========
df = pd.read_csv(INPUT_CSV)
feature_rows = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    wav_path = row["wav_path"]
    utt_id = f"utt_{idx}"

    try:
        emb = extract_wav2vec_features(wav_path)
        feature_row = {
            "utt_id": utt_id,
            "Emotion": row["Emotion"],
            "Utterance": row["Utterance"]
        }
        # Add embedding features as separate columns
        for i, v in enumerate(emb):
            feature_row[f"w2v_{i}"] = v
        feature_rows.append(feature_row)
    except Exception as e:
        print(f"❌ Failed on {utt_id}: {e}")
        continue

# ========== SAVE TO CSV ==========
df_out = pd.DataFrame(feature_rows)
df_out.to_csv(FINAL_OUTPUT, index=False)
print(f"✅ wav2vec features saved: {FINAL_OUTPUT}")
