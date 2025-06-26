import subprocess
import sys
import os

SCRIPTS = [
    ("prepare_meld.py", "meld_audio_text.csv"),
    ("convert_mp4_wav.py", "meld_with_wavs.csv"),
    ("extract_audio_features_wav2vec.py", "meld_audio_wav2vec.csv")
]

SPLITS = ["train", "dev", "test"]
BASE = r"C:\Users\Administrator\Desktop\mohit_project\pipe_data"

def run_script(script, split, expected_file):
    output_path = os.path.join(BASE, split, expected_file)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        print(f"⏩ Already done: {script} ({split})")
        return
    print(f"🚀 Running: {script} --split {split}")
    subprocess.run([sys.executable, script, "--split", split], check=True)

if __name__ == "__main__":
    os.chdir(r"C:\Users\Administrator\Desktop\mohit_project")
    for split in SPLITS:
        print(f"\n🔁 Processing split: {split.upper()}")
        for script, output in SCRIPTS:
            run_script(script, split, output)
