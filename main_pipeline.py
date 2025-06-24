import subprocess
import sys
import os

# === Set Your Working Directory Paths ===
PIPE_DATA_DIR = "C:\\Users\\Administrator\\Desktop\\mohit_project"

# === Scripts to Run ===
SCRIPTS = [
    "prepare_meld.py",
    "convert_mp4_wav.py",
    "extract_audio_features.py"
]

def run_script(script_path):
    print(f"\n🚀 Running {script_path}...")
    try:
        subprocess.run([sys.executable, script_path], check=True)
        print(f"✅ Completed: {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {script_path}\n{e}")

if __name__ == "__main__":
    # Set working directory to pipe_data so relative paths work
    os.chdir(PIPE_DATA_DIR)

    for script in SCRIPTS:
        run_script(script)

    print("\n🏁 All pipeline steps completed.")
