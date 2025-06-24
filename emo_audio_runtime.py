import sounddevice as sd
from scipy.io.wavfile import write
import subprocess
import os
import time

# === CONFIGURATION ===
SMILE_PATH = r"C:/Users/Administrator/Desktop/mohit_project/opensmile-3.0.2-windows-x86_64/opensmile-3.0.2-windows-x86_64/bin/SMILExtract.exe"
CONFIG_PATH = r"C:/Users/Administrator/Desktop/mohit_project/opensmile-3.0.2-windows-x86_64/opensmile-3.0.2-windows-x86_64/config/gemaps/v01a/GeMAPSv01a.conf"
OUTPUT_CSV = "runtime_features.csv"
TEMP_WAV = "temp_input.wav"
DURATION = 5  # seconds
SAMPLE_RATE = 16000  # Hz

def record_audio(filename=TEMP_WAV, duration=DURATION, samplerate=SAMPLE_RATE):
    """Record audio from the microphone and save as WAV."""
    print(f"🎙️ Recording {duration} seconds of audio...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    write(filename, samplerate, recording)
    print(f"✅ Audio saved to {filename}")

def run_opensmile(input_wav, config, output_csv, smile_bin=SMILE_PATH):
    """Run openSMILE to extract audio features."""
    print("🔎 Extracting prosodic features with openSMILE...")
    command = [
        smile_bin,
        "-C", config,
        "-I", input_wav,
        "-O", output_csv
    ]
    try:
        subprocess.run(command, check=True)
        print(f"✅ Features extracted and saved to {output_csv}")
    except subprocess.CalledProcessError as e:
        print("❌ Error running openSMILE:", e)

if __name__ == "__main__":
    record_audio()
    run_opensmile(TEMP_WAV, CONFIG_PATH, OUTPUT_CSV)
