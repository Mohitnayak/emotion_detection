import torch
from transformers import CsmForConditionalGeneration, AutoProcessor, pipeline
from datasets import Audio
import soundfile as sf
import tempfile
import os

# Voice recording
import speech_recognition as sr
import simpleaudio as sa

# Hugging Face model
model_id = "sesame/csm-1b"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
print("🔄 Loading model and processor...")
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map="auto")

# Load speech recognition pipeline for input (can use Whisper if needed)
recognizer = sr.Recognizer()
mic = sr.Microphone()

# Roles for Sesame model
speaker_user = "0"
speaker_bot = "1"
conversation = []

def record_audio():
    with mic as source:
        print("🎙️ Speak now...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    print("🔁 Processing speech...")
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        print("❌ Could not understand audio.")
        return None
    except sr.RequestError:
        print("❌ Speech recognition service error.")
        return None

def play_audio(file_path):
    wave_obj = sa.WaveObject.from_wave_file(file_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()

def save_temp_audio(audio_tensor):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        processor.save_audio(audio_tensor, f.name)
        return f.name

# Main loop
print("🟢 Voice Q&A bot ready. Say something. Say 'exit' to quit.")

while True:
    text_input = record_audio()
    if not text_input:
        continue

    print(f"🧑 You said: {text_input}")
    if text_input.lower() in ["exit", "quit", "stop"]:
        break

    # Add user input to conversation
    conversation.append({
        "role": speaker_user,
        "content": [{"type": "text", "text": text_input}],
    })

    # Prepare model input
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
    ).to(device)

    # Generate spoken output
    with torch.no_grad():
        audio_output = model.generate(**inputs, output_audio=True)

    # Save and play output audio
    output_path = save_temp_audio(audio_output)
    print(f"🤖 Bot responded (audio): {output_path}")
    play_audio(output_path)

    # Append bot's response for context
    conversation.append({
        "role": speaker_bot,
        "content": [{"type": "audio", "path": output_path}]
    })
