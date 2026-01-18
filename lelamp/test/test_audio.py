import sounddevice as sd
import numpy as np
import whisper
import openai
import subprocess
import warnings
from scipy.io.wavfile import write

# ----------------------------
# Setup TTS safely using espeak as fallback
# ----------------------------
tts = None

def init_pyttsx3():
    """Try to initialize pyttsx3 with a working voice"""
    global tts
    try:
        import pyttsx3
        tts = pyttsx3.init()
        tts.setProperty("rate", 165)
        # Don't try to set voices - use the default
        # Setting voices often fails on Linux with espeak backend
        return True
    except Exception as e:
        print(f"pyttsx3 initialization failed: {e}")
        return False

use_pyttsx3 = init_pyttsx3()

def speak(text):
    """Speak text aloud using espeak directly (most reliable on Linux)"""
    # Use espeak directly - it's the most reliable on Linux/Raspberry Pi
    try:
        subprocess.run(['espeak', '-s', '165', text], check=True, 
                      stderr=subprocess.DEVNULL)
        return
    except FileNotFoundError:
        pass
    
    # Try espeak-ng if espeak is not available
    try:
        subprocess.run(['espeak-ng', '-s', '165', text], check=True,
                      stderr=subprocess.DEVNULL)
        return
    except FileNotFoundError:
        pass
    
    # Last resort: try pyttsx3
    global tts, use_pyttsx3
    if use_pyttsx3 and tts:
        try:
            tts.say(text)
            tts.runAndWait()
            return
        except Exception as e:
            use_pyttsx3 = False
    
    print(f"No TTS available. Response: {text}")

# ----------------------------
# Find audio device (Seeed preferred, fallback to default)
# ----------------------------
def get_audio_device(output=True):
    """Find Seeed audio device or fall back to default system device"""
    devices = sd.query_devices()
    
    # First, try to find Seeed device
    seeed_devices = [(i, d) for i, d in enumerate(devices) if "seeed" in d['name'].lower()]
    for i, d in seeed_devices:
        if output and d['max_output_channels'] > 0:
            print(f"Using Seeed output device: {d['name']}")
            return i
        if not output and d['max_input_channels'] > 0:
            print(f"Using Seeed input device: {d['name']}")
            return i
    
    # Fallback to default system device
    try:
        if output:
            default = sd.default.device[1]  # output device
            if default is not None:
                print(f"Seeed not found, using default output device: {devices[default]['name']}")
                return default
        else:
            default = sd.default.device[0]  # input device
            if default is not None:
                print(f"Seeed not found, using default input device: {devices[default]['name']}")
                return default
    except:
        pass
    
    # Last resort: find any device with input/output channels
    for i, d in enumerate(devices):
        if output and d['max_output_channels'] > 0:
            print(f"Using fallback output device: {d['name']}")
            return i
        if not output and d['max_input_channels'] > 0:
            print(f"Using fallback input device: {d['name']}")
            return i
    
    return None

# List available devices for debugging
print("Available audio devices:")
for i, d in enumerate(sd.query_devices()):
    print(f"  {i}: {d['name']} (in: {d['max_input_channels']}, out: {d['max_output_channels']})")
print()

audio_output = get_audio_device(output=True)
audio_input = get_audio_device(output=False)

if audio_input is None:
    raise RuntimeError("No audio input device found!")
if audio_output is None:
    print("Warning: No audio output device found, TTS may not work")

# ----------------------------
# Record audio
# ----------------------------
duration = 5  # seconds
sample_rate = 44100

print("Listening for 5 seconds...")
recording = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    device=audio_input
)
sd.wait()

write("temp_input.wav", sample_rate, recording)

# ----------------------------
# Transcribe with Whisper
# ----------------------------
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

model = whisper.load_model("base", device="cpu")  # options: tiny, small, base, medium, large
result = model.transcribe("temp_input.wav", fp16=False)
user_text = result["text"].strip()
print("You said:", user_text)

# ----------------------------
# Generate lamp response
# ----------------------------
prompt = f"""
You are a gentle, reflective homework helper lamp. The student said: "{user_text}".
Give a thoughtful, encouraging response. Suggest rethinking strategies or hints, 
but do not give the full answer unless asked explicitly.
"""

# Read API key from local file (not tracked by git)
import os
api_key_path = "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"
with open(api_key_path, "r") as f:
    api_key = f.read().strip()

client = openai.OpenAI(api_key=api_key)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "system", "content": prompt}]
)

lamp_response = response.choices[0].message.content.strip()

# ----------------------------
# Speak the response
# ----------------------------
speak(lamp_response)
