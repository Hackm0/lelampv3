import sounddevice as sd
import numpy as np
import whisper
import openai
import subprocess
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
        voices = tts.getProperty('voices')
        # Try each voice until one works
        for voice in voices:
            try:
                tts.setProperty('voice', voice.id)
                return True
            except:
                continue
        return False
    except Exception as e:
        print(f"pyttsx3 initialization failed: {e}")
        return False

use_pyttsx3 = init_pyttsx3()

def speak(text):
    """Speak text aloud using pyttsx3 or fallback to espeak"""
    global tts, use_pyttsx3
    if use_pyttsx3 and tts:
        try:
            tts.say(text)
            tts.runAndWait()
            return
        except Exception as e:
            print(f"pyttsx3 failed, falling back to espeak: {e}")
            use_pyttsx3 = False
    
    # Fallback to espeak (commonly available on Linux/Raspberry Pi)
    try:
        subprocess.run(['espeak', text], check=True)
    except FileNotFoundError:
        # Try espeak-ng if espeak is not available
        try:
            subprocess.run(['espeak-ng', text], check=True)
        except FileNotFoundError:
            print(f"No TTS available. Response: {text}")

# ----------------------------
# Find Seeed audio device
# ----------------------------
def get_seeed_device(output=True):
    devices = sd.query_devices()
    seeed_devices = [(i, d) for i, d in enumerate(devices) if "seeed" in d['name'].lower()]
    for i, d in seeed_devices:
        if output and d['max_output_channels'] > 0:
            return i
        if not output and d['max_input_channels'] > 0:
            return i
    return None

seeed_output = get_seeed_device(output=True)
seeed_input  = get_seeed_device(output=False)

if seeed_output is None or seeed_input is None:
    raise RuntimeError("Seeed audio device not found!")

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
    device=seeed_input
)
sd.wait()

write("temp_input.wav", sample_rate, recording)

# ----------------------------
# Transcribe with Whisper
# ----------------------------
model = whisper.load_model("base")  # options: tiny, small, base, medium, large
result = model.transcribe("temp_input.wav")
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

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "system", "content": prompt}]
)

lamp_response = response.choices[0].message.content.strip()

# ----------------------------
# Speak the response
# ----------------------------
speak(lamp_response)
